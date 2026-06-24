import functools
import itertools
import os
import warnings
import cv2
import skimage
import scipy.fft
import scipy.ndimage
import numpy as np

try:
    import diplib as _dip
except ImportError:  # diplib is an optional speedup; scipy is the fallback
    _dip = None


# Threads scipy.fft uses for the FFTs inside phase_cross_correlation. 1 matches
# scipy's default and ashlar's historical single-threaded behavior; raising it
# (~4 is a good sweet spot on multi-core machines -- diminishing returns beyond,
# and all-cores can regress) speeds the FFT-bound registration with no change in
# results. Override via the ASHLAR_FFT_WORKERS env var or by reassigning here.
FFT_WORKERS = int(os.environ.get("ASHLAR_FFT_WORKERS", "1"))


def _cgroup_cpu_limit():
    """Effective CPU count from a Linux cgroup CPU quota, or None if unlimited.

    Reads cgroup v2 (``cpu.max``) then v1 (``cpu.cfs_quota_us`` /
    ``cpu.cfs_period_us``). ``quota / period`` is the number of cores the
    container is actually allowed, rounded up. Returns None when no quota is set
    or the files are absent (non-Linux, non-containerized).
    """
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return -(-int(quota) // int(period))  # ceil division
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0 and period > 0:
            return -(-quota // period)
    except (OSError, ValueError):
        pass
    return None


def cpu_count():
    """Number of usable CPUs, honoring affinity masks and cgroup CPU quotas.

    Replaces our former reliance on ``joblib.cpu_count()``: respects the
    process's CPU affinity (Linux) and Linux cgroup (v1/v2) quotas so we don't
    over-subscribe threads inside a CPU-limited container. Falls back to the
    logical CPU count where those mechanisms aren't available (macOS, Windows).
    """
    count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            count = len(os.sched_getaffinity(0))
        except OSError:
            pass
    limit = _cgroup_cpu_limit()
    if limit:
        count = min(count, limit)
    return max(1, int(count))


@functools.lru_cache
def _log_kernels(sigma):
    # scipy.ndimage.gaussian_laplace builds separable order-0 and order-2
    # Gaussian-derivative kernels (truncate=4.0). Recover its exact 1D kernels
    # as impulse responses so cv2.sepFilter2D reproduces it bit-for-bit.
    radius = int(4.0 * sigma + 0.5)
    delta = np.zeros(2 * radius + 1)
    delta[radius] = 1.0
    g0 = scipy.ndimage.gaussian_filter1d(delta, sigma, order=0, mode="constant")
    g2 = scipy.ndimage.gaussian_filter1d(delta, sigma, order=2, mode="constant")
    return g0.astype(np.float32), g2.astype(np.float32)


def whiten(img, sigma):
    # Laplacian-of-Gaussian high-pass, evaluated with cv2 (multithreaded SIMD).
    # Output is value-identical to the historical scipy.ndimage implementation
    # (convolve with the uft Laplacian / gaussian_laplace) to float32 precision.
    img = skimage.img_as_float32(img)
    if sigma == 0:
        # The historical kernel is the *negative* discrete Laplacian
        # (skimage.restoration.uft.laplacian); cv2.Laplacian(ksize=1) is the
        # positive Laplacian, so negate to match the sign.
        return -cv2.Laplacian(
            img, cv2.CV_32F, ksize=1, borderType=cv2.BORDER_REFLECT
        )
    # gaussian_laplace = sum over axes of separable order-2 Gaussian-derivative
    # convolutions. cv2.sepFilter2D(kernelX, kernelY) applies a 1-D kernel along
    # x (axis 1) and y (axis 0); the order-2 kernel goes on the differentiated
    # axis, the order-0 (smoothing) kernel on the other.
    g0, g2 = _log_kernels(sigma)
    d_axis0 = cv2.sepFilter2D(img, cv2.CV_32F, g0, g2, borderType=cv2.BORDER_REFLECT)
    d_axis1 = cv2.sepFilter2D(img, cv2.CV_32F, g2, g0, borderType=cv2.BORDER_REFLECT)
    return d_axis0 + d_axis1


@functools.lru_cache
def get_window(shape):
    if isinstance(shape, int) or len(shape) == 1:
        return np.kaiser(shape, beta=2).astype(np.float32)
    else:
        # Build a 2D window by taking the outer product of two 1-D windows.
        wy = np.kaiser(shape[0], beta=2).astype(np.float32)
        wx = np.kaiser(shape[1], beta=2).astype(np.float32)
        window = np.outer(wy, wx)
        return window


def window(img):
    assert img.ndim == 2
    return img * get_window(img.shape)


def _shift_nearest(img, shift):
    # Integer (nearest) shift with zero fill -- equivalent to
    # scipy.ndimage.shift(img, shift, order=0, mode='constant') but a cheap
    # slice/pad instead of the general interpolation routine. Used only to rank
    # the quadrant candidates below, so nearest-pixel placement is sufficient.
    sy = int(np.round(shift[0]))
    sx = int(np.round(shift[1]))
    out = np.zeros_like(img)
    h, w = img.shape
    r0, r1 = max(0, sy), min(h, h + sy)
    c0, c1 = max(0, sx), min(w, w + sx)
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = img[r0 - sy:r1 - sy, c0 - sx:c1 - sx]
    return out


def register(img1, img2, sigma, upsample=10):
    img1w = window(whiten(img1, sigma))
    img2w = window(whiten(img2, sigma))
    with scipy.fft.set_workers(FFT_WORKERS):
        shift = skimage.registration.phase_cross_correlation(
            img1w,
            img2w,
            upsample_factor=upsample,
            normalization=None,
            return_error=False,
        )
    # At this point we may have a shift in the wrong quadrant since the FFT
    # assumes the signal is periodic. We test all four possibilities and return
    # the shift that gives the highest direct correlation (sum of products).
    shape = np.array(img1.shape)
    shift_pos = (shift + shape) % shape
    shift_neg = shift_pos - shape
    shifts = list(itertools.product(*zip(shift_pos, shift_neg)))
    correlations = [
        np.abs(np.sum(img1w * _shift_nearest(img2w, s)))
        for s in shifts
    ]
    idx = np.argmax(correlations)
    shift = shifts[idx]
    correlation = correlations[idx]
    total_amplitude = np.linalg.norm(img1w) * np.linalg.norm(img2w)
    if correlation > 0 and total_amplitude > 0:
        error = -np.log(correlation / total_amplitude)
    else:
        error = np.inf
    return shift, error


def nccw(img1, img2, sigma):
    img1w = whiten(img1, sigma)
    img2w = whiten(img2, sigma)
    correlation = np.abs(np.sum(img1w * img2w))
    total_amplitude = np.linalg.norm(img1w) * np.linalg.norm(img2w)
    if correlation > 0 and total_amplitude > 0:
        error = -np.log(correlation / total_amplitude)
    else:
        error = np.inf
    return error


def crop(img, offset, shape):
    # Note that this only crops to the nearest whole-pixel offset.
    start = offset.round().astype(int)
    end = start + shape
    img = img[start[0]:end[0], start[1]:end[1]]
    return img


# TODO:
# - Deal with ringing from high-frequency elements. The wrapped edges of the
#   image are especially bad, where the wrapping introduces sharp
#   discontinuities. The edge artifacts could be dealt with in several ways
#   (extend the trailing image edge via mirroring, throw away some of the
#   trailing edge of the shifted result) but edges in the "true" image content
#   would require proper pre-filtering. What filter to use, and how to apply it
#   quickly?
# - Can we use real FFT for a ~50% overall speedup? Fourier-space matrices will
#   all be half-size in the last dimension, so FFT is around 50% faster and our
#   fshift calculations will be too.
# - Trailing edge pixels should be zeroed to match the behavior of
#   scipy.ndimage.shift, which we rely on in our maximum-intensity projection.
def fourier_shift(img, shift):
    # Ensure properly aligned complex64 data (fft requires complex to avoid
    # reallocation and copying).
    img = skimage.util.img_as_float32(img)
    img = pyfftw.byte_align(img, dtype=np.complex64)
    # Compute per-axis frequency values according to the Fourier shift theorem.
    # (Read "w" here as "omega".) We pre-multiply as many scalar values as
    # possible on these vectors to avoid operations on the full w matrix below.
    v = np.fft.fftfreq(img.shape[0])
    wy = (2 * np.pi * v * shift[0]).astype(np.float32).reshape(-1, 1)
    u = np.fft.fftfreq(img.shape[1])
    wx = (2 * np.pi * u * shift[1]).astype(np.float32)
    # Add column and row vector to get full expanded matrix of frequencies.
    w = wy + wx
    # We perform an explicit application of Euler's formula with careful
    # management of output arrays to avoid extra memory allocations and copies,
    # squeezing out some speed over the obvious np.exp(-1j*w).
    fshift = np.empty_like(img, dtype=np.complex64)
    np.cos(w, out=fshift.real)
    np.sin(w, out=fshift.imag)
    np.negative(fshift.imag, out=fshift.imag)
    # Perform the FFT, multiply in-place by the shift matrix, then IFFT.
    freq = pyfftw.builders.fft2(img, planner_effort='FFTW_ESTIMATE',
                                avoid_copy=True, auto_align_input=True,
                                auto_contiguous=True)()
    freq *= fshift
    img_s = pyfftw.builders.ifft2(freq, planner_effort='FFTW_ESTIMATE',
                                  avoid_copy=True, auto_align_input=True,
                                  auto_contiguous=True)()
    # Any non-zero imaginary component of the resulting array is due to
    # numerical error, so we can just return the real part.
    # FIXME need to zero out row(s) and column(s) we shifted away from,
    # since at this point we have a cyclic rotation rather than a shift.
    return img_s.real


def register_angle(img1, img2, sigma, upsample=10):
    p1w = whiten(reg_transform_polar(img1), sigma)
    p2w = whiten(reg_transform_polar(img2), sigma)
    with scipy.fft.set_workers(FFT_WORKERS):
        shift = skimage.registration.phase_cross_correlation(
            p1w,
            p2w,
            upsample_factor=upsample,
            normalization=None,
            return_error=False,
        )
    # The output of reg_transform_polar has ambiguous phase (+/- 180 degrees) in
    # the polar axis due to the way it produces a shift-invariant image.  We
    # expect the true angle to be close to zero, so we'll invert anything beyond
    # +/-90 degrees.
    angle = (shift[0] / p1w.shape[0] * 360 + 90) % 180 - 90
    # skimage's warp_polar maps polar angle to the Y-axis in the opposite
    # direction our math expects, so we need to flip the angle's sign.
    angle = -angle
    return angle


def reg_transform_polar(img):
    freq_mag = np.abs(np.fft.fft2(window(img)))
    trans_inv_img = np.fft.fftshift(np.fft.ifft2(freq_mag).real)
    pshape = (360 * 10, round(np.linalg.norm(img.shape) / 2))
    # No window before the polar warp: the 2D Kaiser window is a square (not
    # radially symmetric) envelope that modulates the polar/angular axis we're
    # measuring. Dropping it is neutral-to-better for rotation recovery across
    # test datasets; the radial window below is what actually matters.
    polar_img = skimage.transform.warp_polar(trans_inv_img, output_shape=pshape)
    polar_img = np.clip(polar_img, 0, None) * get_window(polar_img.shape[1])
    return polar_img


def _paste(target, img, pos, func=None):
    """Composite img into target."""
    pos = np.array(pos)
    # Bail out if destination region is out of bounds.
    if np.any(pos >= target.shape[:2]) or np.any(pos + img.shape[:2] < 0):
        return
    pos_f, pos_i = np.modf(pos)
    yi, xi = pos_i.astype('i8')
    # Clip img to the edges of the mosaic.
    if yi < 0:
        img = img[-yi:]
        yi = 0
    if xi < 0:
        img = img[:, -xi:]
        xi = 0
    target_slice = target[yi:yi+img.shape[0], xi:xi+img.shape[1]]
    img = crop_like(img, target_slice)
    # Skip expensive sub-pixel shift if fractional position is zero.
    if pos_f.any():
        if img.ndim == 2:
            img = scipy.ndimage.shift(img, pos_f)
        else:
            for c in range(img.shape[2]):
                img[...,c] = scipy.ndimage.shift(img[...,c], pos_f)
        # For any axis where there is a non-zero subpixel shift, crop out the
        # last row or column of pixels on the "losing" side. These pixels will
        # be darker than normal and will introduce artifacts in most blending
        # modes.
        y1 = None if pos_f[0] <= 0 else 1
        y2 = None if pos_f[0] >= 0 else -1
        x1 = None if pos_f[1] <= 0 else 1
        x2 = None if pos_f[1] >= 0 else -1
        img = img[y1:y2, x1:x2]
        target_slice = target_slice[y1:y2, x1:x2]
        # Exit if image area is zero after subpixel shift.
        if not np.all(img.shape):
            return
    if np.issubdtype(img.dtype, np.floating):
        np.clip(img, 0, 1, img)
    img = dtype_convert(img, target.dtype)
    if func is None:
        target_slice[:] = img
    elif isinstance(func, np.ufunc):
        func(target_slice, img, out=target_slice)
    else:
        target_slice[:] = func(target_slice, img)


def calculate_mosaic_position(position, img_shape, mosaic_shape):
    position = np.array(position)
    position_int = np.floor(position)
    img_shape = np.array(img_shape)
    mosaic_shape = np.array(mosaic_shape)
    _row_start, _col_start = np.ceil(position).astype(int)
    _row_end, _col_end = np.floor(position + img_shape).astype(int)
    # mosaic relevant
    row_start, col_start = np.clip(
        (_row_start, _col_start), (0, 0), mosaic_shape
    ).astype(int)
    row_end, col_end = np.clip(
        (_row_end, _col_end), (0, 0), mosaic_shape
    ).astype(int)
    if (row_start == row_end) or (col_start == col_end):
        return dict(mosaic=None, img=None, translation=None)
    # img relevant
    translation = position - position_int
    row_start_img = row_start - _row_start
    row_end_img = img_shape[0] + row_end - _row_end
    col_start_img = col_start - _col_start
    col_end_img = img_shape[1] + col_end - _col_end
    return {
        'mosaic': [(row_start, row_end), (col_start, col_end)],
        'img': [(row_start_img, row_end_img), (col_start_img, col_end_img)],
        'translation': translation
    }


def subpixel_shift(img, translation):
    """Sub-pixel translate ``img`` by ``translation`` = (row, col), cubic.

    scipy.ndimage.shift(order=3) is a *slow* implementation of cubic B-spline
    interpolation; DIPlib's "bspline" is the same method ~15x faster (matches
    scipy to ~1 count for 99% of pixels), so prefer it when available and fall
    back to scipy otherwise. Returns the input dtype, like scipy.ndimage.shift.
    """
    if _dip is not None:
        # DIPlib dimension 0 is x (numpy's last axis), so pass (col, row).
        shifted = np.asarray(
            _dip.Shift(
                _dip.Image(np.ascontiguousarray(img)),
                [float(translation[1]), float(translation[0])],
                "bspline",
            )
        )
        return shifted.astype(img.dtype, copy=False)
    return scipy.ndimage.shift(img, translation)


def paste(target, img, pos, func=None, decimals=1):
    positions = calculate_mosaic_position(
        # Round the pos to one decimal point because the subpixel shifts are
        # calculated by 10x upsampling and thus only accurate to that level.
        np.around(pos, decimals=decimals), img.shape, target.shape
    )
    pos_mosaic = positions['mosaic']
    pos_img = positions['img']
    translation = positions['translation']
    if pos_mosaic is None:
        return
    img = img[slice(*pos_img[0]), slice(*pos_img[1])]
    if not np.all(translation == 0):
        img = subpixel_shift(img, translation)
        if translation[0] != 0: img = img[1:]
        if translation[1] != 0: img = img[:, 1:]
    target_slice = target[slice(*pos_mosaic[0]), slice(*pos_mosaic[1])]
    assert target_slice.shape == img.shape

    if np.issubdtype(img.dtype, np.floating):
        np.clip(img, 0, 1, img)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*scikit-image 1\.0", FutureWarning)
        img = skimage.util.dtype.convert(img, target.dtype)
    if func is None:
        target_slice[:] = img
    elif isinstance(func, np.ufunc):
        func(target_slice, img, out=target_slice)
    else:
        target_slice[:] = func(target_slice, img)
    target[slice(*pos_mosaic[0]), slice(*pos_mosaic[1])] = target_slice


def pastefunc_blend(target, img):
    """Linear blend based on distance to unfilled space in target."""
    # This should catch actual holes but not the actual unfilled space.
    # FIXME Should generate mask from tile boundaries instead.
    hole_threshold = np.mean(target.shape)
    mask = skimage.morphology.remove_small_holes(target != 0, hole_threshold)
    dist = scipy.ndimage.distance_transform_cdt(mask)
    dmax = dist.max()
    if dmax == 0:
        alpha = 0
    else:
        alpha = dist / dmax
        # Keep target pixel values where img has value 0 (with a 1-pixel
        # dilation to clean up the edge). This is a temporary hack to support
        # image corrections that leave regions of zero pixels around the edge of
        # img, such as barrel correction and rotation.
        # FIXME Should compute the geometry of the source image mask more
        # deliberately and precisely.
        alpha[skimage.morphology.binary_dilation(img == 0)] = 1
    return target * alpha + img * (1 - alpha)


def crop_like(img, target):
    if (img.shape[0] > target.shape[0]):
        img = img[:target.shape[0], :]
    if (img.shape[1] > target.shape[1]):
        img = img[:, :target.shape[1]]
    return img


def dtype_convert(img, dtype):
    """Convert an image to the requested data-type.

    This is just a wrapper around skimage.util.dtype.convert that silences its
    FutureWarning, as Ashlar pins skimage to a version before that planned
    deprecation.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*scikit-image 1\.0", FutureWarning)
        return skimage.util.dtype.convert(img, dtype)


def imsave(fname, arr, **kwargs):
    """Save an image to file.

    This is a wrapper around skimage.io.imsave to force check_contrast=False
    since the contrast check kills us on the huge images we create.

    """

    if "check_contrast" in kwargs:
        warnings.warn("ignoring check_contrast argument -- forcing to False")
    kwargs["check_contrast"] = False

    # We use scikit-image's vendored copy of tifffile directly rather than allow
    # scikit-image to optimistically use a separately-installed copy of tifffile
    # due to bugs and API inconsistencies in the latest pypi-hosted version:
    # * Use of "centimeter" for resolution units instead of "cm"
    # * A bug in writing single-tile planes -- issue #3 on GitHub
    # FIXME Once scikit-image un-vendors tifffile (#4235) AND tifffile fixes #3
    # we can remove this block and use `skimage.io.imsave` directly again. Or we
    # might just want to switch to tifffile.imsave.
    del kwargs["check_contrast"]
    import skimage.external.tifffile
    skimage.external.tifffile.imsave(fname, arr, **kwargs)


def visualize_image(img):
    import skimage.exposure

    img = skimage.exposure.rescale_intensity(img, out_range=(0.0, 1.0))
    return skimage.exposure.equalize_adapthist(img)


def cv2_downscale_local_mean(img, factor):
    """Downsample by integer ``factor`` via area averaging (cv2).

    Equivalent to skimage's downscale_local_mean for interior pixels but much
    faster. The edge is replicated up to a multiple of ``factor`` before an
    exact-factor INTER_AREA resize, which (a) keeps the ceil(dim/factor) output
    size the pyramid level shapes expect and (b) avoids the edge darkening that
    downscale_local_mean's zero-padding produces on odd dimensions.
    """
    assert img.ndim in [2, 3]
    img = np.asarray(img)
    axis_moved = False
    channel_ax = np.argmin(img.shape)
    if (img.ndim == 3) & (channel_ax != 2):
        img = np.moveaxis(img, channel_ax, 2)
        axis_moved = True
    h, w = img.shape[:2]
    ph, pw = (-h) % factor, (-w) % factor
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    simg = cv2.resize(
        img, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA
    )
    if axis_moved:
        simg = np.moveaxis(simg, 2, channel_ax)
    return simg
