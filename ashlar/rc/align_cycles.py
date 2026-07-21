import numpy as np
import scipy.spatial
import skimage.transform

from .. import reg, thumbnail, utils
from . import preproc_reader, rotation_utils


def correct_position(layer_aligner, angle):
    cycle_tform = thumbnail.align_cycles(
        layer_aligner.reference_aligner.reader,
        layer_aligner.reader,
        scale=layer_aligner.reference_aligner.reader.thumbnail_scale,
        angle=angle
    )
    layer_aligner.cycle_tform = cycle_tform
    layer_aligner.corrected_nominal_positions = (
        np.fliplr(cycle_tform(np.fliplr(layer_aligner.reader.metadata.positions)))
    )


def set_pairs(layer_aligner):
    reference_positions = layer_aligner.reference_aligner.metadata.positions
    dist = scipy.spatial.distance.cdist(reference_positions,
                                        layer_aligner.corrected_nominal_positions)
    layer_aligner.reference_idx = np.argmin(dist, 0)
    layer_aligner.reference_positions = reference_positions[layer_aligner.reference_idx]
    layer_aligner.reference_aligner_positions = layer_aligner.reference_aligner.positions[layer_aligner.reference_idx]


def register(layer_aligner, t):
    its, ref_img, img = layer_aligner.overlap(t)
    if np.any(np.array(its.shape) == 0):
        return np.nan
    return utils.register_angle(ref_img, img, layer_aligner.filter_sigma)


def tile_edge_score(reader, channel):
    score = np.zeros(reader.metadata.num_images)
    for i in range(reader.metadata.num_images):
        score[i] = rotation_utils.var_of_laplacian(reader.read(i, channel))
    return score


def refine_angle(layer_aligner, rank=None, top_k=None, sigma=1):
    tiles = np.arange(layer_aligner.reader.metadata.num_images)
    if rank is not None:
        tiles = tiles[rank]
    if top_k is None:
        top_k = len(tiles)
    top_k = min(top_k, len(tiles))
    tiles = tiles[:top_k]
    from concurrent.futures import ThreadPoolExecutor
    img_pairs = list(filter(lambda x: x[0].size > 0, [
        layer_aligner.overlap(t)[1:3]
        for t in tiles
    ]))
    # Rotation estimation uses fixed light smoothing (`sigma`), decoupled from
    # the translation filter_sigma. Ground-truth sweeps (injected rotations on
    # real overlaps, 3 datasets) showed: sigma=0 fails entirely (the sharp
    # Laplacian can't resolve rotation); sigma>=2 over-smooths and biases the
    # small (<1 deg) rotations that actually occur in refinement. sigma=1 gives
    # the least-biased per-tile median there -- its larger per-tile spread is
    # zero-mean and averages out in the nanmedian below.
    # register_angle is numpy/FFT-heavy and releases the GIL, so threads give
    # real parallelism here without forking worker processes (which previously,
    # via joblib's loky backend, left a ~4 GB idle pool resident for the rest of
    # the run). The overlap arrays are already materialized above, so the
    # threaded calls share read-only inputs with no cross-thread state.
    if not img_pairs:
        return np.nan
    # Each register_angle holds a large FFT/warp_polar working set, so cap the
    # pool: a handful of threads saturates these ~30 tiny tasks without the RAM
    # spike that cpu_count concurrent FFTs would cause.
    n_workers = min(len(img_pairs), utils.cpu_count(), 4)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        angles = list(executor.map(
            lambda pair: utils.register_angle(pair[0], pair[1], sigma), img_pairs
        ))
    return np.nanmedian(angles)


def _build_rotated_reader(c2r, c1e, angle, scale):
    """Build a center-cropped, rotated PreprocReader for ``c2r`` at ``angle``.

    Returns the reader (with cycle-corrected tile positions and a matching
    rotated thumbnail) and the corrected nominal positions.
    """
    ori_shape = c2r.metadata.size
    rotation_slice = rotation_utils.compute_slice(ori_shape, angle)
    crop_shape = np.zeros(ori_shape)[rotation_slice].shape

    cycle_tform = thumbnail.align_cycles(
        c1e.reader, c2r, scale=scale, angle=angle,
    )
    corrected_positions = (
        np.fliplr(cycle_tform(np.fliplr(c2r.metadata.centers)))
    )
    corrected_positions += np.multiply(-0.5, crop_shape)

    c2rr = preproc_reader.PreprocReader(
        c2r, angle=angle, center_crop_shape=crop_shape
    )
    c2rr.metadata._positions = corrected_positions
    c2rr.metadata.extent = (
        c2rr.metadata.positions.max(axis=0) + c2rr.metadata.size - c2rr.metadata.origin
    )

    rthumbnail = skimage.transform.rotate(c2r.thumbnail, angle=angle, resize=True)
    t_offset = .5 * np.subtract(rthumbnail.shape, scale * c2rr.metadata.extent)
    ro, co = np.around(t_offset).astype(int)
    slice_r = slice(None) if ro == 0 else slice(ro, -ro)
    slice_c = slice(None) if co == 0 else slice(co, -co)
    c2rr.thumbnail = rthumbnail[slice_r, slice_c]

    return c2rr, corrected_positions


def _make_layer_aligner(c2rr, c1e, corrected_positions, channel, max_shift, filter_sigma):
    la = reg.LayerAligner(
        c2rr, c1e, verbose=True,
        channel=channel, max_shift=max_shift, filter_sigma=filter_sigma,
    )
    la.corrected_nominal_positions = corrected_positions
    set_pairs(la)
    return la


def process_rotated_reader(
    reader, edge_aligner,
    channel=None, max_shift=15,
    filter_sigma=0.0, corner_tol=0.1,
):
    c2r = reader
    c1e = edge_aligner
    c21l = reg.LayerAligner(
        c2r, c1e, verbose=True,
        channel=channel, max_shift=max_shift,
        filter_sigma=filter_sigma
    )
    c21l.make_thumbnail()

    SCALE = c1e.reader.thumbnail_scale

    correct_position(c21l, angle=None)
    set_pairs(c21l)

    def _finalize_unrotated():
        c21l.register_all()
        c21l.calculate_positions()
        c21l.mosaic_shape = c1e.mosaic_shape
        return c21l

    if c21l.cycle_tform.rotation == 0:
        return _finalize_unrotated()

    edgy_scores = tile_edge_score(c2r, c21l.channel)
    rank = np.argsort(edgy_scores)[::-1]
    angle = refine_angle(c21l, rank=rank, top_k=30)
    print(f'\r    refined cycle rotation = {angle:.4f} degrees', flush=True)

    # Convert the corner-displacement budget (px) to an angle threshold: a
    # rotation about a tile center moves the tile's farthest corner by
    # ``half_diag * radians``. A refined angle below this (or NaN, when
    # refinement found no usable overlaps) is not worth the rotated-reader
    # resampling -- fall back to the unrotated positions.
    half_diag = 0.5 * float(np.hypot(*c2r.metadata.size))
    angle_tol = np.degrees(corner_tol / half_diag)
    if not np.isfinite(angle) or abs(angle) < angle_tol:
        return _finalize_unrotated()

    c2rr, corrected_positions = _build_rotated_reader(c2r, c1e, angle, SCALE)
    c21lr = _make_layer_aligner(
        c2rr, c1e, corrected_positions, channel, max_shift, filter_sigma
    )
    c21lr.register_all()
    c21lr.calculate_positions()
    c21lr.mosaic_shape = c1e.mosaic_shape

    return c21lr
