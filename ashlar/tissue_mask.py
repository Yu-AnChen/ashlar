import numpy as np
import skimage.filters
import skimage.transform
import tqdm
import zarr
# from compress_bg.main import entropy_img_to_masks, local_entropy, plot_tissue_mask

from . import thumbnail


def make_mask(
    thumbnail: np.ndarray,
    entropy_kernel_size: int = 5,
    dilation_radius: int = 5,
    plot: bool = False,
    level_center: float = 0.5,
    level_adjust: int = 0,
    figure_title: str = "",
):
    _tt = np.array(thumbnail)
    np.clip(_tt, np.percentile(_tt[_tt > 0], 0).astype(_tt.dtype), None, out=_tt)
    entropy_thumbnail = local_entropy(np.log1p(_tt), kernel_size=entropy_kernel_size)
    thumbnail = np.log1p(thumbnail)

    erange = np.ptp(entropy_thumbnail)
    _threshold = skimage.filters.threshold_otsu(entropy_thumbnail)
    _max = entropy_thumbnail.max() - 0.1 * erange
    _min = entropy_thumbnail.min() + 0.1 * erange
    _threshold = np.clip(_threshold, _min, _max)

    level_center = np.clip(
        level_center, (_min - _threshold) / erange, (_max - _threshold) / erange
    )

    # forcing threshold to be within 10th and 90th percent of the range
    _threshold += level_center * erange

    thresholds = np.concatenate(
        [
            np.linspace(entropy_thumbnail.min(), _threshold, 4)[1:],
            np.linspace(_threshold, entropy_thumbnail.max(), 4)[1:-1],
        ]
    )
    level_adjusts = np.arange(-2, 3, 1)
    masks = entropy_img_to_masks(
        thumbnail, entropy_thumbnail, thresholds, dilation_radius
    )
    mask = masks[list(level_adjusts).index(level_adjust)]

    if plot:
        fig = plot_tissue_mask(
            thumbnail,
            entropy_thumbnail,
            masks,
            thresholds,
            list(level_adjusts).index(level_adjust),
        )
        fig.suptitle(figure_title)

    return mask


def make_reader_mask(reader, thumbnail_px_size=20, **kwargs):
    Affine = skimage.transform.AffineTransform

    metadata = reader.metadata
    px_size = metadata.pixel_size
    factor = px_size / thumbnail_px_size
    timg = thumbnail.make_thumbnail(reader, scale=factor)
    mask = make_mask(timg, **kwargs)

    positions = factor * (metadata.positions - metadata.positions.min(axis=0))

    tile_mask = zarr.group()
    for ii, pp in enumerate(tqdm.tqdm(positions, desc="Making tissue mask")):
        h, w = metadata.tile_size(ii)
        out_shape = np.ceil(factor * metadata.tile_size(ii)).astype("int")
        tform = Affine(translation=pp[::-1])
        tmask = skimage.transform.warp(mask, tform, output_shape=out_shape, order=0)
        tile_mask[str(ii)] = skimage.transform.rescale(tmask, 1 / factor, order=0)[
            :h, :w
        ]
    return tile_mask


import matplotlib.pyplot as plt
import scipy.ndimage
import skimage.exposure
import skimage.filters
import skimage.morphology

INTENSITY_RESCALE_MAX = 1023
INTENSITY_THRESHOLD_P = 25

RM_SMALL_OBJ_FACTOR = 4

PLOT_PAD_FACTOR = 0.01
PLOT_ASPECT_RATIO_THRESHOLD = 1.2
PLOT_FIGSIZE_SCALE = 1.5
PLOT_DPI = 144


def local_entropy(img, kernel_size=5):
    img = skimage.exposure.rescale_intensity(
        img, out_range=(0, INTENSITY_RESCALE_MAX)
    ).astype(np.uint16)
    return skimage.filters.rank.entropy(img, np.ones((kernel_size, kernel_size)))


def entropy_img_to_masks(
    img, entropy_img, thresholds, dilation_radius, img_is_dark_background=True
):
    if not img_is_dark_background:
        img = -1.0 * img
    masks = np.full((len(thresholds), *entropy_img.shape), fill_value=False, dtype=bool)
    footprint = skimage.morphology.disk(radius=dilation_radius)
    for idx, tt in enumerate(thresholds):
        mask = entropy_img > tt
        mask |= img >= np.percentile(img[mask], INTENSITY_THRESHOLD_P)
        skimage.morphology.dilation(mask, footprint=footprint, out=mask)
        scipy.ndimage.binary_fill_holes(mask, output=mask)
        skimage.morphology.remove_small_objects(
            mask, RM_SMALL_OBJ_FACTOR * footprint.sum(), out=mask
        )

        masks[idx] = mask
    return masks


def plot_tissue_mask(img, entropy_img, masks, thresholds, selected_mask_idx):
    # set contrast min to min value that is not 0
    vimg = skimage.exposure.rescale_intensity(
        img,
        in_range=(img[img > 0].min(), img.max()),
        out_range="float",
    )
    # pad images for mask outline drawing
    pad_size = np.ceil(np.max(img.shape) * PLOT_PAD_FACTOR).astype("int")
    vimg = np.pad(vimg, pad_size, constant_values=0)
    ventropy = np.pad(entropy_img, pad_size, constant_values=entropy_img.min())
    vmasks = np.pad(
        masks,
        [(0, 0), (pad_size, pad_size), (pad_size, pad_size)],
        constant_values=False,
    )
    vmask = vmasks[selected_mask_idx]

    subplot_shape = (2, 1)
    if np.divide(*img.shape) > PLOT_ASPECT_RATIO_THRESHOLD:
        subplot_shape = (1, 2)

    fig, axs = plt.subplots(*subplot_shape, sharex=True, sharey=True)

    axs[0].imshow(vimg, cmap="cividis")
    axs[0].contour(vmask, levels=[0.5], colors=["w"], linewidths=1)
    # axs[1].imshow(ventropy, cmap="cividis", interpolation="none")

    _plot_entropy_mask_levels(
        ventropy, vmasks, thresholds, selected_mask_idx, img=vimg, ax=axs[1]
    )
    return fig


def _plot_entropy_mask_levels(
    entropy_img, masks, thresholds, selected_mask_idx=None, img=None, ax=None
):
    import itertools

    import matplotlib.cm
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    assert len(masks) == len(thresholds) == 5

    levels = np.arange(7) - 0.5
    tick_labels = np.arange(5) - 2

    if selected_mask_idx is None:
        selected_mask_idx = 2
    assert selected_mask_idx in range(5)

    if ax is None:
        _, ax = plt.subplots()
    fig = ax.get_figure()

    if img is None:
        img = [[0]]
    ax.imshow(np.log1p(img), cmap="gray")

    ax.contourf(masks.sum(axis=0), cmap="coolwarm_r", levels=levels, alpha=0.75)
    ax.contour(masks[selected_mask_idx], levels=[0.5], colors=["w"], linewidths=1)
    axins = inset_axes(
        ax,
        width=0.1,  # width: .1 inch
        height="100%",  # height: 100%
        loc="lower left",
        bbox_to_anchor=(1.05, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    colors = matplotlib.cm.coolwarm_r(np.linspace(0, 1, 6), alpha=0.75)
    yys = [entropy_img.min()] + list(thresholds) + [entropy_img.max()]

    for pp, cc in zip(itertools.pairwise(yys), colors):
        axins.fill_between([0, 1], *pp, color=cc, step="post")

    axins.set_xlim(0, 1)
    axins.set_ylim(yys[0], yys[-1])
    axins.axes.yaxis.tick_right()
    axins.set_yticks(yys[1:-1], labels=tick_labels)
    axins.set_xticks([])
    axins.axhline(thresholds[selected_mask_idx], color="w", linewidth=3)

    return fig
