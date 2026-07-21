"""Align a later-cycle FOV image to an already-stitched reference mosaic.

The main ``rc`` flow (``align_cycles.process_rotated_reader``) aligns a moving
cycle to a reference that *this pipeline stitched* -- a live ``EdgeAligner`` with
per-FOV tiles, per-tile geometry, and a fitted linear model. This module handles
the other case: the reference is a single already-stitched mosaic image (e.g. the
level-0 array of a ``palom.reader.OmePyramidReader`` pyramid), which has no FOV
tiles and no per-tile geometry.

Strategy: synthesize reference "tiles" by cutting the stitched mosaic into a grid
co-located with the moving cycle's tiles, wrap that in a stand-in ``EdgeAligner``,
then reuse the standard rotation-refine + layer-register flow. After coarse
thumbnail alignment the reference tiles are re-cut at the moving cycle's corrected
positions, so each moving tile pairs 1:1 with the reference crop underneath it and
per-tile registration becomes a plain cross-cycle phase correlation. The reference
frame is the stitched image's own pixel grid (identity linear model, zero origin),
so the assembled output overlays the reference pixel-for-pixel.

Because each moving tile is pinned to solid reference content, placement is
constrained per tissue piece rather than by one global model: disconnected pieces
(with blank gaps between them) can sit at different offsets -- which a single
global constraint would wrongly reject for the minority piece. ``StitchedLayerAligner``
partitions the tiles into pieces and constrains each independently. Inspect a
result with ``summarize_components`` (per-piece offsets and discards) and
``plot_components`` (the piece-connectivity graph over the mosaic).

Usage::

    import palom
    from ashlar import reg
    from ashlar.rc import align_to_stitched

    ref = palom.reader.OmePyramidReader("cycle1.ashlar.ome.tif").pyramid[0][0]
    moving = reg.BioformatsReader("cycle2.rcpnl")
    aligner = align_to_stitched.align_to_stitched(ref, moving, channel=0)
    align_to_stitched.summarize_components(aligner)

    mosaic = reg.Mosaic(aligner, aligner.mosaic_shape, channels=[0], verbose=True)
    reg.PyramidWriter([mosaic], "cycle2-to-stitched.ome.tif", verbose=True).run()
"""

import networkx as nx
import numpy as np
import scipy.spatial.distance
import sklearn.linear_model

from .. import reg, thumbnail
from . import align_cycles

__all__ = [
    "StitchedMetadata",
    "StitchedReader",
    "StitchedLayerAligner",
    "align_to_stitched",
    "summarize_components",
    "plot_components",
    "plot_layer_quality_with_graph",
]


class StitchedMetadata(reg.Metadata):
    """Metadata for reference tiles cut from an already-stitched mosaic.

    Presents a single-channel stitched image as a set of same-size tiles at
    arbitrary ``positions``. ``mosaic_shape`` is the full stitched image shape,
    which the caller uses as the output canvas so the result overlays the
    reference.
    """

    def __init__(self, image, positions, tile_size, pixel_size=None):
        assert image.ndim == 2
        assert len(tile_size) == 2
        self.image = image
        self._positions = np.asarray(positions, dtype=float)
        self._tile_size = np.asarray(tile_size, dtype=int)
        self._pixel_size = pixel_size

    @property
    def _num_images(self):
        return len(self._positions)

    @property
    def num_channels(self):
        return 1

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def pixel_dtype(self):
        return self.image.dtype

    @property
    def mosaic_shape(self):
        return tuple(self.image.shape)

    @property
    def size(self):
        return self._tile_size

    def tile_size(self, i):
        return self._tile_size


class StitchedReader(reg.Reader):
    """Serve tiles cut from an already-stitched mosaic image.

    ``read(series, c)`` returns the crop of the stitched image at
    ``metadata.positions[series]`` with shape ``metadata.size``, zero-padded
    where the crop extends past the image bounds. The channel argument is
    ignored: ``image`` is already a single channel of the stitched mosaic.
    """

    def __init__(self, image, positions, tile_size, pixel_size=None):
        self.metadata = StitchedMetadata(
            image, np.round(positions), tile_size, pixel_size
        )

    @property
    def image(self):
        return self.metadata.image

    def read(self, series, c):
        r, col = np.round(self.metadata.positions[series]).astype(int)
        h, w = self.metadata.size
        rows, cols = self.metadata.mosaic_shape
        dtype = self.metadata.pixel_dtype
        # Clamp the requested window to the image, then pad the clamped-off
        # margin back with zeros. Clamping (rather than slicing with a possibly
        # negative start) is what keeps a tile straddling the top/left edge from
        # silently wrapping to rows/cols at the far side of the image.
        r0, r1 = max(r, 0), min(r + h, rows)
        c0, c1 = max(col, 0), min(col + w, cols)
        if r1 <= r0 or c1 <= c0:
            return np.zeros((h, w), dtype=dtype)
        img = np.asarray(self.image[r0:r1, c0:c1])
        if img.shape != (h, w):
            pad_r = (r0 - r, h - (r1 - r))
            pad_c = (c0 - col, w - (c1 - col))
            img = np.pad(img, (pad_r, pad_c))
        return img.astype(dtype, copy=False)


def _build_stitched_reference(image, moving_reader, pixel_size=None):
    """Wrap ``image`` as a stand-in ``EdgeAligner`` for the reference cycle.

    The synthetic reference tiles are initially placed at the moving cycle's
    nominal grid (shifted to the mosaic origin); coarse alignment corrects the
    residual cycle offset before the tiles are re-cut.
    """
    if pixel_size is None:
        pixel_size = moving_reader.metadata.pixel_size
    reader = StitchedReader(
        image,
        positions=moving_reader.metadata.positions - moving_reader.metadata.origin,
        tile_size=moving_reader.metadata.size,
        pixel_size=pixel_size,
    )
    c1e = reg.EdgeAligner(reader, channel=0)
    # Drop the CachingReader that EdgeAligner wraps around `reader`: this flow
    # re-cuts the synthetic tile positions after coarse alignment, which would
    # leave a series-indexed tile cache returning stale crops.
    c1e.reader = reader
    c1e.make_thumbnail()
    return c1e


def _refresh_thumbnail(reader, channel):
    """Rebuild ``reader.thumbnail`` in place after its tile positions moved.

    ``thumbnail.make_thumbnail`` lays tiles out at ``positions - origin``, so
    pixel (0, 0) of the result means "stage coordinate ``metadata.origin``".
    ``Metadata.origin`` is derived live from ``positions``, so moving the
    positions silently breaks that contract: the cached array still holds
    content anchored at the old origin while every consumer resolves it against
    the new one. Two consumers care. ``thumbnail.align_cycles`` bakes
    ``reader1.metadata.origin`` into the transform it returns, and since the
    shifts it measures are already correct for the old anchor, the trailing
    origin term becomes a pure displacement of the full origin delta -- here the
    coarse cycle shift, well beyond ``max_shift``. ``reg.draw_mosaic_image``
    stretches the thumbnail across the current position extent, so a QC plot
    drawn over a stale one shows tile boxes and arrows against a background
    offset by that same delta.

    Rebuilds at the reader's existing ``thumbnail_scale`` rather than through
    ``EdgeAligner.make_thumbnail``, which recomputes the scale from the new
    positions; that would desync it from the moving cycle's thumbnail (built
    once at the original scale) and from the ``scale`` the caller captured, and
    the two thumbnails must share a scale to be correlated against each other.
    """
    if getattr(reader, "thumbnail", None) is None:
        return
    reader.thumbnail = thumbnail.make_thumbnail(
        reader, channel=channel, scale=reader.thumbnail_scale
    )


def _set_reference_geometry(c1e, positions):
    """Place the synthetic reference tiles at ``positions`` (co-located with the
    moving tiles) and fit an identity linear model.

    ``positions`` are already in the stitched image's pixel frame, so the model
    is identity and the origin is zero -- ``constrain_positions`` then predicts
    each tile at its own corrected nominal position (no reference-model
    distortion, which a real stitched mosaic does not have).
    """
    positions = np.round(positions)
    c1e.reader.metadata._positions = positions
    c1e.positions = positions
    c1e.lr = sklearn.linear_model.LinearRegression()
    c1e.lr.fit(c1e.metadata.positions, c1e.positions)
    c1e.origin = np.array([0, 0])
    c1e.centers = c1e.positions + c1e.metadata.size / 2
    _refresh_thumbnail(c1e.reader, c1e.channel)


class StitchedLayerAligner(reg.LayerAligner):
    """A ``LayerAligner`` that constrains tile positions per tissue piece.

    Against a multi-piece stitched reference each disconnected piece can carry
    its own rigid offset -- different stitchers space the pieces apart
    differently, and with blank gaps there is no image content tying them to a
    single global model. The stock ``constrain_positions`` uses one global
    offset (``median`` of all shifts), so a minority piece whose true offset
    differs from that consensus by more than ``max_shift`` has its correct,
    measured shifts flagged as outliers and discarded.

    This override partitions the tiles into pieces -- connected components of the
    (reused) spatial neighbor graph, keeping only edges between non-discarded
    tiles whose measured shifts agree within ``component_tol`` -- and applies the
    stock offset + residual-rotation constraint independently within each piece.
    With a single piece it reduces exactly to the base behavior.

    ``component_tol`` defaults to ``max_shift_pixels``: a piece only needs to be
    split off when its offset exceeds what the per-piece discard would tolerate
    anyway, so the two thresholds share a scale (see notes in the rc design).

    A component needs at least ``_min_component_size`` mutually-agreeing tiles to
    be *trusted* with its own offset. A lone tile that agrees with no neighbor
    (a singleton) has no local consensus to corroborate it, so it is not trusted
    on its own; it is validated against the nearest trusted piece and kept only
    if it agrees, else discarded -- matching the stock outlier rejection rather
    than blindly trusting an unsupported measurement.
    """

    # A component with fewer than this many mutually-agreeing tiles is not
    # trusted with its own offset (see class docstring). Two independent
    # registrations that agree are the minimum corroboration; a lone tile is not.
    _min_component_size = 2

    def __init__(self, *args, component_tol=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.component_tol = component_tol

    def _component_labels(self, base_discard):
        """Label each tile with a tissue-piece id (-1 if base-discarded).

        Reuses ``self.neighbors_graph`` (EdgeAligner's geometry-only spatial
        adjacency) and keeps an edge only when both endpoints are trustworthy
        and their measured shifts agree within the tolerance. Blank gaps fall
        out on their own: gap tiles are base-discarded and cannot carry an edge.
        Connected components are the candidate tissue pieces; whether each is
        trusted (needs ``_min_component_size`` tiles) is decided by the caller.
        The filtered graph is stored as ``self.component_graph`` for QC plots.
        """
        tol = self.component_tol
        if tol is None:
            tol = self.max_shift_pixels
        n = self.metadata.num_images
        graph = nx.Graph()
        graph.add_nodes_from(np.nonzero(~base_discard)[0].tolist())
        edges = np.array(list(self.neighbors_graph.edges), dtype=int)
        if len(edges):
            i, j = edges[:, 0], edges[:, 1]
            keep = (
                ~base_discard[i]
                & ~base_discard[j]
                & (np.linalg.norm(self.shifts[i] - self.shifts[j], axis=1) <= tol)
            )
            graph.add_edges_from(edges[keep].tolist())
        self.component_graph = graph
        labels = np.full(n, -1, dtype=int)
        for cid, component in enumerate(nx.connected_components(graph)):
            labels[list(component)] = cid
        return labels

    def _constrain_component(self, members, predictions):
        """Stock offset + residual-rotation constraint, restricted to one piece.

        Returns ``(model, extremes, offset, rotation)`` for ``members``: the
        placement for each member, the outlier mask, the piece's median
        translation, and its residual rotation.
        """
        shifts_m = self.shifts[members]
        pos_m = self.positions[members]
        pred_m = predictions[members]

        offset = np.nan_to_num(np.median(shifts_m, axis=0))
        extremes = (
            np.linalg.norm(pos_m - pred_m - offset, axis=1) > self.max_shift_pixels
        )
        model_m = pred_m + offset
        rotation = 0.0
        if (~extremes).sum() >= self._min_kept_for_rotation:
            keep = ~extremes
            Mk, tk = reg._fit_similarity(pred_m[keep], pos_m[keep])
            rot = float(np.degrees(np.arctan2(Mk[1, 0], Mk[0, 0])))
            if abs(rot) >= self._rotation_tol:
                model_m = pred_m @ Mk.T + tk
                # Re-admit tiles flagged only because of the piece's rotation
                # (tangential shift ~ angle * radius can exceed max_shift at the
                # edges), then re-fit on the full lever arm.
                extremes = extremes & (
                    np.linalg.norm(pos_m - model_m, axis=1) > self.max_shift_pixels
                )
                keep = ~extremes
                M, t = reg._fit_similarity(pred_m[keep], pos_m[keep])
                rotation = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
                model_m = pred_m @ M.T + t
        return model_m, extremes, offset, rotation

    def _resolve_loose(self, loose, base_discard, labels, predictions, model, discard):
        """Place tiles that lack a trusted local consensus.

        ``loose`` covers base-discarded tiles (no measurement) and singletons (a
        measurement, but agreeing with no neighbor). Each is placed from the
        nearest trusted piece's *translation* offset -- not its full affine,
        which would extrapolate that piece's rotation across a gap and misplace a
        distant tile. A singleton keeps its own measured position only if it
        agrees with that model within ``max_shift``; otherwise it is discarded.
        """
        trusted_ids = list(self.component_offsets)
        trusted_members = np.nonzero(np.isin(labels, trusted_ids))[0]
        dist = scipy.spatial.distance.cdist(
            self.metadata.positions[loose], self.metadata.positions[trusted_members]
        )
        nearest_cid = labels[trusted_members[np.argmin(dist, axis=1)]]
        for tile, cid in zip(loose, nearest_cid):
            m = predictions[tile] + self.component_offsets[int(cid)]
            model[tile] = m
            agrees = (not base_discard[tile]) and (
                np.linalg.norm(self.positions[tile] - m) <= self.max_shift_pixels
            )
            if not agrees:
                discard[tile] = True

    def constrain_positions(self):
        n = self.metadata.num_images
        position_diffs = np.rint(
            np.absolute(self.positions - self.reference_aligner_positions) * 10
        ) / 10
        # Registering identical cycles/files: nothing to constrain.
        if np.all(position_diffs == 0):
            self.discard = np.full(n, False)
            self.offset = 0
            self.residual_rotation = 0.0
            self.shifts_residual = np.zeros_like(self.shifts)
            self.component_labels = np.zeros(n, dtype=int)
            self.component_offsets = {0: np.zeros(2)}
            self.component_rotations = {0: 0.0}
            return

        # Tiles with no trustworthy shift: camera dark-current background lock
        # (detected from the sigma=0 registration) or a failed registration.
        bg_diffs = np.rint(
            np.absolute(self.bg_positions - self.reference_aligner_positions) * 10
        ) / 10
        base_discard = (bg_diffs == 0).all(axis=1)
        base_discard |= np.isinf(self.errors)

        predictions = self.reference_aligner.lr.predict(
            self.corrected_nominal_positions
        )
        labels = self._component_labels(base_discard)
        sizes = (
            np.bincount(labels[labels >= 0]) if (labels >= 0).any()
            else np.zeros(0, dtype=int)
        )
        trusted_ids = [
            int(cid) for cid in range(len(sizes))
            if sizes[cid] >= self._min_component_size
        ]

        discard = base_discard.copy()
        model = np.full_like(predictions, np.nan)
        self.component_offsets = {}
        self.component_rotations = {}

        if trusted_ids:
            for cid in trusted_ids:
                members = np.nonzero(labels == cid)[0]
                model_m, extremes, offset, rotation = self._constrain_component(
                    members, predictions
                )
                model[members] = model_m
                discard[members[extremes]] = True
                self.component_offsets[cid] = offset
                self.component_rotations[cid] = rotation
            # Everything not in a trusted piece (base-discarded tiles and
            # singletons) is resolved against the nearest trusted piece.
            in_trusted = np.isin(labels, trusted_ids)
            loose = np.nonzero(~in_trusted)[0]
            if len(loose):
                self._resolve_loose(
                    loose, base_discard, labels, predictions, model, discard
                )
            # Collapse non-trusted tiles to a single "loose" label for reporting.
            labels = np.where(in_trusted, labels, -1)
        else:
            # No piece has mutual support: fall back to one global group (the
            # stock single-offset constraint over all registrable tiles).
            members = np.nonzero(~base_discard)[0]
            fill_offset = np.zeros(2)
            if len(members):
                model_m, extremes, offset, rotation = self._constrain_component(
                    members, predictions
                )
                model[members] = model_m
                discard[members[extremes]] = True
                self.component_offsets[0] = offset
                self.component_rotations[0] = rotation
                fill_offset = offset
            bd = np.nonzero(base_discard)[0]
            model[bd] = predictions[bd] + fill_offset
            labels = np.where(base_discard, -1, 0)

        self.component_labels = labels
        # Report the dominant (largest) piece's values through the scalar
        # attributes the QC plot and downstream code read; the per-piece detail
        # lives in the component_* dicts (see summarize_components).
        piece_sizes = {
            cid: int((labels == cid).sum()) for cid in self.component_offsets
        }
        dominant = max(piece_sizes, key=piece_sizes.get) if piece_sizes else None
        self.offset = self.component_offsets[dominant] if dominant is not None else 0
        self.residual_rotation = (
            self.component_rotations[dominant] if dominant is not None else 0.0
        )
        self.discard = discard
        self.shifts_residual = self.positions - model
        self.positions[discard] = model[discard]


def summarize_components(aligner):
    """Print and return a per-piece summary of a ``StitchedLayerAligner`` run.

    One row per trusted tissue piece (tile count, discarded count, the piece's
    offset and residual rotation), plus the count of loose tiles (singletons or
    unregistrable tiles) resolved against the nearest piece.
    """
    labels = aligner.component_labels
    rows = []
    for cid in sorted(aligner.component_offsets):
        members = np.nonzero(labels == cid)[0]
        oy, ox = aligner.component_offsets[cid]
        rows.append({
            "component": int(cid),
            "n_tiles": int(len(members)),
            "n_discard": int(aligner.discard[members].sum()),
            "offset_y": float(oy),
            "offset_x": float(ox),
            "rotation_deg": float(aligner.component_rotations[cid]),
        })
    n_loose = int((labels < 0).sum())
    print(f"    {len(rows)} tissue piece(s); {n_loose} loose tile(s)")
    for r in rows:
        print(
            f"      piece {r['component']:>2}: {r['n_tiles']:>4} tiles, "
            f"{r['n_discard']:>3} discarded, "
            f"offset (y,x)=({r['offset_y']:+8.1f},{r['offset_x']:+8.1f}) px, "
            f"rotation={r['rotation_deg']:+.4f}deg",
            flush=True,
        )
    return rows


def _draw_component_graph(ax, aligner, node_size=60):
    """Overlay the tissue-piece connectivity graph on an existing mosaic axis.

    Faint gray edges are the full spatial neighbor graph; bold blue edges are the
    ones that survived the shift-agreement test (``component_tol``) -- the
    within-piece connections. Nodes are colored by tissue piece; loose tiles
    (base-discarded, or singletons not in a trusted piece) are gray. The gaps
    where a gray neighbor edge has no blue edge on top are the piece boundaries
    the tolerance cut -- what you inspect when tuning ``component_tol``.

    Uses the same ``pos=np.fliplr(centers)`` convention as
    ``reg.plot_layer_quality``, so it can be laid over that plot's axis directly.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    pos = np.fliplr(aligner.centers)
    labels = aligner.component_labels
    nx.draw_networkx_edges(
        aligner.neighbors_graph, pos=pos, ax=ax, edge_color="0.4", width=0.5,
    )
    nx.draw_networkx_edges(
        aligner.component_graph, pos=pos, ax=ax, edge_color="deepskyblue", width=1.5,
    )
    nodes = list(aligner.neighbors_graph.nodes)
    cmap = plt.get_cmap("tab20")
    loose_rgba = mcolors.to_rgba("0.5")
    node_color = [
        loose_rgba if labels[k] < 0 else cmap(int(labels[k]) % cmap.N)
        for k in nodes
    ]
    nx.draw_networkx_nodes(
        aligner.neighbors_graph, pos=pos, ax=ax, nodelist=nodes,
        node_color=node_color, node_size=node_size, edgecolors="k", linewidths=0.3,
    )


def _set_mosaic_limits(ax, aligner):
    """Pin ``ax`` to the mosaic extent ``reg.draw_mosaic_image`` drew into.

    ``imshow`` leaves the axes exactly on the image extent, but quiver arrows and
    the graph's scatter/line collections trigger an autoscale that adds
    matplotlib's default 5% margins. Two QC plots of the same aligner then render
    the same mosaic at different scales -- and since the axes aspect is equal,
    ``bbox_inches="tight"`` turns that into differently sized files as well. Both
    break flipping between the plots to compare them, so every QC plot here ends
    on the same explicit limits regardless of which artists were drawn.
    """
    r0, c0 = aligner.positions.min(axis=0)
    r1, c1 = aligner.positions.max(axis=0) + aligner.metadata.size
    ax.set_xlim(c0 - 0.5, c1 - 0.5)
    ax.set_ylim(r1 - 0.5, r0 - 0.5)


def _component_title(aligner):
    labels = aligner.component_labels
    n_pieces = int(labels.max()) + 1 if (labels >= 0).any() else 0
    return f"{n_pieces} piece(s); {int((labels < 0).sum())} loose tile(s)"


def plot_components(aligner, img=None, ax=None, im_kwargs=None):
    """Plot the tissue-piece connectivity graph over the mosaic.

    See ``_draw_component_graph`` for the graph legend. Reuses
    ``reg.draw_mosaic_image`` for the background.
    """
    import matplotlib.pyplot as plt

    if im_kwargs is None:
        im_kwargs = {}
    if ax is None:
        ax = plt.figure().gca()
    reg.draw_mosaic_image(ax, aligner, img, **im_kwargs)
    _draw_component_graph(ax, aligner)
    _set_mosaic_limits(ax, aligner)
    ax.set_title(_component_title(aligner))
    ax.axis("off")
    ax.figure.set_facecolor("black")
    return ax


def plot_layer_quality_with_graph(aligner, img=None, node_size=40, **kwargs):
    """``reg.plot_layer_quality`` with the tissue-piece graph overlaid.

    One combined QC image: the tile boxes (colored by registration error) and
    shift-residual arrows from ``reg.plot_layer_quality``, plus the piece
    connectivity graph from ``_draw_component_graph`` on the same axis. Nodes are
    drawn smaller here (``node_size``) to sit alongside the arrows. Extra keyword
    arguments pass through to ``reg.plot_layer_quality`` (e.g. ``im_kwargs``,
    ``scale``, ``annotate``).
    """
    import matplotlib.pyplot as plt

    reg.plot_layer_quality(aligner, img=img, **kwargs)
    ax = plt.gca()
    _draw_component_graph(ax, aligner, node_size=node_size)
    _set_mosaic_limits(ax, aligner)
    residual = getattr(aligner, "residual_rotation", 0.0)
    ax.set_title(f"{_component_title(aligner)}  |  residual rot {residual:+.3f}deg")
    return ax


def _make_stitched_layer_aligner(
    reader, edge_aligner, corrected_positions, channel, max_shift, filter_sigma,
    component_tol,
):
    """Like ``align_cycles._make_layer_aligner`` but builds the per-piece
    ``StitchedLayerAligner`` (and threads ``component_tol`` through)."""
    la = StitchedLayerAligner(
        reader, edge_aligner, verbose=True,
        channel=channel, max_shift=max_shift, filter_sigma=filter_sigma,
        component_tol=component_tol,
    )
    la.corrected_nominal_positions = corrected_positions
    align_cycles.set_pairs(la)
    return la


def align_to_stitched(
    reference_image,
    moving_reader,
    *,
    channel=0,
    max_shift=15,
    filter_sigma=0.0,
    pixel_size=None,
    component_tol=None,
    corner_tol=0.1,
):
    """Align ``moving_reader`` (a cycle of FOV tiles) to a stitched reference.

    ``reference_image`` is a 2D array-like (numpy/zarr/dask) holding a single
    channel of the already-stitched reference mosaic -- e.g.
    ``palom.reader.OmePyramidReader(path).pyramid[0][ref_channel]``.

    Returns a registered ``StitchedLayerAligner`` whose ``mosaic_shape`` is the
    full reference shape, ready to feed to ``reg.Mosaic``. Inspect the result
    with ``summarize_components`` to see the detected tissue pieces.

    ``corner_tol`` (pixels) is a corner-displacement budget: the rotated-reader
    resampling is skipped when the refined rotation would move a moving tile's
    farthest corner by less than ``corner_tol`` (or when refinement returns
    NaN), and the unrotated positions are used instead. It is converted to an
    angle threshold from the moving tile's half-diagonal, since ``_build_-
    rotated_reader`` rotates each tile about its own center.
    """
    c2r = moving_reader
    c1e = _build_stitched_reference(reference_image, c2r, pixel_size=pixel_size)

    c21l = StitchedLayerAligner(
        c2r, c1e, verbose=True,
        channel=channel, max_shift=max_shift, filter_sigma=filter_sigma,
        component_tol=component_tol,
    )
    c21l.make_thumbnail()
    scale = c1e.reader.thumbnail_scale

    # Coarse cycle alignment, then re-cut the reference tiles underneath the
    # corrected moving positions so each moving tile pairs 1:1 with its crop.
    align_cycles.correct_position(c21l, angle=None)
    _set_reference_geometry(c1e, c21l.corrected_nominal_positions)
    align_cycles.set_pairs(c21l)

    mosaic_shape = c1e.reader.metadata.mosaic_shape

    def _finalize_unrotated():
        c21l.register_all()
        c21l.calculate_positions()
        c21l.mosaic_shape = mosaic_shape
        return c21l

    if c21l.cycle_tform.rotation == 0:
        return _finalize_unrotated()

    edgy_scores = align_cycles.tile_edge_score(c2r, c21l.channel)
    rank = np.argsort(edgy_scores)[::-1]
    angle = align_cycles.refine_angle(c21l, rank=rank, top_k=30)
    print(f"\r    refined cycle rotation = {angle:.4f} degrees", flush=True)

    # Convert the corner-displacement budget (px) to an angle threshold: a
    # rotation about a tile center moves the tile's farthest corner by
    # ``half_diag * radians``. A refined angle below this (or NaN, when
    # refinement found no usable overlaps) is not worth the rotated-reader
    # resampling -- fall back to the unrotated positions rather than warping by
    # a negligible/bad angle.
    half_diag = 0.5 * float(np.hypot(*c2r.metadata.size))
    angle_tol = np.degrees(corner_tol / half_diag)
    if not np.isfinite(angle) or abs(angle) < angle_tol:
        return _finalize_unrotated()

    c2rr, corrected_positions = align_cycles._build_rotated_reader(
        c2r, c1e, angle, scale
    )
    # Re-cut the reference tiles again, now at the rotated corrected positions.
    _set_reference_geometry(c1e, corrected_positions)
    c21lr = _make_stitched_layer_aligner(
        c2rr, c1e, corrected_positions, channel, max_shift, filter_sigma,
        component_tol,
    )
    c21lr.register_all()
    c21lr.calculate_positions()
    c21lr.mosaic_shape = mosaic_shape
    return c21lr
