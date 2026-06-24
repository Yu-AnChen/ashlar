import numpy as np
import skimage.transform

from .. import reg, transform, utils


class PreprocMetadata:
    """Metadata shim exposing preprocessing-adjusted geometry.

    Wraps another Metadata and overrides only tile size and positions (and the
    quantities derived from them), delegating everything else. This leaves the
    underlying reader's metadata unmodified, unlike mutating it in place.
    """

    def __init__(self, wrapped, positions, size):
        self._wrapped = wrapped
        self._positions = np.asarray(positions, dtype=float)
        self._size = np.asarray(size, dtype=int)

    def __getattr__(self, name):
        # Don't delegate dunders -- otherwise pickle would pick up the wrapped
        # object's __setstate__/__getstate__ and corrupt the round-trip. Also
        # guard _wrapped itself to avoid infinite recursion before it's set
        # (e.g. while unpickling, before __dict__ is restored).
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name == "_wrapped":
            raise AttributeError(name)
        return getattr(self._wrapped, name)

    @property
    def positions(self):
        return self._positions

    @property
    def size(self):
        return self._size

    def tile_size(self, i):
        return self._size

    @property
    def centers(self):
        return self._positions + self._size / 2

    @property
    def origin(self):
        return self._positions.min(axis=0)


class PreprocReader(reg.Reader):
    """Wraps a reader to apply per-tile preprocessing.

    Applies barrel correction, axis flips, rotation, and center cropping to
    each tile, and exposes the resulting geometry through a ``PreprocMetadata``
    shim. The wrapped reader and its metadata are left untouched.
    """

    def __init__(
        self, reader, *,
        flip_x=False, flip_y=False, angle=0, barrel_k=0,
        center_crop_shape=None, flip_pos_x=False, flip_pos_y=False,
    ):
        self.reader = reader
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.angle = angle
        self.barrel_k = barrel_k
        self.center_crop_shape = center_crop_shape

        positions = np.array(reader.metadata.positions, dtype=float)
        if flip_pos_y:
            positions = positions * [-1, 1]
        if flip_pos_x:
            positions = positions * [1, -1]

        size = np.array(reader.metadata.size, dtype=int)
        self.offsets = np.zeros(2, dtype=int)
        if center_crop_shape is not None:
            self.offsets = ((size - center_crop_shape) // 2).astype(int)
            size = np.array(center_crop_shape, dtype=int)
        positions = positions + self.offsets

        self._metadata = PreprocMetadata(reader.metadata, positions, size)

    @property
    def metadata(self):
        return self._metadata

    @property
    def path(self):
        return self.reader.path

    def validity_mask(self):
        """Geometric coverage of real data after preprocessing (0=border, 1=data).

        Applies the same flip/rotation/crop as ``read`` to a ones array, so the
        rotation/barrel zero-border is described by geometry rather than guessed
        from pixel values. Fractional at the antialiased edge (partial coverage).
        """
        m = np.ones(np.array(self.reader.metadata.size, dtype=int), dtype=np.float32)
        if self.barrel_k != 0:
            m = transform.barrel_correction(m, self.barrel_k, cval=0)
        if self.flip_x:
            m = np.fliplr(m)
        if self.flip_y:
            m = np.flipud(m)
        if self.angle != 0:
            m = skimage.transform.rotate(m, self.angle, preserve_range=True)
        if self.offsets.any():
            m = utils.crop(m, self.offsets, self.metadata.size)
        return m.astype(np.float32)

    def read(self, series, c):
        img = self.reader.read(series=series, c=c)
        dtype = self.metadata.pixel_dtype
        if self.barrel_k != 0:
            img = transform.barrel_correction(img, self.barrel_k)
            img = utils.dtype_convert(img, dtype)
        if self.flip_x:
            img = np.fliplr(img)
        if self.flip_y:
            img = np.flipud(img)
        if self.angle != 0:
            img = skimage.transform.rotate(
                img, self.angle, preserve_range=True
            ).astype(dtype)
        if self.offsets.any():
            # Crop to the exact target shape (utils.crop takes an explicit
            # shape), which keeps the returned image consistent with
            # metadata.size regardless of odd/even parity.
            img = utils.crop(img, self.offsets, self.metadata.size)
        return img
