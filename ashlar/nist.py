from __future__ import division, print_function
import re
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib
import numpy as np
import skimage.io
from . import reg


# Experimental classes for reading sample TIFF datasets from the NIST
# Information System Group's MIST stitching tool.


class NistMetadata(reg.Metadata):

    def __init__(self, path, pattern, overlap, overlapY):
        self.path = pathlib.Path(path)
        self.pattern = pattern
        self.overlap = overlap
        self.overlapY = overlapY
        self._enumerate_tiles()

    def _enumerate_tiles(self):
        regex = re.sub(r'{([^:}]+)(?:[^}]*)}', r'(?P<\1>.*?)',
                       self.pattern.replace('.', '\.'))
        rows = set()
        cols = set()
        n = 0
        for p in self.path.iterdir():
            match = re.match(regex, p.name)
            if match:
                gd = match.groupdict()
                rows.add(int(gd['row']))
                cols.add(int(gd['col']))
                n += 1
        if n != len(rows) * len(cols):
            raise Exception("Tiles do not form a full rectangular grid")
        self._actual_num_images = n
        self.height = len(rows)
        self.width = len(cols)
        self.row_offset = min(rows)
        self.col_offset = min(cols)
        path = self.path / self.pattern.format(
            row=self.row_offset, col=self.col_offset
        )
        img = skimage.io.imread(path)
        self._tile_size = np.array(img.shape)[0:2]


    @property
    def _num_images(self):
        return self._actual_num_images

    @property
    def num_channels(self):
        return 4

    @property
    def pixel_size(self):
        return np.ones(2)

    @property
    def pixel_dtype(self):
        return np.uint16

    def tile_position(self, i):
        row, col = self.tile_rc(i)
        overlapYX = [self.overlapY, self.overlap] if self.overlapY != 0 else [self.overlap, self.overlap]
        # return [row, col] * self.tile_size(i) * (1 - np.array(overlapYX))
        return [row, col] * self.tile_size(i) * (1 - self.overlap)

    def tile_size(self, i):
        return self._tile_size

    def tile_rc(self, i):
        row = i // self.width + self.row_offset
        col = i % self.width + self.col_offset
        return row, col


class NistReader(reg.Reader):

    def __init__(self, path, pattern='img_Phase_r{row:03}_c{col:03}.tif',
                 overlap=0.1, overlapY=0):
        self.path = pathlib.Path(path)
        self.pattern = pattern
        self.overlap = overlap
        self.overlapY = overlapY
        self.metadata = NistMetadata(self.path, self.pattern, overlap, overlapY)

    def read(self, series, c):
        # assert c == 0, "Channel must be 0"
        img = skimage.io.imread(str(self.path / self.filename(series)))
        return img[:,:,c] \
        # if img.ndim == 2 else img[:, :, 0]

    def filename(self, series):
        row, col = self.metadata.tile_rc(series)
        return self.pattern.format(row=row, col=col)

