import openslide
import numpy as np
import pathlib
from ashlar import reg
from PIL import Image
from skimage import transform
from skimage.color import hax_from_rgb, separate_stains
from skimage.exposure import rescale_intensity

class SvsMetadata(reg.Metadata):

    def __init__(self, path, tile_size, overlap, pixel_size):
        self.path = pathlib.Path(path)
        self._tile_size = np.array(tile_size)
        self.overlap = overlap
        self._pixel_size = pixel_size
        self.deconstruct_slide()
        pass
    
    def deconstruct_slide(self):
        self.slide = openslide.open_slide(str(self.path))
        m_shape = np.array(self.slide.dimensions[::-1])
        rc = (m_shape - self._tile_size) / (self._tile_size * (1 - self.overlap))
        rc = np.ceil(rc).astype(np.int64) + 1
        num_r, num_c = rc
        self._positions = [   
            self._tile_size * (1 - self.overlap) * np.unravel_index(i, (num_r, num_c))
            for i in range(num_r * num_c)
        ]
        np.arange(num_r * num_c)
        self._positions = np.array(self._positions)

    @property
    def _num_images(self):
        return len(self._positions)

    @property
    def num_channels(self):
        return 6

    @property
    def pixel_size(self):
        return 1.0

    @property
    def pixel_dtype(self):
        return np.dtype(np.uint8)

    def tile_size(self, i):
        return self._tile_size

    # def image_path(self, series, c):
    #     return self.base_path / self.image_paths[series, c]

class SvsReader(reg.Reader):

    def __init__(self, path, tile_size, overlap, pixel_size):
        self.metadata = SvsMetadata(path, tile_size, overlap, pixel_size)
        self.path = pathlib.Path(path)
        self._init_slide()

    def _init_slide(self):
        self.slide = self.metadata.slide
        thumbnail = self.slide.read_region(
            (0, 0), 2, self.slide.level_dimensions[2]
        )
        hax_t = separate_stains(
            np.array(thumbnail.convert(mode='RGB')),
            hax_from_rgb
        )
        self.aec_upper = hax_t[..., 1].max()
        self.aec_lower = hax_t[..., 1].min()
        self.hem_upper = hax_t[..., 0].max()
        self.hem_lower = hax_t[..., 0].min()
        self.thumbnail = np.array(
            thumbnail.convert(mode='L')
        )
        self.thumbnail = transform.rescale(
            self.thumbnail, self.slide.level_downsamples[2] / 20
        )

    def read(self, series, c):
        position = self.metadata.positions[series]
        rounded_p = np.floor(position).astype(np.int64)
        img_pixel = self.slide.read_region(
            rounded_p[::-1], level=0, size=(self.metadata._tile_size + 1)
        )
        diff = (position - rounded_p)[::-1]
        img_subpixel = img_pixel.crop(
            (*diff, *(diff + self.metadata._tile_size[::-1]))
        )
        if c == 0:
            # return grayscale inverted image
            img_grayscale = np.array(
                img_subpixel.convert(mode='L')
            )
            img_grayscale[img_grayscale == 0] = 255
            return np.invert(img_grayscale)
        elif c in [1, 2, 3]:
            # return [R, G, B] image
            return np.array(
                img_subpixel.convert(mode='RGB')
            )[..., c - 1]
        elif c in [4, 5]:
            # return [AEC, HEM] image
            hax = separate_stains(
                np.array(img_subpixel.convert(mode='RGB')),
                hax_from_rgb
            )
            if c == 4:
                return rescale_intensity(
                    hax[..., 1], in_range=(self.aec_lower, self.aec_upper),
                    out_range=(0, 1)
                )
            elif c == 5:
                return rescale_intensity(
                    hax[..., 0], in_range=(self.hem_lower, self.hem_upper),
                    out_range=(0, 1)
                )


import pathlib
import sklearn

settings = {
    'svs_dir': 'H:/2019NOV-TNP_PILOT/TNP SARDANA images (mIHC)/75682/Raw images',
    'ref_slide_pattern': '*HEM*',
}

svs_paths = sorted(pathlib.Path(settings['svs_dir']).glob('*.svs'))

# Identify the ref slide to for the shape of the mosaic
ref_slide = None
for svs in svs_paths:
    if svs.match(settings['ref_slide_pattern']):
        print('Reference file found {}'.format(svs.name))
        ref_slide = svs
        svs_paths.remove(svs)
        break

c1r = SvsReader(str(ref_slide), (250, 250), 0.02, 1.)
c1e = reg.EdgeAligner(c1r, verbose=True, max_shift=50)
c1e.positions = c1e.metadata.positions

c1e.lr = sklearn.linear_model.LinearRegression()
c1e.lr.fit(c1e.metadata.positions, c1e.positions)
c1e.origin = c1e.positions.min(axis=0)
c1e.centers = c1e.positions + c1e.metadata.size / 2

m_shape = c1r.slide.dimensions[::-1]
out_name = 'ashlar/' + c1r.path.name.replace(
    '.svs', '-c{channel}-t250-o0.02-m30.tif'
)

reg.Mosaic(c1e, m_shape, out_name, verbose=True, channels=[0, 5], combined=True, tile_size=1024).run()

for i in svs_paths[7:]:
    print('Processing', i.name)
    c2r = SvsReader(str(i), (250, 250), 0.02, 1.)
    c21l = reg.LayerAligner(c2r, c1e, verbose=True, max_shift=30)
    c21l.run()
    out_name_layer = 'ashlar/' + c2r.path.name.replace(
        '.svs', '-c{channel}-t250-o0.02-m30.tif'
    )   
    reg.Mosaic(c21l, m_shape, out_name_layer, verbose=True, channels=[0, 4], combined=True, tile_size=1024).run()

reg.build_pyramid('75682-t250-o0.02-m30-norm.ome.tif', int(2 * (len(svs_paths) + 1)), c1e.mosaic_shape, c1r.metadata.pixel_dtype, c1r.metadata.pixel_size, 1024, verbose=True)
