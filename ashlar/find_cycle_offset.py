from __future__ import division, print_function
import os
import skimage.io
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib
from . import reg

fft2 = reg.fft2
whiten = reg.whiten

def path_to_date(path):
    return os.path.getmtime(str(path))

def find_cycle_offset(thumbnail_dir, factor=10):
    # thumbnail_dir = path
    thumbnail_files = sorted(
        pathlib.Path(thumbnail_dir).glob('*mosaic_c*.tif'),
        key=path_to_date
    )

    cycle_offsts = [0, 0]
    ref_thumb = skimage.io.imread(str(thumbnail_files[0]))
    ref = fft2(whiten(ref_thumb))
    for thumb in thumbnail_files[1:]:
        thumb_img = skimage.io.imread(str(thumb))
        shift, error, _= skimage.feature.register_translation(
            ref, fft2(whiten(thumb_img)), 1, 'fourier'
        )
        cycle_offsts += list(shift*10)

    return ' '.join([str(o) for o in cycle_offsts])



