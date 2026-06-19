import skimage.data
import skimage.util
import numpy as np
import tifffile
import skimage.filters

img = skimage.data.astronaut()[..., 0]
imgs = skimage.util.view_as_windows(img, window_shape=(100, 100), step=90)

h, w, *_ = imgs.shape
pixel_size = 0.3

positions = np.mgrid[:h, :w].reshape(2, -1).T * 90
random_gaussian_sigma = np.random.random(h * w) * 3

with tifffile.TiffWriter("01-ref.ome.tif", bigtiff=True) as tif:
    for ii, p, ss in zip(
        imgs.reshape(-1, *(100, 100)), positions, random_gaussian_sigma
    ):
        ii = skimage.filters.gaussian(ii, ss)
        ii = skimage.util.img_as_ubyte(ii)
        ii = np.array([ii] * 5, dtype=np.uint16)
        metadata = {
            "Pixels": {
                "PhysicalSizeX": pixel_size,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_size,
                "PhysicalSizeYUnit": "µm",
            },
            "Plane": {
                "PositionX": [p[1] * pixel_size] * ii.shape[0],
                "PositionY": [p[0] * pixel_size] * ii.shape[0],
            },
        }
        tif.write(ii, metadata=metadata)

with tifffile.TiffWriter("02-bg.ome.tif", bigtiff=True) as tif:
    for ii, p, ss in zip(
        imgs.reshape(-1, *(100, 100)), positions, random_gaussian_sigma
    ):
        ii = skimage.filters.gaussian(ii, ss)
        ii = skimage.util.img_as_ubyte(ii)
        ii = ii.astype(np.uint16)
        ii = np.array([ii + 105] * 5, dtype=np.uint16)
        metadata = {
            "Pixels": {
                "PhysicalSizeX": pixel_size,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_size,
                "PhysicalSizeYUnit": "µm",
            },
            "Plane": {
                "PositionX": [p[1] * pixel_size] * ii.shape[0],
                "PositionY": [p[0] * pixel_size] * ii.shape[0],
            },
        }
        tif.write(ii, metadata=metadata)

with tifffile.TiffWriter("03-ab.ome.tif", bigtiff=True) as tif:
    for ii, p, ss in zip(
        imgs.reshape(-1, *(100, 100)), positions, random_gaussian_sigma
    ):
        ii = skimage.filters.gaussian(ii, ss)
        ii = skimage.util.img_as_ubyte(ii)
        ii = ii.astype(np.float32)
        ii = np.array([ii, ii * 5, ii * 10, ii / 5, ii / 10])
        ii += 105
        ii = np.round(ii).astype(np.uint16)
        metadata = {
            "Pixels": {
                "PhysicalSizeX": pixel_size,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_size,
                "PhysicalSizeYUnit": "µm",
            },
            "Plane": {
                "PositionX": [p[1] * pixel_size] * ii.shape[0],
                "PositionY": [p[0] * pixel_size] * ii.shape[0],
            },
        }
        tif.write(ii, metadata=metadata)


# ---------------------------------------------------------------------------- #
#                     export fiducial channel and compress                     #
# ---------------------------------------------------------------------------- #
from ashlar import reg
import pathlib
import tifffile
import numpy as np
import tqdm

out_dir = pathlib.Path(r"C:\Users\yc296\Desktop\ashlar-rotation-data")
out_dir.mkdir(exist_ok=True, parents=True)

files = r"""
\\research.files.med.harvard.edu\HITS\lsp-data\cycif-production\215_Kidney_Xenotransplant_Human\26-06-04_XenoChimerismCycle3\Pysed2\LSP65674a_Nephrectomy_001_A107_Xeno_wHTG_Cycle1_v1_001503.pysed.ome.tif
\\research.files.med.harvard.edu\HITS\lsp-data\cycif-production\215_Kidney_Xenotransplant_Human\26-06-04_XenoChimerismCycle3\Pysed2\LSP65674a_Nephrectomy_002_A107_Xeno_wHTG_Cycle2_v2_001512.pysed.ome.tif
\\research.files.med.harvard.edu\HITS\lsp-data\cycif-production\215_Kidney_Xenotransplant_Human\26-06-04_XenoChimerismCycle3\Pysed2\LSP65674a_Nephrectomy_003_A107_Xeno_wHTG_Cycle3_v2_002155.pysed.ome.tif
""".strip().split("\n")

for ff in files[:]:
    c1r = reg.BioformatsReader(ff)
    out_path = out_dir / pathlib.Path(ff).name

    positions = c1r.metadata.positions
    positions *= [-1, 1]
    pixel_size = c1r.metadata.pixel_size
    use_channels = [0]
    with tifffile.TiffWriter(out_path, bigtiff=True) as tif:
        for ii, pp in enumerate(tqdm.tqdm(positions)):
            img = np.asarray([c1r.read(ii, cc) for cc in use_channels])
            metadata = {
                "Pixels": {
                    "PhysicalSizeX": pixel_size,
                    "PhysicalSizeXUnit": "\u00b5m",
                    "PhysicalSizeY": pixel_size,
                    "PhysicalSizeYUnit": "\u00b5m",
                },
                "Plane": {
                    "PositionX": [pp[1] * pixel_size] * len(img),
                    "PositionY": [pp[0] * pixel_size] * len(img),
                },
            }
            tif.write(img, metadata=metadata, compression="zstd", predictor=True)
