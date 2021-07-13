from joblib import Parallel, delayed
import tifffile
import cv2
import numpy as np
import pathlib


def whiten_norm(img, sigma):
    g_img = cv2.GaussianBlur(img.astype(float), (0, 0), sigma)
    log_img = cv2.Laplacian(g_img, cv2.CV_64F, ksize=1)

    return np.linalg.norm(log_img)


def best_z_to_ome(
    reader,
    output_dir='.'
):

    def wrap(i):
        return [
            whiten_norm(reader.read(i, 0, z=z), 1)
            for z in reader.metadata.z_map.keys()
        ]
    edgy_scores = np.array(
        Parallel(verbose=1, n_jobs=-1)(
            delayed(wrap)(i) 
            for i in range(reader.metadata.num_images)
        )
    )
    
    # edgy_scores = np.array([
    #     [
    #         whiten_norm(reader.read(i, 0, z=z), 1) 
    #         for z in reader.metadata.z_map.keys()
    #     ]
    #     for i in range(reader.metadata.num_images)
    # ])
    
    tile_best_z = np.argmax(edgy_scores, axis=1)
    print(tile_best_z)

    pixel_size = reader.metadata.pixel_size
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f'{reader.path.name}.ome.tif'

    print('Write to', str(output_path))
    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        for p, z, s in zip(
            reader.metadata.positions, tile_best_z, range(reader.metadata.num_images)
        ):
            img = np.array([
                reader.read(s, c, z)
                for c in range(reader.metadata.num_channels)
            ])
            positions = {
                'Pixels': {
                    'PhysicalSizeX': pixel_size,
                    'PhysicalSizeXUnit': 'µm',
                    'PhysicalSizeY': pixel_size,
                    'PhysicalSizeYUnit': 'µm'
                },
                'Plane': {
                    'PositionX': [p[1]*pixel_size]*img.shape[0],
                    'PositionY': [p[0]*pixel_size]*img.shape[0]
                }
            }
            tif.write(img, metadata=positions)
