import pathlib
from . import fileseries
from . import codex_best_z
from . import reg


def run_1():
    input_dir_parent = r'V:\YC-20210607-codex_penn_oldridge\Helm6939_PB3182021\Helm6939_PB3182021'
    input_dirs = sorted(filter(
        lambda x: x.is_dir(), 
        pathlib.Path(input_dir_parent).glob('*')
    ))
    series_pattern = '1_{series}_Z{z}_CH{channel}.tif'

    for i in input_dirs:
        reader = fileseries.FileSeriesReader(
            str(i),
            series_pattern,
            0.3, 3, 3,
            'snake', 'horizontal',
            0.3
        )
        codex_best_z.best_z_to_ome(
            reader=reader,
            output_dir=r'V:\YC-20210607-codex_penn_oldridge\mcmicro\Helm6939_PB3182021\raw'
        )


def run_2():
    input_dir_parent = r'V:\YC-20210607-codex_penn_oldridge\FFPE_TonsilNewStageTest_2020-09-02_toShare'
    input_dirs = sorted(filter(
        lambda x: x.is_dir(), 
        pathlib.Path(input_dir_parent).glob('*')
    ))
    series_pattern = 'TileScan 1--Stage{series}--Z{z}--C{channel}.tif'

    for i in input_dirs:
        reader = fileseries.FileSeriesReader(
            str(i),
            series_pattern,
            0.1, 21, 13,
            'snake', 'horizontal',
            0.3225
        )
        codex_best_z.best_z_to_ome(
            reader=reader,
            output_dir=r'V:\YC-20210607-codex_penn_oldridge\mcmicro\FFPE_TonsilNewStageTest_2020-09-02\raw'
        )


def run_3():
    input_dir = r'Y:\sorger\data\computation\Yu-An\YC-20210610-codex_sascha_exp_243_no_binning'
    input_dir = pathlib.Path(input_dir)
    input_czis = sorted(input_dir.glob('*.czi'))

    for i in input_czis:
        reader = reg.BioformatsReader(str(i))
        codex_best_z.best_z_to_ome(
            reader=reader,
            output_dir=input_dir / 'raw'
        )