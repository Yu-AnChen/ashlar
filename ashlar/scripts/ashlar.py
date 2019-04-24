from __future__ import print_function
import sys
import argparse
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib
from .. import __version__ as VERSION
from .. import reg
from .. import thumbnail
import numpy as np
import pandas as pd


def main(argv=sys.argv):

    parser = argparse.ArgumentParser(
        description='Stitch and align one or more multi-series images'
    )
    parser.add_argument(
        'filepaths', metavar='FILE', nargs='*',
        help='an image file to be processed (one file per cycle)'
    )
    parser.add_argument(
        '-o', '--output', dest='output', default='.', metavar='DIR',
        help='write output image files to DIR; default is the current directory'
    )
    parser.add_argument(
        '-c', '--align-channel', dest='align_channel', nargs='?', type=int,
        default='0', metavar='CHANNEL',
        help=('align images using channel number CHANNEL; numbering starts'
              ' at 0')
    )
    parser.add_argument(
        '--output-channels', nargs='*', type=int, metavar='CHANNEL',
        help=('output only channels listed in CHANNELS; numbering starts at 0')
    )
    parser.add_argument(
        '-m', '--maximum-shift', type=float, default=15, metavar='SHIFT',
        help='maximum allowed per-tile corrective shift in microns'
    )
    arg_f_default = 'cycle_{cycle}_channel_{channel}.tif'
    parser.add_argument(
        '-f', '--filename-format', dest='filename_format',
        default=arg_f_default, metavar='FORMAT',
        help=('use FORMAT to generate output filenames, with {{cycle}} and'
              ' {{channel}} as required placeholders for the cycle and channel'
              ' numbers; default is {default}'.format(default=arg_f_default))
    )
    parser.add_argument(
        '--pyramid', default=False, action='store_true',
        help='write output as a single pyramidal TIFF'
    )
    # Implement default-value logic ourselves so we can detect when the user
    # has explicitly set a value.
    tile_size_default = 1024
    parser.add_argument(
        '--tile-size', type=int, default=None, metavar='PIXELS',
        help=('set tile width and height to PIXELS (pyramid output only);'
              ' default is {default}'.format(default=tile_size_default))
    )
    parser.add_argument(
        '--ffp', metavar='FILE', nargs='*',
        help=('read flat field profile image from FILES; if specified must'
              ' be one common file for all cycles or one file for each cycle')
    )
    parser.add_argument(
        '--dfp', metavar='FILE', nargs='*',
        help=('read dark field profile image from FILES; if specified must'
              ' be one common file for all cycles or one file for each cycle')
    )
    parser.add_argument(
        '--plates', default=False, action='store_true',
        help='enable plate mode for HTS data'
    )
    parser.add_argument(
        '-q', '--quiet', dest='quiet', default=False, action='store_true',
        help='suppress progress display'
    )
    parser.add_argument(
        '--version', dest='version', default=False, action='store_true',
        help='print version'
    )
    args = parser.parse_args(argv[1:])

    if args.version:
        print('ashlar {}'.format(VERSION))
        return 0

    if len(args.filepaths) == 0:
        parser.print_usage()
        return 1

    filepaths = args.filepaths
    if len(filepaths) == 1:
        path = pathlib.Path(filepaths[0])
        if path.is_dir():
            filepaths = sorted(str(p) for p in path.glob('*rcpnl'))

    output_path = pathlib.Path(args.output)
    if not output_path.exists():
        print("Output directory '{}' does not exist".format(output_path))
        return 1

    if args.tile_size and not args.pyramid:
        print("--tile-size can only be used with --pyramid")
        return 1
    if args.tile_size is None:
        # Implement default value logic as mentioned in argparser setup above.
        args.tile_size = tile_size_default

    ffp_paths = args.ffp
    if ffp_paths:
        if len(ffp_paths) not in (0, 1, len(filepaths)):
            print("Wrong number of flat-field profiles. Must be 1, or {}"
                  " (number of input files)".format(len(filepaths)))
            return 1
        if len(ffp_paths) == 1:
            ffp_paths = ffp_paths * len(filepaths)

    dfp_paths = args.dfp
    if dfp_paths:
        if len(dfp_paths) not in (0, 1, len(filepaths)):
            print("Wrong number of dark-field profiles. Must be 1, or {}"
                  " (number of input files)".format(len(filepaths)))
            return 1
        if len(dfp_paths) == 1:
            dfp_paths = dfp_paths * len(filepaths)

    aligner_args = {}
    aligner_args['channel'] = args.align_channel
    aligner_args['verbose'] = not args.quiet
    aligner_args['max_shift'] = args.maximum_shift

    mosaic_args = {}
    if args.output_channels:
        mosaic_args['channels'] = args.output_channels
    if args.pyramid:
        mosaic_args['tile_size'] = args.tile_size
    if args.quiet is False:
        mosaic_args['verbose'] = True

    try:
        if args.plates:
            return process_plates(
                filepaths, output_path, args.filename_format, ffp_paths,
                dfp_paths, aligner_args, mosaic_args, args.pyramid, args.quiet
            )
        else:
            mosaic_path_format = str(output_path / args.filename_format)
            return process_single(
                filepaths, mosaic_path_format, ffp_paths, dfp_paths,
                aligner_args, mosaic_args, args.pyramid, args.quiet
            )
    except ProcessingError as e:
        print(e.message)
        return 1


def process_single(
    filepaths, mosaic_path_format, ffp_paths, dfp_paths,
    aligner_args, mosaic_args, pyramid, quiet, plate=None, well=None
):

    output_path_0 = format_cycle(mosaic_path_format, 0)
    if pyramid:
        if output_path_0 != mosaic_path_format:
            raise ProcessingError(
                "For pyramid output, please use -f to specify an output"
                " filename without {cycle} or {channel} placeholders"
            )

    mosaic_args = mosaic_args.copy()
    if pyramid:
        mosaic_args['combined'] = True
    num_channels = 0

    if not quiet:
        print('Cycle 0:')
        print('    reading %s' % filepaths[0])
    reader = reg.BioformatsReader(filepaths[0], plate=plate, well=well)
    edge_aligner = reg.EdgeAligner(reader, **aligner_args)
    edge_aligner.run()

    csv_path = pathlib.Path(mosaic_path_format).with_name(
        'p_label_positions_cycle{:1}.csv'.format(1)
    )
    ea = edge_aligner
    df1 = pd.DataFrame(
        np.hstack((ea.positions, ea.connected_tiles_mask, ea.pure_prediction_tiles)),
        columns=[
            'pos_y', 'pos_x', 'tree_ids', 'pure_prediction'
        ],
    )
    df1.to_csv(csv_path)

    mshape = edge_aligner.mosaic_shape
    mosaic_args_final = mosaic_args.copy()
    mosaic_args_final['first'] = True
    if ffp_paths:
        mosaic_args_final['ffp_path'] = ffp_paths[0]
    if dfp_paths:
        mosaic_args_final['dfp_path'] = dfp_paths[0]
    mosaic = reg.Mosaic(
        edge_aligner, mshape, output_path_0, **mosaic_args_final
    )
    mosaic.run()
    num_channels += len(mosaic.channels)

    reader.thumbnail_img = thumbnail.thumbnail(
        reader, channel=aligner_args['channel']
    )
    ref_reader = reader

    for cycle, filepath in enumerate(filepaths[1:], 1):
        if not quiet:
            print('Cycle %d:' % cycle)
            print('    reading %s' % filepath)
        reader = reg.BioformatsReader(filepath, plate=plate, well=well)
        cycle_offset = thumbnail.calculate_cycle_offset(
            ref_reader, reader,
            channel=aligner_args['channel'], save=(False, True)
        )
        reader.metadata.positions
        reader.metadata._positions += cycle_offset
        layer_aligner = reg.LayerAligner(reader, edge_aligner, **aligner_args)
        layer_aligner.cycle_offset = cycle_offset
        layer_aligner.run()

        csv_path_layer = pathlib.Path(mosaic_path_format).with_name(
            'p_label_positions_cycle{:1}.csv'.format(cycle + 1)
        )
        la = layer_aligner
        df2 = pd.DataFrame(
            np.hstack((
                la.positions, la.reference_idx.reshape(-1, 1), 
                la.shifts, la.errors.reshape(-1, 1), la.pure_prediction_tiles
            )),
            columns=[
                'pos_y', 'pos_x', 'ref_idx', 
                'shifts_y', 'shifts_x', 'errors', 'pure_prediction'
            ],
        )
        df2.to_csv(csv_path_layer)

        mosaic_args_final = mosaic_args.copy()
        if ffp_paths:
            mosaic_args_final['ffp_path'] = ffp_paths[cycle]
        if dfp_paths:
            mosaic_args_final['dfp_path'] = dfp_paths[cycle]
        mosaic = reg.Mosaic(
            layer_aligner, mshape, format_cycle(mosaic_path_format, cycle),
            **mosaic_args_final
        )
        mosaic.run()
        num_channels += len(mosaic.channels)

    if pyramid:
        print("Building pyramid")
        reg.build_pyramid(
            output_path_0, num_channels, mshape, reader.metadata.pixel_dtype,
            reader.metadata.pixel_size, mosaic_args['tile_size'], not quiet
        )

    return 0


def process_plates(
    filepaths, output_path, filename_format, ffp_paths, dfp_paths,
    aligner_args, mosaic_args, pyramid, quiet
):

    metadata = reg.BioformatsMetadata(filepaths[0])
    if metadata.num_plates == 0:
        # FIXME raise ProcessingError here instead?
        print("Dataset does not contain plate information.")
        return 1

    for p, plate_name in enumerate(metadata.plate_names):
        print("Plate {} ({})\n==========\n".format(p, plate_name))
        for w, well_name in enumerate(metadata.well_names[p]):
            print("Well {}\n-----".format(well_name))
            if len(metadata.plate_well_series[p][w]) > 0:
                well_path = output_path / plate_name / well_name
                well_path.mkdir(parents=True, exist_ok=True)
                mosaic_path_format = str(well_path / filename_format)
                process_single(
                    filepaths, mosaic_path_format, ffp_paths, dfp_paths,
                    aligner_args, mosaic_args, pyramid, quiet, plate=p, well=w
                )
            else:
                print("Skipping -- No images found.")
            print()
        print()

    return 0


def format_cycle(f, cycle):
    return f.format(cycle=cycle, channel='{channel}')


class ProcessingError(RuntimeError):
    pass


if __name__ == '__main__':
    sys.exit(main())
