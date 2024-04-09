import argparse
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import os
import glob
import random
import sys
import pandas as pd
from more_itertools import unique_everseen
from tqdm.auto import tqdm
from random import shuffle


def add_io_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('-input_dir_to_transcribe', default=None, help='file list')
    parser.add_argument('-input_file_to_transcribe', default=None, help='one file')

    # params for if we're reading file names from a CSV
    parser.add_argument('-input_files_map', help='CSV of files to transcribe', default=None)
    parser.add_argument('-file_col_name', help='column name for file', default='file')
    parser.add_argument('-split', help='split', default=None)
    parser.add_argument('-split_col_name', help='column name for split', default='split')
    parser.add_argument('-dataset', help='dataset', default=None)
    parser.add_argument('-dataset_col_name', help='column name for dataset', default='dataset')

    # some algorithms only take a certain file format (e.g. MP3 or WAV)
    parser.add_argument('-input_file_format', default=None,
                        help='Required input format ["mp3", "wav"]. '
                             'E.g. (I think) hFT only takes in WAV files.'
                        )
    parser.add_argument('-output_dir', help='output directory')
    parser.add_argument('-output_file', default=None, help='output file')
    parser.add_argument('-start_index', help='start index', type=int, default=None)
    parser.add_argument('-end_index', help='end index', type=int, default=None)
    return parser


def check_and_convert_between_mp3_and_wav(input_fname, current_fmt='mp3', desired_fmt='wav'):
    input_fmt, output_fmt = f'.{current_fmt}', f'.{desired_fmt}'
    output_file = input_fname.replace(input_fmt, output_fmt)
    if not os.path.exists(input_fname):
        print(f'converting {input_fname}: {input_fmt}->{output_fmt}...')
        try:
            if input_fmt == 'mp3':
                sound = AudioSegment.from_mp3(input_fname)
                sound.export(output_file, format="wav")
            else:
                sound = AudioSegment.from_wav(input_fname)
                sound.export(output_file, format="mp3")
        except CouldntDecodeError:
            print('failed to convert ' + input_fname)
            return None
    return output_file


def get_files_to_transcribe(args):
    """
    Helper function to get the files to transcribe.
    Reads in the files from a CSV, a directory, or a single file.
        (if CSV is provided, then the input directory serves to give us a starting-point for the files.)
        (otherwise, we just glob all the files in the directory.)

    Returns list of tuples (input_file, output_file).
    Output file the same as input file, with "_transcribed.midi".
       If no output directory is provided, it is placed in the same directory.
       Otherwise, it is placed in the output directory.
       The same file hierarchy is maintained.

    :param args: argparse.ArgumentParser
    :return

    """
    # get files to transcribe

    # if just one filename is provided, format it as a list
    if args.input_file_to_transcribe is not None:
        files_to_transcribe = [args.input_file_to_transcribe]

    # get a list of files from a CSV
    elif args.input_files_map is not None:
        files_to_transcribe = pd.read_csv(args.input_files_map)
        if args.split is not None:
            files_to_transcribe = files_to_transcribe.loc[lambda df: df[args.split_col_name] == args.split]
        if args.dataset is not None:
            files_to_transcribe = files_to_transcribe.loc[lambda df: df[args.dataset_col_name] == args.dataset]
        files_to_transcribe = files_to_transcribe[args.file_col_name].tolist()
        if args.input_dir_to_transcribe is not None:
            files_to_transcribe = list(map(lambda x: os.path.join(args.input_dir_to_transcribe, x), files_to_transcribe))

    # get all files in a directory
    elif args.input_dir_to_transcribe is not None:
        files_to_transcribe = (
                glob.glob(os.path.join(args.input_dir_to_transcribe, '**', '*.mp3'), recursive=True) +
                glob.glob(os.path.join(args.input_dir_to_transcribe, '**', '*.wav'), recursive=True)
        )

    # convert file-types
    if args.input_file_format is not None:
        # make sure all the files of mp3 are converted to wav, or v.v.
        other_fmt = 'mp3' if args.input_file_format == 'wav' else 'wav'
        files_to_convert = list(filter(lambda x: os.path.splitext(x)[1] == other_fmt, files_to_transcribe))
        print(f'converting {len(files_to_convert)} files...')
        for f in files_to_convert:
            check_and_convert_between_mp3_and_wav(f, current_fmt=other_fmt, desired_fmt=args.input_file_format)
    else:
        # input format doesn't matter, so we just want 1 of each
        files_to_transcribe = list(unique_everseen(files_to_transcribe, key=lambda x: os.path.splitext(x)[0]))

    # apply cutoffs
    if (args.start_index is not None) or (args.end_index is not None):
        if args.start_index is None:
            args.start_index = 0
        if args.end_index is None:
            args.end_index = len(files_to_transcribe)
        files_to_transcribe = files_to_transcribe[args.start_index:args.end_index]

    # format output
    if args.output_file is not None:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        return (files_to_transcribe[0], args.output_file)

    # if the output directory is not provided, then we just put the output files in the same directory
    # otherwise, we output to the output directory, preserving the hierarchy of the original files.
    output_files = list(map(lambda x: f"{os.path.splitext(x)[0]}_transcribed.midi", files_to_transcribe))
    if args.output_dir is not None:
        if args.input_dir_to_transcribe is not None:
            output_files = list(map(lambda x: x[len(args.input_dir_to_transcribe):], output_files))
        output_files = list(map(lambda x: os.path.join(args.output_dir, x), output_files))
        for o in output_files:
            os.makedirs(os.path.dirname(o), exist_ok=True)

    # shuffle
    output = list(zip(files_to_transcribe, output_files))
    output = list(filter(lambda x: not os.path.exists(x[1]), output))
    random.shuffle(output)
    return output



