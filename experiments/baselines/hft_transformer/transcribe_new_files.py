#! python
import os
import argparse
import json
import sys
import glob
import random
import torch
here = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(here, 'model'))
import hft_amt as amt
import time
from random import shuffle
sys.path.append(os.path.join(here, '../..'))
import loader_util
from tqdm.auto import tqdm

_AMT = None
def get_AMT(config_file=None, model_file=None):
    global _AMT
    if _AMT is None:
        if config_file is None:
            config_file = os.path.join(here, 'model_files/config-aug.json')
        if model_file is None:
            if torch.cuda.is_available():
                model_file = os.path.join(here, 'model_files/model-with-aug-data_006_009.pkl')
            else:
                model_file = os.path.join(here, 'model_files/model-with-aug-data_006_009_cpu.bin')
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if torch.cuda.is_available():
            _AMT = amt.AMT(config, model_file, verbose_flag=False)
        else:
            model = torch.load(model_file, map_location=torch.device('cpu'))
            _AMT = amt.AMT(config, model_path=None, verbose_flag=False)
            _AMT.model = model
    return _AMT



def transcribe_file(
        fname,
        output_fname,
        mode='combination',
        thred_mpe=0.5,
        thred_onset=0.5,
        thred_offset=0.5,
        n_stride=0,
        ablation=False,
        AMT=None
):
    if AMT is None:
        AMT = get_AMT()
    now_start = time.time()
    a_feature = AMT.wav2feature(fname)
    print(f'READING ELAPSED TIME: {time.time() - now_start}')
    now_read = time.time()
    # transcript
    if n_stride > 0:
        output = AMT.transcript_stride(a_feature, n_stride, mode=mode, ablation_flag=ablation)
    else:
        output = AMT.transcript(a_feature, mode=mode, ablation_flag=ablation)
    (output_1st_onset, output_1st_offset, output_1st_mpe, output_1st_velocity,
     output_2nd_onset, output_2nd_offset, output_2nd_mpe, output_2nd_velocity) = output
    print(f'TRANSCRIPTION ELAPSED TIME: {time.time() - now_read}')
    print(f'TOTAL ELAPSED TIME: {time.time() - now_start}')
    # note (mpe2note)
    a_note_1st_predict = AMT.mpe2note(
        a_onset=output_1st_onset,
        a_offset=output_1st_offset,
        a_mpe=output_1st_mpe,
        a_velocity=output_1st_velocity,
        thred_onset=thred_onset,
        thred_offset=thred_offset,
        thred_mpe=thred_mpe,
        mode_velocity='ignore_zero',
        mode_offset='shorter'
    )

    a_note_2nd_predict = AMT.mpe2note(
        a_onset=output_2nd_onset,
        a_offset=output_2nd_offset,
        a_mpe=output_2nd_mpe,
        a_velocity=output_2nd_velocity,
        thred_onset=thred_onset,
        thred_offset=thred_offset,
        thred_mpe=thred_mpe,
        mode_velocity='ignore_zero',
        mode_offset='shorter'
    )

    AMT.note2midi(a_note_2nd_predict, output_fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # necessary arguments
    parser = loader_util.add_io_arguments(parser)
    parser.add_argument('-f_config', help='config json file', default=None)
    parser.add_argument('-model_file', help='input model file', default=None)
    # parameters
    parser.add_argument('-mode', help='mode to transcript (combination|single)', default='combination')
    parser.add_argument('-thred_mpe', help='threshold value for mpe detection', type=float, default=0.5)
    parser.add_argument('-thred_onset', help='threshold value for onset detection', type=float, default=0.5)
    parser.add_argument('-thred_offset', help='threshold value for offset detection', type=float, default=0.5)
    parser.add_argument('-n_stride', help='number of samples for offset', type=int, default=0)
    parser.add_argument('-ablation', help='ablation mode', action='store_true')
    args = parser.parse_args()

    assert (args.input_dir_to_transcribe is not None) or (args.input_file_to_transcribe is not None), "input file or directory is not specified"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    a_list = loader_util.get_files_to_transcribe(args)

    # load model
    AMT = get_AMT(args.f_config, args.model_file)

    long_filename_counter = 0
    for input_fname, output_fname in tqdm(a_list):
        if os.path.exists(output_fname):
            continue

        print(f'transcribing {input_fname} -> {output_fname}')
        try:
            transcribe_file(
                input_fname,
                output_fname,
                args.mode,
                args.thred_mpe,
                args.thred_onset,
                args.thred_offset,
                args.n_stride,
                args.ablation,
                AMT,
            )
            now = time.time()
            print(f'ELAPSED TIME: {time.time() - now}')
        except Exception as e:
            print(e)
            continue

    print('** done **')


"""
e.g. usage:

python evaluation/transcribe_new_files.py \
    -input_dir_to_transcribe evaluation/glenn-gould-bach-data \
    -output_dir hft-evaluation-data/ \

python baselines/hft_transformer/transcribe_new_files.py \
    -input_dir_to_transcribe ../../music-corpora/ \
    -input_files_map other-dataset-splits.csv \
    -split_col_name split \
    -split test \
    -output_dir hft-dtw-evaluation-data/ \
    -file_col_name audio_path 
"""
