#! python
import os
import argparse
import json
import sys
import glob
from baselines.hft_transformer.src import amt
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import random
import torch
here = os.path.dirname(os.path.abspath(__file__))


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

def check_and_convert_mp3_to_wav(fname):
    wav_file = fname.replace('.mp3', '.wav')
    if not os.path.exists(wav_file):
        print('converting ' + fname + ' to .wav...')
        try:
            sound = AudioSegment.from_mp3(fname)
            sound.export(fname.replace('.mp3', '.wav'), format="wav")
        except CouldntDecodeError:
            print('failed to convert ' + fname)
            return None
    return wav_file


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

    a_feature = AMT.wav2feature(fname)

    # transcript
    if n_stride > 0:
        output = AMT.transcript_stride(a_feature, n_stride, mode=mode, ablation_flag=ablation)
    else:
        output = AMT.transcript(a_feature, mode=mode, ablation_flag=ablation)
    (output_1st_onset, output_1st_offset, output_1st_mpe, output_1st_velocity,
     output_2nd_onset, output_2nd_offset, output_2nd_mpe, output_2nd_velocity) = output

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
    parser.add_argument('-input_dir_to_transcribe', default=None, help='file list')
    parser.add_argument('-input_file_to_transcribe', default=None, help='one file')
    parser.add_argument('-output_dir', help='output directory')
    parser.add_argument('-output_file', default=None, help='output file')
    parser.add_argument('-f_config', help='config json file', default=None)
    parser.add_argument('-model_file', help='input model file', default=None)
    parser.add_argument('-start_index', help='start index', type=int, default=None)
    parser.add_argument('-end_index', help='end index', type=int, default=None)
    parser.add_argument('-skip_transcribe_mp3', action='store_true', default=False)
    # parameters
    parser.add_argument('-mode', help='mode to transcript (combination|single)', default='combination')
    parser.add_argument('-thred_mpe', help='threshold value for mpe detection', type=float, default=0.5)
    parser.add_argument('-thred_onset', help='threshold value for onset detection', type=float, default=0.5)
    parser.add_argument('-thred_offset', help='threshold value for offset detection', type=float, default=0.5)
    parser.add_argument('-n_stride', help='number of samples for offset', type=int, default=0)
    parser.add_argument('-ablation', help='ablation mode', action='store_true')
    args = parser.parse_args()

    assert (args.input_dir_to_transcribe is not None) or (args.input_file_to_transcribe is not None), "input file or directory is not specified"

    if args.input_dir_to_transcribe is not None:
        if not args.skip_transcribe_mp3:
            # list file
            a_mp3s = (
                    glob.glob(os.path.join(args.input_dir_to_transcribe, '*.mp3')) +
                    glob.glob(os.path.join(args.input_dir_to_transcribe, '*', '*.mp3'))
            )
            print(f'transcribing {len(a_mp3s)} files: [{str(a_mp3s)}]...')
            list(map(check_and_convert_mp3_to_wav, a_mp3s))

        a_list = (
            glob.glob(os.path.join(args.input_dir_to_transcribe, '*.wav')) +
            glob.glob(os.path.join(args.input_dir_to_transcribe, '*', '*.wav'))
        )
        if (args.start_index is not None) or (args.end_index is not None):
            if args.start_index is None:
                args.start_index = 0
            if args.end_index is None:
                args.end_index = len(a_list)
            a_list = a_list[args.start_index:args.end_index]
        # shuffle a_list
        random.shuffle(a_list)

    elif args.input_file_to_transcribe is not None:
        args.input_file_to_transcribe = check_and_convert_mp3_to_wav(args.input_file_to_transcribe)
        if args.input_file_to_transcribe is None:
            sys.exit()
        a_list = [args.input_file_to_transcribe]
        print(f'transcribing {str(a_list)} files...')

    # load model
    AMT = get_AMT(args.f_config, args.model_file)

    long_filename_counter = 0
    for fname in a_list:
        if args.output_file is not None:
            output_fname = args.output_file
        else:
            output_fname = fname.replace('.wav', '')
            if len(output_fname) > 200:
                output_fname = output_fname[:200] + f'_fnabbrev-{long_filename_counter}'
            output_fname += '_transcribed.mid'
            output_fname = os.path.join(args.output_dir, os.path.basename(output_fname))
            if os.path.exists(output_fname):
                continue

        print('[' + fname + ']')
        try:
            transcribe_file(
                fname,
                output_fname,
                args.mode,
                args.thred_mpe,
                args.thred_onset,
                args.thred_offset,
                args.n_stride,
                args.ablation,
                AMT,
            )
        except Exception as e:
            print(e)
            continue

    print('** done **')


"""
e.g. usage:

python evaluation/transcribe_new_files.py \
    -input_dir_to_transcribe evaluation/glenn-gould-bach-data \
    -output_dir hft-evaluation-data/ \
"""
