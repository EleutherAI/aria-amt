"""
Helper script to get and augment MAESTRO test according to `-augmentation_config` and `-apply_augmentation` flags.
"""
import pandas as pd
import os
import shutil
from amt.audio import AudioTransform, pad_or_trim
from amt.data import get_wav_mid_segments, load_config
import json
import librosa
import torch
import torchaudio
from tqdm.auto import tqdm

SAMPLE_RATE = load_config()['audio']['sample_rate']
AUG_BATCH_SIZE = 100

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-split', type=str, required=True, default='test', help='Split to print out.')
    parser.add_argument('-dataset', type=str, default=None, help='Dataset to use.')
    parser.add_argument('-input_file_dir', type=str, default=None, help='Directory of the dataset to use.')
    parser.add_argument(
        '-input_splits_file',
        type=str,
        required=True,
        help='Directory of the MAESTRO dataset.'
    )
    parser.add_argument('-midi_col_name', type=str, default=None, help='Column name for MIDI files.')
    parser.add_argument('-audio_col_name', type=str, default=None, help='Column name for audio files.')
    parser.add_argument('-output_dir', type=str, required=True, help='Output directory.')
    parser.add_argument('-apply_augmentation', action='store_true', default=False, help='Apply augmentation to the files.')
    parser.add_argument('-augmentation_config', type=str, default=None, help='Path to the augmentation config file.')
    parser.add_argument('-device', type=str, default='cpu', help='Device to use.')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    audio_transformer = None
    if args.apply_augmentation:
        aug_config = json.load(open(args.augmentation_config))
        audio_transformer = AudioTransform(**aug_config).to(args.device)

    # Load the split
    input_files_to_process = pd.read_csv(args.input_splits_file)
    if args.split is not None:
        input_files_to_process = input_files_to_process.loc[lambda df: df['split'] == args.split]
    if args.dataset is not None:
        input_files_to_process = input_files_to_process.loc[lambda df: df['dataset'] == args.dataset]

    # Process the files
    for _, row in tqdm(
        input_files_to_process.iterrows(),
        total=len(input_files_to_process),
        desc=f'Processing {args.split} split'
    ):
        # copy MIDI file into the output directory
        if args.midi_col_name is not None:
            midi_outfile = os.path.basename(row[args.midi_col_name])
            fname, ext = os.path.splitext(midi_outfile)
            midi_outfile = f'{fname}_gold{ext}'
            midi_outfile = os.path.join(args.output_dir, midi_outfile)
            if not os.path.exists(midi_outfile):
                shutil.copy(
                    os.path.join(args.input_file_dir, row['midi_filename']),
                    os.path.join(args.output_dir, midi_outfile)
                )

        # either just vanilla copy the audio file, or apply augmentation
        if args.audio_col_name is not None:
            audio_outfile = os.path.basename(row[args.audio_col_name])
            audio_outfile = os.path.join(args.output_dir, audio_outfile)
            audio_input_file = os.path.join(args.input_file_dir, row[args.audio_col_name])
            if not os.path.exists(audio_outfile):
                if args.apply_augmentation:
                    try:
                        segments = get_wav_mid_segments(audio_input_file)
                        segments = list(map(lambda x: x[0], segments))
                        aug_wav_parts = []
                        for i in range(0, len(segments), AUG_BATCH_SIZE):
                            batch_to_augment = torch.vstack(segments[i:i + AUG_BATCH_SIZE]).to(args.device)
                            mel = audio_transformer(batch_to_augment)
                            aug_wav = audio_transformer.inverse_log_mel(mel)
                            aug_wav_parts.append(aug_wav)
                        aug_wav = torch.vstack(aug_wav_parts)
                        aug_wav = aug_wav.reshape(1, -1).cpu()
                        torchaudio.save(audio_outfile, src=aug_wav, sample_rate=SAMPLE_RATE)
                    except Exception as e:
                        print(f'Failed to augment {audio_input_file}: {e}')
                else:
                    shutil.copy(
                        os.path.join(audio_input_file),
                        os.path.join(args.output_dir, audio_outfile)
                    )
