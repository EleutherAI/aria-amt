# Helper script to print out all files in the desired split of the MAESTRO dataset.
import pandas as pd
import os
import shutil
from amt.audio import AudioTransform

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-split', type=str, required=True, help='Split to print out.')
    parser.add_argument('-maestro_dir', type=str, required=True, help='Directory of the MAESTRO dataset.')
    parser.add_argument('-output_dir', type=str, required=True, help='Output directory.')
    parser.add_argument('-apply_augmentation', action='store_true', default=False, help='Apply augmentation to the files.')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.apply_augmentation:
        audio_transformer = AudioTransform(
            reverb_factor=1,
            min_snr=20,
            max_snr=50,
            max_dist_gain=25,
            min_dist_gain=0,
            noise_ratio=0.95,
            reverb_ratio=0.95,
            applause_ratio=0.01,
            bandpass_ratio=0.15,
            distort_ratio=0.15,
            reduce_ratio=0.01,
            detune_ratio=0.1,
            detune_max_shift=0.15,
            spec_aug_ratio=0.5,
        )

    # Load the split
    maestro_df = pd.read_csv(os.path.join(args.maestro_dir, 'maestro-v3.0.0.csv'))
    split_df = maestro_df.loc[lambda df: df['split'] == args.split]
    for _, row in split_df.iterrows():
        shutil.copy(
            os.path.join(args.maestro_dir, row['audio_filename']),
            os.path.join(args.output_dir, row['audio_filename'])
        )

