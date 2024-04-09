#!/bin/sh
#SBATCH --output=dtw_transcription__%x.%j.out
#SBATCH --error=dtw_transcription__%x.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=100GB
#SBATCH --cpus-per-gpu=20
#SBATCH --partition=isi


python baselines/hft_transformer/transcribe_new_files.py \
    -input_dir_to_transcribe ../../music-corpora/ \
    -input_files_map other-dataset-splits.csv \
    -split_col_name split \
    -split test \
    -output_dir hft_transformer-evaluation-data/ \
    -file_col_name audio_path

python baselines/giantmidi/transcribe_new_files.py \
    -input_dir_to_transcribe ../../music-corpora/ \
    -input_files_map other-dataset-splits.csv \
    -split_col_name split \
    -split test \
    -output_dir giantmidi-evaluation-data/ \
    -file_col_name audio_path

conda activate py311
python baselines/google_t5/transcribe_new_files.py \
    -input_dir_to_transcribe ../../music-corpora/ \
    -input_files_map other-dataset-splits.csv \
    -split_col_name split \
    -split test \
    -output_dir google-evaluation-data/ \
    -file_col_name audio_path