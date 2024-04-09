#!/bin/sh
#SBATCH --output=aug_2__%x.%j.out
#SBATCH --error=aug_2__%x.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=100GB
#SBATCH --cpus-per-gpu=20
#SBATCH --partition=isi

conda activate py311
PROJ_DIR=/project/jonmay_231/spangher/Projects/aria-amt
OUTPUT_DIR="$PROJ_DIR/experiments/aug_2_files"

# process data
if [ ! -d "$OUTPUT_DIR" ]; then
  python process_maestro.py \
    -split test \
    -maestro_dir "$PROJ_DIR/../maestro-v3.0.0/maestro-v3.0.0.csv" \
    -output_dir $OUTPUT_DIR \
    -split test \
    -midi_col_name 'midi_filename' \
    -audio_col_name 'audio_filename' \
    -apply_augmentation \
    -augmentation_config "$PROJ_DIR/experiments/augmentation_configs/config_2.json" \
    -device 'cuda:0'
fi

source /home1/${USER}/.bashrc
conda activate py311

## run google inference
#echo "Running google inference"
#GOOGLE_OUTPUT_DIR="$OUTPUT_DIR/google_t5_transcriptions"
##if [ ! -d "$GOOGLE_OUTPUT_DIR" ]; then
#python baselines/google_t5/transcribe_new_files.py \
#    -input_dir_to_transcribe $OUTPUT_DIR \
#    -output_dir $GOOGLE_OUTPUT_DIR
##fi
#
#echo "Running giant midi inference"
#GIANT_MIDI_OUTPUT_DIR="$OUTPUT_DIR/giant_midi_transcriptions"
#python baselines/giantmidi/transcribe_new_files.py \
#    -input_dir_to_transcribe $OUTPUT_DIR \
#    -output_dir $GIANT_MIDI_OUTPUT_DIR

echo "Running hft inference"
HFT_OUTPUT_DIR="$OUTPUT_DIR/hft_transcriptions"
python baselines/hft_transformer/transcribe_new_files.py \
    -input_dir_to_transcribe $OUTPUT_DIR \
    -output_dir $HFT_OUTPUT_DIR