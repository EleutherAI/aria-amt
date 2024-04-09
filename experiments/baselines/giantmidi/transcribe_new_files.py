import os
import argparse
import time
import torch
import piano_transcription_inference
import glob
from more_itertools import unique_everseen
from  tqdm.auto import tqdm
from random import shuffle
import sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '../..'))
import loader_util


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    parser = loader_util.add_io_arguments(parser)
    args = parser.parse_args()

    files_to_transcribe = loader_util.get_files_to_transcribe(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transcriptor = piano_transcription_inference.PianoTranscription(device=device)

    # Transcriptor
    for n, (input_fname, output_fname) in tqdm(enumerate(files_to_transcribe), total=len(files_to_transcribe)):
        if os.path.exists(output_fname):
            continue

        now_start = time.time()
        (audio, _) = (piano_transcription_inference
                            .load_audio(input_fname, sr=piano_transcription_inference.sample_rate, mono=True))
        print(f'READING ELAPSED TIME: {time.time() - now_start}')
        now_read = time.time()
        try:
            # Transcribe
            transcribed_dict = transcriptor.transcribe(audio, output_fname)
        except:
            print('Failed for this audio!')
        print(f'TRANSCRIPTION ELAPSED TIME: {time.time() - now_read}')
        print(f'TOTAL ELAPSED TIME: {time.time() - now_start}')



"""
python transcribe_new_files.py \
     --input_dir_to_transcribe /mnt/data10/spangher/aira-dl/hFT-Transformer/evaluation/glenn-gould-bach-data \
     --output_dir /mnt/data10/spangher/aira-dl/hFT-Transformer/evaluation/glenn-gould-bach-data/kong-model
"""