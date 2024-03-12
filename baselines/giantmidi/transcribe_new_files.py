import os
import argparse
import time
import torch
import piano_transcription_inference
import glob


def transcribe_piano(mp3s_dir, midis_dir, begin_index=None, end_index=None):
    """Transcribe piano solo mp3s to midi files."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(midis_dir, exist_ok=True)

    # Transcriptor
    transcriptor = piano_transcription_inference.PianoTranscription(device=device)

    transcribe_time = time.time()
    for n, mp3_path in enumerate(glob.glob(os.path.join(mp3s_dir, '*.mp3'))[begin_index:end_index]):
        print(n, mp3_path)
        midi_file = os.path.basename(mp3_path).replace('.mp3', '.midi')
        midi_path = os.path.join(midis_dir, midi_file)
        if os.path.exists(midi_path):
            continue

        (audio, _) = (
            piano_transcription_inference
                .load_audio(mp3_path, sr=piano_transcription_inference.sample_rate, mono=True)
        )

        try:
            # Transcribe
            transcribed_dict = transcriptor.transcribe(audio, midi_path)
            print(transcribed_dict)
        except:
            print('Failed for this audio!')

    print('Time: {:.3f} s'.format(time.time() - transcribe_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    parser.add_argument('--mp3s_dir', type=str, required=True, help='')
    parser.add_argument('--midis_dir', type=str, required=True, help='')
    parser.add_argument(
        '--begin_index', type=int, required=False,
        help='File num., of an ordered list of files, to start transcribing from.', default=None
    )
    parser.add_argument(
        '--end_index', type=int, required=False, default=None,
        help='File num., of an ordered list of files, to end transcription.'
    )

    # Parse arguments
    args = parser.parse_args()
    transcribe_piano(
        mp3s_dir=args.mp3s_dir,
        midis_dir=args.midis_dir,
        begin_index=args.begin_index,
        end_index=args.end_index
    )

"""
python transcribe_new_files.py \
     transcribe_piano \
     --mp3s_dir /mnt/data10/spangher/aira-dl/hFT-Transformer/evaluation/glenn-gould-bach-data \
     --midis_dir /mnt/data10/spangher/aira-dl/hFT-Transformer/evaluation/glenn-gould-bach-data/kong-model
"""