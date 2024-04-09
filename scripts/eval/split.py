import csv
import random
import glob
import argparse
import os


def get_matched_paths(audio_dir: str, mid_dir: str):
    # Assume that the files have the same path relative to their directory
    res = []
    mid_paths = glob.glob(os.path.join(mid_dir, "**/*.mid"), recursive=True)
    print(f"found {len(mid_paths)} mid files")

    audio_dir_last = os.path.basename(audio_dir)
    mid_dir_last = os.path.basename(mid_dir)

    for mid_path in mid_paths:
        input_rel_path = os.path.relpath(mid_path, mid_dir)

        mp3_rel_path = os.path.splitext(input_rel_path)[0] + ".mp3"
        mp3_path = os.path.join(audio_dir, mp3_rel_path)

        # Check if the corresponding .mp3 file exists
        if os.path.isfile(mp3_path):
            matched_mid_path = os.path.join(mid_dir_last, input_rel_path)
            matched_mp3_path = os.path.join(audio_dir_last, mp3_rel_path)

            res.append((matched_mp3_path, matched_mid_path))

    print(f"found {len(res)} matched mp3-midi pairs")
    assert len(mid_paths) == len(res), "audio files missing"

    return res


def create_csv(matched_paths, csv_path):
    split_csv = open(csv_path, "w")
    csv_writer = csv.writer(split_csv)
    csv_writer.writerow(["mid_path", "audio_path", "split"])

    for audio_path, mid_path in matched_paths:
        if random.random() < 0.1:
            csv_writer.writerow([mid_path, audio_path, "test"])
        else:
            csv_writer.writerow([mid_path, audio_path, "train"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mid_dir", type=str)
    parser.add_argument("-audio_dir", type=str)
    parser.add_argument("-csv_path", type=str)
    args = parser.parse_args()

    matched_paths = get_matched_paths(args.audio_dir, args.mid_dir)

    create_csv(matched_paths, args.csv_path)
