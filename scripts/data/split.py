import csv
import random
import glob
import argparse
import os

from typing import Callable


def guitarset_file_hook(audio_path: str):
    base, ext = os.path.splitext(audio_path)

    return [base + "_mic" + ext, base + "_mix" + ext]


def gaps_file_hook(audio_path: str):
    base, ext = os.path.splitext(audio_path)
    assert base.endswith("-fine-aligned")

    return [base[: -len("-fine-aligned")] + ext]


def get_hook(hook_name: str):
    name_to_fn = {
        "guitarset": guitarset_file_hook,
        "gaps": gaps_file_hook,
    }

    return name_to_fn[hook_name]


def get_matched_paths(
    audio_dir: str,
    mid_dir: str,
    midi_ex: str,
    audio_ex: str,
    audio_hook: Callable | None = None,
):
    # Assume that the files have the same path relative to their directory
    res = []
    mid_paths = glob.glob(
        os.path.join(mid_dir, f"**/*.{midi_ex}"), recursive=True
    )
    print(f"found {len(mid_paths)} .{midi_ex} files")

    audio_dir_last = os.path.basename(audio_dir)
    mid_dir_last = os.path.basename(mid_dir)

    for mid_path in mid_paths:
        input_rel_path = os.path.relpath(mid_path, mid_dir)

        audio_rel_path = os.path.splitext(input_rel_path)[0] + f".{audio_ex}"
        audio_path = os.path.join(audio_dir, audio_rel_path)

        if audio_hook is not None:
            audio_paths = audio_hook(audio_path)
            audio_rel_paths = audio_hook(audio_rel_path)
        else:
            audio_paths = [audio_path]
            audio_rel_paths = [audio_rel_path]

        for _audio_path, _audio_rel_path in zip(audio_paths, audio_rel_paths):
            if os.path.isfile(_audio_path):
                matched_mid_path = os.path.join(mid_dir_last, input_rel_path)
                matched_audio_path = os.path.join(
                    audio_dir_last, _audio_rel_path
                )

                # print((matched_audio_path, matched_mid_path))
                res.append((matched_audio_path, matched_mid_path))

    print(f"found {len(res)} matched audio-midi pairs")
    assert len(mid_paths) <= len(res), "audio files missing"

    return res


def create_csv(matched_paths, csv_path, ratio):
    split_csv = open(csv_path, "w")
    csv_writer = csv.writer(split_csv)
    csv_writer.writerow(["mid_path", "audio_path", "split"])

    for audio_path, mid_path in matched_paths:
        if random.random() < ratio:
            csv_writer.writerow([mid_path, audio_path, "test"])
        else:
            csv_writer.writerow([mid_path, audio_path, "train"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mid_dir", type=str)
    parser.add_argument("-audio_dir", type=str)
    parser.add_argument("-csv_path", type=str)
    parser.add_argument(
        "-midi_ex",
        type=str,
        choices=["mid", "midi"],
        default="mid",
        help="File extension of the MIDI files",
    )
    parser.add_argument(
        "-audio_ex",
        type=str,
        choices=["mp3", "wav"],
        default="mp3",
        help="File extension of the audio files",
    )
    parser.add_argument(
        "-hook",
        type=str,
        choices=["guitarset", "gaps"],
        help="use dataset specific hook for audio filenames",
        required=False,
    )
    parser.add_argument("-ratio", type=int, default=0.1)
    args = parser.parse_args()

    matched_paths = get_matched_paths(
        args.audio_dir,
        args.mid_dir,
        args.midi_ex,
        args.audio_ex,
        audio_hook=get_hook(args.hook) if args.hook else None,
    )

    create_csv(matched_paths, args.csv_path, args.ratio)
