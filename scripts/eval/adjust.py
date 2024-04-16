import argparse
import glob
import os

from aria.data.midi import MidiDict


def get_matched_paths(orig_dir: str, adj_dir: str):
    # Assume that the files have the same path relative to their directory
    res = []
    orig_paths = glob.glob(os.path.join(orig_dir, "**/*.mid"), recursive=True)
    print(f"found {len(orig_paths)} mid files")

    for mid_path in orig_paths:
        orig_rel_path = os.path.relpath(mid_path, orig_dir)
        adj_path = os.path.join(adj_dir, orig_rel_path)
        orig_path = os.path.join(orig_dir, orig_rel_path)

        res.append((os.path.abspath(orig_path), os.path.abspath(adj_path)))

    print(f"found {len(res)} matched mp3-midi pairs")
    assert len(orig_paths) == len(res)

    return res


def adjust_mid(orig_path: str, adj_path: str):
    assert os.path.isfile(adj_path) is False
    mid_dict = MidiDict.from_midi(orig_path)
    mid_dict.resolve_pedal()
    mid = mid_dict.to_midi()

    os.makedirs(os.path.dirname(adj_path), exist_ok=True)
    mid.save(adj_path)


def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate MP3 files based on audio content."
    )
    parser.add_argument(
        "orig_dir", type=str, help="Directory to scan for duplicate MP3 files."
    )
    parser.add_argument(
        "adj_dir", type=str, help="Directory to scan for duplicate MP3 files."
    )
    args = parser.parse_args()

    matched_paths = get_matched_paths(
        orig_dir=args.orig_dir, adj_dir=args.adj_dir
    )

    for orig_path, adj_path in matched_paths:
        adjust_mid(orig_path, adj_path)


if __name__ == "__main__":
    main()
