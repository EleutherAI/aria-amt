#!/usr/bin/env python3

import argparse
import sys
import os

from csv import DictReader


# TODO: Implement a way of inferring the tokenizer name automatically
def _parse_maestro_args():
    argp = argparse.ArgumentParser(prog="amt maestro")
    argp.add_argument("dir", help="MAESTRO directory path")
    argp.add_argument("csv", help="MAESTRO csv path")
    argp.add_argument("-train", help="train save path", required=True)
    argp.add_argument("-val", help="val save path", required=True)
    argp.add_argument("-test", help="test save path", required=True)
    argp.add_argument(
        "-mp",
        help="number of processes to use",
        type=int,
        required=False,
    )

    return argp.parse_args(sys.argv[2:])


def build_maestro(args):
    from amt.data import AmtDataset

    assert os.path.isdir(args.dir), "MAESTRO directory not found"
    assert os.path.isfile(args.csv), "MAESTRO csv not found"
    if (
        os.path.isfile(args.train)
        or os.path.isfile(args.val)
        or os.path.isfile(args.test)
    ):
        print("Dataset files already exist - overwriting")

    matched_paths_train = []
    matched_paths_val = []
    matched_paths_test = []
    with open(args.csv, "r") as f:
        dict_reader = DictReader(f)
        for entry in dict_reader:
            audio_path = os.path.normpath(
                os.path.join(args.dir, entry["audio_filename"])
            )
            midi_path = os.path.normpath(
                os.path.join(args.dir, entry["midi_filename"])
            )

            if not os.path.isfile(audio_path) or not os.path.isfile(audio_path):
                print("File missing - skipping")
                print(audio_path)
                print(midi_path)
                continue

            if entry["split"] == "train":
                matched_paths_train.append((audio_path, midi_path))
            elif entry["split"] == "validation":
                matched_paths_val.append((audio_path, midi_path))
            elif entry["split"] == "test":
                matched_paths_test.append((audio_path, midi_path))
            else:
                print("Invalid split")

    print(f"Building {args.train}")
    AmtDataset.build(
        matched_load_paths=matched_paths_train,
        save_path=args.train,
        num_processes=args.mp,
    )
    print(f"Building {args.val}")
    AmtDataset.build(
        matched_load_paths=matched_paths_val,
        save_path=args.val,
        num_processes=args.mp,
    )
    print(f"Building {args.test}")
    AmtDataset.build(
        matched_load_paths=matched_paths_test,
        save_path=args.test,
        num_processes=args.mp,
    )


def main():
    # Nested argparse inspired by - https://shorturl.at/kuKW0
    parser = argparse.ArgumentParser(usage="amt <command> [<args>]")
    parser.add_argument(
        "command",
        help="command to run",
        choices=("maestro",),
    )

    # parse_args defaults to [1:] for args, but you need to
    # exclude the rest of the args too, or validation will fail
    args = parser.parse_args(sys.argv[1:2])

    if not hasattr(args, "command"):
        parser.print_help()
        print("Unrecognized command")
        exit(1)
    elif args.command == "maestro":
        build_maestro(args=_parse_maestro_args())
    else:
        print("Unrecognized command")
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
