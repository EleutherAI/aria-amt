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


def _parse_transcribe_args():
    argp = argparse.ArgumentParser(prog="amt transcribe")
    argp.add_argument("model_name", help="name of model config file")
    argp.add_argument("cp", help="checkpoint path")
    argp.add_argument("load_path", help="wav file load path")
    argp.add_argument("save_path", help="midi file save path")

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


def transcribe(args):
    from torch.cuda import is_available as cuda_is_available
    from amt.tokenizer import AmtTokenizer
    from amt.inference import greedy_sample
    from amt.config import load_model_config
    from amt.model import ModelConfig, AmtEncoderDecoder
    from aria.data.midi import MidiDict
    from aria.utils import _load_weight

    assert os.path.isfile(args.load_path), "audio file not found"
    assert os.path.isfile(args.cp), "model checkpoint file not found"

    if not cuda_is_available():
        print("CUDA device is not available. Using CPU instead.")
        device = "cpu"
    else:
        device = "cuda"

    tokenizer = AmtTokenizer()
    model_config = ModelConfig(**load_model_config(args.model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model = AmtEncoderDecoder(model_config)
    model_state = _load_weight(ckpt_path=args.cp, device=device)
    model.load_state_dict(model_state)

    mid_dict = greedy_sample(model=model, audio_path=args.load_path)
    mid = mid_dict.to_midi()
    mid.save(args.save_path)


def main():
    # Nested argparse inspired by - https://shorturl.at/kuKW0
    parser = argparse.ArgumentParser(usage="amt <command> [<args>]")
    parser.add_argument(
        "command",
        help="command to run",
        choices=("maestro", "transcribe"),
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
    elif args.command == "transcribe":
        transcribe(args=_parse_transcribe_args())
    else:
        print("Unrecognized command")
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
