#!/usr/bin/env python3

import argparse
import sys
import os
import glob

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
    argp.add_argument("-load_path", help="path to mp3/wav file", required=False)
    argp.add_argument(
        "-load_dir", help="dir containing mp3/wav files", required=False
    )
    argp.add_argument("-save_dir", help="dir to save midi files", required=True)
    argp.add_argument(
        "-multi_gpu", help="use all GPUs", action="store_true", default=False
    )
    argp.add_argument("-bs", help="batch size", type=int, default=16)

    return argp.parse_args(sys.argv[2:])


def build_maestro(args):
    from amt.data import AmtDataset

    assert os.path.isdir(args.dir), "MAESTRO directory not found"
    assert os.path.isfile(args.csv), "MAESTRO csv not found"
    if os.path.isfile(args.train):
        print(f"Dataset file already exists at {args.train} - removing")
        os.remove(args.train)
    if os.path.isfile(args.val):
        print(f"Dataset file already exists at {args.val} - removing")
        os.remove(args.val)
    if os.path.isfile(args.test):
        print(f"Dataset file already exists at {args.test} - removing")
        os.remove(args.test)

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
    import torch
    from torch.cuda import is_available as cuda_is_available
    from amt.tokenizer import AmtTokenizer
    from amt.infer import batch_transcribe
    from amt.config import load_model_config
    from amt.model import ModelConfig, AmtEncoderDecoder
    from aria.utils import _load_weight

    assert cuda_is_available(), "CUDA device not found"
    assert os.path.isfile(args.cp), "model checkpoint file not found"
    assert args.load_path or args.load_dir, "must give either load path or dir"
    if args.load_path:
        assert os.path.isfile(args.load_path), f"audio file not found: {args.load_path}"
        trans_mode = "single"
    if args.load_dir:
        assert os.path.isdir(args.load_dir), "load directory doesn't exist"
        trans_mode = "batch"
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    assert os.path.isdir(args.save_dir), "save dir doesn't exist"

    # Setup model
    tokenizer = AmtTokenizer()
    model_config = ModelConfig(**load_model_config(args.model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model = AmtEncoderDecoder(model_config)
    model_state = _load_weight(ckpt_path=args.cp)

    # Fix keys in compiled model checkpoint
    _model_state = {}
    for k, v in model_state.items():
        if k.startswith("_orig_mod."):
            _model_state[k[len("_orig_mod.") :]] = v
        else:
            _model_state[k] = v
    model_state = _model_state
    model.load_state_dict(model_state)
    torch.multiprocessing.set_start_method("spawn")

    if trans_mode == "batch":
        found_wav = glob.glob(
            os.path.join(args.load_dir, "**/*.wav"), recursive=True
        )
        found_mp3 = glob.glob(
            os.path.join(args.load_dir, "**/*.mp3"), recursive=True
        )
        print(f"Found {len(found_mp3)} mp3 and {len(found_wav)} wav files")
        file_paths = found_mp3 + found_wav
    else:
        file_paths = [args.load_path]

    if args.multi_gpu:
        # Generate chunks
        gpu_ids = [
            int(id) for id in os.getenv("CUDA_VISIBLE_DEVICES").split(",")
        ]
        num_gpus = len(gpu_ids)
        print(f"Visible gpu_ids: {gpu_ids}")

        chunk_size = (len(file_paths) // num_gpus) + 1
        chunks = [
            file_paths[i : i + chunk_size]
            for i in range(0, len(file_paths), chunk_size)
        ]
        print(f"Split {len(file_paths)} files into {len(chunks)} chunks")

        processes = []
        for idx, chunk in enumerate(chunks):
            print(
                f"Starting process on cuda-{idx}: {len(chunk)} files to process"
            )
            process = torch.multiprocessing.Process(
                target=batch_transcribe,
                args=(
                    chunk,
                    model,
                    args.save_dir,
                    args.bs,
                    gpu_ids[idx],
                    args.load_dir,
                ),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

    else:
        batch_transcribe(
            file_paths=file_paths,
            model=model,
            save_dir=args.save_dir,
            batch_size=args.bs,
            input_dir=args.load_dir,
        )


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
