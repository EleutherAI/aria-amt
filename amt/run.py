#!/usr/bin/env python3

import argparse
import sys
import os
import glob

from csv import DictReader


# TODO: Implement a way of inferring the tokenizer name automatically
def _add_maestro_args(subparser):
    subparser.add_argument("dir", help="MAESTRO directory path")
    subparser.add_argument("csv", help="MAESTRO csv path")
    subparser.add_argument("-train", help="train save path", required=True)
    subparser.add_argument("-val", help="val save path", required=True)
    subparser.add_argument("-test", help="test save path", required=True)
    subparser.add_argument(
        "-mp",
        help="number of processes to use",
        type=int,
        required=False,
    )


def _add_transcribe_args(subparser):
    subparser.add_argument("model_name", help="name of model config file")
    subparser.add_argument('checkpoint_path', help="checkpoint path")
    subparser.add_argument("-load_path", help="path to mp3/wav file", required=False)
    subparser.add_argument(
        "-load_dir", help="dir containing mp3/wav files", required=False
    )
    subparser.add_argument("-save_dir", help="dir to save midi files", required=True)
    subparser.add_argument(
        "-multi_gpu", help="use all GPUs", action="store_true", default=False
    )
    subparser.add_argument("-bs", help="batch size", type=int, default=16)


def build_maestro(maestro_dir, maestro_csv_file, train_file, val_file, test_file, num_procs):
    from amt.data import AmtDataset

    assert os.path.isdir(maestro_dir), "MAESTRO directory not found"
    assert os.path.isfile(maestro_csv_file), "MAESTRO csv not found"
    if os.path.isfile(train_file):
        print(f"Dataset file already exists at {train_file} - removing")
        os.remove(train_file)
    if os.path.isfile(val_file):
        print(f"Dataset file already exists at {val_file} - removing")
        os.remove(val_file)
    if os.path.isfile(test_file):
        print(f"Dataset file already exists at {test_file} - removing")
        os.remove(test_file)

    matched_paths_train = []
    matched_paths_val = []
    matched_paths_test = []
    with open(maestro_csv_file, "r") as f:
        dict_reader = DictReader(f)
        for entry in dict_reader:
            audio_path = os.path.normpath(
                os.path.join(maestro_dir, entry["audio_filename"])
            )
            midi_path = os.path.normpath(
                os.path.join(maestro_dir, entry["midi_filename"])
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

    print(f"Building {train_file}")
    AmtDataset.build(
        matched_load_paths=matched_paths_train,
        save_path=train_file,
        num_processes=num_procs,
    )
    print(f"Building {val_file}")
    AmtDataset.build(
        matched_load_paths=matched_paths_val,
        save_path=val_file,
        num_processes=num_procs,
    )
    print(f"Building {test_file}")
    AmtDataset.build(
        matched_load_paths=matched_paths_test,
        save_path=test_file,
        num_processes=num_procs,
    )


def transcribe(
        model_name, checkpoint_path, save_dir, load_path=None, load_dir=None,
        batch_size=16, multi_gpu=False,
        augment=None,
):
    """
    Transcribe audio files to midi using the given model and checkpoint.

    Parameters
    ----------
    model_name : str
        Name of the model config file
    checkpoint_path : str
        Path to the model checkpoint
    save_dir : str
        Directory to save the transcribed midi files
    load_path : str
        Name of the audio file to transcribe (if specified, don't specify load_dir)
    load_dir : str
        Directory containing audio files to transcribe (if specified, don't specify load_path)
    batch_size : int
        Batch size to use for transcription
    multi_gpu : bool
        Use all available GPUs for transcription
    augment : str
        Augment the audio files before transcribing. This is used for evaluation. This tests the robustness of the model.
    """
    import torch
    from torch.cuda import is_available as cuda_is_available
    from amt.tokenizer import AmtTokenizer
    from amt.infer import batch_transcribe
    from amt.config import load_model_config
    from amt.model import ModelConfig, AmtEncoderDecoder
    from aria.utils import _load_weight

    assert cuda_is_available(), "CUDA device not found"
    assert os.path.isfile(checkpoint_path), "model checkpoint file not found"
    assert load_path or load_dir, "must give either load path or dir"
    if load_path:
        assert os.path.isfile(
            load_path
        ), f"audio file not found: {load_path}"
        trans_mode = "single"
    if load_dir:
        assert os.path.isdir(load_dir), "load directory doesn't exist"
        trans_mode = "batch"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    assert os.path.isdir(save_dir), "save dir doesn't exist"

    # Setup model
    tokenizer = AmtTokenizer()
    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model = AmtEncoderDecoder(model_config)
    model_state = _load_weight(ckpt_path=checkpoint_path)

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
            os.path.join(load_dir, "**/*.wav"), recursive=True
        )
        found_mp3 = glob.glob(
            os.path.join(load_dir, "**/*.mp3"), recursive=True
        )
        print(f"Found {len(found_mp3)} mp3 and {len(found_wav)} wav files")
        file_paths = found_mp3 + found_wav
    else:
        file_paths = [load_path]

    if multi_gpu:
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
                    save_dir,
                    batch_size,
                    gpu_ids[idx],
                    load_dir,
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
            save_dir=save_dir,
            batch_size=batch_size,
            input_dir=load_dir,
        )


def main():
    # Nested argparse inspired by - https://shorturl.at/kuKW0
    parser = argparse.ArgumentParser(usage="amt <command> [<args>]")
    subparsers = parser.add_subparsers(help="sub-command help")
    # add maestro and transcribe subparsers
    subparser_maestro = subparsers.add_parser("maestro", help="Commands to build the maestro dataset.")
    subparser_transcribe = subparsers.add_parser("transcribe", help="Commands to run transcription.")
    _add_maestro_args(subparser_maestro)
    _add_transcribe_args(subparser_transcribe)

    args = parser.parse_args()

    if not hasattr(args, "command"):
        parser.print_help()
        print("Unrecognized command")
        exit(1)
    elif args.command == "maestro":
        build_maestro(
            maestro_dir=args.dir,
            maestro_csv_file=args.csv,
            train_file=args.train,
            val_file=args.val,
            test_file=args.test,
            num_procs=args.mp,
        )
    elif args.command == "transcribe":
        transcribe(
            model_name=args.model_name,
            checkpoint_path=args.checkpoint_path,
            load_path=args.load_path,
            load_dir=args.load_dir,
            save_dir=args.save_dir,
            batch_size=args.bs,
            multi_gpu=args.multi_gpu,
        )
    else:
        print("Unrecognized command")
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
