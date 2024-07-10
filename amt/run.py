#!/usr/bin/env python3

import argparse
import os
import glob

from csv import DictReader

# V2 ideas:
# - When we retrain, perhaps provide many examples of non piano-audio, matched with a
#   special token which denotes a non-piano audio segment.
# - We could additionally occasionally splice a piano segment with non piano audio
#   and task the model with detecting this using the tokenizer.
# - Retrain with much larger synth dataset


# TODO: Implement a way of inferring the tokenizer name automatically
def _add_maestro_args(subparser):
    subparser.add_argument("dir", help="MAESTRO directory path")
    subparser.add_argument("-train", help="train save path", required=True)
    subparser.add_argument("-val", help="val save path", required=True)
    subparser.add_argument("-test", help="test save path", required=True)
    subparser.add_argument(
        "-mp",
        help="number of processes to use",
        type=int,
        default=1,
    )


def _add_matched_args(subparser):
    subparser.add_argument("audio", help="audio directory path")
    subparser.add_argument("mid", help="midi directory path")
    subparser.add_argument("csv", help="path to split.csv")
    subparser.add_argument("-train", help="train save path", required=False)
    subparser.add_argument("-val", help="val save path", required=False)
    subparser.add_argument("-test", help="test save path", required=False)
    subparser.add_argument(
        "-mp",
        help="number of processes to use",
        type=int,
        default=1,
    )


def _add_synth_args(subparser):
    subparser.add_argument("dir", help="Directory containing MIDIs")
    subparser.add_argument("csv", help="Split csv")
    subparser.add_argument("-train", help="train save path", required=True)
    subparser.add_argument("-test", help="test save path", required=True)
    subparser.add_argument(
        "-mp",
        help="number of processes to use",
        type=int,
        default=1,
    )


def _add_transcribe_args(subparser):
    subparser.add_argument("model_name", help="name of model config file")
    subparser.add_argument("checkpoint_path", help="checkpoint path")
    subparser.add_argument(
        "-load_path", help="path to mp3/wav file", required=False
    )
    subparser.add_argument(
        "-load_dir", help="dir containing mp3/wav files", required=False
    )
    subparser.add_argument(
        "-maestro",
        help="get file paths from maestro val/test sets",
        action="store_true",
        default=False,
    )
    subparser.add_argument(
        "-save_dir", help="dir to save midi files", required=True
    )
    subparser.add_argument(
        "-multi_gpu", help="use all GPUs", action="store_true", default=False
    )
    subparser.add_argument(
        "-q8",
        help="apply int8 quantization on weights",
        action="store_true",
        default=False,
    )
    subparser.add_argument(
        "-compile",
        help="use the pytorch compiler to generate a cuda graph",
        action="store_true",
        default=False,
    )
    subparser.add_argument(
        "-max_autotune",
        help="use mode=max_autotune when compiling",
        action="store_true",
        default=False,
    )
    subparser.add_argument("-bs", help="batch size", type=int, default=16)


def get_synth_mid_paths(mid_dir: str, csv_path: str):
    assert os.path.isdir(mid_dir), "directory doesn't exist"
    assert os.path.isfile(csv_path), "csv not found"

    train_paths = []
    test_paths = []
    with open(csv_path, "r") as f:
        dict_reader = DictReader(f)
        for entry in dict_reader:
            mid_path = os.path.normpath(
                os.path.join(mid_dir, entry["mid_path"])
            )

            assert os.path.isfile(mid_path), "file missing"
            if entry["split"] == "train":
                train_paths.append(mid_path)
            elif entry["split"] == "test":
                test_paths.append(mid_path)
            else:
                raise ValueError("Invalid split")

    return train_paths, test_paths


def build_synth(
    mid_dir: str,
    csv_path: str,
    train_path: str,
    test_path: str,
    num_procs: int,
):
    from amt.data import AmtDataset, pianoteq_cmd_fn

    if os.path.isfile(train_path):
        print(f"Dataset file already exists at {train_path} - removing")
        os.remove(train_path)
    if os.path.isfile(test_path):
        print(f"Dataset file already exists at {test_path} - removing")
        os.remove(test_path)

    (
        train_paths,
        test_paths,
    ) = get_synth_mid_paths(mid_dir, csv_path)

    print(f"Found {len(train_paths)} train and {len(test_paths)} test paths")

    print(f"Building {train_path}")
    AmtDataset.build(
        load_paths=train_paths,
        save_path=train_path,
        num_processes=num_procs,
        cli_cmd_fn=pianoteq_cmd_fn,
    )
    print(f"Building {test_path}")
    AmtDataset.build(
        load_paths=test_paths,
        save_path=test_path,
        num_processes=num_procs,
        cli_cmd_fn=pianoteq_cmd_fn,
    )


def _get_matched_maestro_paths(maestro_dir):
    assert os.path.isdir(maestro_dir), "MAESTRO directory not found"

    maestro_csv_path = os.path.join(maestro_dir, "maestro-v3.0.0.csv")
    assert os.path.isfile(maestro_csv_path), "MAESTRO csv not found"

    matched_paths_train = []
    matched_paths_val = []
    matched_paths_test = []
    with open(maestro_csv_path, "r") as f:
        dict_reader = DictReader(f)
        for entry in dict_reader:
            audio_path = os.path.normpath(
                os.path.join(maestro_dir, entry["audio_filename"])
            )
            midi_path = os.path.normpath(
                os.path.join(maestro_dir, entry["midi_filename"])
            )

            if not os.path.isfile(audio_path) or not os.path.isfile(midi_path):
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

    return matched_paths_train, matched_paths_val, matched_paths_test


def _get_matched_paths(audio_dir: str, mid_dir: str, split_csv_path: str):
    assert os.path.isdir(audio_dir), "audio dir not found"
    assert os.path.isdir(mid_dir), "mid dir not found"
    assert os.path.isfile(split_csv_path), "split csv not found"

    matched_paths_train = []
    matched_paths_val = []
    matched_paths_test = []
    with open(split_csv_path, "r") as f:
        dict_reader = DictReader(f)
        for entry in dict_reader:
            audio_path = os.path.normpath(
                os.path.join(audio_dir, entry["audio_path"])
            )
            mid_path = os.path.normpath(
                os.path.join(mid_dir, entry["mid_path"])
            )

            if not os.path.isfile(audio_path) or not os.path.isfile(mid_path):
                raise FileNotFoundError(
                    f"File pair missing: {(audio_path, mid_path)}"
                )

            if entry["split"] == "train":
                matched_paths_train.append((audio_path, mid_path))
            elif entry["split"] == "val":
                matched_paths_val.append((audio_path, mid_path))
            elif entry["split"] == "test":
                matched_paths_test.append((audio_path, mid_path))
            else:
                raise ValueError("Invalid split")

    return matched_paths_train, matched_paths_val, matched_paths_test


def _build_from_matched_paths(
    matched_paths_train: list,
    matched_paths_val: list,
    matched_paths_test: list,
    train_path: str | None = None,
    val_path: str | None = None,
    test_path: str | None = None,
    num_procs: int = 1,
):
    from amt.data import AmtDataset

    if train_path is None:
        pass
    elif len(matched_paths_train) >= 1:
        if os.path.isfile(train_path):
            input(
                f"Dataset file already exists at {train_path} - Press enter to continue (^C to quit)"
            )
            os.remove(train_path)
        print(f"Building {train_path}")
        AmtDataset.build(
            load_paths=matched_paths_train,
            save_path=train_path,
            num_processes=num_procs,
        )
    if val_path is None:
        pass
    elif len(matched_paths_val) >= 1:
        if os.path.isfile(val_path):
            input(
                f"Dataset file already exists at {val_path} - Press enter to continue (^C to quit)"
            )
            os.remove(val_path)
        print(f"Building {val_path}")
        AmtDataset.build(
            load_paths=matched_paths_val,
            save_path=val_path,
            num_processes=num_procs,
        )
    if test_path is None:
        pass
    elif len(matched_paths_test) >= 1 and test_path:
        if os.path.isfile(test_path):
            input(
                f"Dataset file already exists at {test_path} - Press enter to continue (^C to quit)"
            )
            os.remove(test_path)
        print(f"Building {test_path}")
        AmtDataset.build(
            load_paths=matched_paths_test,
            save_path=test_path,
            num_processes=num_procs,
        )


def build_from_csv(
    audio_dir: str,
    mid_dir: str,
    split_csv_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
    num_procs: int,
):
    (
        matched_paths_train,
        matched_paths_val,
        matched_paths_test,
    ) = _get_matched_paths(
        audio_dir=audio_dir,
        mid_dir=mid_dir,
        split_csv_path=split_csv_path,
    )

    print(
        f"Found {len(matched_paths_train)}, {len(matched_paths_val)}, {len(matched_paths_test)} train, val, and test paths"
    )

    _build_from_matched_paths(
        matched_paths_train=matched_paths_train,
        matched_paths_val=matched_paths_val,
        matched_paths_test=matched_paths_test,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        num_procs=num_procs,
    )


def build_maestro(
    maestro_dir: str,
    train_path: str,
    val_path: str,
    test_path: str,
    num_procs: int,
):
    (
        matched_paths_train,
        matched_paths_val,
        matched_paths_test,
    ) = _get_matched_maestro_paths(maestro_dir=maestro_dir)

    _build_from_matched_paths(
        matched_paths_train=matched_paths_train,
        matched_paths_val=matched_paths_val,
        matched_paths_test=matched_paths_test,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        num_procs=num_procs,
    )


def transcribe(
    model_name: str,
    checkpoint_path: str,
    save_dir: str,
    load_path: str | None = None,
    load_dir: str | None = None,
    maestro: bool = False,
    batch_size: int = 8,
    multi_gpu: bool = False,
    quantize: bool = False,
    compile_mode: str | bool = False,
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
    from torch.cuda import is_available as cuda_is_available
    from amt.tokenizer import AmtTokenizer
    from amt.inference.transcribe import batch_transcribe
    from amt.config import load_model_config
    from amt.inference.model import ModelConfig, AmtEncoderDecoder
    from aria.utils import _load_weight

    assert cuda_is_available(), "CUDA device not found"
    assert os.path.isfile(checkpoint_path), "model checkpoint file not found"
    assert load_path or load_dir, "must give either load path or dir"
    if load_path:
        assert os.path.isfile(load_path), f"audio file not found: {load_path}"
        trans_mode = "single"
    if load_dir:
        assert os.path.isdir(load_dir), "load directory doesn't exist"
        if maestro is True:
            trans_mode = "maestro"
        else:
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

    if trans_mode == "batch":
        found_wav = glob.glob(
            os.path.join(load_dir, "**/*.wav"), recursive=True
        )
        found_mp3 = glob.glob(
            os.path.join(load_dir, "**/*.mp[34]"), recursive=True
        )
        print(f"Found {len(found_mp3)} mp3 and {len(found_wav)} wav files")
        file_paths = found_mp3 + found_wav
    elif trans_mode == "maestro":
        matched_train_paths, matched_val_paths, matched_test_paths = (
            _get_matched_maestro_paths(load_dir)
        )
        train_mp3_paths = [ap for ap, mp in matched_train_paths]
        val_mp3_paths = [ap for ap, mp in matched_val_paths]
        test_mp3_paths = [ap for ap, mp in matched_test_paths]
        file_paths = test_mp3_paths
        assert len(file_paths) == 177, "Invalid maestro files"
    else:
        file_paths = [load_path]
        batch_size = 1

    if multi_gpu:
        gpu_ids = [
            int(id) for id in os.getenv("CUDA_VISIBLE_DEVICES").split(",")
        ]
        print(f"Visible gpu_ids: {gpu_ids}")
        batch_transcribe(
            file_paths=file_paths,
            model=model,
            save_dir=save_dir,
            batch_size=batch_size,
            input_dir=load_dir,
            gpu_ids=gpu_ids,
            quantize=quantize,
            compile_mode=compile_mode,
        )

    else:
        batch_transcribe(
            file_paths=file_paths,
            model=model,
            save_dir=save_dir,
            batch_size=batch_size,
            input_dir=load_dir,
            quantize=quantize,
            compile_mode=compile_mode,
        )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="sub-command help", dest="command")

    subparser_maestro = subparsers.add_parser(
        "build-maestro", help="Commands to build the maestro dataset."
    )
    subparser_matched = subparsers.add_parser(
        "build-matched", help="Commands to build dataset from matched paths."
    )
    subparser_synth = subparsers.add_parser(
        "build-synth", help="Commands to build the synthetic dataset."
    )
    subparser_transcribe = subparsers.add_parser(
        "transcribe", help="Commands to run transcription."
    )
    _add_maestro_args(subparser_maestro)
    _add_matched_args(subparser_matched)
    _add_synth_args(subparser_synth)
    _add_transcribe_args(subparser_transcribe)

    args = parser.parse_args()

    if not hasattr(args, "command"):
        parser.print_help()
        print("Unrecognized command")
        exit(1)
    elif args.command == "build-maestro":
        build_maestro(
            maestro_dir=args.dir,
            train_path=args.train,
            val_path=args.val,
            test_path=args.test,
            num_procs=args.mp,
        )
    elif args.command == "build-matched":
        build_from_csv(
            audio_dir=args.audio,
            mid_dir=args.mid,
            split_csv_path=args.csv,
            train_path=args.train,
            val_path=args.val,
            test_path=args.test,
            num_procs=args.mp,
        )
    elif args.command == "build-synth":
        build_synth(
            mid_dir=args.dir,
            csv_path=args.csv,
            train_path=args.train,
            test_path=args.test,
            num_procs=args.mp,
        )
    elif args.command == "transcribe":
        transcribe(
            model_name=args.model_name,
            checkpoint_path=args.checkpoint_path,
            load_path=args.load_path,
            load_dir=args.load_dir,
            maestro=args.maestro,
            save_dir=args.save_dir,
            batch_size=args.bs,
            multi_gpu=args.multi_gpu,
            quantize=args.q8,
            compile_mode=(
                "max-autotune"
                if args.compile and args.max_autotune
                else "reduce-overhead" if args.compile else False
            ),
        )
    else:
        print("Unrecognized command")
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
