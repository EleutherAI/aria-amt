import mmap
import os
import io
import base64
import shutil
import orjson
import torch
import torchaudio

from multiprocessing import Pool, Queue, Process
from typing import Callable

from aria.data.midi import MidiDict
from amt.tokenizer import AmtTokenizer
from amt.config import load_config
from amt.audio import pad_or_trim


def get_wav_mid_segments(
    audio_path: str,
    mid_path: str = "",
    return_json: bool = False,
    stride_factor: int | None = None,
):
    """This function yields tuples of matched log mel spectrograms and
    tokenized sequences (np.array, list). If it is given only an audio path
    then it will return an empty list for the mid_feature
    """
    tokenizer = AmtTokenizer()
    config = load_config()
    sample_rate = config["audio"]["sample_rate"]
    chunk_len = config["audio"]["chunk_len"]
    num_samples = sample_rate * chunk_len
    samples_per_ms = sample_rate // 1000

    if not stride_factor:
        stride_factor = config["data"]["stride_factor"]

    if not os.path.isfile(audio_path):
        return None

    # Load midi if required
    if mid_path == "":
        midi_dict = None
    elif not os.path.isfile(mid_path):
        return None
    else:
        midi_dict = MidiDict.from_midi(mid_path)

    # Load audio
    wav, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(
            waveform=wav,
            orig_freq=sr,
            new_freq=sample_rate,
        ).mean(0)

    # Create features
    total_samples = wav.shape[-1]
    res = []
    for idx in range(
        0,
        total_samples - (num_samples - num_samples // stride_factor),
        num_samples // stride_factor,
    ):
        audio_feature = pad_or_trim(wav[idx:], length=num_samples)
        if midi_dict is not None:
            mid_feature = tokenizer._tokenize_midi_dict(
                midi_dict=midi_dict,
                start_ms=idx // samples_per_ms,
                end_ms=(idx + num_samples) / samples_per_ms,
            )
        else:
            mid_feature = []

        if return_json is True:
            audio_feature = audio_feature.tolist()

        res.append((audio_feature, mid_feature))

    return res


def write_features(audio_path: str, mid_path: str, save_path: str):
    features = get_wav_mid_segments(
        audio_path=audio_path,
        mid_path=mid_path,
        return_json=False,
    )

    # Father forgive me for I have sinned
    with open(save_path, mode="a") as file:
        for wav, seq in features:
            # Encode wav using b64 to avoid newlines
            wav_buffer = io.BytesIO()
            torch.save(wav, wav_buffer)
            wav_buffer.seek(0)
            wav_bytes = wav_buffer.read()
            wav_str = base64.b64encode(wav_bytes).decode("utf-8")
            file.write(wav_str)
            file.write("\n")

            seq_bytes = orjson.dumps(seq)
            seq_str = base64.b64encode(seq_bytes).decode("utf-8")
            file.write(seq_str)
            file.write("\n")


def get_synth_audio(cli_cmd_fn: str, mid_path: str, wav_path: str):
    _cmd = cli_cmd_fn(mid_path, wav_path)
    os.system(_cmd)


def write_synth_features(cli_cmd_fn: Callable, mid_path: str, save_path: str):
    audio_path_temp = f"{os.getpid()}_temp.wav"

    try:
        get_synth_audio(
            cli_cmd=cli_cmd_fn, mid_path=mid_path, wav_path=audio_path_temp
        )
    except:
        if os.path.isfile(audio_path_temp):
            os.remove(audio_path_temp)
        return
    else:
        features = get_wav_mid_segments(
            audio_path=audio_path_temp,
            mid_path=mid_path,
            return_json=False,
        )
        os.remove(audio_path_temp)

    with open(save_path, mode="a") as file:
        for wav, seq in features:
            wav_buffer = io.BytesIO()
            torch.save(wav, wav_buffer)
            wav_buffer.seek(0)
            wav_bytes = wav_buffer.read()
            wav_str = base64.b64encode(wav_bytes).decode("utf-8")
            file.write(wav_str)
            file.write("\n")

            seq_bytes = orjson.dumps(seq)
            seq_str = base64.b64encode(seq_bytes).decode("utf-8")
            file.write(seq_str)
            file.write("\n")


def build_worker_fn(load_path_queue, save_path_queue, _save_path: str):
    dirname, basename = os.path.split(_save_path)
    worker_save_path = os.path.join(dirname, str(os.getpid()) + basename)

    while not load_path_queue.empty():
        audio_path, mid_path = load_path_queue.get()
        write_features(audio_path, mid_path, worker_save_path)

    print("Worker", os.getpid(), "finished")
    save_path_queue.put(worker_save_path)


def build_synth_worker_fn(
    cli_cmd: Callable,
    load_path_queue,
    save_path_queue,
    _save_path: str,
):
    dirname, basename = os.path.split(_save_path)
    worker_save_path = os.path.join(dirname, str(os.getpid()) + basename)

    while not load_path_queue.empty():
        mid_path = load_path_queue.get()
        write_synth_features(cli_cmd, mid_path, worker_save_path)

    save_path_queue.put(worker_save_path)


class AmtDataset(torch.utils.data.Dataset):
    def __init__(self, load_path: str):
        self.tokenizer = AmtTokenizer(return_tensors=True)
        self.config = load_config()["data"]
        self.mixup_fn = self.tokenizer.export_msg_mixup()
        self.file_buff = open(load_path, mode="r")
        self.file_mmap = mmap.mmap(
            self.file_buff.fileno(), 0, access=mmap.ACCESS_READ
        )

        index_path = AmtDataset._get_index_path(load_path=load_path)
        if os.path.isfile(index_path) is True:
            self.index = self._load_index(load_path=index_path)
        else:
            print("Calculating index...")
            self.index = self._build_index()
            print(
                f"Index of length {len(self.index)} calculated, saving to {index_path}"
            )
            self._save_index(index=self.index, save_path=index_path)

    def close(self):
        if self.file_buff:
            self.file_buff.close()
        if self.file_mmap:
            self.file_mmap.close()

    def __del__(self):
        self.close()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        def _format(tok):
            # This is required because json formats tuples into lists
            if isinstance(tok, list):
                return tuple(tok)
            return tok

        self.file_mmap.seek(self.index[idx])

        # Load data from line
        wav = torch.load(
            io.BytesIO(base64.b64decode(self.file_mmap.readline()))
        )
        _seq = orjson.loads(base64.b64decode(self.file_mmap.readline()))

        _seq = [_format(tok) for tok in _seq]  # Format seq
        _seq = self.mixup_fn(_seq)  # Data augmentation

        src = self.tokenizer.trunc_seq(
            seq=_seq,
            seq_len=self.config["max_seq_len"],
        )
        tgt = self.tokenizer.trunc_seq(
            seq=_seq[1:],
            seq_len=self.config["max_seq_len"],
        )

        return wav, self.tokenizer.encode(src), self.tokenizer.encode(tgt)

    def _build_index(self):
        self.file_mmap.seek(0)
        index = []
        while True:
            pos = self.file_mmap.tell()
            self.file_mmap.readline()
            if self.file_mmap.readline() == b"":
                break
            else:
                index.append(pos)

        return index

    def _save_index(self, index: list[int], save_path: str):
        with open(save_path, "w") as file:
            for idx in index:
                file.write(f"{idx}\n")

    def _load_index(self, load_path: str):
        with open(load_path, "r") as file:
            return [int(line.strip()) for line in file]

    @staticmethod
    def _get_index_path(load_path: str):
        return (
            f"{load_path.rsplit('.', 1)[0]}_index.{load_path.rsplit('.', 1)[1]}"
        )

    def _build_index(self):
        self.file_mmap.seek(0)
        index = []
        pos = 0
        while True:
            pos_buff = pos

            pos = self.file_mmap.find(b"\n", pos)
            if pos == -1:
                break
            pos = self.file_mmap.find(b"\n", pos + 1)
            if pos == -1:
                break

            index.append(pos_buff)
            pos += 1

        return index

    @classmethod
    def build(
        cls,
        load_paths: list,
        save_path: str,
        cli_cmd_fn: Callable | None = None,
        num_processes: int = 1,
    ):
        assert os.path.isfile(save_path) is False, f"{save_path} already exists"

        index_path = AmtDataset._get_index_path(load_path=save_path)
        if os.path.isfile(index_path):
            print(f"Removing existing index file at {index_path}")
            os.remove(AmtDataset._get_index_path(load_path=save_path))

        save_path_queue = Queue()
        load_path_queue = Queue()
        for entry in load_paths:
            load_path_queue.put(entry)

        if cli_cmd_fn is None:
            # Build matched audio-midi dataset
            assert len(load_paths[0]) == 2, "Invalid load paths"
            print("Building matched audio-midi dataset")
            worker_processes = [
                Process(
                    target=build_worker_fn,
                    args=(
                        load_path_queue,
                        save_path_queue,
                        save_path,
                    ),
                )
                for _ in range(num_processes)
            ]
        else:
            # Build synthetic dataset
            assert len(load_paths[0]) == 1, "Invalid load paths"
            print("Building synthetic dataset")
            worker_processes = [
                Process(
                    target=build_synth_worker_fn,
                    args=(
                        cli_cmd_fn,
                        load_path_queue,
                        save_path_queue,
                        save_path,
                    ),
                )
                for _ in range(num_processes)
            ]

        for p in worker_processes:
            p.start()
        for p in worker_processes:
            p.join()

        sharded_save_paths = []
        while not save_path_queue.empty():
            try:
                _path = save_path_queue.get_nowait()
                sharded_save_paths.append(_path)
            except Queue.Empty:
                break

        # This is bad, however cat is fast
        if shutil.which("cat") is None:
            print("The GNU cat command is not available")
        else:
            print("Concatinating sharded dataset files")
            shell_cmd = f"cat "
            for _path in sharded_save_paths:
                shell_cmd += f"{_path} "
                print()
            shell_cmd += f">> {save_path}"

            os.system(shell_cmd)
            for _path in sharded_save_paths:
                os.remove(_path)

        # Create index by loading object
        AmtDataset(load_path=save_path)
