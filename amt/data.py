import mmap
import os
import io
import random
import shlex
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


def _check_onset_threshold(seq: list, onset: int):
    for tok_1, tok_2 in zip(seq, seq[1:]):
        if isinstance(tok_1, tuple) and tok_1[0] in ("on", "off"):
            _onset = tok_2[1]
            if _onset > onset:
                return True

    return False


def get_wav_mid_segments(
    audio_path: str,
    mid_path: str = "",
    return_json: bool = False,
    stride_factor: int | None = None,
    pad_last=False,
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
    pad_factor = 2 if pad_last is True else 1
    res = []
    for idx in range(
        0,
        total_samples
        - (num_samples - pad_factor * (num_samples // stride_factor)),
        num_samples // stride_factor,
    ):
        audio_feature = pad_or_trim(wav[idx:], length=num_samples)
        if midi_dict is not None:
            mid_feature = tokenizer._tokenize_midi_dict(
                midi_dict=midi_dict,
                start_ms=idx // samples_per_ms,
                end_ms=(idx + num_samples) / samples_per_ms,
                max_pedal_len_ms=15000,
            )

            # Hardcoded to 5s
            if _check_onset_threshold(mid_feature, 5000) is False:
                print("No note messages after 5s - skipping")
                continue

        else:
            mid_feature = []

        if return_json is True:
            audio_feature = audio_feature.tolist()

        res.append((audio_feature, mid_feature))

    return res


def pianoteq_cmd_fn(mid_path: str, wav_path: str):
    presets = [
        "C. Bechstein DG Prelude",
        "C. Bechstein DG Sweet",
        "C. Bechstein DG Felt I",
        "C. Bechstein DG Felt II",
        "C. Bechstein DG D 282",
        "C. Bechstein DG Recording 1",
        "C. Bechstein DG Recording 2",
        "C. Bechstein DG Recording 3",
        "C. Bechstein DG Cinematic",
        "C. Bechstein DG Snappy",
        "C. Bechstein DG Venue",
        "C. Bechstein DG Player",
        "HB Steinway D Blues",
        "HB Steinway D Pop",
        "HB Steinway D New Age",
        "HB Steinway D Prelude",
        "HB Steinway D Felt I",
        "HB Steinway D Felt II",
        "HB Steinway Model D",
        "HB Steinway D Classical Recording",
        "HB Steinway D Jazz Recording",
        "HB Steinway D Chamber Recording",
        "HB Steinway D Studio Recording",
        "HB Steinway D Intimate",
        "HB Steinway D Cinematic",
        "HB Steinway D Close Mic Classical",
        "HB Steinway D Close Mic Jazz",
        "HB Steinway D Player Wide",
        "HB Steinway D Player Clean",
        "HB Steinway D Trio",
        "HB Steinway D Duo",
        "HB Steinway D Cabaret",
        "HB Steinway D Bright",
        "HB Steinway D Hyper Bright",
    ]

    preset = random.choice(presets)

    # Safely quote the preset name, MIDI path, and WAV path
    safe_preset = shlex.quote(preset)
    safe_mid_path = shlex.quote(mid_path)
    safe_wav_path = shlex.quote(wav_path)

    executable_path = "/mnt/ssd-1/aria/pianoteq/x86-64bit/Pianoteq 8 STAGE"
    command = f'"{executable_path}" --preset {safe_preset} --midi {safe_mid_path} --wav {safe_wav_path}'

    return command


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
            cli_cmd_fn=cli_cmd_fn, mid_path=mid_path, wav_path=audio_path_temp
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

        if os.path.isfile(audio_path_temp):
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
        try:
            write_synth_features(cli_cmd, mid_path, worker_save_path)
        except Exception as e:
            print("Failed")
            print(e)

    save_path_queue.put(worker_save_path)


class AmtDataset(torch.utils.data.Dataset):
    def __init__(self, load_paths: str | list):
        self.tokenizer = AmtTokenizer(return_tensors=True)
        self.config = load_config()["data"]
        self.mixup_fn = self.tokenizer.export_msg_mixup()

        if isinstance(load_paths, str):
            load_paths = [load_paths]
        self.file_buffs = []
        self.file_mmaps = []
        self.index = []

        for path in load_paths:
            buff = open(path, mode="r")
            self.file_buffs.append(buff)
            mmap_obj = mmap.mmap(buff.fileno(), 0, access=mmap.ACCESS_READ)
            self.file_mmaps.append(mmap_obj)

            index_path = AmtDataset._get_index_path(load_path=path)
            if os.path.isfile(index_path):
                _index = self._load_index(load_path=index_path)
            else:
                print("Calculating index...")
                _index = self._build_index(mmap_obj)
                print(
                    f"Index of length {len(_index)} calculated, saving to {index_path}"
                )
                self._save_index(index=_index, save_path=index_path)

            self.index.extend(
                [(len(self.file_mmaps) - 1, pos) for pos in _index]
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        def _format(tok):
            # This is required because json formats tuples into lists
            if isinstance(tok, list):
                return tuple(tok)
            return tok

        file_id, pos = self.index[idx]
        mmap_obj = self.file_mmaps[file_id]
        mmap_obj.seek(pos)

        # Load data from line
        wav = torch.load(io.BytesIO(base64.b64decode(mmap_obj.readline())))
        _seq = orjson.loads(base64.b64decode(mmap_obj.readline()))

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

        return wav, self.tokenizer.encode(src), self.tokenizer.encode(tgt), idx

    def close(self):
        for buff in self.file_buffs:
            buff.close()
        for mmap in self.file_mmaps:
            mmap.close()

    def __del__(self):
        self.close()

    def _save_index(self, index: list, save_path: str):
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

    def _build_index(self, mmap_obj):
        mmap_obj.seek(0)
        index = []
        pos = 0
        while True:
            pos_buff = pos

            pos = mmap_obj.find(b"\n", pos)
            if pos == -1:
                break
            pos = mmap_obj.find(b"\n", pos + 1)
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
            assert isinstance(load_paths[0], str), "Invalid load paths"
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
            for _path in sharded_save_paths:
                shell_cmd = f"cat {_path} >> {save_path}"
                os.system(shell_cmd)
                os.remove(_path)

        # Create index by loading object
        AmtDataset(load_paths=save_path)
