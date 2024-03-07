import mmap
import os
import shutil
import orjson
import torch
import torchaudio

from multiprocessing import Pool

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
    for idx in range(0, total_samples, num_samples // stride_factor):
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


def write_features(args):
    audio_path, mid_path, save_path = args
    features = get_wav_mid_segments(
        audio_path=audio_path,
        mid_path=mid_path,
        return_json=False,
    )
    dirname, basename = os.path.split(save_path)
    proc_save_path = os.path.join(dirname, str(os.getpid()) + basename)

    with open(proc_save_path, mode="ab") as file:
        for wav, seq in features:
            file.write(
                orjson.dumps(
                    wav.numpy(),
                    option=orjson.OPT_SERIALIZE_NUMPY,
                )
            )
            file.write(b"\n")
            file.write(orjson.dumps(seq))
            file.write(b"\n")

    return proc_save_path


class AmtDataset(torch.utils.data.Dataset):
    def __init__(self, load_path: str):
        self.tokenizer = AmtTokenizer(return_tensors=True)
        self.config = load_config()["data"]
        self.mixup_fn = self.tokenizer.export_msg_mixup()
        self.file_buff = open(load_path, mode="r")
        self.file_mmap = mmap.mmap(
            self.file_buff.fileno(), 0, access=mmap.ACCESS_READ
        )
        self.index = self._build_index()

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
        wav = torch.tensor(orjson.loads(self.file_mmap.readline()))
        _seq = orjson.loads(self.file_mmap.readline())

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

    @classmethod
    def build(
        cls,
        matched_load_paths: list[tuple[str, str]],
        save_path: str,
        num_processes: int = 1,
    ):
        assert os.path.isfile(save_path) is False, f"{save_path} already exists"
        num_paths = len(matched_load_paths)
        with Pool(processes=num_processes) as pool:
            sharded_save_paths = []
            res = pool.imap_unordered(
                write_features,
                ((ap, mp, save_path) for ap, mp in matched_load_paths),
            )
            for idx, proc_save_path in enumerate(res):
                if idx % 10 == 0 and idx != 0:
                    print(f"Finished {idx}/{num_paths}")
                if proc_save_path not in sharded_save_paths:
                    sharded_save_paths.append(proc_save_path)

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
