import mmap
import os
import json
import jsonlines
import torch

from typing import Callable
from multiprocessing import Pool

from aria.data.midi import MidiDict
from amt.tokenizer import AmtTokenizer
from amt.config import load_config
from amt.audio import (
    log_mel_spectrogram,
    pad_or_trim,
    N_FRAMES,
)

config = load_config()["data"]
STRIDE_FACTOR = config["stride_factor"]


def get_features(audio_path: str, mid_path: str):
    """This function yields tuples of matched log mel spectrograms and
    tokenized sequences (np.array, list).
    """
    tokenizer = AmtTokenizer()

    if not os.path.isfile(audio_path) or not os.path.isfile(mid_path):
        return None

    try:
        midi_dict = MidiDict.from_midi(mid_path)
        log_spec = log_mel_spectrogram(audio=audio_path)
    except Exception as e:
        print("Failed to convert files into features")
        return None

    _, total_frames = log_spec.shape
    res = []
    for start_frame in range(0, total_frames, N_FRAMES // STRIDE_FACTOR):
        audio_feature = pad_or_trim(log_spec[:, start_frame:], length=N_FRAMES)
        mid_feature = tokenizer._tokenize_midi_dict(
            midi_dict=midi_dict,
            start_ms=start_frame * 10,
            end_ms=(start_frame + N_FRAMES) * 10,
        )
        res.append((audio_feature, mid_feature))

    return res


def get_features_mp(args):
    """Multiprocessing wrapper for get_features"""
    res = get_features(*args)
    if res is None:
        return False, None
    else:
        return True, res


class AmtDataset(torch.utils.data.Dataset):
    def __init__(self, load_path: str):
        self.tokenizer = AmtTokenizer(return_tensors=True)
        self.aug_fn = self.tokenizer.export_msg_mixup()
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

        # This isn't going to load properly
        spec, seq = json.loads(self.file_mmap.readline())  # Load data from line

        spec = torch.tensor(spec)  # Format spectrogram into tensor
        seq = [_format(tok) for tok in seq]  # Format seq
        seq = self.aug_fn(seq)  # Data augmentation

        src = seq
        tgt = seq[1:] + [self.tokenizer.pad_tok]

        return spec, self.tokenizer.encode(src), self.tokenizer.encode(tgt)

    def _build_index(self):
        self.file_mmap.seek(0)
        index = []
        while True:
            pos = self.file_mmap.tell()
            line_buffer = self.file_mmap.readline()
            if line_buffer == b"":
                break
            else:
                index.append(pos)

        return index

    @classmethod
    def build(
        cls,
        matched_load_paths: list[tuple[str, str]],
        save_path: str,
        audio_aug_hook: Callable | None = None,
    ):
        def _get_features(_matched_load_paths: list):
            with Pool(4) as pool:
                results = pool.imap(get_features_mp, _matched_load_paths)
                num_paths = len(_matched_load_paths)
                for idx, (success, res) in enumerate(results):
                    if idx % 50 == 0 and idx != 0:
                        print(f"Processed audio-mid pairs: {idx}/{num_paths}")

                    if success == False:
                        continue
                    for _audio_feature, _mid_feature in res:
                        yield _audio_feature.tolist(), _mid_feature

        with jsonlines.open(save_path, mode="w") as writer:
            for audio_feature, mid_feature in _get_features(matched_load_paths):
                writer.write([audio_feature, mid_feature])
