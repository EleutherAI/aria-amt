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
from midi2audio import FluidSynth
import random


class SyntheticMidiHandler:
    def __init__(self, soundfont_path: str, soundfont_prob_dict: dict = None, num_wavs_per_midi: int = 1):
        """
        File to load MIDI files and convert them to audio.

        Parameters
        ----------
        soundfont_path : str
            Path to the directory containing soundfont files.
        soundfont_prob_dict : dict, optional
            Dictionary containing the probability of using a soundfont file.
            The keys are the soundfont file names and the values are the
            probability of using the soundfont file. If none is given, then
            a uniform distribution is used.
        num_wavs_per_midi : int, optional
            Number of audio files to generate per MIDI file.
        """

        self.soundfont_path = soundfont_path
        self.soundfont_prob_dict = soundfont_prob_dict
        self.num_wavs_per_midi = num_wavs_per_midi

        self.fs_objs = self._load_soundfonts()
        self.soundfont_cumul_prob_dict = self._get_cumulative_prob_dict()

    def _load_soundfonts(self):
        """Loads the soundfonts into fluidsynth objects."""
        fs_files = os.listdir(self.soundfont_path)
        fs_objs = {}
        for fs_file in fs_files:
            fs_objs[fs_file] = FluidSynth(fs_file)
        return fs_objs

    def _get_cumulative_prob_dict(self):
        """Returns a dictionary with the cumulative probabilities of the soundfonts.
        Used for sampling the soundfonts.
        """
        if self.soundfont_prob_dict is None:
            self.soundfont_prob_dict = {k: 1 / len(self.fs_objs) for k in self.fs_objs.keys()}
        self.soundfont_prob_dict = {k: v / sum(self.soundfont_prob_dict.values())
                                    for k, v in self.soundfont_prob_dict.items()}
        cumul_prob_dict = {}
        cumul_prob = 0
        for k, v in self.soundfont_prob_dict.items():
            cumul_prob_dict[k] = (cumul_prob, cumul_prob + v)
            cumul_prob += v
        return cumul_prob_dict

    def _sample_soundfont(self):
        """Samples a soundfont file."""
        rand_num = random.random()
        for k, (v_s, v_e) in self.soundfont_cumul_prob_dict.items():
            if (rand_num >= v_s) and (rand_num < v_e):
                return self.fs_objs[k]

    def get_wav(self, midi_path: str, save_path: str):
        """
        Converts a MIDI file to audio.

        Parameters
        ----------
        midi_path : str
            Path to the MIDI file.
        save_path : str
            Path to save the audio file.
        """
        for i in range(self.num_wavs_per_midi):
            soundfont = self._sample_soundfont()
            if self.num_wavs_per_midi > 1:
                save_path = save_path[:-4] + f"_{i}.wav"
            soundfont.midi_to_audio(midi_path, save_path)


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
        total_samples - (num_samples - (num_samples // stride_factor)),
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

    @classmethod
    def build(
        cls,
        matched_load_paths: list[tuple[str, str]],
        save_path: str,
        num_processes: int = 1,
    ):
        assert os.path.isfile(save_path) is False, f"{save_path} already exists"

        index_path = AmtDataset._get_index_path(load_path=save_path)
        if os.path.isfile(index_path):
            print(f"Removing existing index file at {index_path}")
            os.remove(AmtDataset._get_index_path(load_path=save_path))

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

        # Create index by loading object
        AmtDataset(load_path=save_path)

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
