import unittest
import logging
import os
import torch
import torchaudio
import matplotlib.pyplot as plt

from amt.data import get_paired_wav_mid_segments, AmtDataset
from amt.tokenizer import AmtTokenizer
from amt.audio import AudioTransform
from ariautils.midi import MidiDict


logging.basicConfig(level=logging.INFO)
if os.path.isdir("tests/test_results") is False:
    os.mkdir("tests/test_results")

MAESTRO_PATH = "/home/loubb/work/aria-amt/temp/train.txt"


def plot_spec(
    mel: torch.Tensor,
    name: str | int,
    onsets: list = [],
    offsets: list = [],
):
    # mel: [height, width]

    height, width = mel.shape
    fig_width, fig_height = width // 100, height // 100
    plt.figure(figsize=(fig_width, fig_height), dpi=100)
    plt.imshow(
        mel, aspect="auto", origin="lower", cmap="viridis", interpolation="none"
    )

    line_width_in_points = 1 / 100 * 72

    for x in onsets:
        plt.axvline(
            x=x,
            color="red",
            alpha=0.5,
            linewidth=line_width_in_points,
        )
    for x in offsets:
        plt.axvline(
            x=x,
            color="purple",
            alpha=0.5,
            linewidth=line_width_in_points,
        )

    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f"tests/test_results/{name}.png", dpi=100)
    plt.close()


# Need to test this properly, have issues turning mel_spec back into audio
class TestDataGen(unittest.TestCase):
    def test_wav_mid_segments(self):
        tokenizer = AmtTokenizer()
        for idx, (wav, seq) in enumerate(
            get_paired_wav_mid_segments(
                audio_path="tests/test_data/147.wav",
                mid_path="tests/test_data/147.mid",
                stride_factor=6,
            )
        ):
            print(wav.shape, len(seq))
            torchaudio.save(
                f"tests/test_results/{idx}.wav", wav.unsqueeze(0), 16000
            )
            print(idx)
            tokenizer.detokenize(seq, 30000).to_midi().save(
                f"tests/test_results/{idx}.mid"
            )


class TestAmtDataset(unittest.TestCase):
    def test_build(self):
        matched_paths = [
            ("tests/test_data/maestro.wav", "tests/test_data/maestro1.mid")
            for _ in range(3)
        ]
        if os.path.isfile("tests/test_results/dataset.jsonl"):
            os.remove("tests/test_results/dataset.jsonl")

        AmtDataset.build(
            load_paths=matched_paths,
            save_path="tests/test_results/dataset.jsonl",
        )

        dataset = AmtDataset("tests/test_results/dataset.jsonl")
        tokenizer = AmtTokenizer()
        for idx, (wav, src, tgt, idx) in enumerate(dataset):
            print(wav.shape, src.shape, tgt.shape)
            src_decoded = tokenizer.decode(src.tolist())
            tgt_decoded = tokenizer.decode(tgt.tolist())
            self.assertListEqual(src_decoded[1:], tgt_decoded[:-1])

            mid = tokenizer.detokenize(src_decoded, len_ms=30000).to_midi()
            mid.save(f"tests/test_results/trunc_{idx}.mid")

    def test_build_multiple(self):
        matched_paths = [
            ("tests/test_data/maestro.wav", "tests/test_data/maestro1.mid")
            for _ in range(2)
        ]
        if os.path.isfile("tests/test_results/dataset_1.jsonl"):
            os.remove("tests/test_results/dataset_1.jsonl")
        if os.path.isfile("tests/test_results/dataset_2.jsonl"):
            os.remove("tests/test_results/dataset_2.jsonl")

        AmtDataset.build(
            load_paths=matched_paths,
            save_path="tests/test_results/dataset_1.jsonl",
        )

        AmtDataset.build(
            load_paths=matched_paths,
            save_path="tests/test_results/dataset_2.jsonl",
        )

        dataset = AmtDataset(
            [
                "tests/test_results/dataset_1.jsonl",
                "tests/test_results/dataset_2.jsonl",
            ]
        )

        for idx, (wav, src, tgt, idx) in enumerate(dataset):
            print(wav.shape, src.shape, tgt.shape)

    def test_maestro(self):
        if not os.path.isfile(MAESTRO_PATH):
            return

        tokenizer = AmtTokenizer()
        audio_transform = AudioTransform()
        dataset = AmtDataset(load_paths=MAESTRO_PATH)
        print(f"Dataset length: {len(dataset)}")
        for idx, (wav, src, tgt, __idx) in enumerate(dataset):
            src_dec, tgt_dec = tokenizer.decode(src.tolist()), tokenizer.decode(
                tgt.tolist()
            )

            if idx % 7 == 0 and idx < 100:
                print(idx)
                src_mid_dict = tokenizer.detokenize(
                    src_dec,
                    len_ms=30000,
                )

                src_mid = src_mid_dict.to_midi()
                src_mid.save(f"tests/test_results/dataset_{idx}.mid")
                torchaudio.save(
                    f"tests/test_results/wav_{idx}.wav", wav.unsqueeze(0), 16000
                )
                torchaudio.save(
                    f"tests/test_results/wav_aug_{idx}.wav",
                    audio_transform.aug_wav(wav.unsqueeze(0)),
                    16000,
                )
                plot_spec(
                    audio_transform(wav.unsqueeze(0)).squeeze(0), f"mel_{idx}"
                )

            self.assertTrue(tokenizer.unk_tok not in src_dec)
            self.assertTrue(tokenizer.unk_tok not in tgt_dec)
            for src_tok, tgt_tok in zip(src_dec[1:], tgt_dec):
                self.assertEqual(src_tok, tgt_tok)


# TODO: Port these over to new spectrogram format (audio transform)
class TestAug(unittest.TestCase):
    def test_spec(self):
        SAMPLE_RATE, CHUNK_LEN = 16000, 30
        audio_transform = AudioTransform()
        wav, sr = torchaudio.load("tests/test_data/maestro.wav")
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE).mean(
            0, keepdim=True
        )[:, : SAMPLE_RATE * CHUNK_LEN]

        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=2048,
            hop_length=160,
            power=1,
            n_iter=64,
        )

        spec = audio_transform.spec_transform(wav)
        shift_spec = audio_transform.shift_spec(spec, 1)
        shift_wav = griffin_lim(shift_spec[..., :384])
        torchaudio.save("tests/test_results/orig.wav", wav, SAMPLE_RATE)
        torchaudio.save("tests/test_results/shift.wav", shift_wav, SAMPLE_RATE)

    def test_pitch_aug(self):
        tokenizer = AmtTokenizer()
        tensor_pitch_aug_fn = tokenizer.export_tensor_pitch_aug()
        mid_dict = MidiDict.from_midi("tests/test_data/maestro2.mid")
        seq = tokenizer.tokenize(mid_dict, 0, 30000)
        src = torch.tensor(tokenizer.encode(tokenizer.trunc_seq(seq, 4096)))
        tgt = torch.tensor(tokenizer.encode(tokenizer.trunc_seq(seq[1:], 4096)))

        src = torch.stack((src, src, src))
        tgt = torch.stack((tgt, tgt, tgt))
        src_aug = tensor_pitch_aug_fn(src.clone(), shift=1)
        tgt_aug = tensor_pitch_aug_fn(tgt.clone(), shift=1)

        src_aug_dec = tokenizer.decode(src_aug[1].tolist())
        tgt_aug_dec = tokenizer.decode(tgt_aug[2].tolist())
        print(seq[:20])
        print(src_aug_dec[:20])
        print(tgt_aug_dec[:20])

        for tok, aug_tok in zip(seq, src_aug_dec):
            if type(tok) is tuple and aug_tok[0] in {"on", "off"}:
                self.assertEqual(tok[1] + 1, aug_tok[1])

        for src_tok, tgt_tok in zip(src_aug_dec[1:], tgt_aug_dec):
            self.assertEqual(src_tok, tgt_tok)

    def test_detune(self):
        SAMPLE_RATE, CHUNK_LEN = 16000, 30
        audio_transform = AudioTransform()
        wav, sr = torchaudio.load("tests/test_data/maestro.wav")
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE).mean(
            0, keepdim=True
        )[:, : SAMPLE_RATE * CHUNK_LEN]

        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=2048,
            hop_length=160,
            power=1,
            n_iter=64,
        )

        spec = audio_transform.spec_transform(wav)
        shift_spec = audio_transform.detune_spec(spec)
        shift_wav = griffin_lim(shift_spec)
        gl_wav = griffin_lim(spec[..., :384])
        torchaudio.save("tests/test_results/orig.wav", wav, SAMPLE_RATE)
        torchaudio.save("tests/test_results/orig_gl.wav", gl_wav, SAMPLE_RATE)
        torchaudio.save("tests/test_results/detune.wav", shift_wav, SAMPLE_RATE)

    def test_mels(self):
        audio_transform = AudioTransform()
        SAMPLE_RATE, N_FFT, CHUNK_LEN = (
            audio_transform.sample_rate,
            1,
            30,
        )
        wav, sr = torchaudio.load("tests/test_data/maestro.wav")
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE).mean(
            0, keepdim=True
        )[:, : SAMPLE_RATE * CHUNK_LEN]

        wavs = torch.stack((wav[0], wav[0], wav[0]))
        mels = audio_transform(wavs)
        for idx in range(mels.shape[0]):
            plot_spec(
                mels[idx],
                f"{mels[0].shape[0]}-{N_FFT}-{SAMPLE_RATE}",
            )
            break

    def test_distortion(self):
        SAMPLE_RATE, CHUNK_LEN = 16000, 30
        audio_transform = AudioTransform()
        wav, sr = torchaudio.load("tests/test_data/maestro.wav")
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE).mean(
            0, keepdim=True
        )[:, : SAMPLE_RATE * CHUNK_LEN]

        torchaudio.save("tests/test_results/orig.wav", wav, SAMPLE_RATE)
        res = audio_transform.apply_distortion(wav)
        torchaudio.save("tests/test_results/dist.wav", res, SAMPLE_RATE)

    def test_bandpass(self):
        SAMPLE_RATE, CHUNK_LEN = 16000, 30
        audio_transform = AudioTransform()
        wav, sr = torchaudio.load("tests/test_data/147.wav")
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE).mean(
            0, keepdim=True
        )[:, : SAMPLE_RATE * CHUNK_LEN]

        torchaudio.save("tests/test_results/orig.wav", wav, SAMPLE_RATE)
        res = audio_transform.apply_bandpass(wav)
        torchaudio.save("tests/test_results/bandpass.wav", res, SAMPLE_RATE)

    def test_applause(self):
        SAMPLE_RATE, CHUNK_LEN = 16000, 30
        audio_transform = AudioTransform()
        wav, sr = torchaudio.load("tests/test_data/maestro.wav")
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE).mean(
            0, keepdim=True
        )[:, : SAMPLE_RATE * CHUNK_LEN]

        torchaudio.save("tests/test_results/orig.wav", wav, SAMPLE_RATE)
        res = audio_transform.apply_applause(wav)
        torchaudio.save("tests/test_results/applause.wav", res, SAMPLE_RATE)

    def test_reduction(self):
        SAMPLE_RATE, CHUNK_LEN = 16000, 30
        audio_transform = AudioTransform()
        wav, sr = torchaudio.load("tests/test_data/maestro.wav")
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE).mean(
            0, keepdim=True
        )[:, : SAMPLE_RATE * CHUNK_LEN]

        torchaudio.save("tests/test_results/orig.wav", wav, SAMPLE_RATE)
        res = audio_transform.apply_reduction(wav)
        torchaudio.save("tests/test_results/reduction.wav", res, SAMPLE_RATE)

    def test_noise(self):
        SAMPLE_RATE, CHUNK_LEN = 16000, 30
        audio_transform = AudioTransform()
        wav, sr = torchaudio.load("tests/test_data/maestro.wav")
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE).mean(
            0, keepdim=True
        )[:, : SAMPLE_RATE * CHUNK_LEN]

        torchaudio.save("tests/test_results/orig.wav", wav, SAMPLE_RATE)
        res = audio_transform.apply_noise(wav)
        torchaudio.save("tests/test_results/noise.wav", res, SAMPLE_RATE)


if __name__ == "__main__":
    unittest.main()
