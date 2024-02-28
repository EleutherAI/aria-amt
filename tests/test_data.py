import unittest
import logging
import os
import torch
import torchaudio
import matplotlib.pyplot as plt

from amt.data import get_wav_mid_segments, AmtDataset
from amt.tokenizer import AmtTokenizer
from amt.audio import AudioTransform
from aria.data.midi import MidiDict


logging.basicConfig(level=logging.INFO)
if os.path.isdir("tests/test_results") is False:
    os.mkdir("tests/test_results")

MAESTRO_PATH = "/weka/proj-aria/aria-amt/data/maestro/val.jsonl"


# Need to test this properly, have issues turning mel_spec back into audio
class TestDataGen(unittest.TestCase):
    def test_wav_mid_segments(self):
        for log_spec, seq in get_wav_mid_segments(
            audio_path="tests/test_data/147.wav",
            mid_path="tests/test_data/147.mid",
        ):
            print(log_spec.shape, len(seq))


class TestAmtDataset(unittest.TestCase):
    def test_build(self):
        matched_paths = [
            ("tests/test_data/147.wav", "tests/test_data/147.mid")
            for _ in range(3)
        ]
        if os.path.isfile("tests/test_results/dataset.jsonl"):
            os.remove("tests/test_results/dataset.jsonl")

        AmtDataset.build(
            matched_load_paths=matched_paths,
            save_path="tests/test_results/dataset.jsonl",
        )

        dataset = AmtDataset("tests/test_results/dataset.jsonl")
        tokenizer = AmtTokenizer()
        for idx, (spec, src, tgt) in enumerate(dataset):
            print(spec.shape, src.shape, tgt.shape)
            src_decoded = tokenizer.decode(src)
            tgt_decoded = tokenizer.decode(tgt)
            self.assertListEqual(src_decoded[1:], tgt_decoded[:-1])

            mid = tokenizer._detokenize_midi_dict(
                src_decoded, len_ms=30000
            ).to_midi()
            mid.save(f"tests/test_results/trunc_{idx}.mid")

    def test_maestro(self):
        if not os.path.isfile(MAESTRO_PATH):
            return

        tokenizer = AmtTokenizer()
        dataset = AmtDataset(load_path=MAESTRO_PATH)
        for idx, (mel, src, tgt) in enumerate(dataset):
            src_dec, tgt_dec = tokenizer.decode(src), tokenizer.decode(tgt)
            if (idx + 1) % 100 == 0:
                break
            if idx % 7 == 0:
                src_mid_dict = tokenizer._detokenize_midi_dict(
                    src_dec,
                    len_ms=30000,
                )

                src_mid = src_mid_dict.to_midi()
                if idx % 10 == 0:
                    src_mid.save(f"tests/test_results/dataset_{idx}.mid")

            self.assertTrue(tokenizer.unk_tok not in src_dec)
            self.assertTrue(tokenizer.unk_tok not in tgt_dec)
            for src_tok, tgt_tok in zip(src_dec[1:], tgt_dec):
                self.assertEqual(src_tok, tgt_tok)


class TestAug(unittest.TestCase):
    def plot_mel(self, mel: torch.Tensor, idx: int):
        plt.figure(figsize=(10, 4))
        plt.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel Spectrogram")
        plt.tight_layout()
        plt.savefig(f"tests/test_results/mel{idx}.png")
        plt.close()

    def test_mels(self):
        SAMPLE_RATE, CHUNK_LEN = 16000, 30
        tokenizer = AmtTokenizer(return_tensors=True)
        mid_dict = MidiDict.from_midi("tests/test_data/maestro2.mid")
        seq = tokenizer._tokenize_midi_dict(mid_dict, 0, 30000)
        seq = tokenizer.encode(seq)
        seqs = torch.stack((seq, seq, seq))
        src = seqs[:, :-1].clone()
        tgt = seqs[:, 1:].clone()

        audio_transform = AudioTransform()
        wav, sr = torchaudio.load("tests/test_data/maestro.wav")
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE).mean(
            0, keepdim=True
        )[:, : SAMPLE_RATE * CHUNK_LEN]
        wav_shift = audio_transform.aug_pitch(wav)
        wav_aug = audio_transform.aug_wav(wav_shift)
        torchaudio.save("tests/test_results/orig.wav", wav, SAMPLE_RATE)
        torchaudio.save(
            "tests/test_results/pitch_aug.wav", wav_shift, SAMPLE_RATE
        )
        torchaudio.save("tests/test_results/aug.wav", wav_aug, SAMPLE_RATE)

        wavs = torch.stack((wav[0], wav[0], wav[0]))
        mels, (src_aug, tgt_aug) = audio_transform(wavs, src, tgt)
        for idx in range(mels.shape[0]):
            self.plot_mel(mels[idx], idx)

        src_aug, tgt_aug = src_aug[0], tgt_aug[0]
        for idx in range(src_aug.shape[0] - 1):
            self.assertEqual(src_aug[idx + 1].item(), tgt_aug[idx].item())

        seq = tokenizer.decode(src_aug)
        mid = tokenizer._detokenize_midi_dict(seq, 30000).to_midi()
        mid.save("tests/test_results/mid_aug.mid")


if __name__ == "__main__":
    unittest.main()
