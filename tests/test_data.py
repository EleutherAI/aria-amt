import unittest
import logging
import os

from amt.data import get_features, AmtDataset
from amt.tokenizer import AmtTokenizer
from aria.data.midi import MidiDict


logging.basicConfig(level=logging.INFO)
if os.path.isdir("tests/test_results") is False:
    os.mkdir("tests/test_results")


# Need to test this properly, have issues turning mel_spec back into audio
class TestDataGen(unittest.TestCase):
    def test_feature_gen(self):
        for log_spec, seq in get_features(
            audio_path="tests/test_data/147.wav",
            mid_path="tests/test_data/147.mid",
        ):
            print(log_spec.shape, len(seq))


class TestAmtDataset(unittest.TestCase):
    def test_build(self):
        matched_paths = [("tests/test_data/147.wav", "tests/test_data/147.mid")]
        AmtDataset.build(
            matched_load_paths=matched_paths,
            save_path="tests/test_results/dataset.jsonl",
        )

        dataset = AmtDataset("tests/test_results/dataset.jsonl")
        tokenizer = AmtTokenizer()
        for idx, (spec, src, tgt) in enumerate(dataset):
            print(spec.shape, src.shape, tgt.shape)
            decoded = tokenizer.decode(src)
            mid = tokenizer._detokenize_midi_dict(
                decoded, len_ms=30000
            ).to_midi()
            mid.save(f"tests/test_results/trunc_{idx}.mid")


if __name__ == "__main__":
    unittest.main()
