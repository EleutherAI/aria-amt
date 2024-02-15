import unittest
import logging
import os

from amt.tokenizer import AmtTokenizer
from aria.data.midi import MidiDict

if os.path.isdir("tests/test_results") is False:
    os.mkdir("tests/test_results")


class TestAmtTokenizer(unittest.TestCase):
    def test_tokenize(self):
        def _tokenize_detokenize(mid_name: str):
            START = 5000
            END = 10000

            tokenizer = AmtTokenizer()
            midi_dict = MidiDict.from_midi(f"tests/test_data/{mid_name}")
            tokenized_seq = tokenizer._tokenize_midi_dict(
                midi_dict=midi_dict,
                start_ms=START,
                end_ms=END,
            )
            logging.info(f"{mid_name} tokenized:")
            logging.info(tokenized_seq)

            _midi_dict = tokenizer._detokenize_midi_dict(
                tokenized_seq, END - START
            )
            _mid = _midi_dict.to_midi()
            _mid.save(f"tests/test_results/{mid_name}")
            logging.info(f"{mid_name} note_msgs:")
            for msg in _midi_dict.note_msgs:
                logging.info(msg)

        _tokenize_detokenize(mid_name="arabesque.mid")
        _tokenize_detokenize(mid_name="bach.mid")
        _tokenize_detokenize(mid_name="beethoven_moonlight.mid")


if __name__ == "__main__":
    if os.path.isdir("tests/test_results") is False:
        os.mkdir("tests/test_results")

    logging.basicConfig(level=logging.INFO)
    unittest.main()
