import unittest
import logging
import os

from amt.tokenizer import AmtTokenizer
from aria.tokenizer import AbsTokenizer
from aria.data.midi import MidiDict

logging.basicConfig(level=logging.INFO)
if os.path.isdir("tests/test_results") is False:
    os.mkdir("tests/test_results")


# Add test for unk tok


class TestAmtTokenizer(unittest.TestCase):
    def test_tokenize(self):
        def _tokenize_detokenize(mid_name: str, start: int, end: int):
            length = end - start
            tokenizer = AmtTokenizer()
            midi_dict = MidiDict.from_midi(f"tests/test_data/{mid_name}")

            logging.info(f"tokenizing {mid_name} in range ({start}, {end})...")
            tokenized_seq = tokenizer._tokenize_midi_dict(midi_dict, start, end)
            tokenized_seq = tokenizer.decode(tokenizer.encode(tokenized_seq))
            self.assertTrue(tokenizer.unk_tok not in tokenized_seq)
            _midi_dict = tokenizer._detokenize_midi_dict(tokenized_seq, length)
            _mid = _midi_dict.to_midi()
            _mid.save(f"tests/test_results/{start}_{end}_{mid_name}")

        _tokenize_detokenize("basic.mid", start=0, end=30000)
        _tokenize_detokenize("147.mid", start=0, end=30000)
        _tokenize_detokenize("beethoven_moonlight.mid", start=0, end=30000)

        for _idx in range(5):
            START = _idx * 25000
            END = (_idx + 1) * 25000
            _tokenize_detokenize("maestro1.mid", start=START, end=END)
            _tokenize_detokenize("maestro2.mid", start=START, end=END)
            _tokenize_detokenize("maestro3.mid", start=START, end=END)

    def test_aug(self):
        def aug(_midi_dict: MidiDict, _start_ms: int, _end_ms: int):
            _tokenized_seq = tokenizer._tokenize_midi_dict(
                midi_dict=_midi_dict,
                start_ms=_start_ms,
                end_ms=_end_ms,
            )

            aug_fn = tokenizer.export_msg_mixup()
            _aug_tokenized_seq = aug_fn(_tokenized_seq)
            self.assertEqual(len(_tokenized_seq), len(_aug_tokenized_seq))

            return _tokenized_seq, _aug_tokenized_seq

        DELTA_MS = 5000
        tokenizer = AmtTokenizer()
        midi_dict = MidiDict.from_midi("tests/test_data/maestro2.mid")
        __end_ms = midi_dict.note_msgs[-1]["data"]["end"]

        for idx, __start_ms in enumerate(range(0, __end_ms, DELTA_MS)):
            tokenized_seq, aug_tokenized_seq = aug(
                midi_dict, __start_ms, __start_ms + DELTA_MS
            )

            self.assertEqual(
                len(
                    tokenizer._detokenize_midi_dict(
                        tokenized_seq, DELTA_MS
                    ).note_msgs
                ),
                len(
                    tokenizer._detokenize_midi_dict(
                        aug_tokenized_seq, DELTA_MS
                    ).note_msgs
                ),
            )

            if idx == 0 or idx == 1:
                logging.info(
                    f"msg mixup: {tokenized_seq} ->\n{aug_tokenized_seq}"
                )

                _midi_dict = tokenizer._detokenize_midi_dict(
                    tokenized_seq, DELTA_MS
                )
                _mid = _midi_dict.to_midi()
                _mid.save(f"tests/test_results/maestro2_orig.mid")

                _midi_dict = tokenizer._detokenize_midi_dict(
                    aug_tokenized_seq, DELTA_MS
                )
                _mid = _midi_dict.to_midi()
                _mid.save(f"tests/test_results/maestro2_aug.mid")


if __name__ == "__main__":
    unittest.main()
