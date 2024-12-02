import unittest
import logging
import torch
import os

from amt.tokenizer import AmtTokenizer
from ariautils.midi import MidiDict

logging.basicConfig(level=logging.INFO)
if os.path.isdir("tests/test_results") is False:
    os.mkdir("tests/test_results")


class TestAmtTokenizer(unittest.TestCase):
    def test_tokenize(self):
        def _tokenize_detokenize(mid_name: str, start: int, end: int):
            length = end - start
            tokenizer = AmtTokenizer()
            midi_dict = MidiDict.from_midi(f"tests/test_data/{mid_name}")

            logging.info(f"tokenizing {mid_name} in range ({start}, {end})...")
            tokenized_seq = tokenizer.tokenize(midi_dict, start, end)
            tokenized_seq = tokenizer.decode(tokenizer.encode(tokenized_seq))
            self.assertTrue(tokenizer.unk_tok not in tokenized_seq)
            _midi_dict = tokenizer.detokenize(tokenized_seq, length)
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

    def test_eos_tok(self):
        tokenizer = AmtTokenizer()
        midi_dict = MidiDict.from_midi(f"tests/test_data/maestro1.mid")

        cnt = 0
        while True:
            seq = tokenizer.tokenize(
                midi_dict, start_ms=cnt * 10000, end_ms=(cnt * 10000) + 30000
            )
            if len(seq) <= 2:
                self.assertEqual(seq[-1], tokenizer.eos_tok)
                break
            else:
                cnt += 1

    def test_pitch_aug(self):
        tokenizer = AmtTokenizer()
        tensor_pitch_aug = tokenizer.export_tensor_pitch_aug()

        midi_dict_1 = MidiDict.from_midi("tests/test_data/maestro1.mid")
        midi_dict_2 = MidiDict.from_midi("tests/test_data/maestro2.mid")
        midi_dict_3 = MidiDict.from_midi("tests/test_data/maestro3.mid")
        seq_1 = tokenizer.tokenize(midi_dict_1, 0, 30000)
        seq_1 = tokenizer.trunc_seq(seq_1, 2048)
        seq_2 = tokenizer.trunc_seq(
            tokenizer.tokenize(midi_dict_2, 0, 30000), 2048
        )
        seq_2 = tokenizer.trunc_seq(seq_2, 2048)
        seq_3 = tokenizer.trunc_seq(
            tokenizer.tokenize(midi_dict_3, 0, 30000), 2048
        )
        seq_3 = tokenizer.trunc_seq(seq_3, 2048)

        seqs = torch.stack(
            (
                torch.tensor(tokenizer.encode(seq_1)),
                torch.tensor(tokenizer.encode(seq_2)),
                torch.tensor(tokenizer.encode(seq_3)),
            )
        )
        aug_seqs = tensor_pitch_aug(seqs, shift=2)

        midi_dict_1_aug = tokenizer.detokenize(
            tokenizer.decode(aug_seqs[0].tolist()), 30000
        )
        midi_dict_2_aug = tokenizer.detokenize(
            tokenizer.decode(aug_seqs[1].tolist()), 30000
        )
        midi_dict_3_aug = tokenizer.detokenize(
            tokenizer.decode(aug_seqs[2].tolist()), 30000
        )
        midi_dict_1_aug.to_midi().save("tests/test_results/pitch1.mid")
        midi_dict_2_aug.to_midi().save("tests/test_results/pitch2.mid")
        midi_dict_3_aug.to_midi().save("tests/test_results/pitch3.mid")

    def test_aug(self):
        def aug(_midi_dict: MidiDict, _start_ms: int, _end_ms: int):
            _tokenized_seq = tokenizer.tokenize(
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
                len(tokenizer.detokenize(tokenized_seq, DELTA_MS).note_msgs),
                len(
                    tokenizer.detokenize(aug_tokenized_seq, DELTA_MS).note_msgs
                ),
            )

            if idx == 0 or idx == 1:
                logging.info(
                    f"msg mixup: {tokenized_seq} ->\n{aug_tokenized_seq}"
                )

                _midi_dict = tokenizer.detokenize(tokenized_seq, DELTA_MS)
                _mid = _midi_dict.to_midi()
                _mid.save(f"tests/test_results/maestro2_orig.mid")

                _midi_dict = tokenizer.detokenize(aug_tokenized_seq, DELTA_MS)
                _mid = _midi_dict.to_midi()
                _mid.save(f"tests/test_results/maestro2_aug.mid")


if __name__ == "__main__":
    unittest.main()
