import os
import random
import torch

from amt.model import AmtEncoderDecoder
from amt.tokenizer import AmtTokenizer
from amt.data import get_features
from amt.config import load_config
from aria.data.midi import MidiDict

config = load_config()
MAX_SEQ_LEN = config["data"]["max_seq_len"]
LEN_MS = 30000  # This should not be hardcoded

# TODO: Implement this with KV-caching, see the whisper inference file


def greedy_sample(
    model: AmtEncoderDecoder,
    audio_path: str,
    temp: float,  # Needed?
    top_p: float,  # Needed?
):
    # 1. Get audio features
    # 2. For each audio features, do greedy decoding starting each new segment
    #    with the prev tokens set correctly previously
    # 3. Add the results to one buffer sequence, with onsets potentially out of
    #    range, this should not be an issue the way that detokenize_midi_dict
    #    is currently implemented
    # 4.
    def _process_segment(
        audio_seg: torch.tensor,
        prefix: list,
        model: AmtEncoderDecoder,
        tokenizer: AmtTokenizer = AmtTokenizer(),
    ):
        start_idx = len(prefix)
        pad_id = tokenizer.pad_id
        seq = tokenizer.encode(tokenizer.trunc_seq(prefix, MAX_SEQ_LEN))

        for idx in range(start_idx, MAX_SEQ_LEN - 1):
            logits = model.forward(mel=audio_seg, tokens=seq)
            next_tok_id = logits[idx, :].argmax()

            if next_tok_id == pad_id:
                break
            else:
                seq[idx + 1] = next_tok_id

        if idx == MAX_SEQ_LEN - 1:
            print("WARNING: Ran out of context when generating sequence")

        seq = tokenizer.decode(seq)
        _, unclosed_notes = tokenizer._detokenize_midi_dict(
            tokenized_seq=seq, len_ms=LEN_MS
        )

        return seq, unclosed_notes

    tokenizer = AmtTokenizer()
    audio_segments = [f for f, _ in get_features(audio_path=audio_path)]
    _unclosed_notes = []
    concat_seq = []
    _onset_adj = 0
    for idx, _audio_seg in enumerate(audio_segments):
        _seq = [("prev", k) for k, _ in _unclosed_notes.items()]
        random.shuffle(_seq)

        _seq, _unclosed_notes = _process_segment(
            audio_seg=_audio_seg,
            prefix=_seq,
            model=model,
            tokenizer=tokenizer,
        )

        for tok in _seq:
            if type(tok) is tuple and tok[0] == "onset":
                _onset_orig = tok[1]
                _onset_adj = _onset_orig + (idx * LEN_MS)
                concat_seq.append(("onset", _onset_adj))
            elif tok is tokenizer.pad_tok:
                break
            else:
                concat_seq.append(tok)

    return tokenizer._detokenize_midi_dict(
        tokenized_seq=concat_seq,
        len_ms=_onset_adj,
    )
