import os
import random
import torch

from tqdm import tqdm

from amt.model import AmtEncoderDecoder
from amt.tokenizer import AmtTokenizer
from amt.data import get_features
from amt.config import load_config
from aria.data.midi import MidiDict


# TODO: Implement this with KV-caching, see the whisper inference file

# Due to the autoregressive nature, a good inference algorithm should use some
# sort of branching to make sure that we don't miss notes, ect... Implement this
# next week -- Exciting problem (checkout other inference algos)

# Implement maximum note len =5s
# Implement either beam search or decoding initial onset note on first


def greedy_sample(
    model: AmtEncoderDecoder,
    audio_path: str,
    device: str,
):
    LEN_MS = 30000  # This should not be hardcoded
    MAX_SEQ_LEN = model.dims.n_text_ctx

    def _process_segment(
        audio_seg: torch.tensor,
        prefix: list,
        model: AmtEncoderDecoder,
        tokenizer: AmtTokenizer = AmtTokenizer(),
    ):
        start_idx = len(prefix)
        pad_id = tokenizer.pad_id
        eos_id = tokenizer.tok_to_id[tokenizer.eos_tok]
        audio_seg = audio_seg.unsqueeze(0).to(device)
        seq = tokenizer.encode(tokenizer.trunc_seq(prefix, MAX_SEQ_LEN))
        seq = torch.tensor(seq).unsqueeze(0).to(device)
        audio_feature = model.embed_audio(mel=audio_seg)

        for idx in (
            pbar := tqdm(
                range(start_idx, MAX_SEQ_LEN - 1),
                total=MAX_SEQ_LEN - (start_idx + 1),
                leave=False,
            )
        ):
            logits = model.logits(
                audio_features=audio_feature, tokens=seq[:, :idx]
            )
            next_tok_id = torch.argmax(logits[0, -1], dim=-1)

            seq[0, idx] = next_tok_id
            if next_tok_id == pad_id or next_tok_id == eos_id:
                break

        if idx == MAX_SEQ_LEN - 2:
            print("WARNING: Ran out of context when generating sequence")

        seq = tokenizer.decode(seq[0, :])
        _, unclosed_notes = tokenizer._detokenize_midi_dict(
            tokenized_seq=seq,
            len_ms=LEN_MS,
            return_unclosed_notes=True,
        )

        return seq, unclosed_notes

    audio_segments = [f for f, _ in get_features(audio_path=audio_path)]
    print(f"{len(audio_segments)} audio segments to process...")

    model.to(device)
    model.eval()
    tokenizer = AmtTokenizer()
    _unclosed_notes = []
    concat_seq = [tokenizer.bos_tok]
    _onset_adj = 0
    for idx, _audio_seg in enumerate(audio_segments):
        _seq = [("prev", p) for p in _unclosed_notes] + [tokenizer.bos_tok]

        _seq, _unclosed_notes = _process_segment(
            audio_seg=_audio_seg,
            prefix=_seq,
            model=model,
            tokenizer=tokenizer,
        )
        random.shuffle(_unclosed_notes)

        # DEBUG
        __midi_dict = tokenizer._detokenize_midi_dict(_seq, 30000)
        __midi = __midi_dict.to_midi()
        __midi.save(f"/weka/proj-aria/aria-amt/samples/res{idx}.mid")

        print(f"Done {idx + 1}/{len(audio_segments)}")
        for tok in _seq:
            if type(tok) is tuple and tok[0] == "onset":
                _onset_orig = tok[1]
                _onset_adj = _onset_orig + (idx * LEN_MS)
                concat_seq.append(("onset", _onset_adj))
            elif type(tok) is tuple and tok[0] == "prev":
                continue
            elif tok is tokenizer.bos_tok:
                continue
            elif tok is tokenizer.pad_tok or tok is tokenizer.eos_tok:
                break
            else:
                concat_seq.append(tok)

    return tokenizer._detokenize_midi_dict(
        tokenized_seq=concat_seq,
        len_ms=_onset_adj,
    )
