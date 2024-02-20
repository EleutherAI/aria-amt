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

        for idx in (
            pbar := tqdm(
                range(start_idx, MAX_SEQ_LEN - 1),
                total=MAX_SEQ_LEN - (start_idx + 1),
                leave=False,
            )
        ):
            logits = model.forward(mel=audio_seg, tokens=seq[:, :idx])
            probs = torch.softmax(logits[0, -1], dim=-1)
            next_tok_id = torch.multinomial(probs / 0.001, num_samples=1)

            # Debug logging:
            # print(f"input seq shape: {seq[:, :idx].shape}")
            # print(f"logits shape: {logits.shape}")
            # print(f"probs shape: {probs.shape}")
            # print(int(next_tok_id), tokenizer.id_to_tok[int(next_tok_id)])

            if next_tok_id == pad_id or next_tok_id == eos_id:
                break
            else:
                seq[0, idx] = next_tok_id

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
    concat_seq = []
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

        print(f"Done {idx}/{len(audio_segments)}:\n{_seq}")

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
