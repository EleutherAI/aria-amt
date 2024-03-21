import random
import os
import copy
import functools

from torch import Tensor
from collections import defaultdict

from aria.data.midi import MidiDict, get_duration_ms
from aria.tokenizer import Tokenizer
from amt.config import load_config


DEBUG = os.getenv("DEBUG")


class AmtTokenizer(Tokenizer):
    """MidiDict tokenizer designed for AMT"""

    def __init__(self, return_tensors: bool = False):
        super().__init__(return_tensors)
        self.config = load_config()["tokenizer"]
        self.name = "amt"

        self.time_step = self.config["time_quantization"]["step"]
        self.num_steps = self.config["time_quantization"]["num_steps"]
        self.onset_time_quantizations = [
            i * self.time_step for i in range(self.num_steps + 1)
        ]
        self.max_onset = self.onset_time_quantizations[-1]

        # Calculate velocity quantizations
        self.default_velocity = self.config["velocity_quantization"]["default"]
        self.velocity_step = self.config["velocity_quantization"]["step"]
        self.velocity_quantizations = [
            i * self.velocity_step
            for i in range(int(127 / self.velocity_step) + 1)
        ]
        self.max_velocity = self.velocity_quantizations[-1]

        # Build vocab
        self.prev_tokens = [("prev", i) for i in range(128)]
        self.note_on_tokens = [("on", i) for i in range(128)]
        self.note_off_tokens = [("off", i) for i in range(128)]
        self.pedal_tokens = [("pedal", 0), ("pedal", 1), ("prev", "pedal")]
        self.velocity_tokens = [("vel", i) for i in self.velocity_quantizations]
        self.onset_tokens = [
            ("onset", i) for i in self.onset_time_quantizations
        ]

        self.add_tokens_to_vocab(
            self.special_tokens
            + self.prev_tokens
            + self.note_on_tokens
            + self.note_off_tokens
            + self.pedal_tokens
            + self.velocity_tokens
            + self.onset_tokens
        )
        self.pad_id = self.tok_to_id[self.pad_tok]

    def _get_inference_ids(self):
        return [
            self.tok_to_id[tok]
            for tok in self.velocity_tokens + self.onset_tokens
        ]

    def _quantize_onset(self, time: int):
        # This function will return values res >= 0 (inc. 0)
        return self._find_closest_int(time, self.onset_time_quantizations)

    def _quantize_velocity(self, velocity: int):
        # This function will return values in the range 0 < res =< 127
        velocity_quantized = self._find_closest_int(
            velocity, self.velocity_quantizations
        )

        if velocity_quantized == 0 and velocity != 0:
            return self.velocity_step
        else:
            return velocity_quantized

    # TODO:
    # - I need to make this method more robust, as it will have to handle
    #   an arbitrary MIDI file
    def _tokenize_midi_dict(
        self,
        midi_dict: MidiDict,
        start_ms: int,
        end_ms: int,
        max_pedal_len_ms: int | None = None
    ):
        assert (
            end_ms - start_ms <= self.max_onset
        ), "Invalid values for start_ms, end_ms"

        if midi_dict.pedal_resolved is False:
            midi_dict.resolve_pedal()  # Important !!
        pedal_intervals = midi_dict._build_pedal_intervals()

        if len(pedal_intervals.keys()) > 1:
            print("Warning: midi_dict has more than one pedal channel")
        if len(midi_dict.instrument_msgs) > 1:
            print("Warning: midi_dict has more than one instrument msg")
        pedal_intervals = pedal_intervals[0]

        last_msg_ms = -1
        on_off_notes = []
        prev_toks = []
        for msg in midi_dict.note_msgs:
            _pitch = msg["data"]["pitch"]
            _velocity = msg["data"]["velocity"]
            _start_tick = msg["data"]["start"]
            _end_tick = msg["data"]["end"]

            note_start_ms = get_duration_ms(
                start_tick=0,
                end_tick=_start_tick,
                tempo_msgs=midi_dict.tempo_msgs,
                ticks_per_beat=midi_dict.ticks_per_beat,
            )
            note_end_ms = get_duration_ms(
                start_tick=0,
                end_tick=_end_tick,
                tempo_msgs=midi_dict.tempo_msgs,
                ticks_per_beat=midi_dict.ticks_per_beat,
            )

            if note_end_ms > last_msg_ms:
                last_msg_ms = note_end_ms

            rel_note_start_ms_q = self._quantize_onset(note_start_ms - start_ms)
            rel_note_end_ms_q = self._quantize_onset(note_end_ms - start_ms)
            velocity_q = self._quantize_velocity(_velocity)

            # This shouldn't be needed !
            if note_start_ms == note_end_ms:
                continue
            elif note_start_ms >= note_end_ms:
                continue

            assert note_start_ms <= note_end_ms, "Error"
            if note_end_ms <= start_ms or note_start_ms >= end_ms:  # Skip
                continue
            elif (
                note_start_ms < start_ms and _pitch not in prev_toks
            ):  # Add to prev notes
                prev_toks.append(_pitch)
                if note_end_ms < end_ms:
                    on_off_notes.append(
                        ("off", _pitch, rel_note_end_ms_q, None)
                    )
            else:  # Add to on_off_msgs
                # Skip notes with no duration or duplicate notes
                if rel_note_start_ms_q == rel_note_end_ms_q:
                    continue
                if (
                    "on",
                    _pitch,
                    rel_note_start_ms_q,
                    velocity_q,
                ) in on_off_notes:
                    continue
                on_off_notes.append(
                    ("on", _pitch, rel_note_start_ms_q, velocity_q)
                )
                if note_end_ms < end_ms:
                    on_off_notes.append(
                        ("off", _pitch, rel_note_end_ms_q, None)
                    )

        on_off_pedal = []
        for pedal_on_tick, pedal_off_tick in pedal_intervals:
            pedal_on_ms = get_duration_ms(
                start_tick=0,
                end_tick=pedal_on_tick,
                tempo_msgs=midi_dict.tempo_msgs,
                ticks_per_beat=midi_dict.ticks_per_beat,
            )
            pedal_off_ms = get_duration_ms(
                start_tick=0,
                end_tick=pedal_off_tick,
                tempo_msgs=midi_dict.tempo_msgs,
                ticks_per_beat=midi_dict.ticks_per_beat,
            )
            
            if max_pedal_len_ms is not None:
                pedal_off_ms = min(pedal_off_ms, pedal_on_ms + max_pedal_len_ms)

            rel_on_ms_q = self._quantize_onset(pedal_on_ms - start_ms)
            rel_off_ms_q = self._quantize_onset(pedal_off_ms - start_ms)

            # On message
            if pedal_off_ms <= start_ms or pedal_on_ms >= end_ms:
                continue
            elif pedal_on_ms < start_ms and pedal_off_ms >= start_ms:
                prev_toks.append("pedal")
            else:
                on_off_pedal.append(("pedal", 1, rel_on_ms_q, None))

            # Off message
            if pedal_off_ms <= start_ms or pedal_off_ms >= end_ms:
                continue
            else:
                on_off_pedal.append(("pedal", 0, rel_off_ms_q, None))

        on_off_combined = on_off_notes + on_off_pedal
        on_off_combined.sort(
            key=lambda x: (
                x[2],
                (0 if x[0] == "pedal" else 1 if x[0] == "off" else 2),
            )
        )
        random.shuffle(prev_toks)

        tokenized_seq = []
        for tok in on_off_combined:
            _type, _val, _onset, _velocity = tok
            if _type == "on":
                tokenized_seq.append(("on", _val))
                tokenized_seq.append(("onset", _onset))
                tokenized_seq.append(("vel", _velocity))
            elif _type == "off":
                tokenized_seq.append(("off", _val))
                tokenized_seq.append(("onset", _onset))
            elif _type == "pedal":
                if _val == 0:
                    tokenized_seq.append(("pedal", _val))
                    tokenized_seq.append(("onset", _onset))
                elif _val:
                    tokenized_seq.append(("pedal", _val))
                    tokenized_seq.append(("onset", _onset))

        prefix = [("prev", p) for p in prev_toks]

        # Add eos_tok only if segment includes end of midi_dict
        if last_msg_ms < end_ms:
            return prefix + [self.bos_tok] + tokenized_seq + [self.eos_tok]
        else:
            return prefix + [self.bos_tok] + tokenized_seq

    def _detokenize_midi_dict(
        self,
        tokenized_seq: list,
        len_ms: int,
        return_unclosed_notes: bool = False,
    ):
        # NOTE: These values chosen so that 1000 ticks = 1000ms, allowing us to
        # skip converting between ticks and ms
        assert len_ms > 0, "len_ms must be positive"
        TICKS_PER_BEAT = 500
        TEMPO = 500000

        tokenized_seq = copy.deepcopy(tokenized_seq)

        if self.eos_tok in tokenized_seq:
            tokenized_seq = tokenized_seq[: tokenized_seq.index(self.eos_tok)]
        if self.pad_tok in tokenized_seq:
            tokenized_seq = tokenized_seq[: tokenized_seq.index(self.pad_tok)]

        meta_msgs = []
        pedal_msgs = []
        note_msgs = []
        tempo_msgs = [{"type": "tempo", "data": TEMPO, "tick": 0}]
        instrument_msgs = [
            {
                "type": "instrument",
                "data": 0,
                "tick": 0,
                "channel": 0,
            }
        ]

        # Process prev tokens
        notes_to_close = {}
        for idx, tok in enumerate(tokenized_seq):
            if tok == self.bos_tok:
                break
            elif type(tok) == tuple and tok[0] == "prev":
                if tok[1] in notes_to_close.keys():
                    print(f"Duplicate 'prev' token: {tok[1]}")
                    if DEBUG:
                        raise Exception

                if tok[1] == "pedal":
                    pedal_msgs.append(
                        {
                            "type": "pedal",
                            "data": 1,
                            "tick": 0,
                            "channel": 0,
                        }
                    )
                elif isinstance(tok[1], int):
                    notes_to_close[tok[1]] = (0, self.default_velocity)
                else:
                    print(f"Invalid 'prev' token: {tok}")
                    if DEBUG:
                        raise Exception
            else:
                raise Exception(
                    f"Invalid note sequence at position {idx}: {tok, tokenized_seq[:idx]}"
                )

        # Process notes
        for tok_1, tok_2, tok_3 in zip(
            tokenized_seq[idx + 1 :],
            tokenized_seq[idx + 2 :],
            tokenized_seq[idx + 3 :] + [(None, None)],
        ):
            tok_1_type, tok_1_data = tok_1
            tok_2_type, tok_2_data = tok_2
            tok_3_type, tok_3_data = tok_3

            if tok_1_type == "prev":
                notes_to_close[tok_1_data] = (0, self.default_velocity)
                print("Unexpected token order: 'prev' seen after '<S>'")
                raise ValueError
            elif tok_1_type == "pedal":
                _pedal_data = tok_1_data
                _tick = tok_2_data
                pedal_msgs.append(
                    {
                        "type": "pedal",
                        "data": _pedal_data,
                        "tick": _tick,
                        "channel": 0,
                    }
                )
            elif tok_1_type == "on":
                if (tok_2_type, tok_3_type) != ("onset", "vel"):
                    print("Unexpected token order:", tok_1, tok_2, tok_3)
                    raise ValueError
                else:
                    notes_to_close[tok_1_data] = (tok_2_data, tok_3_data)
            elif tok_1_type == "off":
                if tok_2_type != "onset":
                    print("Unexpected token order:", tok_1, tok_2, tok_3)
                    if DEBUG:
                        raise Exception
                else:
                    # Process note and add to note msgs
                    note_to_close = notes_to_close.pop(tok_1_data, None)
                    if note_to_close is None:
                        print(
                            f"No 'on' token corresponding to 'off' token: {tok_1, tok_2}"
                        )
                        if DEBUG:
                            raise Exception
                        continue
                    else:
                        _pitch = tok_1_data
                        _start_tick, _velocity = note_to_close
                        _end_tick = tok_2_data

                        if _end_tick - _start_tick > 5000:
                            _end_tick = _start_tick + 5000

                        note_msgs.append(
                            {
                                "type": "note",
                                "data": {
                                    "pitch": _pitch,
                                    "start": _start_tick,
                                    "end": _end_tick,
                                    "velocity": _velocity,
                                },
                                "tick": _start_tick,
                                "channel": 0,
                            }
                        )

        # Process remaining notes with no off token
        for k, v in notes_to_close.items():
            _pitch = k
            _start_tick, _velocity = v
            _end_tick = len_ms

            if _end_tick - _start_tick > 5000:
                _end_tick = _start_tick + 5000

            note_msgs.append(
                {
                    "type": "note",
                    "data": {
                        "pitch": _pitch,
                        "start": _start_tick,
                        "end": _end_tick,
                        "velocity": _velocity,
                    },
                    "tick": _start_tick,
                    "channel": 0,
                }
            )

        midi_dict = MidiDict(
            meta_msgs=meta_msgs,
            tempo_msgs=tempo_msgs,
            pedal_msgs=pedal_msgs,
            instrument_msgs=instrument_msgs,
            note_msgs=note_msgs,
            ticks_per_beat=TICKS_PER_BEAT,
            metadata={},
        )

        if return_unclosed_notes is True:
            return midi_dict, [p for p, _ in notes_to_close.items()]
        else:
            return midi_dict

    def trunc_seq(self, seq: list, seq_len: int):
        """Truncate or pad sequence to feature sequence length."""
        seq += [self.pad_tok] * (seq_len - len(seq))

        return seq[:seq_len]

    def export_data_aug(self):
        return [self.export_msg_mixup()]

    def export_msg_mixup(self):
        def msg_mixup(src: list):
            # Process bos, eos, and pad tokens
            orig_len = len(src)
            seen_pad_tok = False
            seen_eos_tok = False
            if self.pad_tok in src:
                seen_pad_tok = True
            if self.eos_tok in src:
                src = src[: src.index(self.eos_tok)]
                seen_eos_tok = True

            # Reorder prev tokens
            res = []
            idx = 0
            for idx, tok in enumerate(src):
                if tok == self.bos_tok:
                    break
                elif type(tok) == tuple and tok[0] != "prev":
                    print("Missing BOS token when processing prefix")
                    break
                elif type(tok) == tuple and tok[0] == "prev":
                    res.append(tok)
                else:
                    print(f"Unexpected token when processing prefix: {tok}")
                    if DEBUG:
                        raise Exception

            random.shuffle(res)  # Only includes prev toks
            res.append(self.bos_tok)  # Beggining of sequence

            buffer = defaultdict(lambda: defaultdict(list))
            for tok_1, tok_2, tok_3 in zip(
                src[idx + 1 :],
                src[idx + 2 :],
                src[idx + 3 :] + [(None, None)],
            ):
                if tok_2 == self.pad_tok:
                    seen_pad_tok = True
                    break

                tok_1_type, tok_1_data = tok_1
                tok_2_type, tok_2_data = tok_2

                if tok_1_type == "on":
                    _onset = tok_2_data
                    buffer[_onset]["on"].append((tok_1, tok_2, tok_3))
                elif tok_1_type == "off":
                    _onset = tok_2_data
                    buffer[_onset]["off"].append((tok_1, tok_2))
                elif tok_1_type == "pedal":
                    _onset = tok_2_data
                    buffer[_onset]["pedal"].append((tok_1, tok_2))
                else:
                    pass

            # Shuffle order and re-append to result
            for k, v in sorted(buffer.items()):
                off_pedal_combined = v["off"] + v["pedal"]
                random.shuffle(off_pedal_combined)
                random.shuffle(v["on"])
                for item in off_pedal_combined:
                    res.append(item[0])  # Off or pedal
                    res.append(item[1])  # Onset
                for item in v["on"]:
                    res.append(item[0])  # Pitch
                    res.append(item[1])  # Onset
                    res.append(item[2])  # Velocity

            if seen_eos_tok is True:
                res += [self.eos_tok]
            if seen_pad_tok is True:
                return self.trunc_seq(res, orig_len)
            else:
                return res

        return msg_mixup

    def export_tensor_pitch_aug(self):
        def tensor_pitch_aug(
            seq: Tensor,
            shift: int,
            tok_to_id: dict,
            id_to_tok: dict,
            pad_tok: str,
            unk_tok: str,
        ):
            """This acts on (batched) tensors, applying pitch aug in place"""
            if shift == 0:
                return seq

            batch_size, seq_len = seq.shape
            for i in range(batch_size):
                for j in range(seq_len):
                    tok = id_to_tok[seq[i, j].item()]
                    if type(tok) is tuple and tok[0] in {"on", "off"}:
                        msg_type, pitch = tok
                        seq[i, j] = tok_to_id.get(
                            (msg_type, pitch + shift), unk_tok
                        )
                    elif tok == pad_tok:
                        break

            return seq

        return functools.partial(
            tensor_pitch_aug,
            tok_to_id=self.tok_to_id,
            id_to_tok=self.id_to_tok,
            pad_tok=self.pad_tok,
            unk_tok=self.unk_tok,
        )
