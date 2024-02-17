import random

from collections import defaultdict

from aria.data.midi import MidiDict, get_duration_ms
from aria.tokenizer import Tokenizer
from amt.config import load_config


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
        self.velocity_tokens = [("vel", i) for i in self.velocity_quantizations]
        self.onset_tokens = [
            ("onset", i) for i in self.onset_time_quantizations
        ]

        self.add_tokens_to_vocab(
            self.special_tokens
            + self.prev_tokens
            + self.note_on_tokens
            + self.note_off_tokens
            + self.velocity_tokens
            + self.onset_tokens
        )
        self.pad_id = self.tok_to_id[self.pad_tok]

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

    # Go back and make sure that the use of < or <= are as we want
    # There might also be issues with very rapid notes, make sure this is
    # working as intended
    def _tokenize_midi_dict(
        self,
        midi_dict: MidiDict,
        start_ms: int,
        end_ms: int,
    ):
        channel_to_pedal_intervals = self._build_pedal_intervals(midi_dict)
        prev_notes = []
        on_off_notes = []
        for msg in midi_dict.note_msgs:
            _channel = msg["channel"]
            _pitch = msg["data"]["pitch"]
            _velocity = msg["data"]["velocity"]
            _start_tick = msg["data"]["start"]
            _end_tick = msg["data"]["end"]

            # Update end tick if affected by pedal
            for pedal_interval in channel_to_pedal_intervals[_channel]:
                pedal_start, pedal_end = (
                    pedal_interval[0],
                    pedal_interval[1],
                )
                if (
                    pedal_start <= _start_tick < pedal_end
                    and _end_tick < pedal_end
                ):
                    _end_tick = pedal_end

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

            rel_note_start_ms_q = self._quantize_onset(note_start_ms - start_ms)
            rel_note_end_ms_q = self._quantize_onset(note_end_ms - start_ms)
            velocity_q = self._quantize_velocity(_velocity)

            if note_end_ms <= start_ms or note_start_ms >= end_ms:  # Skip
                continue
            elif (
                note_start_ms < start_ms and _pitch not in prev_notes
            ):  # Add to prev notes
                prev_notes.append(_pitch)
                if note_end_ms < end_ms:
                    on_off_notes.append(
                        ("off", _pitch, rel_note_end_ms_q, None)
                    )
            else:  # Add to on_off_msgs
                # Skip notes with no duration or duplicate notes
                if rel_note_start_ms_q == rel_note_end_ms_q:
                    continue
                elif (
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

        on_off_notes.sort(key=lambda x: (x[2], x[0] == "on"))  # Check
        random.shuffle(prev_notes)

        tokenized_seq = []
        note_status = {}
        for pitch in prev_notes:
            tokenized_seq.append(("prev", pitch))
            note_status[pitch] = True
        for note in on_off_notes:
            _type, _pitch, _onset, _velocity = note
            if _type == "on":
                if note_status.get(_pitch) == True:
                    # If note already on, turn it off first
                    tokenized_seq.append(("off", _pitch))
                    tokenized_seq.append(("onset", _onset))

                tokenized_seq.append(("on", _pitch))
                tokenized_seq.append(("onset", _onset))
                tokenized_seq.append(("vel", _velocity))
                note_status[_pitch] = True
            elif _type == "off":
                if note_status.get(_pitch) == False:
                    # If note not on, skip
                    continue
                else:
                    tokenized_seq.append(("off", _pitch))
                    tokenized_seq.append(("onset", _onset))
                    note_status[_pitch] = False

        return tokenized_seq

    def _detokenize_midi_dict(self, tokenized_seq: list, len_ms: int):
        # NOTE: These values chosen so that 1000 ticks = 1000ms, allowing us to
        # skip converting between ticks and ms
        TICKS_PER_BEAT = 500
        TEMPO = 500000

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

        # Process notes
        notes_to_close = {}
        for tok_1, tok_2, tok_3 in zip(
            tokenized_seq[:],
            tokenized_seq[1:],
            tokenized_seq[2:] + [(None, None)],
        ):
            tok_1_type, tok_1_data = tok_1
            tok_2_type, tok_2_data = tok_2
            tok_3_type, tok_3_data = tok_3

            if tok_1_type == "prev":
                notes_to_close[tok_1_data] = (0, self.default_velocity)
            elif tok_1_type == "on":
                if (tok_2_type, tok_3_type) != ("onset", "vel"):
                    print("Unexpected token order")
                else:
                    notes_to_close[tok_1_data] = (tok_2_data, tok_3_data)
            elif tok_1_type == "off":
                if tok_2_type != "onset":
                    print("Unexpected token order")
                else:
                    # Process note and add to note msgs
                    note_to_close = notes_to_close.pop(tok_1_data, None)
                    if note_to_close is None:
                        print(f"No 'on' token corresponding to 'off' token")
                        continue
                    else:
                        _pitch = tok_1_data
                        _start_tick, _velocity = note_to_close
                        _end_tick = tok_2_data
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

        return MidiDict(
            meta_msgs=meta_msgs,
            tempo_msgs=tempo_msgs,
            pedal_msgs=pedal_msgs,
            instrument_msgs=instrument_msgs,
            note_msgs=note_msgs,
            ticks_per_beat=TICKS_PER_BEAT,
            metadata={},
        )

    def export_data_aug(self):
        return [self.export_msg_mixup()]

    def export_msg_mixup(self):
        def msg_mixup(src: list):
            # Reorder prev tokens
            res = []
            idx = 0
            for idx, tok in enumerate(src):
                tok_type, tok_data = tok
                if tok_type != "prev":
                    break
                else:
                    res.append(tok)

            random.shuffle(res)  # Only includes prev toks
            buffer = defaultdict(lambda: defaultdict(list))
            for tok_1, tok_2, tok_3 in zip(
                src[idx:],
                src[idx + 1 :],
                src[idx + 2 :] + [(None, None)],
            ):
                tok_1_type, tok_1_data = tok_1
                tok_2_type, tok_2_data = tok_2
                if tok_1_type == "on":
                    _onset = tok_2_data
                    buffer[_onset]["on"].append((tok_1, tok_2, tok_3))
                elif tok_1_type == "off":
                    _onset = tok_2_data
                    buffer[_onset]["off"].append((tok_1, tok_2))
                else:
                    pass

            # Shuffle order and re-append to result
            for k, v in sorted(buffer.items()):
                random.shuffle(v["on"])
                random.shuffle(v["off"])
                for item in v["off"]:
                    res.append(item[0])  # Pitch
                    res.append(item[1])  # Onset
                for item in v["on"]:
                    res.append(item[0])  # Pitch
                    res.append(item[1])  # Onset
                    res.append(item[2])  # Velocity

            return res

        return msg_mixup
