import os
import signal
import time
import copy
import random
import logging
import traceback
import threading
import torch
import concurrent
import multiprocessing
import torch.multiprocessing as torch_multiprocessing
import torch._dynamo.config
import torch._inductor.config
import numpy as np

from multiprocessing import Queue, Manager
from multiprocessing.synchronize import Lock as LockType
from queue import Empty
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Deque
from tqdm import tqdm
from functools import wraps
from torch.cuda import is_bf16_supported
from librosa.effects import _signal_to_frame_nonsilent

from amt.inference.model import AmtEncoderDecoder
from amt.tokenizer import AmtTokenizer
from amt.audio import AudioTransform, SAMPLE_RATE
from amt.data import get_wav_segments

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True

MAX_SEQ_LEN = 4096
MAX_BLOCK_LEN = 4096
LEN_MS = 30000
STRIDE_FACTOR = 3
CHUNK_LEN_MS = LEN_MS // STRIDE_FACTOR


# TODO: Implement continuous batching in a torch.compile friendly way


def _setup_logger(name: str | None = None):
    logger_name = f"[{name}] " if name else ""
    logger = logging.getLogger(__name__)
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        f"[%(asctime)s] {logger_name}%(process)d: [%(levelname)s] %(message)s",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler("transcribe.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logging.getLogger(__name__)


@torch.jit.script
def get_static_mask():
    # The values are hardcoded here for the pytorch jit - manually update
    col_indices = torch.arange(3419, device="cuda").unsqueeze(0)
    mask_a = col_indices >= 392
    mask_b = col_indices <= 3418
    return col_indices, mask_a & mask_b


@torch.jit.script
def recalculate_tok_ids(
    logits: torch.Tensor,
    tok_ids: torch.Tensor,
):
    probs = torch.softmax(logits, dim=-1)

    # Mask out all non-onset/vel tok_ids
    col_indices, interval_mask = get_static_mask()

    # Mask out tok_ids larger than 30ms from original tok_id
    tok_ids_expanded = tok_ids.unsqueeze(1)
    mask_c = col_indices <= tok_ids_expanded + 2
    mask_d = col_indices >= tok_ids_expanded - 2
    beam_mask = mask_c & mask_d

    # Don't mask out the original tok_id (required for non-onset/vel toks)
    tok_id_mask = torch.zeros_like(probs, dtype=torch.bool)
    tok_id_mask.scatter_(1, tok_ids_expanded, 1)

    # Combine and calculate probs
    combined_mask = (interval_mask & beam_mask) | tok_id_mask
    probs[~combined_mask] = 0

    # Calculate expected value
    weighted_idxs = probs * torch.arange(
        probs.size(1), device=probs.device
    ).float().unsqueeze(0)
    idx_evs = (
        (weighted_idxs.sum(dim=1) / (probs.sum(dim=1) + 1e-9))
        .round()
        .to(torch.long)
    )

    return idx_evs


# Changes seq and eos_idxs in place - tok_ids hardcoded
@torch.jit.script
def update_seq_end_idxs_(
    next_tok_ids: torch.Tensor,
    seq: torch.Tensor,
    eos_idxs: torch.Tensor,
    prefix_lens: torch.Tensor,
    idx: int,
):
    # Update eos_idxs if next tok is eos_tok and not eos_id < idx
    eos_mask = (next_tok_ids == 1) & (eos_idxs > idx)
    eos_idxs[eos_mask] = idx

    # Update eos_idxs if next tok in onset > 20000
    offset_mask = (next_tok_ids >= 2418) & (eos_idxs > idx)
    eos_idxs[offset_mask] = idx - 2

    # Don't update toks in prefix or after eos_idx
    insert_mask = (prefix_lens <= idx) & (eos_idxs >= idx)
    seq[insert_mask, idx] = next_tok_ids[insert_mask]


def optional_bf16_autocast(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_bf16_supported():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return func(*args, **kwargs)
        else:
            # TODO: We are using float instead of float16 due to strange bug
            with torch.autocast("cuda", dtype=torch.float):
                return func(*args, **kwargs)

    return wrapper


@torch.inference_mode()
def decode_token(
    model: AmtEncoderDecoder,
    x: torch.Tensor,
    xa: torch.Tensor,
    x_input_pos: torch.Tensor,
    xa_input_pos: torch.Tensor,
):
    logits = model.decoder.forward(
        x=x,
        xa=xa,
        x_input_pos=x_input_pos,
        xa_input_pos=xa_input_pos,
    )[:, -1]
    next_tok_ids = torch.argmax(logits, dim=-1)

    return logits, next_tok_ids


@torch.inference_mode()
def prefill(
    model: AmtEncoderDecoder,
    x: torch.Tensor,
    xa: torch.Tensor,
    x_input_pos: torch.Tensor,
    xa_input_pos: torch.Tensor,
):
    # This is the same as decode_token and is separate for compilation reasons
    logits = model.decoder.forward(
        x=x,
        xa=xa,
        x_input_pos=x_input_pos,
        xa_input_pos=xa_input_pos,
    )[:, -1]
    next_tok_ids = torch.argmax(logits, dim=-1)

    return logits, next_tok_ids


# This is not used anywhere but may come in handy one days
def calculate_input_pos(prefix_lens: torch.Tensor, prefill: bool):
    # Given prefix lens e.g. [67, 2, 9], generates the input positions,
    # truncate to the left with -1
    max_pos = torch.max(prefix_lens)
    pos_idxs = torch.stack(
        [
            torch.cat(
                [
                    torch.full((max_pos - pos,), -1, dtype=torch.long),
                    torch.arange(pos),
                ]
            )
            for pos in prefix_lens
        ]
    )

    return pos_idxs


@optional_bf16_autocast
@torch.inference_mode()
def process_segments(
    tasks: List,
    model: AmtEncoderDecoder,
    audio_transform: AudioTransform,
    tokenizer: AmtTokenizer,
    logger: logging.Logger,
):
    log_mels = audio_transform.log_mel(
        torch.stack([audio_seg.cuda() for (audio_seg, prefix), _ in tasks])
    )
    audio_features = model.encoder(xa=log_mels)

    raw_prefixes = [prefix for (audio_seg, prefix), _ in tasks]
    prefix_lens = torch.tensor(
        [len(prefix) for prefix in raw_prefixes], dtype=torch.int
    ).cuda()
    min_prefix_len = min(prefix_lens).item()
    prefixes = [
        tokenizer.trunc_seq(prefix, MAX_BLOCK_LEN) for prefix in raw_prefixes
    ]
    seq = torch.stack([tokenizer.encode(prefix) for prefix in prefixes]).cuda()
    eos_idxs = torch.tensor(
        [MAX_BLOCK_LEN for _ in prefixes], dtype=torch.int
    ).cuda()

    for idx in (
        pbar := tqdm(
            range(min_prefix_len, MAX_BLOCK_LEN - 1),
            total=MAX_BLOCK_LEN - (min_prefix_len + 1),
            leave=False,
        )
    ):
        # for idx in range(min_prefix_len, MAX_BLOCK_LEN - 1):
        if idx == min_prefix_len:
            logits, next_tok_ids = prefill(
                model,
                x=seq[:, :idx],
                xa=audio_features,
                x_input_pos=torch.arange(0, idx, device=seq.device),
                xa_input_pos=torch.arange(
                    0, audio_features.shape[1], device=seq.device
                ),
            )
        else:
            with torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.MATH
            ):
                logits, next_tok_ids = decode_token(
                    model,
                    x=seq[:, idx - 1 : idx],
                    xa=audio_features,
                    x_input_pos=torch.tensor(
                        [idx - 1], device=seq.device, dtype=torch.int
                    ),
                    xa_input_pos=torch.tensor(
                        [], device=seq.device, dtype=torch.int
                    ),
                )
        assert not torch.isnan(logits).any(), "NaN seen in logits"

        logits[:, 389] *= 1.05  # Increase pedal-off msg logits
        next_tok_ids = torch.argmax(logits, dim=-1)

        next_tok_ids = recalculate_tok_ids(
            logits=logits,
            tok_ids=next_tok_ids,
        )
        update_seq_end_idxs_(
            next_tok_ids=next_tok_ids,
            seq=seq,
            eos_idxs=eos_idxs,
            prefix_lens=prefix_lens,
            idx=idx,
        )

        if all(_idx <= idx for _idx in eos_idxs):
            break

    # If there is a context length overflow, we need to have some special logic
    # to make sure that a sequence of the correct format is returned. Right now
    # it messes things up somehow
    if not all(_idx <= idx for _idx in eos_idxs):
        logger.warning("Context length overflow when transcribing segment(s)")

    results = [
        tokenizer.decode(seq[_idx, : eos_idxs[_idx] + 1])
        for _idx in range(seq.shape[0])
    ]

    return results


@torch.inference_mode()
def gpu_manager(
    gpu_batch_queue: Queue,
    gpu_waiting_dict: dict,
    gpu_waiting_dict_lock: LockType,
    result_queue: Queue,
    model: AmtEncoderDecoder,
    batch_size: int,
    compile_mode: str | bool = False,
    gpu_id: int | None = None,
):
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)

    if gpu_id is not None:
        logger = _setup_logger(name=f"GPU-{gpu_id}")
    else:
        logger = _setup_logger(name=f"GPU")
        gpu_id = 0

    logger.info("Started GPU manager")

    model.decoder.setup_cache(
        batch_size=batch_size,
        max_seq_len=MAX_BLOCK_LEN,
        dtype=torch.bfloat16 if is_bf16_supported() else torch.float,
    )
    model.cuda()
    model.eval()
    if compile_mode is not False:
        global decode_token
        decode_token = torch.compile(
            decode_token,
            mode=compile_mode,
            fullgraph=True,
        )

    audio_transform = AudioTransform().cuda()
    tokenizer = AmtTokenizer(return_tensors=True)

    try:
        while True:
            try:
                with gpu_waiting_dict_lock:
                    gpu_waiting_dict[gpu_id] = time.time()
                batch = gpu_batch_queue.get(timeout=60)
                with gpu_waiting_dict_lock:
                    del gpu_waiting_dict[gpu_id]
            except Empty as e:
                with gpu_waiting_dict_lock:
                    del gpu_waiting_dict[gpu_id]
                raise e
            else:
                try:
                    results = process_segments(
                        tasks=batch,
                        model=model,
                        audio_transform=audio_transform,
                        tokenizer=tokenizer,
                        logger=logger,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to process batch: {traceback.format_exc()}"
                    )
                    raise e
                else:
                    # pid = -1 when its a pad sequence
                    for result, (_, pid) in zip(results, batch):
                        if pid != -1:
                            result_queue.put((result, pid))

    except Exception as e:
        logger.error(f"GPU manager failed with exception: {e}")
    finally:
        del gpu_waiting_dict[gpu_id]
        logger.info(f"GPU manager terminated")


def _find_min_diff_batch(tasks: List, batch_size: int):
    prefix_lens = [
        (len(prefix), idx) for idx, ((audio_seg, prefix), _) in enumerate(tasks)
    ]
    prefix_lens.sort(key=lambda x: x[0])

    min_diff = float("inf")
    start_idx = 0

    # Iterate through the array to find the batch with the min difference
    for _idx in range(len(prefix_lens) - batch_size + 1):
        current_diff = (
            prefix_lens[_idx + batch_size - 1][0] - prefix_lens[_idx][0]
        )
        if current_diff < min_diff:
            min_diff = current_diff
            start_idx = _idx

    return [
        orig_idx
        for prefix_lens, orig_idx in prefix_lens[
            start_idx : start_idx + batch_size
        ]
    ]


# NOTE:
# - For some reason copying gpu_waiting_dict is not working properly and is
#   leading to race conditions. I've implemented a lock to stop it.
# - The size of gpu_batch_queue decreases before the code for deleting the
#   corresponding entry in gpu_waiting_dict gets processed. Adding a short
#   sleep is a workaround
def gpu_batch_manager(
    gpu_task_queue: Queue,
    gpu_batch_queue: Queue,
    gpu_waiting_dict: dict,
    gpu_waiting_dict_lock: LockType,
    batch_size: int,
    max_wait_time: float = 0.25,
    min_batch_size: int = 1,
):
    logger = _setup_logger(name="B")
    logger.info("Started batch manager")

    tasks: Deque[Tuple[object, int]] = deque()
    gpu_wait_time = 0

    try:
        while True:
            try:
                while not gpu_task_queue.empty():
                    task, pid = gpu_task_queue.get_nowait()
                    tasks.append((task, pid))
            except Empty:
                pass

            with gpu_waiting_dict_lock:
                curr_time = time.time()
                num_tasks_in_batch_queue = gpu_batch_queue.qsize()
                num_gpus_waiting = len(gpu_waiting_dict)
                gpu_wait_time = (
                    max(
                        [
                            curr_time - wait_time_abs
                            for gpu_id, wait_time_abs in gpu_waiting_dict.items()
                        ]
                    )
                    if gpu_waiting_dict
                    else 0.0
                )

            should_create_batch = (
                len(tasks) >= 4 * batch_size
                or (
                    num_gpus_waiting > num_tasks_in_batch_queue
                    and len(tasks) >= batch_size
                )
                or (
                    num_gpus_waiting > num_tasks_in_batch_queue
                    and len(tasks) >= min_batch_size
                    and gpu_wait_time > max_wait_time
                )
            )

            if should_create_batch:
                logger.debug(
                    f"Creating batch: "
                    f"num_gpus_waiting={num_gpus_waiting}, "
                    f"gpu_wait_time={round(gpu_wait_time, 4)}s, "
                    f"num_tasks_ready={len(tasks)}, "
                    f"num_batches_ready={num_tasks_in_batch_queue}"
                )
                batch = create_batch(tasks, batch_size, min_batch_size, logger)
                gpu_batch_queue.put(batch)
                time.sleep(0.025)

    except Exception as e:
        logger.error(f"GPU batch manager failed with exception: {e}")
    finally:
        logger.info("GPU batch manager terminated")


def create_batch(
    tasks: Deque[Tuple[object, int]],
    batch_size: int,
    min_batch_size: int,
    logger: logging.Logger,
):
    assert len(tasks) >= min_batch_size, "Insufficient number of tasks"

    if len(tasks) < batch_size:
        logger.info(f"Creating a partial padded batch with {len(tasks)} tasks")
        batch_idxs = list(range(len(tasks)))
        batch = [tasks.popleft() for _ in batch_idxs]

        while len(batch) < batch_size:
            pad_task, _ = batch[0]
            batch.append((pad_task, -1))
    else:
        batch_idxs = _find_min_diff_batch(list(tasks), batch_size)
        batch = [tasks[idx] for idx in batch_idxs]
        for idx in sorted(batch_idxs, reverse=True):
            del tasks[idx]

    return batch


def _shift_onset(seq: List, shift_ms: int):
    res = []
    for tok in seq:
        if type(tok) is tuple and tok[0] == "onset":
            res.append(("onset", tok[1] + shift_ms))
        else:
            res.append(tok)

    return res


def _truncate_seq(
    seq: List,
    start_ms: int,
    end_ms: int,
    tokenizer: AmtTokenizer = AmtTokenizer(),
):
    # Truncates and shifts a sequence by retokenizing the underlying midi_dict
    if start_ms == end_ms:
        _mid_dict, unclosed_notes = tokenizer._detokenize_midi_dict(
            seq, start_ms, return_unclosed_notes=True
        )
        random.shuffle(unclosed_notes)
        return [("prev", p) for p in unclosed_notes] + [tokenizer.bos_tok]
    else:
        _mid_dict = tokenizer._detokenize_midi_dict(seq, LEN_MS)
        if len(_mid_dict.note_msgs) == 0:
            return [tokenizer.bos_tok]
        else:
            # The end_ms - 1 is a workaround to get rid of the off msgs
            res = tokenizer._tokenize_midi_dict(_mid_dict, start_ms, end_ms - 1)

        if res[-1] == tokenizer.eos_tok:
            res.pop()
        return res


# TODO: Add detection for pedal messages which occur before notes are played
def _process_silent_intervals(
    seq: List,
    intervals: List,
    tokenizer: AmtTokenizer,
):
    def adjust_onset(_onset: int):
        # Adjusts the onset according to the silence intervals
        for start, end in intervals:
            if start <= _onset <= end:
                return start

        return _onset

    if len(intervals) == 0:
        return seq

    res = []
    logger = logging.getLogger(__name__)
    active_notes = {pitch: False for pitch in range(0, 127)}
    active_notes["pedal"] = False

    for tok_1, tok_2, tok_3 in zip(
        seq,
        seq[1:] + [tokenizer.pad_tok],
        seq[2:] + [tokenizer.pad_tok, tokenizer.pad_tok],
    ):
        if isinstance(tok_1, tuple) is False:
            res.append(tok_1)
            continue
        elif tok_1[0] == "prev":
            res.append(tok_1)
            active_notes[tok_1[1]] = True
            continue
        elif tok_1[0] in {"onset", "vel"}:
            continue

        if tok_1[0] == "pedal":
            note_type = "on" if tok_1[1] == 1 else "off"
            note_val = "pedal"
        elif tok_1[0] in {"on", "off"}:
            note_type = tok_1[0]
            note_val = tok_1[1]

        if note_type == "on":
            # Check that the rest of the tokens are valid
            if isinstance(tok_2, tuple) is False:
                logger.debug(f"Invalid token sequence {tok_1}, {tok_2}")
                continue
            if note_val != "pedal" and isinstance(tok_3, tuple) is False:
                logger.debug(
                    f"Invalid token sequence {tok_1}, {tok_2}, {tok_3}"
                )
                continue

            # Don't add on if note is already on
            if active_notes[note_val] is True:
                continue

            # Calculate adjusted onset and add if conditions are met
            onset = tok_2[1]
            onset_adj = adjust_onset(onset)
            if onset != onset_adj:
                continue
            else:
                active_notes[note_val] = True
                res.append(tok_1)
                res.append(tok_2)
                if note_val != "pedal":
                    res.append(tok_3)

        elif note_type == "off":
            # Check that the rest of the tokens are valid
            if isinstance(tok_2, tuple) is False and tok_2[0] != "onset":
                logger.debug(f"Invalid token sequence {tok_1}, {tok_2}")
                continue

            # Don't add on if note is not on
            if active_notes[note_val] is False:
                continue

            # Add note with adjusted offset
            offset = tok_2[1]
            offset_adj = adjust_onset(offset)
            if offset != offset_adj:
                logger.debug(
                    f"Adjusted offset of {tok_1}, {tok_2} -> {offset_adj}"
                )
            res.append(tok_1)
            res.append(("onset", tokenizer._quantize_onset(offset_adj)))
            active_notes[note_val] = False

    return res


def _get_silent_intervals(wav: torch.Tensor):
    FRAME_LEN = 2048
    HOP_LEN = 512
    MIN_WINDOW_S = 5
    MIN_WINDOW_STEPS = (SAMPLE_RATE // HOP_LEN) * MIN_WINDOW_S + 1
    MS_PER_HOP = int((HOP_LEN * 1e3) / SAMPLE_RATE)

    non_silent = _signal_to_frame_nonsilent(
        wav.numpy(),
        frame_length=FRAME_LEN,
        hop_length=HOP_LEN,
        top_db=45,
        ref=np.max,
    )
    non_silent = np.concatenate(([True], non_silent, [True]))

    edges = np.diff(non_silent.astype(int))
    starts = np.where(edges == -1)[0]
    ends = np.where(edges == 1)[0]

    # Calculate lengths
    lengths = ends - starts

    # Filter intervals by minimum length
    valid = lengths > MIN_WINDOW_STEPS
    silent_intervals = [
        (start * MS_PER_HOP, (end - 1) * MS_PER_HOP)
        for start, end, vl in zip(starts, ends, valid)
        if vl
    ]

    return silent_intervals


def transcribe_file(
    file_path,
    gpu_task_queue: Queue,
    result_queue: Queue,
    pid: int,
    tokenizer: AmtTokenizer = AmtTokenizer(),
    segment: Tuple[int, int] | None = None,
):
    logger = logging.getLogger(__name__)

    logger.info(f"Getting wav segments: {file_path}")

    seq = [tokenizer.bos_tok]
    concat_seq = [tokenizer.bos_tok]
    idx = 0
    for curr_audio_segment in get_wav_segments(
        audio_path=file_path,
        stride_factor=STRIDE_FACTOR,
        pad_last=True,
        segment=segment,
    ):
        init_idx = len(seq)
        # Add to gpu queue and wait for results
        silent_intervals = _get_silent_intervals(curr_audio_segment)
        input_seq = copy.deepcopy(seq)
        gpu_task_queue.put(((curr_audio_segment, seq), pid))
        while True:
            try:
                gpu_result = result_queue.get(timeout=0.01)
            except Exception as e:
                pass
            else:
                if gpu_result[1] == pid:
                    seq = gpu_result[0]
                    break
                else:
                    result_queue.put(gpu_result)

        if len(silent_intervals) > 0:
            logger.debug(
                f"Seen silent intervals in audio chunk {idx}: {silent_intervals}"
            )

        seq_adj = _process_silent_intervals(
            seq, intervals=silent_intervals, tokenizer=tokenizer
        )

        if len(seq_adj) < len(seq) - 15:
            logger.info(
                f"Removed tokens ({len(seq)} -> {len(seq_adj)}) "
                f"in audio chunk {idx} according to silence in intervals: "
                f"{silent_intervals}",
            )
            seq = seq_adj

        try:
            next_seq = _truncate_seq(
                seq,
                CHUNK_LEN_MS,
                LEN_MS - CHUNK_LEN_MS,
            )
        except Exception as e:
            logger.info(
                f"Failed to reconcile sequences for audio chunk {idx}: {file_path}"
            )
            logger.debug(traceback.format_exc())

            try:
                seq = _truncate_seq(
                    input_seq,
                    CHUNK_LEN_MS - 2,
                    CHUNK_LEN_MS,
                )
            except Exception as e:
                seq = [tokenizer.bos_tok]
                logger.info(
                    f"Failed to recover prompt, proceeding with default: {seq}"
                )
            else:
                logger.info(f"Proceeding with prompt: {seq}")

        else:
            if seq[-1] == tokenizer.eos_tok:
                logger.info(f"Seen eos_tok in audio chunk {idx}: {file_path}")
                seq = seq[:-1]

            if len(next_seq) == 1:
                logger.info(
                    f"Skipping audio chunk {idx} (silence): {file_path}"
                )
                seq = [tokenizer.bos_tok]
            else:
                concat_seq += _shift_onset(
                    seq[init_idx:],
                    idx * CHUNK_LEN_MS,
                )
                seq = next_seq

        idx += 1

    return concat_seq


def get_save_path(
    file_path: str,
    input_dir: str | None,
    save_dir: str,
    idx_str: int | str = "",
):
    if input_dir is None:
        save_path = os.path.join(
            save_dir,
            os.path.splitext(os.path.basename(file_path))[0] + f"{idx_str}.mid",
        )
    else:
        input_rel_path = os.path.relpath(file_path, input_dir)
        save_path = os.path.join(
            save_dir, os.path.splitext(input_rel_path)[0] + f"{idx_str}.mid"
        )
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

    return save_path


def process_file(
    file_path: str,
    file_queue: Queue,
    gpu_task_queue: Queue,
    result_queue: Queue,
    tokenizer: AmtTokenizer,
    save_dir: str,
    input_dir: str,
    logger: logging.Logger,
    segments: List[Tuple[int, Tuple[int, int]]] | None = None,
):
    def _save_seq(_seq: List, _save_path: str):
        if os.path.exists(_save_path):
            logger.info(f"Already exists {_save_path} - overwriting")

        for tok in _seq[::-1]:
            if type(tok) is tuple and tok[0] == "onset":
                last_onset = tok[1]
                break

        try:
            mid_dict = tokenizer._detokenize_midi_dict(
                tokenized_seq=_seq,
                len_ms=last_onset,
            )
            mid = mid_dict.to_midi()
            mid.save(_save_path)
        except Exception as e:
            logger.error(f"Failed to save {_save_path}")
            logger.debug(traceback.format_exc())
            logger.debug(_seq)

    def remove_failures_from_queue_(_queue: Queue, _pid: int):
        _buff = []
        while True:
            try:
                _buff.append(_queue.get(timeout=5))
            except Exception:
                break

        num_removed = 0
        for _task, __pid in _buff:
            if _pid != __pid:
                _queue.put((_task, __pid))
            else:
                num_removed += 1

        return num_removed

    pid = threading.get_ident()
    if segments is None:
        # process_file and get_wav_segments will interpret segment=None as
        # processing the entire file
        segments = [(None, None)]

    if len(segments) == 0:
        logger.info(f"No segments to transcribe, skipping file: {file_path}")

    for idx, segment in segments:
        idx_str = f"_{idx}" if idx is not None else ""
        save_path = get_save_path(file_path, input_dir, save_dir, idx_str)

        try:
            seq = transcribe_file(
                file_path,
                gpu_task_queue,
                result_queue,
                pid=pid,
                segment=segment,
            )
        except Exception as e:
            logger.error(
                f"Failed to process {file_path} segment {idx}: {traceback.format_exc()}"
            )
            task_rmv_cnt = remove_failures_from_queue_(gpu_task_queue, pid)
            res_rmv_cnt = remove_failures_from_queue_(result_queue, pid)
            logger.info(f"Removed {task_rmv_cnt} from task queue")
            logger.info(f"Removed {res_rmv_cnt} from result queue")
            continue

        logger.info(
            f"Finished file: {file_path} (segment: {idx if idx is not None else 'full'})"
        )
        if len(seq) < 500:
            logger.info(
                f"Skipping seq - too short (segment {idx if idx is not None else 'full'})"
            )
        else:
            logger.debug(
                f"Saving seq of length {len(seq)} from file: {file_path} (segment: {idx if idx is not None else 'full'})"
            )
            _save_seq(seq, save_path)

    logger.info(f"{file_queue.qsize()} file(s) remaining in queue")


def cleanup_processes(child_pids: List):
    for pid in child_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError as e:
            pass
        except Exception as e:
            print(f"Failed to kill child process: {e}")


def watchdog(main_pids: List, child_pids: List):
    while True:
        if not all(os.path.exists(f"/proc/{pid}") for pid in main_pids):
            print("Watchdog cleaning up children...")
            cleanup_processes(child_pids=child_pids)
            print("Watchdog exit.")
            return

        time.sleep(1)


def worker(
    file_queue: Queue,
    gpu_task_queue: Queue,
    result_queue: Queue,
    save_dir: str,
    input_dir: str | None = None,
    tasks_per_worker: int = 5,
):
    logger = _setup_logger(name="F")
    tokenizer = AmtTokenizer()

    def process_file_wrapper():
        while True:
            try:
                file_to_process = file_queue.get(timeout=15)
            except Empty as e:
                if file_queue.empty():
                    logger.info("File queue empty")
                    break
                else:
                    # I'm pretty sure empty is thrown due to timeout too
                    logger.info("Processes timed out waiting for file queue")
                    continue

            process_file(
                file_path=file_to_process["path"],
                file_queue=file_queue,
                gpu_task_queue=gpu_task_queue,
                result_queue=result_queue,
                tokenizer=tokenizer,
                save_dir=save_dir,
                input_dir=input_dir,
                logger=logger,
                segments=file_to_process.get("segments", None),
            )

            if file_queue.empty():
                return

    try:
        with ThreadPoolExecutor(max_workers=tasks_per_worker) as executor:
            futures = [
                executor.submit(process_file_wrapper)
                for _ in range(tasks_per_worker)
            ]
            concurrent.futures.wait(futures)
    except Exception as e:
        logger.error(f"File worker failed with exception: {e}")
    finally:
        logger.info("File worker terminated")


def batch_transcribe(
    files_to_process: List[dict],
    model: AmtEncoderDecoder,
    save_dir: str,
    batch_size: int = 8,
    input_dir: str | None = None,
    gpu_ids: int | None = None,
    num_workers: int | None = None,
    quantize: bool = False,
    compile_mode: str | bool = False,
):
    assert os.name == "posix", "UNIX/LINUX is the only supported OS"
    assert compile_mode in {
        "reduce-overhead",
        "max-autotune",
        False,
    }, "Invalid value for compile_mode"

    torch.multiprocessing.set_start_method("spawn")
    num_gpus = len(gpu_ids) if gpu_ids is not None else 1
    logger = _setup_logger()

    if os.path.isfile("transcribe.log"):
        os.remove("transcribe.log")

    if quantize is True:
        logger.info("Quantizing decoder weights to int8")
        model.decoder = quantize_int8(model.decoder)

    file_queue = Queue()
    sorted(
        files_to_process, key=lambda x: os.path.getsize(x["path"]), reverse=True
    )
    for file_to_process in files_to_process:
        if "segments" in file_to_process:
            # Process files with segments
            unsaved_segments = []
            for idx, segment in enumerate(file_to_process["segments"]):
                segment_save_path = get_save_path(
                    file_to_process["path"],
                    input_dir,
                    save_dir,
                    idx_str=f"_{idx}",
                )
                if not os.path.isfile(segment_save_path):
                    unsaved_segments.append((idx, segment))

            if unsaved_segments:
                file_to_process["segments"] = unsaved_segments
                file_queue.put(file_to_process)
        else:
            # Process files without segments (whole file)
            if not os.path.isfile(
                get_save_path(file_to_process["path"], input_dir, save_dir)
            ):
                file_queue.put(file_to_process)

    logger.info(
        f"Files to process: {file_queue.qsize()}/{len(files_to_process)}"
    )

    if file_queue.qsize() == 0:
        logger.info("No files to process")
        return

    if num_workers is None:
        num_workers = min(
            min(batch_size, multiprocessing.cpu_count() - num_gpus),
            file_queue.qsize(),
        )
    num_processes_per_worker = min(
        5 * (batch_size // num_workers), file_queue.qsize() // num_workers
    )

    mp_manager = Manager()
    gpu_waiting_dict = mp_manager.dict()
    gpu_waiting_dict_lock = mp_manager.Lock()
    gpu_batch_queue = Queue()
    gpu_task_queue = Queue()
    result_queue = Queue()

    child_pids = []
    logger.info(
        f"Creating {num_workers} file worker(s) with {num_processes_per_worker} sub-processes"
    )
    worker_processes = [
        multiprocessing.Process(
            target=worker,
            args=(
                file_queue,
                gpu_task_queue,
                result_queue,
                save_dir,
                input_dir,
                num_processes_per_worker,
            ),
        )
        for _ in range(num_workers)
    ]

    for p in worker_processes:
        p.start()
        child_pids.append(p.pid)

    gpu_batch_manager_process = multiprocessing.Process(
        target=gpu_batch_manager,
        args=(
            gpu_task_queue,
            gpu_batch_queue,
            gpu_waiting_dict,
            gpu_waiting_dict_lock,
            batch_size,
        ),
    )
    gpu_batch_manager_process.start()
    child_pids.append(gpu_batch_manager_process.pid)

    start_time = time.time()
    if num_gpus > 1:
        gpu_manager_processes = [
            torch_multiprocessing.Process(
                target=gpu_manager,
                args=(
                    gpu_batch_queue,
                    gpu_waiting_dict,
                    gpu_waiting_dict_lock,
                    result_queue,
                    model,
                    batch_size,
                    compile_mode,
                    gpu_id,
                ),
            )
            for gpu_id in range(len(gpu_ids))
        ]
        for p in gpu_manager_processes:
            p.start()
            child_pids.append(p.pid)

        watchdog_process = multiprocessing.Process(
            target=watchdog,
            args=(
                [
                    os.getpid(),
                    gpu_batch_manager_process.pid,
                ]
                + [p.pid for p in gpu_manager_processes],
                child_pids,
            ),
        )
        watchdog_process.start()
    else:
        _gpu_manager_process = torch_multiprocessing.Process(
            target=gpu_manager,
            args=(
                gpu_batch_queue,
                gpu_waiting_dict,
                gpu_waiting_dict_lock,
                result_queue,
                model,
                batch_size,
                compile_mode,
            ),
        )
        _gpu_manager_process.start()
        child_pids.append(_gpu_manager_process.pid)
        gpu_manager_processes = [_gpu_manager_process]

        watchdog_process = multiprocessing.Process(
            target=watchdog,
            args=(
                [
                    os.getpid(),
                    gpu_batch_manager_process.pid,
                    _gpu_manager_process.pid,
                ],
                child_pids,
            ),
        )
        watchdog_process.start()

    try:
        for p in worker_processes:
            p.join()

        if gpu_manager_processes is not None:
            for p in gpu_manager_processes:
                p.terminate()
                p.join()

    except KeyboardInterrupt:
        logger.info("Main process received KeyboardInterrupt")
        logger.info("Cleaning up child processes")
        cleanup_processes(child_pids=child_pids)
        logger.info("Complete")
    finally:
        watchdog_process.terminate()
        watchdog_process.join()
        gpu_batch_manager_process.terminate()
        gpu_batch_manager_process.join()
        file_queue.close()
        file_queue.join_thread()
        gpu_task_queue.close()
        gpu_task_queue.join_thread()
        gpu_batch_queue.close()
        gpu_batch_queue.join_thread()
        result_queue.close()
        result_queue.join_thread()

    time_taken_s = int(time.time() - start_time)
    logger.info(
        f"Took {int(time_taken_s // 60)}m {time_taken_s % 60}s to transcribe files"
    )


def quantize_int8(model: torch.nn.Module):
    from amt.inference.quantize import WeightOnlyInt8QuantHandler

    quantizer = WeightOnlyInt8QuantHandler(model)
    int8_state_dict = quantizer.create_quantized_state_dict()
    _model = quantizer.convert_for_runtime()
    _model.load_state_dict(int8_state_dict)

    return _model
