import os
import sys
import signal
import time
import random
import logging
import traceback
import threading
import torch
import torch.multiprocessing as multiprocessing
import torch._dynamo.config
import torch._inductor.config

from torch.multiprocessing import Queue
from tqdm import tqdm
from functools import wraps
from torch.cuda import is_bf16_supported

from amt.inference.model import AmtEncoderDecoder
from amt.tokenizer import AmtTokenizer
from amt.audio import AudioTransform
from amt.data import get_wav_mid_segments

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True

MAX_SEQ_LEN = 4096
MAX_BLOCK_LEN = 4096
LEN_MS = 30000
STRIDE_FACTOR = 3
CHUNK_LEN_MS = LEN_MS // STRIDE_FACTOR


def _setup_logger(name: str | None = None):
    logger_name = f"[{name}] " if name else ""
    logger = logging.getLogger(__name__)
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # Adjust the formatter to include the name before the PID if provided
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
    mask_c = col_indices <= tok_ids_expanded + 3
    mask_d = col_indices >= tok_ids_expanded - 3
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
            with torch.autocast("cuda", dtype=torch.float32):
                return func(*args, **kwargs)

    return wrapper


@torch.no_grad()
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


@optional_bf16_autocast
@torch.no_grad()
def process_segments(
    tasks: list,
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

    # for idx in (
    #     pbar := tqdm(
    #         range(min_prefix_len, MAX_BLOCK_LEN - 1),
    #         total=MAX_BLOCK_LEN - (min_prefix_len + 1),
    #         leave=False,
    #     )
    # ):
    for idx in range(min_prefix_len, MAX_BLOCK_LEN - 1):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            if idx == min_prefix_len:
                logits, next_tok_ids = decode_token(
                    model,
                    x=seq[:, :idx],
                    xa=audio_features,
                    x_input_pos=torch.arange(0, idx, device=seq.device),
                    xa_input_pos=torch.arange(
                        0, audio_features.shape[1], device=seq.device
                    ),
                )
            else:
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
        logger.warning("Context length overflow when transcribing segment")

    results = [
        tokenizer.decode(seq[_idx, : eos_idxs[_idx] + 1])
        for _idx in range(seq.shape[0])
    ]

    return results


def gpu_manager(
    gpu_batch_queue: Queue,
    result_queue: Queue,
    model: AmtEncoderDecoder,
    batch_size: int,
    compile: bool = False,
    gpu_id: int | None = None,
):
    logger = _setup_logger(name="GPU")
    logger.info("Started GPU manager")

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model.decoder.setup_cache(batch_size=batch_size, max_seq_len=MAX_BLOCK_LEN)
    model.cuda()
    model.eval()
    if compile is True:
        global decode_token, recalculate_tok_ids
        if batch_size == 1:
            recalculate_tok_ids = torch.compile(
                recalculate_tok_ids, mode="max-autotune-no-cudagraphs"
            )
        decode_token = torch.compile(
            decode_token,
            mode="reduce-overhead",
            # mode="max-autotune",
            fullgraph=True,
        )

    audio_transform = AudioTransform().cuda()
    tokenizer = AmtTokenizer(return_tensors=True)

    try:
        while True:
            try:
                batch = gpu_batch_queue.get(timeout=30)
            except Exception as e:
                logger.info(f"GPU timed out waiting for batch")
                break
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
                            result_queue.put({"result": result, "pid": pid})

    except Exception as e:
        logger.error(f"GPU manager failed with exception: {e}")
    finally:
        logger.info(f"GPU manager terminated")


def _find_min_diff_batch(tasks: list, batch_size: int):
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


def gpu_batch_manager(
    gpu_task_queue: Queue,
    gpu_batch_queue: Queue,
    batch_size: int,
):
    logger = _setup_logger(name="B")
    logger.info("Started batch manager")
    try:
        tasks = []
        while True:
            try:
                task, pid = gpu_task_queue.get(timeout=0.2)
            except Exception as e:
                pass
            else:
                tasks.append((task, pid))
                continue

            # No tasks in queue -> check gpu batch queue
            if gpu_batch_queue.empty() is False:
                continue
            elif len(tasks) == 0:
                continue

            # Get new batch and add to batch queue
            if len(tasks) < batch_size:
                logger.warning(
                    f"Not enough tasks ({len(tasks)}) - padding batch"
                )
            while len(tasks) < batch_size:
                _pad_task, _pid = tasks[0]
                tasks.append((_pad_task, -1))

            assert len(tasks) >= batch_size, "batch error"
            new_batch_idxs = _find_min_diff_batch(
                tasks,
                batch_size=batch_size,
            )
            gpu_batch_queue.put([tasks[_idx] for _idx in new_batch_idxs])
            tasks = [
                task
                for _idx, task in enumerate(tasks)
                if _idx not in new_batch_idxs
            ]
    except Exception as e:
        logger.error(f"GPU batch manager failed with exception: {e}")
    finally:
        logger.info(f"GPU batch manager terminated")


def _shift_onset(seq: list, shift_ms: int):
    res = []
    for tok in seq:
        if type(tok) is tuple and tok[0] == "onset":
            res.append(("onset", tok[1] + shift_ms))
        else:
            res.append(tok)

    return res


def _truncate_seq(
    seq: list,
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
            res = tokenizer._tokenize_midi_dict(_mid_dict, start_ms, end_ms - 1)

        if res[-1] == tokenizer.eos_tok:
            res.pop()
        return res


def transcribe_file(
    file_path,
    gpu_task_queue: Queue,
    result_queue: Queue,
    pid: int,
    tokenizer: AmtTokenizer = AmtTokenizer(),
):
    logger = logging.getLogger(__name__)

    logger.info(f"Getting wav segments: {file_path}")
    audio_segments = [
        f
        for f, _ in get_wav_mid_segments(
            audio_path=file_path,
            stride_factor=STRIDE_FACTOR,
            pad_last=True,
        )
    ]

    res = []
    seq = [tokenizer.bos_tok]
    concat_seq = [tokenizer.bos_tok]
    idx = 0
    while audio_segments:
        init_idx = len(seq)

        # Add to gpu queue and wait for results
        gpu_task_queue.put(((audio_segments.pop(0), seq), pid))
        while True:
            try:
                gpu_result = result_queue.get(timeout=0.1)
            except Exception as e:
                pass
            else:
                if gpu_result["pid"] == pid:
                    seq = gpu_result["result"]
                    break
                else:
                    result_queue.put(gpu_result)

        try:
            next_seq = _truncate_seq(
                seq,
                CHUNK_LEN_MS,
                LEN_MS - CHUNK_LEN_MS,
            )
        except Exception as e:
            logger.info(
                f"Skipping segment {idx} (failed to transcribe): {file_path}"
            )
            logger.debug(traceback.format_exc())
            seq = [tokenizer.bos_tok]
        else:
            if seq[-1] == tokenizer.eos_tok:
                logger.info(f"Seen eos_tok at segment {idx}: {file_path}")
                seq = seq[:-1]

            if len(next_seq) == 1:
                logger.info(f"Skipping segment {idx} (silence): {file_path}")
                seq = [tokenizer.bos_tok]
            else:
                concat_seq += _shift_onset(
                    seq[init_idx:],
                    idx * CHUNK_LEN_MS,
                )
                seq = next_seq

        idx += 1

    res.append(concat_seq)

    return res


def get_save_path(
    file_path: str,
    input_dir: str,
    save_dir: str,
    idx: int | str = "",
):
    if input_dir is None:
        save_path = os.path.join(
            save_dir,
            os.path.splitext(os.path.basename(file_path))[0] + f"{idx}.mid",
        )
    else:
        input_rel_path = os.path.relpath(file_path, input_dir)
        save_path = os.path.join(
            save_dir, os.path.splitext(input_rel_path)[0] + f"{idx}.mid"
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
):
    def _save_seq(_seq: list, _save_path: str):
        if os.path.exists(_save_path):
            logger.info(f"Already exists {_save_path} - overwriting")

        for tok in _seq[::-1]:
            if type(tok) is tuple and tok[0] == "onset":
                last_onset = tok[1]
                break

        try:
            mid_dict = tokenizer._detokenize_midi_dict(
                tokenized_seq=_seq, len_ms=last_onset
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
    try:
        seqs = transcribe_file(file_path, gpu_task_queue, result_queue, pid=pid)
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {traceback.format_exc()}")
        task_rmv_cnt = remove_failures_from_queue_(gpu_task_queue, pid)
        res_rmv_cnt = remove_failures_from_queue_(result_queue, pid)
        logger.info(f"Removed {task_rmv_cnt} from task queue")
        logger.info(f"Removed {res_rmv_cnt} from result queue")
        return

    logger.info(f"Finished file: {file_path}")
    for seq in seqs:
        if len(seq) < 500:
            logger.info("Skipping seq - too short")
        else:
            logger.debug(
                f"Saving seq of length {len(seq)} from file: {file_path}"
            )

            _save_seq(seq, get_save_path(file_path, input_dir, save_dir))

    logger.info(f"{file_queue.qsize()} file(s) remaining in queue")


def watchdog(main_gpu_pid: int, child_pids: list):
    while True:
        if not os.path.exists(f"/proc/{main_gpu_pid}"):
            print("Cleaning up children...")
            for pid in child_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        else:
            print(f"{main_gpu_pid} still alive")
        time.sleep(1)


def worker(
    file_queue: Queue,
    gpu_task_queue: Queue,
    result_queue: Queue,
    save_dir: str,
    input_dir: str | None = None,
    tasks_per_worker: int = 1,
):
    logger = _setup_logger(name="F")
    tokenizer = AmtTokenizer()
    threads = []
    try:
        while not file_queue.empty() or any(t.is_alive() for t in threads):
            while len(threads) < tasks_per_worker and not file_queue.empty():
                logging.info("Starting worker")
                file_path = file_queue.get()
                t = threading.Thread(
                    target=process_file,
                    args=(
                        file_path,
                        file_queue,
                        gpu_task_queue,
                        result_queue,
                        tokenizer,
                        save_dir,
                        input_dir,
                        logger,
                    ),
                )
                t.start()
                threads.append(t)

            threads = [t for t in threads if t.is_alive()]

            time.sleep(0.1)

        for t in threads:
            t.join()

    except Exception as e:
        logger.error(f"File worker failed with exception: {e}")
    finally:
        logger.info(f"File worker terminated")


# Needs to test this for multi-gpu
def batch_transcribe(
    file_paths: list,
    model: AmtEncoderDecoder,
    save_dir: str,
    batch_size: int = 16,
    input_dir: str | None = None,
    gpu_ids: int | None = None,
    quantize: bool = False,
    compile: bool = False,
):
    assert os.name == "posix", "UNIX/LINUX is the only supported OS"
    torch.multiprocessing.set_start_method("spawn")
    num_gpus = len(gpu_ids) if gpu_ids is not None else 1
    logger = _setup_logger()

    if os.path.isfile("transcribe.log"):
        os.remove("transcribe.log")

    if quantize is True:
        logger.info("Quantising decoder weights to int8")
        model.decoder = quantize_int8(model.decoder)

    file_queue = Queue()
    for file_path in file_paths:
        if (
            os.path.isfile(get_save_path(file_path, input_dir, save_dir))
            is False
        ):
            file_queue.put(file_path)

    logger.info(f"Files to process: {file_queue.qsize()}/{len(file_paths)}")

    num_workers = min(
        min(batch_size * num_gpus, multiprocessing.cpu_count() - num_gpus),
        file_queue.qsize(),
    )

    gpu_task_queue = Queue()
    gpu_batch_queue = Queue()
    result_queue = Queue()

    child_pids = []
    logger.info(f"Creating {num_workers} file worker(s)")
    worker_processes = [
        multiprocessing.Process(
            target=worker,
            args=(
                file_queue,
                gpu_task_queue,
                result_queue,
                save_dir,
                input_dir,
                3,
            ),
        )
        for _ in range(num_workers)
    ]

    for p in worker_processes:
        p.start()
        child_pids.append(p.pid)

    gpu_batch_manager_process = multiprocessing.Process(
        target=gpu_batch_manager,
        args=(gpu_task_queue, gpu_batch_queue, batch_size),
    )
    gpu_batch_manager_process.start()
    child_pids.append(gpu_batch_manager_process.pid)

    time.sleep(5)
    start_time = time.time()

    if num_gpus > 1:
        gpu_manager_processes = [
            multiprocessing.Process(
                target=gpu_manager,
                args=(
                    gpu_batch_queue,
                    result_queue,
                    model,
                    batch_size,
                    compile,
                    gpu_id,
                ),
            )
            for gpu_id in gpu_ids
        ]
        for p in gpu_manager_processes:
            p.start()
        watchdog_process = multiprocessing.Process(
            target=watchdog, args=(gpu_batch_manager_process[0].pid, child_pids)
        )
        watchdog_process.start()
    else:
        gpu_manager_processes = None
        watchdog_process = multiprocessing.Process(
            target=watchdog, args=(os.getpid(), child_pids)
        )
        watchdog_process.start()
        gpu_manager(
            gpu_batch_queue,
            result_queue,
            model,
            batch_size,
            compile,
        )

    if gpu_manager_processes is not None:
        for p in gpu_manager_processes:
            p.join()

    for p in worker_processes:
        p.terminate()
        p.join()

    gpu_batch_manager_process.terminate()
    gpu_batch_manager_process.join()
    watchdog_process.terminate()
    watchdog_process.join()

    print("Took", (time.time() - start_time) / 60, "mins to transcribe files")


def quantize_int8(model: torch.nn.Module):
    from amt.inference.quantize import WeightOnlyInt8QuantHandler

    quantizer = WeightOnlyInt8QuantHandler(model)
    int8_state_dict = quantizer.create_quantized_state_dict()
    _model = quantizer.convert_for_runtime()
    _model.load_state_dict(int8_state_dict)

    return _model
