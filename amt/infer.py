import os
import time
import random
import torch
import torch.multiprocessing as multiprocessing

from torch.multiprocessing import Queue
from tqdm import tqdm

from amt.model import AmtEncoderDecoder
from amt.tokenizer import AmtTokenizer
from amt.audio import AudioTransform
from amt.data import get_wav_mid_segments

MAX_SEQ_LEN = 4096
LEN_MS = 30000
STRIDE_FACTOR = 3
CHUNK_LEN_MS = LEN_MS // STRIDE_FACTOR
BEAM = 3
ONSET_TOLERANCE = 50
VEL_TOLERANCE = 50


# TODO: Profile and fix gpu util


def calculate_vel(
    logits: torch.Tensor,
    init_vel: int,
    tokenizer: AmtTokenizer = AmtTokenizer(),
):
    probs, idxs = torch.topk(torch.softmax(logits, dim=-1), BEAM)
    vels = [tokenizer.id_to_tok[idx.item()] for idx in idxs]

    # Get rid of outliers
    for idx in range(BEAM):
        vel = vels[idx]
        if type(vel) is not tuple:
            vels[idx] = 0
            probs[idx] = 0.0
        elif vel[0] != "vel":
            vels[idx] = 0
            probs[idx] = 0.0
        elif (vel[1] < init_vel - VEL_TOLERANCE / 2) or (
            vel[1] > init_vel + VEL_TOLERANCE / 2
        ):
            vels[idx] = vels[idx][1]
            probs[idx] = 0.0
        else:
            vels[idx] = vels[idx][1]

    vels = torch.tensor(vels).to(probs.device)
    new_vel = torch.sum(vels * probs) / torch.sum(probs)
    new_vel = round(new_vel.item() / 5) * 5

    return tokenizer.tok_to_id[("vel", new_vel)]


def calculate_onset(
    logits: torch.Tensor,
    init_onset: int,
    tokenizer: AmtTokenizer = AmtTokenizer(),
):
    probs, idxs = torch.topk(torch.softmax(logits, dim=-1), BEAM)
    onsets = [tokenizer.id_to_tok[idx.item()] for idx in idxs]

    # Get rid of outliers
    for idx in range(BEAM):
        onset = onsets[idx]
        if type(onset) is not tuple:
            onsets[idx] = 0
            probs[idx] = 0.0
        elif onset[0] != "onset":
            onsets[idx] = 0
            probs[idx] = 0.0
        elif (onset[1] < init_onset - ONSET_TOLERANCE / 2) or (
            onset[1] > init_onset + ONSET_TOLERANCE / 2
        ):
            onsets[idx] = onsets[idx][1]
            probs[idx] = 0.0
        else:
            onsets[idx] = onsets[idx][1]

    onsets = torch.tensor(onsets).to(probs.device)
    new_onset = torch.sum(onsets * probs) / torch.sum(probs)
    new_onset = round(new_onset.item() / 10) * 10

    return tokenizer.tok_to_id[("onset", new_onset)]


@torch.autocast("cuda", dtype=torch.bfloat16)
def process_segments(
    tasks: list,
    model: AmtEncoderDecoder,
    audio_transform: AudioTransform,
    tokenizer: AmtTokenizer,
):
    audio_segs = torch.stack(
        [audio_seg for (audio_seg, prefix), _ in tasks]
    ).cuda()
    log_mels = audio_transform.log_mel(audio_segs)
    audio_features = model.embed_audio(mel=log_mels)

    raw_prefixes = [prefix for (audio_seg, prefix), _ in tasks]
    prefix_lens = [len(prefix) for prefix in raw_prefixes]
    min_prefix_len = min(prefix_lens)
    prefixes = [
        tokenizer.trunc_seq(prefix, MAX_SEQ_LEN) for prefix in raw_prefixes
    ]
    seq = torch.stack([tokenizer.encode(prefix) for prefix in prefixes]).cuda()
    eos_seen = [False for _ in prefixes]

    kv_cache = model.get_empty_cache()

    for idx in (
        pbar := tqdm(
            range(min_prefix_len, MAX_SEQ_LEN - 1),
            total=MAX_SEQ_LEN - (min_prefix_len + 1),
            leave=False,
        )
    ):
        # for idx in range(min_prefix_len, MAX_SEQ_LEN - 1):
        if idx == min_prefix_len:
            logits = model.decoder(
                xa=audio_features,
                x=seq[:, :idx],
                kv_cache=kv_cache,
            )
        else:
            logits = model.decoder(
                xa=audio_features,
                x=seq[:, idx - 1 : idx],
                kv_cache=kv_cache,
            )

        next_tok_ids = torch.argmax(logits[:, -1], dim=-1)

        for batch_idx in range(logits.shape[0]):
            if eos_seen[batch_idx] is not False:
                # End already seen, add pad token
                tok_id = tokenizer.pad_id
            elif idx >= prefix_lens[batch_idx]:
                # New token required, recalculated if needed
                tok_id = next_tok_ids[batch_idx].item()
                tok = tokenizer.id_to_tok[tok_id]
                if type(tok) is tuple and tok[0] == "onset":
                    # If onset token, recalculate
                    tok_id = calculate_onset(logits[batch_idx, -1], tok[1])
                elif type(tok) is tuple and tok[0] == "vel":
                    # If velocity token, recalculate
                    tok_id = calculate_vel(logits[batch_idx, -1], tok[1])

            else:
                # Still in prefix tokens, do nothing
                tok_id = tokenizer.tok_to_id[prefixes[batch_idx][idx]]

            seq[batch_idx, idx] = tok_id
            if tokenizer.id_to_tok[tok_id] == tokenizer.eos_tok:
                eos_seen[batch_idx] = idx

        if all(eos_seen):
            break

    if not all(eos_seen):
        print("WARNING: OVERFLOW")
        for _idx in range(seq.shape[0]):
            if eos_seen[_idx] == False:
                eos_seen[_idx] = MAX_SEQ_LEN

    results = [
        tokenizer.decode(seq[_idx, : eos_seen[_idx] + 1])
        for _idx in range(seq.shape[0])
    ]

    return results


def gpu_manager(
    gpu_task_queue: Queue,
    result_queue: Queue,
    model: AmtEncoderDecoder,
    batch_size: int,
):
    # model.compile()
    audio_transform = AudioTransform().cuda()
    tokenizer = AmtTokenizer(return_tensors=True)
    process_pid = multiprocessing.current_process().pid

    wait_for_batch = True
    batch = []
    while True:
        try:
            task, pid = gpu_task_queue.get(timeout=5)
        except:
            print(f"{process_pid}: GPU task timeout")
            if len(batch) == 0:
                print(f"{process_pid}: Finished GPU tasks")
                return
            else:
                wait_for_batch = False
        else:
            batch.append((task, pid))

        if len(batch) == batch_size or (
            len(batch) > 0 and wait_for_batch is False
        ):
            # Process batch on GPU
            results = process_segments(
                tasks=[task for task in batch],
                model=model,
                audio_transform=audio_transform,
                tokenizer=tokenizer,
            )
            for result, (_, pid) in zip(results, batch):
                result_queue.put({"result": result, "pid": pid})
            batch.clear()


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
    if start_ms == end_ms:
        _mid_dict, unclosed_notes = tokenizer._detokenize_midi_dict(
            seq, start_ms, return_unclosed_notes=True
        )
        random.shuffle(unclosed_notes)
        return [("prev", p) for p in unclosed_notes] + [tokenizer.bos_tok]
    else:
        _mid_dict = tokenizer._detokenize_midi_dict(seq, LEN_MS)
        try:
            res = tokenizer._tokenize_midi_dict(_mid_dict, start_ms, end_ms - 1)
        except:
            return ["<S>"]
        else:
            return res[: res.index(tokenizer.eos_tok)]


def process_file(
    file_path,
    gpu_task_queue: Queue,
    result_queue: Queue,
    tokenizer: AmtTokenizer = AmtTokenizer(),
):
    process_pid = multiprocessing.current_process().pid
    print(f"{process_pid}: Getting wav segments")
    audio_segments = [
        f
        for f, _ in get_wav_mid_segments(
            audio_path=file_path, stride_factor=STRIDE_FACTOR
        )
    ]
    seq = ["<S>"]
    res = ["<S>"]
    for idx, audio_seg in enumerate(audio_segments):
        init_idx = len(seq)

        # Add to gpu queue and wait for results
        gpu_task_queue.put(((audio_seg, seq), process_pid))
        while True:
            gpu_result = result_queue.get()
            if gpu_result["pid"] == process_pid:
                seq = gpu_result["result"]
                break
            else:
                result_queue.put(gpu_result)

        res += _shift_onset(
            seq[init_idx : seq.index(tokenizer.eos_tok)],
            idx * CHUNK_LEN_MS,
        )

        if idx == len(audio_segments) - 1:
            break
        else:
            seq = _truncate_seq(seq, CHUNK_LEN_MS, LEN_MS)
            if len(seq) == 1:
                print(f"{process_pid}: exiting early")
                return res

    return res


def worker(
    file_queue: Queue,
    gpu_task_queue: Queue,
    result_queue: Queue,
    save_dir: str,
    input_dir: str | None = None,
):
    def _get_save_path(_file_path: str):
        if input_dir is None:
            save_path = os.path.join(
                save_dir,
                os.path.splitext(os.path.basename(file_path))[0] + ".mid",
            )
        else:
            input_rel_path = os.path.relpath(_file_path, input_dir)
            save_path = os.path.join(
                save_dir, os.path.splitext(input_rel_path)[0] + ".mid"
            )
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

        return save_path

    pid = multiprocessing.current_process().pid
    tokenizer = AmtTokenizer()
    files_processed = 0
    while not file_queue.empty():
        file_path = file_queue.get()
        save_path = _get_save_path(file_path)
        if os.path.exists(save_path):
            print(f"{pid}: {save_path} already exists, overwriting")

        try:
            res = process_file(file_path, gpu_task_queue, result_queue)
        except Exception as e:
            print(f"{pid}: Failed to transcribe {file_path}")
            continue

        files_processed += 1

        for tok in res[::-1]:
            if type(tok) is tuple and tok[0] == "onset":
                last_onset = tok[1]
                break

        try:
            mid_dict = tokenizer._detokenize_midi_dict(
                tokenized_seq=res, len_ms=last_onset
            )
            mid = mid_dict.to_midi()
            mid.save(save_path)
        except Exception as e:
            print(f"{pid}: Failed to detokenize with error {e}")
        else:
            print(f"{pid}: Finished file {files_processed} - {file_path}")
            print(f"{pid}: {file_queue.qsize()} file(s) remaining in queue")


def batch_transcribe(
    file_paths: list,
    model: AmtEncoderDecoder,
    save_dir: str,
    batch_size: int = 16,
    gpu_id: int | None = None,
    input_dir: str | None = None,
):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model.cuda()
    model.eval()
    file_queue = Queue()
    for file_path in file_paths:
        file_queue.put(file_path)

    gpu_task_queue = Queue()
    result_queue = Queue()

    worker_processes = [
        multiprocessing.Process(
            target=worker,
            args=(
                file_queue,
                gpu_task_queue,
                result_queue,
                save_dir,
                input_dir,
            ),
        )
        for _ in range(batch_size + 1)
    ]
    for p in worker_processes:
        p.start()

    time.sleep(10)
    gpu_manager_process = multiprocessing.Process(
        target=gpu_manager,
        args=(gpu_task_queue, result_queue, model, batch_size),
    )
    gpu_manager_process.start()

    for p in worker_processes:
        p.join()

    gpu_manager_process.join()
