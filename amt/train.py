import os
import sys
import csv
import random
import traceback
import functools
import argparse
import logging
import torch
import torchaudio
import accelerate

from torch import nn as nn
from torch.utils.data import DataLoader

from accelerate.logging import get_logger
from safetensors.torch import load_file
from logging.handlers import RotatingFileHandler
from tqdm import tqdm

from amt.tokenizer import AmtTokenizer
from amt.model import AmtEncoderDecoder, ModelConfig
from amt.audio import AudioTransform
from amt.data import AmtDataset
from amt.config import load_model_config
from aria.utils import _load_weight

GRADIENT_ACC_STEPS = 2

# ----- USAGE -----
#
# This script is meant to be run using the huggingface accelerate cli, see:
#
# https://huggingface.co/docs/accelerate/basic_tutorials/launch
# https://huggingface.co/docs/accelerate/package_reference/cli
#
# For example usage you could run the pre-training script with:
#
# accelerate launch [arguments] amt/train.py pretrain \
#   small \
#   data/train.jsonl \
#   data/val.jsonl \
#   -epochs 10 \
#   -bs 4 \
#   -workers 8
#
# You could resume a run from an accelerate checkpoint with:
#
# accelerate launch [arguments] amt/train.py resume \
#   small \
#   pretrain \
#   data/train.jsonl \
#   data/val.jsonl \
#   -cdir models/epoch5_step0 \
#   -rstep 0 \
#   -repoch 5 \
#   -epochs 5 \
#   -bs 4 \
#   -workers 8


def setup_logger(project_dir: str):
    # Get logger and reset all handlers
    logger = logging.getLogger(__name__)
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s: [%(levelname)s] %(message)s",
    )

    fh = RotatingFileHandler(
        os.path.join(project_dir, "logs.txt"), backupCount=5, maxBytes=1024**3
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return get_logger(__name__)  # using accelerate.logging.get_logger()


def setup_project_dir(project_dir: str | None):
    if not project_dir:
        # Create project directory
        if not os.path.isdir("./experiments"):
            os.mkdir("./experiments")

        project_dirs = [
            _dir
            for _dir in os.listdir("./experiments")
            if os.path.isdir(os.path.join("experiments", _dir))
        ]

        ind = 0
        while True:
            if str(ind) not in project_dirs:
                break
            else:
                ind += 1

        project_dir_abs = os.path.abspath(os.path.join("experiments", str(ind)))
        assert not os.path.isdir(project_dir_abs)
        os.mkdir(project_dir_abs)

    elif project_dir:
        # Run checks on project directory
        if os.path.isdir(project_dir):
            assert (
                len(os.listdir(project_dir)) == 0
            ), "Provided project directory is not empty"
            project_dir_abs = os.path.abspath(project_dir)
        elif os.path.isfile(project_dir):
            raise FileExistsError(
                "The provided path points toward an existing file"
            )
        else:
            try:
                os.mkdir(project_dir)
            except Exception as e:
                raise e(f"Failed to create project directory at {project_dir}")
        project_dir_abs = os.path.abspath(project_dir)

    os.mkdir(os.path.join(project_dir_abs, "checkpoints"))

    return project_dir_abs


def _get_optim(
    lr: float,
    model: nn.Module,
    num_epochs: int,
    steps_per_epoch: int,
    warmup: int = 100,
    end_ratio: int = 0.1,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    warmup_lrs = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1,
        total_iters=warmup,
    )
    linear_decay_lrs = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=end_ratio,
        total_iters=(num_epochs * steps_per_epoch) - warmup,
    )

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_lrs, linear_decay_lrs],
        milestones=[warmup],
    )

    return optimizer, lr_scheduler


def get_pretrain_optim(
    model: nn.Module,
    num_epochs: int,
    steps_per_epoch: int,
):
    LR = 5e-4
    END_RATIO = 0.1
    WARMUP_STEPS = 1000

    return _get_optim(
        lr=LR,
        model=model,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        warmup=WARMUP_STEPS,
        end_ratio=END_RATIO,
    )


def get_finetune_optim(
    model: nn.Module,
    num_epochs: int,
    steps_per_epoch: int,
):
    LR = 1e-4
    END_RATIO = 0.1
    WARMUP_STEPS = 1000

    return _get_optim(
        lr=LR,
        model=model,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        warmup=WARMUP_STEPS,
        end_ratio=END_RATIO,
    )


def get_dataloaders(
    train_data_paths: str,
    val_data_path: str,
    batch_size: int,
    num_workers: int,
):
    logger = get_logger(__name__)
    logger.info("Indexing datasets...")
    train_dataset = AmtDataset(load_paths=train_data_paths)
    val_dataset = AmtDataset(load_paths=val_data_path)
    logger.info(
        f"Loaded datasets with length: train={len(train_dataset)}; val={len(val_dataset)}"
    )

    # Pitch aug (to the sequence tensors) must be applied in the train
    # dataloader as it needs to be done to every element in the batch equally.
    # Having this code running on the main process was causing a bottleneck.
    # Furthermore, distortion runs very slowly on the gpu, so we do it in
    # the dataloader instead.
    tensor_pitch_aug = AmtTokenizer().export_tensor_pitch_aug()
    audio_transform = AudioTransform()

    def _collate_fn(seqs, max_pitch_shift: int):
        wav, src, tgt, idxs = torch.utils.data.default_collate(seqs)

        # Pitch aug
        pitch_shift = random.randint(-max_pitch_shift, max_pitch_shift)
        src = tensor_pitch_aug(seq=src, shift=pitch_shift)
        tgt = tensor_pitch_aug(seq=tgt, shift=pitch_shift)

        # Distortion
        wav = audio_transform.distortion_aug_cpu(wav)

        return wav, src, tgt, pitch_shift, idxs

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=functools.partial(_collate_fn, max_pitch_shift=5),
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return train_dataloader, val_dataloader


def plot_spec(mel: torch.Tensor, name: str | int):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title("(mel)-Spectrogram")
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def _debug(wav, mel, src, tgt, idx):
    print("Running debug", idx)
    for _idx in range(wav.shape[0]):
        if os.path.isdir(f"debug/{idx}") is False:
            os.makedirs(f"debug/{idx}")
        torchaudio.save(
            f"debug/{idx}/wav_{_idx}.wav", wav[_idx].unsqueeze(0).cpu(), 16000
        )
        plot_spec(mel[_idx].cpu(), f"debug/{idx}/mel_{_idx}.png")
        tokenizer = AmtTokenizer()
        src_dec = tokenizer.decode(src[_idx])
        mid_dict = tokenizer._detokenize_midi_dict(src_dec, 30000)
        mid = mid_dict.to_midi()
        mid.save(f"debug/{idx}/mid_{_idx}.mid")


def _train(
    epochs: int,
    accelerator: accelerate.Accelerator,
    model: AmtEncoderDecoder,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    audio_transform: AudioTransform,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    steps_per_checkpoint: int | None = None,
    resume_step: int | None = None,
    resume_epoch: int | None = None,
    project_dir: str | None = None,
):
    def make_checkpoint(_accelerator, _epoch: int, _step: int):
        checkpoint_dir = os.path.join(
            project_dir,
            "checkpoints",
            f"epoch{_epoch}_step{_step}",
        )

        logger.info(
            f"EPOCH {_epoch}/{epochs + start_epoch}: Saving checkpoint - {checkpoint_dir}"
        )
        _accelerator.save_state(checkpoint_dir)

    def get_max_norm(named_parameters):
        max_grad_norm = {"val": 0.0}
        for name, parameter in named_parameters:
            if parameter.grad is not None and parameter.requires_grad is True:
                grad_norm = parameter.grad.data.norm(2).item()
                # logger.debug(f"{name}: {grad_norm}")
                if grad_norm >= max_grad_norm["val"]:
                    max_grad_norm["name"] = name
                    max_grad_norm["val"] = grad_norm

        return max_grad_norm

    # This is all slightly messy as train_loop and val_loop make use of the
    # variables in the wider scope. Perhaps refactor this at some point.
    def train_loop(
        dataloader: DataLoader,
        _epoch: int,
        _resume_step: int = 0,
    ):
        avg_train_loss = 0
        trailing_loss = 0
        loss_buffer = []

        try:
            lr_for_print = "{:.2e}".format(scheduler.get_last_lr()[0])
        except Exception:
            pass
        else:
            lr_for_print = "{:.2e}".format(optimizer.param_groups[-1]["lr"])

        model.train()
        grad_norm = 0.0
        for __step, batch in (
            pbar := tqdm(
                enumerate(dataloader),
                total=len(dataloader) + _resume_step,
                initial=_resume_step,
                leave=False,
            )
        ):
            with accelerator.accumulate(model):
                step = __step + _resume_step + 1
                wav, src, tgt, pitch_shift, idxs = batch

                with torch.no_grad():
                    mel = audio_transform.forward(wav, shift=pitch_shift)

                logits = model(mel, src)  # (b_sz, s_len, v_sz)
                logits = logits.transpose(
                    1, 2
                )  # Transpose for CrossEntropyLoss
                loss = loss_fn(logits, tgt)

                # Calculate statistics
                loss_buffer.append(accelerator.gather(loss).mean(dim=0).item())
                trailing_loss = sum(loss_buffer[-TRAILING_LOSS_STEPS:]) / len(
                    loss_buffer[-TRAILING_LOSS_STEPS:]
                )
                avg_train_loss = sum(loss_buffer) / len(loss_buffer)

                # Backwards step
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), 1.0
                    ).item()
                optimizer.step()
                optimizer.zero_grad()

                # Logging
                logger.debug(
                    f"EPOCH {_epoch} STEP {step}: "
                    f"lr={lr_for_print}, "
                    f"loss={round(loss_buffer[-1], 4)}, "
                    f"trailing_loss={round(trailing_loss, 4)}, "
                    f"average_loss={round(avg_train_loss, 4)}, "
                    f"grad_norm={round(grad_norm, 4)}"
                )
                if accelerator.is_main_process:
                    loss_writer.writerow([_epoch, step, loss_buffer[-1]])

                pbar.set_postfix_str(
                    f"lr={lr_for_print}, "
                    f"loss={round(loss_buffer[-1], 4)}, "
                    f"trailing={round(trailing_loss, 4)}, "
                    f"grad_norm={round(grad_norm, 4)}"
                )

                if scheduler:
                    scheduler.step()
                    lr_for_print = "{:.2e}".format(scheduler.get_last_lr()[0])

                if steps_per_checkpoint:
                    if step % steps_per_checkpoint == 0:
                        make_checkpoint(
                            _accelerator=accelerator,
                            _epoch=_epoch,
                            _step=step,
                        )

        logger.info(
            f"EPOCH {_epoch}/{epochs + start_epoch}: Finished training - "
            f"average_loss={round(avg_train_loss, 4)}"
        )

        return avg_train_loss

    @torch.no_grad()
    def val_loop(dataloader, _epoch: int, aug: bool):
        loss_buffer = []
        model.eval()
        for step, batch in (
            pbar := tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                leave=False,
            )
        ):
            wav, src, tgt, idxs = batch

            if aug == False:
                mel = audio_transform.log_mel(wav)
            elif aug == True:
                # Apply aug without distortion or spec-augment
                mel = audio_transform.log_mel(
                    audio_transform.aug_wav(wav), detune=True
                )
            else:
                raise TypeError

            logits = model(mel, src)
            logits = logits.transpose(1, 2)  # Transpose for CrossEntropyLoss
            loss = loss_fn(logits, tgt)

            # Logging
            loss_buffer.append(accelerator.gather(loss).mean(dim=0).item())
            avg_val_loss = sum(loss_buffer) / len(loss_buffer)
            pbar.set_postfix_str(f"average_loss={round(avg_val_loss, 4)}")

        # EPOCH
        logger.info(
            f"EPOCH {_epoch}/{epochs + start_epoch}: Finished evaluation "
            f"{'(aug)' if aug is True else ''} - "
            f"average_loss={round(avg_val_loss, 4)}"
        )

        return avg_val_loss

    if steps_per_checkpoint:
        assert (
            steps_per_checkpoint > 1
        ), "Invalid checkpoint mode value (too small)"

    TRAILING_LOSS_STEPS = 100
    PAD_ID = train_dataloader.dataset.tokenizer.pad_id
    logger = get_logger(__name__)  # Accelerate logger
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    logger.info(
        f"Model has "
        f"{'{:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad))} "
        "parameters"
    )

    if accelerator.is_main_process:
        loss_csv = open(os.path.join(project_dir, "loss.csv"), "w")
        loss_writer = csv.writer(loss_csv)
        loss_writer.writerow(["epoch", "step", "loss"])
        epoch_csv = open(os.path.join(project_dir, "epoch.csv"), "w")
        epoch_writer = csv.writer(epoch_csv)
        epoch_writer.writerow(
            ["epoch", "avg_train_loss", "avg_val_loss", "avg_val_loss_aug"]
        )

    if resume_epoch is not None:
        start_epoch = resume_epoch + 1
    else:
        start_epoch = 0

    if resume_step is not None:
        assert resume_epoch is not None, "Must provide resume epoch"
        logger.info(
            f"Resuming training from step {resume_step} - logging as EPOCH {resume_epoch}"
        )
        skipped_dataloader = accelerator.skip_first_batches(
            dataloader=train_dataloader,
            num_batches=resume_step,
        )

        avg_train_loss = train_loop(
            dataloader=skipped_dataloader,
            _epoch=resume_epoch,
            _resume_step=resume_step,
        )
        avg_val_loss = val_loop(
            dataloader=val_dataloader, _epoch=resume_epoch, aug=False
        )
        avg_val_loss_aug = val_loop(
            dataloader=val_dataloader, _epoch=resume_epoch, aug=True
        )
        if accelerator.is_main_process:
            epoch_writer.writerow(
                [resume_epoch, avg_train_loss, avg_val_loss, avg_val_loss_aug]
            )
            epoch_csv.flush()
            make_checkpoint(
                _accelerator=accelerator, _epoch=start_epoch, _step=0
            )

    for epoch in range(start_epoch, epochs + start_epoch):
        try:
            avg_train_loss = train_loop(
                dataloader=train_dataloader, _epoch=epoch
            )
            avg_val_loss = val_loop(
                dataloader=val_dataloader, _epoch=epoch, aug=False
            )
            avg_val_loss_aug = val_loop(
                dataloader=val_dataloader, _epoch=epoch, aug=True
            )
            if accelerator.is_main_process:
                epoch_writer.writerow(
                    [epoch, avg_train_loss, avg_val_loss, avg_val_loss_aug]
                )
                epoch_csv.flush()
                make_checkpoint(
                    _accelerator=accelerator, _epoch=epoch + 1, _step=0
                )
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise e

    logging.shutdown()
    if accelerator.is_main_process:
        loss_csv.close()
        epoch_csv.close()


# NOTE: Any differences observed when resuming training are most likely the
# result of randomness inherent to the data-augmentation. I'm currently unsure
# how to register and restore this random state during checkpointing.
def resume_train(
    model_name: str,
    train_data_paths: str,
    val_data_path: str,
    mode: str,
    num_workers: int,
    batch_size: int,
    epochs: int,
    checkpoint_dir: str,
    resume_epoch: int,
    resume_step: int,
    steps_per_checkpoint: int | None = None,
    project_dir: str = None,
):
    # Validate inputs
    assert mode in {"pretrain", "finetune"}, "Invalid mode"
    assert 0 < num_workers <= 128, "Too many workers"
    assert epochs > 0, "Invalid number of epochs"
    assert batch_size > 0, "Invalid batch size"
    assert torch.cuda.is_available() is True, "CUDA not available"
    assert os.path.isdir(checkpoint_dir), f"No dir at {checkpoint_dir}"
    for _path in train_data_paths:
        assert os.path.isfile(_path), f"No file found at {_path}"
    assert os.path.isfile(val_data_path), f"No file found at {val_data_path}"

    tokenizer = AmtTokenizer()
    accelerator = accelerate.Accelerator(
        project_dir=project_dir, gradient_accumulation_steps=GRADIENT_ACC_STEPS
    )
    if accelerator.is_main_process:
        project_dir = setup_project_dir(project_dir)
        logger = setup_logger(project_dir)

    logger = get_logger(__name__)
    logger.info(f"Using project directory {project_dir} ")
    logger.warning(
        "Please insure that the training config and resume step are set "
        "correctly, the script does not currently check that this is the case. "
        "If the previous checkpoint was saved at step n, then resume_step "
        "should be n. If there is a mismatch between the batch size then the "
        "script will resume at the wrong step. It is also important that the "
        "same distributed setup is used for training."
    )
    logger.info(
        f"Using training config: "
        f"model_name={model_name}, "
        f"mode={mode}, "
        f"epochs={epochs}, "
        f"num_proc={accelerator.num_processes}, "
        f"batch_size={batch_size}, "
        f"grad_acc_steps={GRADIENT_ACC_STEPS}, "
        f"num_workers={num_workers}, "
        f"checkpoint_dir={checkpoint_dir}, "
        f"resume_step={resume_step}, "
        f"resume_epoch={resume_epoch}"
    )
    if steps_per_checkpoint:
        logger.info(f"Creating checkpoints every {steps_per_checkpoint}")

    # Init model
    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model = AmtEncoderDecoder(model_config)
    model = torch.compile(model)
    audio_transform = AudioTransform().to(accelerator.device)
    logger.info(f"Loaded model with config: {load_model_config(model_name)}")
    logger.info(f"Loaded transform with config: {audio_transform.get_params()}")

    train_dataloader, val_dataloader = get_dataloaders(
        train_data_paths=train_data_paths,
        val_data_path=val_data_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    assert (
        model_config.n_text_ctx
        == train_dataloader.dataset.config["max_seq_len"]
    ), "seq_len mismatch between dataset and model"

    if mode == "pretrain":
        optimizer, scheduler = get_pretrain_optim(
            model,
            num_epochs=epochs,
            steps_per_epoch=len(train_dataloader) // GRADIENT_ACC_STEPS,
        )
    elif mode == "finetune":
        optimizer, scheduler = get_finetune_optim(
            model,
            num_epochs=epochs,
            steps_per_epoch=len(train_dataloader) // GRADIENT_ACC_STEPS,
        )
    else:
        raise Exception

    (
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
    ) = accelerator.prepare(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
    )

    try:
        accelerator.load_state(checkpoint_dir)
    except Exception as e:
        raise Exception(
            f"Failed to load checkpoint: {e}\n"
            "This could be due to a mismatch between the tokenizer used "
            "to build the pre-training and fine-tuning datasets"
        )
    logger.info(f"Loaded checkpoint at {checkpoint_dir}")
    logger.info("Starting train job")

    _train(
        epochs=epochs,
        accelerator=accelerator,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        audio_transform=audio_transform,
        optimizer=optimizer,
        scheduler=scheduler,
        steps_per_checkpoint=steps_per_checkpoint,
        resume_step=resume_step,
        resume_epoch=resume_epoch,
        project_dir=project_dir,
    )


def train(
    model_name: str,
    train_data_paths: str,
    val_data_path: str,
    mode: str,
    num_workers: int,
    batch_size: int,
    epochs: int,
    finetune_cp_path: str | None = None,  # loads ft optimizer and cp
    steps_per_checkpoint: int | None = None,
    project_dir: str = None,
):
    # Validate inputs
    assert mode in {"pretrain", "finetune"}, "Invalid mode"
    assert 0 < num_workers <= 128, "Too many workers"
    assert epochs > 0, "Invalid number of epochs"
    assert batch_size > 0, "Invalid batch size"
    assert torch.cuda.is_available() is True, "CUDA not available"
    for _path in train_data_paths:
        assert os.path.isfile(_path), f"No file found at {_path}"
    assert os.path.isfile(val_data_path), f"No file found at {val_data_path}"
    if mode == "finetune":
        assert os.path.isfile(finetune_cp_path), "Invalid checkpoint path"

    tokenizer = AmtTokenizer()
    accelerator = accelerate.Accelerator(
        project_dir=project_dir, gradient_accumulation_steps=GRADIENT_ACC_STEPS
    )
    if accelerator.is_main_process:
        project_dir = setup_project_dir(project_dir)
        logger = setup_logger(project_dir)

    logger = get_logger(__name__)
    logger.info(f"Using project directory {project_dir}")
    logger.info(
        f"Using training config: "
        f"model_name={model_name}, "
        f"mode={mode}, "
        f"epochs={epochs}, "
        f"num_proc={accelerator.num_processes}, "
        f"batch_size={batch_size}, "
        f"grad_acc_steps={GRADIENT_ACC_STEPS}, "
        f"num_workers={num_workers}"
    )

    # Init model
    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model = AmtEncoderDecoder(model_config)
    model = torch.compile(model)
    audio_transform = AudioTransform().to(accelerator.device)
    logger.info(f"Loaded model with config: {load_model_config(model_name)}")
    logger.info(f"Loaded transform with config: {audio_transform.get_params()}")
    if mode == "finetune":
        try:
            model.load_state_dict(_load_weight(finetune_cp_path))
        except Exception as e:
            raise Exception(f"Failed to load checkpoint: {e}")
        logger.info(
            f"Loaded finetune checkpoint located at: {finetune_cp_path}"
        )

    train_dataloader, val_dataloader = get_dataloaders(
        train_data_paths=train_data_paths,
        val_data_path=val_data_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    assert (
        model_config.n_text_ctx
        == train_dataloader.dataset.config["max_seq_len"]
    ), "seq_len mismatch between dataset and model"

    if mode == "pretrain":
        optimizer, scheduler = get_pretrain_optim(
            model,
            num_epochs=epochs,
            steps_per_epoch=len(train_dataloader) // GRADIENT_ACC_STEPS,
        )
    elif mode == "finetune":
        optimizer, scheduler = get_finetune_optim(
            model,
            num_epochs=epochs,
            steps_per_epoch=len(train_dataloader) // GRADIENT_ACC_STEPS,
        )
    else:
        raise Exception

    (
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
    ) = accelerator.prepare(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
    )

    logger.info(
        f"Starting {'finetune' if finetune_cp_path else 'pretrain'} job"
    )
    _train(
        epochs=epochs,
        accelerator=accelerator,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        audio_transform=audio_transform,
        optimizer=optimizer,
        scheduler=scheduler,
        steps_per_checkpoint=steps_per_checkpoint,
        project_dir=project_dir,
    )


def convert_cp_from_safetensors(checkpoint_path: str, save_path: str):
    d = load_file(checkpoint_path)
    key = list(d.keys())[0]
    gap = len(key.split(".")[0])
    d = {s[gap + 1 :]: v for s, v in d.items()}
    torch.save(d, save_path)


def convert_cp_from_accelerate(
    model_name: str,
    checkpoint_dir: str,
    save_path: str,
):
    tokenizer = AmtTokenizer()
    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model = AmtEncoderDecoder(model_config)

    accelerator = accelerate.Accelerator()
    model = accelerator.prepare(model)
    accelerator.load_state(checkpoint_dir)
    torch.save(model.state_dict(), save_path)


def parse_resume_args():
    argp = argparse.ArgumentParser(prog="python amt/train.py resume")
    argp.add_argument("model", help="name of model config file")
    argp.add_argument("resume_mode", help="training mode", choices=["pt", "ft"])
    argp.add_argument("-train_data", nargs="+", help="paths to train data")
    argp.add_argument("-val_data", help="path to val data")
    argp.add_argument("-cdir", help="checkpoint dir", type=str, required=True)
    argp.add_argument("-rstep", help="resume step", type=int, required=True)
    argp.add_argument("-repoch", help="resume epoch", type=int, required=True)
    argp.add_argument("-epochs", help="train epochs", type=int, required=True)
    argp.add_argument("-bs", help="batch size", type=int, default=32)
    argp.add_argument("-workers", help="number workers", type=int, default=1)
    argp.add_argument("-pdir", help="project dir", type=str, required=False)
    argp.add_argument(
        "-spc", help="steps per checkpoint", type=int, required=False
    )

    return argp.parse_args(sys.argv[2:])


def parse_train_args():
    argp = argparse.ArgumentParser(prog="python amt/train.py pretrain")
    argp.add_argument("model", help="name of model config file")
    argp.add_argument("-train_data", nargs="+", help="paths to train data")
    argp.add_argument("-val_data", help="path to val dir")
    argp.add_argument(
        "-cpath", help="resuming checkpoint", type=str, required=False
    )
    argp.add_argument("-epochs", help="train epochs", type=int, required=True)
    argp.add_argument("-bs", help="batch size", type=int, default=32)
    argp.add_argument("-workers", help="number workers", type=int, default=1)
    argp.add_argument("-pdir", help="project dir", type=str, required=False)
    argp.add_argument(
        "-spc", help="steps per checkpoint", type=int, required=False
    )

    return argp.parse_args(sys.argv[2:])


if __name__ == "__main__":
    # Nested argparse inspired by - https://shorturl.at/kuKW0
    parser = argparse.ArgumentParser(
        usage="python amt/train.py <command> [<args>]"
    )
    parser.add_argument(
        "mode", help="training mode", choices=("pretrain", "finetune", "resume")
    )

    args = parser.parse_args(sys.argv[1:2])
    if not hasattr(args, "mode"):
        parser.print_help()
        print("Unrecognized command")
        exit(1)
    elif args.mode == "pretrain":
        train_args = parse_train_args()
        train(
            model_name=train_args.model,
            train_data_paths=train_args.train_data,
            val_data_path=train_args.val_data,
            mode="pretrain",
            num_workers=train_args.workers,
            batch_size=train_args.bs,
            epochs=train_args.epochs,
            steps_per_checkpoint=train_args.spc,
            project_dir=train_args.pdir,
        )
    elif args.mode == "finetune":
        train_args = parse_train_args()
        train(
            model_name=train_args.model,
            train_data_paths=train_args.train_data,
            val_data_path=train_args.val_data,
            mode="finetune",
            num_workers=train_args.workers,
            batch_size=train_args.bs,
            epochs=train_args.epochs,
            finetune_cp_path=train_args.cpath,
            steps_per_checkpoint=train_args.spc,
            project_dir=train_args.pdir,
        )
    elif args.mode == "resume":
        resume_args = parse_resume_args()
        resume_train(
            model_name=resume_args.model,
            train_data_paths=resume_args.train_data,
            val_data_path=resume_args.val_data,
            mode="pretrain" if resume_args.resume_mode == "pt" else "finetune",
            num_workers=resume_args.workers,
            batch_size=resume_args.bs,
            epochs=resume_args.epochs,
            checkpoint_dir=resume_args.cdir,
            resume_step=resume_args.rstep,
            resume_epoch=resume_args.repoch,
            steps_per_checkpoint=resume_args.spc,
            project_dir=resume_args.pdir,
        )
    else:
        print("Unrecognized command")
        parser.print_help()
        exit(1)
