"""
Training script for TinyRecursiveModel.

Performs Supervised Fine-Tuning (SFT) on AdaptLLM task data
using a causal language-modelling objective with answer-only loss.

Usage (single GPU):
    python train.py

Usage (multi-GPU via accelerate):
    accelerate launch --num_processes 2 train.py task_name=NER

Hydra config is defined in configs/train.yaml.
"""

import logging
import math
import os
import time
from pathlib import Path

import hydra
import hydra.utils as hu
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.dataset_readers.train_dsr import TrainDatasetReader, collate_fn
from src.models.model import get_custom_model

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Linear warmup → cosine decay LR schedule."""
    import math

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


def compute_loss(model, batch, vocab_size: int, n_supervision_steps: int = 2) -> torch.Tensor:
    """
    Run model forward with deep supervision and compute cross-entropy loss
    only on non-ignored label positions (answer tokens).

    The TinyRecursiveModel.forward() signature expects `targets` for training.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    # Clamp labels to valid range (keep -100 ignore tokens)
    valid_mask = labels != -100
    labels_clamped = labels.clone()
    labels_clamped[valid_mask] = labels_clamped[valid_mask].clamp(0, vocab_size - 1)

    loss = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        targets=labels_clamped,
        n_supervision_steps=n_supervision_steps,
    )
    return loss


def save_checkpoint(model, optimizer, scheduler, epoch: int, step: int, output_dir: str, name: str = "checkpoint"):
    """Save model state dict and training state."""
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"{name}_ep{epoch}_step{step}.pt")
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        ckpt_path,
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    print(f"[Checkpoint] Saved: {ckpt_path}")
    return ckpt_path


def load_checkpoint(model, optimizer, scheduler, ckpt_path: str, device):
    """Load from a saved checkpoint and return (epoch, step)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = ckpt.get("epoch", 0)
    step = ckpt.get("step", 0)
    logger.info("Loaded checkpoint from %s (epoch=%d, step=%d)", ckpt_path, epoch, step)
    return epoch, step


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Training config:\n%s", OmegaConf.to_yaml(cfg))

    accelerator = Accelerator(mixed_precision=cfg.get("mixed_precision", "no"))

    # ── Build dataset ──────────────────────────────────────────────────
    task_names = cfg.task_name.split("+")

    # Build one dataset per task and concatenate
    from torch.utils.data import ConcatDataset

    datasets = []
    for task_name in task_names:
        ds = TrainDatasetReader(
            model_name=cfg.model_name,
            task_name=task_name,
            split=cfg.get("train_split", "train"),
            max_length=cfg.max_length,
            generate_max_len=cfg.generate_max_len,
            cache_dir=cfg.get("cache_dir", None),
            add_bos_token=cfg.get("add_bos_token", False),
            tokenizer_name=cfg.get("tokenizer_name", None),
        )
        datasets.append(ds)

    train_dataset = ConcatDataset(datasets)
    tokenizer = datasets[0].tokenizer
    pad_token_id = tokenizer.pad_token_id or 0

    def _collate(batch):
        return collate_fn(batch, pad_token_id=pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        collate_fn=_collate,
        pin_memory=True,
    )

    # ── Build model ────────────────────────────────────────────────────
    model = get_custom_model(
        model_type=cfg.model_name,
        vocab_size=cfg.vocab_size,
        dim=cfg.get("dim", 256),
        n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("n_layers", 2),
        max_seq_len=cfg.max_length,
        n_latent_recursions=cfg.get("n_latent_recursions", 3),
        n_improvement_cycles=cfg.get("n_improvement_cycles", 2),
        dropout=cfg.get("dropout", 0.1),
        adapter_dropout=cfg.get("adapter_dropout", 0.0),
        use_task_adapter=cfg.get("use_task_adapter", True),
        use_checkpoint=cfg.get("use_checkpoint", False),
        use_less_is_more=cfg.get("use_less_is_more", False),
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    logger.info("Model parameters: %d", total_params)

    # ── Optimizer & scheduler ─────────────────────────────────────────
    lr = cfg.get("learning_rate", 3e-4)
    weight_decay = cfg.get("weight_decay", 0.01)

    # Use depth-scaled optimizer groups if available
    if hasattr(model, "get_depth_scaled_optimizer_groups"):
        param_groups = model.get_depth_scaled_optimizer_groups(base_lr=lr)
    else:
        param_groups = [{"params": list(model.parameters()), "lr": lr}]

    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    num_epochs = cfg.get("num_epochs", 5)
    num_steps_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)
    total_steps = num_epochs * num_steps_per_epoch
    warmup_steps = cfg.get("warmup_steps", max(100, total_steps // 10))

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Accelerate prepare ────────────────────────────────────────────
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # ── Resume from checkpoint ────────────────────────────────────────
    output_dir = cfg.get("output_dir", "./checkpoints")
    resume_ckpt = cfg.get("resume_checkpoint", None)
    start_epoch, global_step = 0, 0
    if resume_ckpt and os.path.isfile(resume_ckpt):
        raw_model = accelerator.unwrap_model(model)
        start_epoch, global_step = load_checkpoint(
            raw_model, optimizer, scheduler, resume_ckpt, accelerator.device
        )

    # ── Training ─────────────────────────────────────────────────────
    log_every = cfg.get("log_every", 10)
    save_every = cfg.get("save_every", 200)
    n_supervision_steps = cfg.get("n_supervision_steps", 2)
    vocab_size = cfg.vocab_size

    best_loss = float("inf")
    best_ckpt_path = None

    model.train()
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            raw_model = accelerator.unwrap_model(model)
            loss = compute_loss(raw_model, batch, vocab_size=vocab_size, n_supervision_steps=n_supervision_steps)

            # Skip NaN/Inf losses (can happen early in training with random weights)
            if not torch.isfinite(loss):
                logger.warning("Non-finite loss at step %d: %s — skipping batch", global_step, loss.item())
                optimizer.zero_grad()
                continue

            accelerator.backward(loss)

            # Gradient clipping
            if cfg.get("grad_clip", 1.0) > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()
            n_batches += 1

            if accelerator.is_main_process and global_step % log_every == 0:
                avg_loss = epoch_loss / n_batches
                lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr
                elapsed = time.time() - t0
                print(
                    f"[Epoch {epoch+1}/{num_epochs}] step={global_step} "
                    f"loss={loss.item():.4f} avg={avg_loss:.4f} "
                    f"lr={lr_now:.2e} elapsed={elapsed:.1f}s"
                )
                logger.info(
                    "epoch=%d step=%d loss=%.4f avg_loss=%.4f lr=%.2e",
                    epoch + 1, global_step, loss.item(), avg_loss, lr_now,
                )

            if accelerator.is_main_process and global_step % save_every == 0:
                raw_model = accelerator.unwrap_model(model)
                save_checkpoint(raw_model, optimizer, scheduler, epoch + 1, global_step, output_dir)

        # End of epoch
        avg_epoch_loss = epoch_loss / max(1, n_batches)
        logger.info("Epoch %d done. avg_loss=%.4f", epoch + 1, avg_epoch_loss)
        if accelerator.is_main_process:
            print(f"\n[Epoch {epoch+1}/{num_epochs}] DONE — avg_loss={avg_epoch_loss:.4f}\n")

            # Save best checkpoint
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                raw_model = accelerator.unwrap_model(model)
                best_ckpt_path = save_checkpoint(
                    raw_model, optimizer, scheduler, epoch + 1, global_step, output_dir, name="best"
                )
                print(f"[Best] New best checkpoint: {best_ckpt_path} (loss={best_loss:.4f})")

    # ── Final save ────────────────────────────────────────────────────
    if accelerator.is_main_process:
        raw_model = accelerator.unwrap_model(model)
        final_path = save_checkpoint(raw_model, optimizer, scheduler, num_epochs, global_step, output_dir, name="final")
        print(f"\n[Training complete] Final checkpoint: {final_path}")
        print(f"[Training complete] Best checkpoint:  {best_ckpt_path} (loss={best_loss:.4f})")
        logger.info("Training complete. Final=%s Best=%s", final_path, best_ckpt_path)


if __name__ == "__main__":
    main()
