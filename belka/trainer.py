"""Shared training / validation loop with OneCycle + AMP + EMA.

Model-agnostic: it only assumes ``model(input_ids, attention_mask, prot_emb)``
and reads ``model.EXCLUDE_PREFIXES`` to decide which parameters to checkpoint.
Validation uses the EMA weights and Average Precision for model selection.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from .ema import ModelEMA


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def filtered_state_dict(model, exclude_prefixes: tuple[str, ...]) -> dict:
    sd = _unwrap(model).state_dict()
    if not exclude_prefixes:
        return sd
    return {k: v for k, v in sd.items() if not any(k.startswith(p) for p in exclude_prefixes)}


def load_checkpoint(model, path: str, device, exclude_prefixes: tuple[str, ...]) -> None:
    """Load a checkpoint, asserting the trainable parameters actually loaded.

    This guards against the silent ``strict=False`` failure where a key-prefix
    mismatch (e.g. DataParallel ``module.``) leaves the model at random init.
    """
    sd = torch.load(path, map_location=device)
    missing, unexpected = _unwrap(model).load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:5]}"
    bad = [k for k in missing if not any(k.startswith(p) for p in exclude_prefixes)]
    assert not bad, f"trainable keys missing from checkpoint: {bad[:5]}"


def train_and_validate(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    *,
    epochs: int = 15,
    max_lr: float = 5e-4,
    ema_decay: float = 0.999,
    save_name: str = "checkpoints/best.pth",
    time_limit_hours: float = 8.0,
    grad_clip: float = 1.0,
) -> float:
    os.makedirs(os.path.dirname(save_name) or ".", exist_ok=True)
    exclude = getattr(_unwrap(model), "EXCLUDE_PREFIXES", ())

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader),
        epochs=epochs, pct_start=0.1, anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda")
    ema = ModelEMA(model, decay=ema_decay)
    best_ap, t0 = 0.0, time.time()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]")
        for batch in pbar:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            prot = batch["prot_emb"].to(device, non_blocking=True)
            lab = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(input_ids=ids, attention_mask=mask, prot_emb=prot)
                loss = criterion(logits.float(), lab.view_as(logits).float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # ---- validation on EMA weights ----
        backup = ema.swap_in(model)
        model.eval()
        preds, labs, vloss = [], [], 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [val-ema]"):
                ids = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                prot = batch["prot_emb"].to(device, non_blocking=True)
                lab = batch["labels"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(input_ids=ids, attention_mask=mask, prot_emb=prot)
                    vloss += criterion(logits.float(), lab.view_as(logits).float()).item()
                preds.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
                labs.append(lab.cpu().numpy().ravel())
        ema.swap_out(model, backup)

        ap = average_precision_score(np.concatenate(labs), np.concatenate(preds))
        print(f"\nEpoch {epoch + 1}: train {running / len(train_loader):.4f} | "
              f"val {vloss / len(val_loader):.4f} | AP(EMA) {ap:.4f}")

        if ap > best_ap:
            best_ap = ap
            backup = ema.swap_in(model)
            torch.save(filtered_state_dict(model, exclude), save_name)
            ema.swap_out(model, backup)
            print(f"  saved {save_name} (AP={best_ap:.4f})")

        if (time.time() - t0) / 3600 > time_limit_hours:
            print("time limit reached, stopping early")
            break

    return best_ap
