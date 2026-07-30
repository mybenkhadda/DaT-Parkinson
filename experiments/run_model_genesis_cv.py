"""Model Genesis self-supervised pretraining, then fine-tuning, per CV fold.

Leakage rule (per project convention): self-supervised pretraining for a
given fold uses ONLY that fold's training-subset images -- the held-out
validation examinations for that fold are never touched during pretraining,
exactly like the supervised step. This means pretraining is repeated once
per fold (5x), not once globally, which is the scientifically correct
(if more expensive) approach for a valid OOF comparison.

Pretraining budget is deliberately bounded (max 20 epochs, patience 5) to
keep 5x-repeated pretraining tractable -- documented explicitly rather than
silently under-training and claiming a fair comparison to a fully-converged
setup.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variant_common as vc

PRETRAIN_MAX_EPOCHS = 20
PRETRAIN_PATIENCE = 5
PRETRAIN_LR = 1e-3
PRETRAIN_BATCH_SIZE = 8
PRETRAIN_VAL_FRACTION = 0.1

FINETUNE_ENCODER_LR = 1e-4
FINETUNE_HEAD_LR = 1e-3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pretrain_encoder(pretrain_pool_uids: list[str], fold: int, seed: int = 42):
    """Self-supervised reconstruction pretraining. `pretrain_pool_uids` must
    already exclude the current fold's held-out validation examinations --
    enforced by the caller."""
    rng = np.random.RandomState(seed + fold)
    shuffled = rng.permutation(pretrain_pool_uids)
    n_val = max(1, int(len(shuffled) * PRETRAIN_VAL_FRACTION))
    pretrain_val_uids = shuffled[:n_val].tolist()
    pretrain_train_uids = shuffled[n_val:].tolist()

    train_ds = vc.ReconstructionDataset(pretrain_train_uids, seed=seed + fold)
    val_ds = vc.ReconstructionDataset(pretrain_val_uids, seed=seed + fold + 1000)
    train_loader = DataLoader(train_ds, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=PRETRAIN_BATCH_SIZE, shuffle=False, num_workers=0)

    model = vc.GenesisAutoencoder().to(vc.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR, weight_decay=vc.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.MSELoss()
    scaler = vc.make_grad_scaler(vc.DEVICE)

    best_val_loss = float("inf")
    patience_left = PRETRAIN_PATIENCE
    ckpt_path = vc.VARIANTS_DIR / f"model_genesis_pretrain_fold{fold}_best.pt"
    history = []

    for epoch in range(PRETRAIN_MAX_EPOCHS):
        t0 = time.time()
        model.train()
        train_loss, n_seen = 0.0, 0
        for x, target in train_loader:
            x, target = x.to(vc.DEVICE), target.to(vc.DEVICE)
            optimizer.zero_grad()
            use_amp = scaler is not None and vc.DEVICE == "cuda"
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    recon = model(x)
                    loss = criterion(recon, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), vc.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                recon = model(x)
                loss = criterion(recon, target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), vc.GRAD_CLIP_NORM)
                optimizer.step()
            train_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        train_loss /= max(n_seen, 1)

        model.eval()
        val_loss, n_val_seen = 0.0, 0
        with torch.no_grad():
            for x, target in val_loader:
                x, target = x.to(vc.DEVICE), target.to(vc.DEVICE)
                recon = model(x)
                loss = criterion(recon, target)
                val_loss += loss.item() * x.size(0)
                n_val_seen += x.size(0)
        val_loss /= max(n_val_seen, 1)
        scheduler.step(val_loss)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_recon_loss": train_loss, "val_recon_loss": val_loss, "epoch_seconds": elapsed})

        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            patience_left = PRETRAIN_PATIENCE
            torch.save(model.encoder.state_dict(), ckpt_path)
        else:
            patience_left -= 1
        log(f"  [Genesis-pretrain fold {fold}] epoch {epoch:02d} train_recon={train_loss:.5f} "
            f"val_recon={val_loss:.5f} ({elapsed:.1f}s){'  *best*' if improved else ''}")
        if patience_left <= 0:
            log(f"  [Genesis-pretrain fold {fold}] early stopping at epoch {epoch}")
            break

    return ckpt_path, pd.DataFrame(history), best_val_loss


def build_param_groups(model: vc.GenesisClassifier):
    backbone_params = [p for _, p in model.backbone_named_parameters()]
    return [
        {"params": backbone_params, "lr": FINETUNE_ENCODER_LR},
        {"params": model.head_parameters(), "lr": FINETUNE_HEAD_LR},
    ]


def main():
    vc.seed_everything()
    folds_df = vc.load_cv_folds()
    n_folds = folds_df["fold"].nunique()

    oof_map: dict[str, float] = {}
    all_finetune_histories = {}
    all_pretrain_histories = {}
    fold_timing = {}

    for fold in range(n_folds):
        train_df = folds_df[folds_df["fold"] != fold].reset_index(drop=True)
        val_df = folds_df[folds_df["fold"] == fold].reset_index(drop=True)

        log(f"=== Fold {fold}: self-supervised pretraining on {len(train_df)} training-only examinations "
            f"(fold {fold}'s {len(val_df)} held-out examinations excluded) ===")
        t0 = time.time()
        pretrain_ckpt_path, pretrain_history, best_recon_loss = pretrain_encoder(
            train_df["uid"].tolist(), fold=fold)
        pretrain_seconds = time.time() - t0
        all_pretrain_histories[fold] = pretrain_history.to_dict(orient="records")
        log(f"Fold {fold} pretraining done in {pretrain_seconds:.0f}s (best val recon loss={best_recon_loss:.5f})")

        pos_frac = train_df["is_pathologic"].mean()
        class_weight_pos = float((1 - pos_frac) / max(pos_frac, 1e-6))
        finetune_ckpt_path = vc.VARIANTS_DIR / f"model_genesis_finetune_fold{fold}_best.pt"

        def _build_loaders(bs, train_df=train_df, val_df=val_df):
            return vc.build_loaders_3d(train_df, val_df, bs, num_workers=0, aug_cfg=vc.AUG_CFG)

        def _build_model(pretrain_ckpt_path=pretrain_ckpt_path):
            model = vc.GenesisClassifier()
            encoder_state = torch.load(pretrain_ckpt_path, map_location="cpu", weights_only=True)
            missing_unexpected = model.encoder.load_state_dict(encoder_state, strict=True)
            return model

        log(f"=== Fold {fold}: fine-tuning classifier from the pretrained encoder ===")
        t1 = time.time()
        model, finetune_history, best_epoch, best_val_loss = vc.train_fold_with_oom_backoff(
            _build_loaders, _build_model, build_param_groups, vc.BATCH_SIZE_3D, min_batch_size=2,
            max_epochs=vc.MAX_EPOCHS, patience=vc.EARLY_STOPPING_PATIENCE, grad_clip=vc.GRAD_CLIP_NORM,
            ckpt_path=finetune_ckpt_path, device=vc.DEVICE, class_weight_pos=class_weight_pos,
            model_tag=f"Genesis-finetune fold {fold}", log_fn=log,
        )
        finetune_seconds = time.time() - t1
        all_finetune_histories[fold] = finetune_history.to_dict(orient="records")
        fold_timing[fold] = {"pretrain_seconds": pretrain_seconds, "finetune_seconds": finetune_seconds}
        log(f"Fold {fold} fine-tuning done in {finetune_seconds:.0f}s (best epoch {best_epoch}, val_loss={best_val_loss:.4f})")

        eval_loader = DataLoader(vc.Volume3DDataset(val_df, augment=False), batch_size=vc.BATCH_SIZE_3D, shuffle=False)
        model.eval()
        with torch.no_grad():
            for x, y, uids in eval_loader:
                probs = torch.sigmoid(model(x.to(vc.DEVICE))).cpu().numpy()
                for u, p in zip(uids, probs):
                    oof_map[u] = float(p)

        del model
        if vc.DEVICE == "cuda":
            torch.cuda.empty_cache()

        pd.DataFrame({"uid": list(oof_map.keys()), "model_genesis_pretrained": list(oof_map.values())}).to_csv(
            vc.VARIANTS_DIR / "model_genesis_oof.csv", index=False)
        with open(vc.VARIANTS_DIR / "model_genesis_pretrain_histories.json", "w") as f:
            json.dump(all_pretrain_histories, f, indent=2)
        with open(vc.VARIANTS_DIR / "model_genesis_finetune_histories.json", "w") as f:
            json.dump(all_finetune_histories, f, indent=2)
        with open(vc.VARIANTS_DIR / "model_genesis_timing.json", "w") as f:
            json.dump(fold_timing, f, indent=2)

    oof_df = pd.DataFrame({"uid": list(oof_map.keys()), "model_genesis_pretrained": list(oof_map.values())})
    oof_df = oof_df.merge(folds_df[["uid", "is_pathologic", "fold"]], on="uid")
    oof_df.to_csv(vc.VARIANTS_DIR / "model_genesis_oof.csv", index=False)

    metrics = vc.compute_binary_metrics(oof_df["is_pathologic"], oof_df["model_genesis_pretrained"])
    log(f"Model Genesis OOF metrics (all {n_folds} folds): {json.dumps(metrics, indent=2, default=str)}")
    total_seconds = sum(v["pretrain_seconds"] + v["finetune_seconds"] for v in fold_timing.values())
    log(f"Total time (pretrain+finetune, all folds): {total_seconds:.0f}s")
    log("Model Genesis run complete.")


if __name__ == "__main__":
    main()
