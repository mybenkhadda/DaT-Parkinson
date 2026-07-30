"""MedicalNet-pretrained 3D CNN: 5-fold fine-tuning on the existing
cv_folds.csv split (never re-derived), using the real Tencent/MedicalNet
ResNet-18 pretrained backbone (MIT license, weights + checksum recorded
below), with a lower learning rate on the pretrained backbone than on the
new classification head, and the exact same training stabilization as
Path C (Adam, ReduceLROnPlateau, early stopping on val log loss, AMP,
gradient clipping, best-checkpoint-only).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variant_common as vc

WEIGHTS_PATH = vc.REPO_ROOT / "artifacts" / "pretrained_weights" / "medicalnet_resnet18_23dataset.pth"
BACKBONE_LR = 1e-4   # lower than the head, per the task
HEAD_LR = 1e-3        # same magnitude as CFG["lr_3d"] used for Path C


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_param_groups(model: vc.MedicalNetResNet3D):
    backbone_params = [p for _, p in model.backbone_named_parameters()]
    return [
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": model.head_parameters(), "lr": HEAD_LR},
    ]


def main():
    vc.seed_everything()
    folds_df = vc.load_cv_folds()
    n_folds = folds_df["fold"].nunique()

    weight_checksum = sha256_of(WEIGHTS_PATH)
    provenance = {
        "source_repo": "https://github.com/Tencent/MedicalNet",
        "weight_file": "resnet_18_23dataset.pth",
        "downloaded_from": "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18",
        "license": "MIT",
        "sha256": weight_checksum,
        "architecture": "ResNet-18 (BasicBlock, shortcut_type='A' parameter-free), verified: "
                         "0 unexpected keys, only fc.weight/fc.bias missing on load",
        "backbone_lr": BACKBONE_LR,
        "head_lr": HEAD_LR,
    }
    log(f"Weight provenance: {json.dumps(provenance, indent=2)}")

    oof_map: dict[str, float] = {}
    all_histories = {}
    fold_timing = {}

    for fold in range(n_folds):
        train_df = folds_df[folds_df["fold"] != fold].reset_index(drop=True)
        val_df = folds_df[folds_df["fold"] == fold].reset_index(drop=True)
        pos_frac = train_df["is_pathologic"].mean()
        class_weight_pos = float((1 - pos_frac) / max(pos_frac, 1e-6))
        ckpt_path = vc.VARIANTS_DIR / f"medicalnet_fold{fold}_best.pt"

        def _build_loaders(bs, train_df=train_df, val_df=val_df):
            return vc.build_loaders_3d(train_df, val_df, bs, num_workers=0, aug_cfg=vc.AUG_CFG)

        def _build_model():
            model = vc.MedicalNetResNet3D()
            report = vc.load_medicalnet_pretrained(model, WEIGHTS_PATH)
            assert report["unexpected_keys"] == [], f"unexpected keys: {report['unexpected_keys']}"
            assert set(report["missing_keys"]) == {"fc.weight", "fc.bias"}, \
                f"unexpected missing keys: {report['missing_keys']}"
            return model

        t0 = time.time()
        model, history, best_epoch, best_val_loss = vc.train_fold_with_oom_backoff(
            _build_loaders, _build_model, build_param_groups, vc.BATCH_SIZE_3D, min_batch_size=2,
            max_epochs=vc.MAX_EPOCHS, patience=vc.EARLY_STOPPING_PATIENCE, grad_clip=vc.GRAD_CLIP_NORM,
            ckpt_path=ckpt_path, device=vc.DEVICE, class_weight_pos=class_weight_pos,
            model_tag=f"MedicalNet fold {fold}", log_fn=log,
        )
        fold_seconds = time.time() - t0
        fold_timing[fold] = fold_seconds
        all_histories[fold] = history.to_dict(orient="records")
        log(f"Fold {fold} done in {fold_seconds:.0f}s (best epoch {best_epoch}, val_loss={best_val_loss:.4f})")

        eval_loader = torch.utils.data.DataLoader(
            vc.Volume3DDataset(val_df, augment=False), batch_size=vc.BATCH_SIZE_3D, shuffle=False)
        model.eval()
        with torch.no_grad():
            for x, y, uids in eval_loader:
                probs = torch.sigmoid(model(x.to(vc.DEVICE))).cpu().numpy()
                for u, p in zip(uids, probs):
                    oof_map[u] = float(p)

        del model
        if vc.DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Persist incrementally so a later fold's crash doesn't lose earlier folds.
        pd.DataFrame({"uid": list(oof_map.keys()), "medicalnet_pretrained": list(oof_map.values())}).to_csv(
            vc.VARIANTS_DIR / "medicalnet_oof.csv", index=False)
        with open(vc.VARIANTS_DIR / "medicalnet_histories.json", "w") as f:
            json.dump(all_histories, f, indent=2)
        with open(vc.VARIANTS_DIR / "medicalnet_provenance.json", "w") as f:
            json.dump({**provenance, "fold_timing_seconds": fold_timing}, f, indent=2)

    oof_df = pd.DataFrame({"uid": list(oof_map.keys()), "medicalnet_pretrained": list(oof_map.values())})
    oof_df = oof_df.merge(folds_df[["uid", "is_pathologic", "fold"]], on="uid")
    oof_df.to_csv(vc.VARIANTS_DIR / "medicalnet_oof.csv", index=False)

    metrics = vc.compute_binary_metrics(oof_df["is_pathologic"], oof_df["medicalnet_pretrained"])
    log(f"MedicalNet OOF metrics (all {n_folds} folds): {json.dumps(metrics, indent=2, default=str)}")
    log(f"Total training time: {sum(fold_timing.values()):.0f}s across {n_folds} folds")
    log("MedicalNet fine-tuning run complete.")


if __name__ == "__main__":
    main()
