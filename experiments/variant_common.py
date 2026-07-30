"""Shared code for the three new model variants (MedicalNet fine-tuning,
Model Genesis self-supervised pretraining, registration-based Path A
features). Reuses the existing preprocessing cache and cv_folds.csv fold
assignments verbatim -- no re-deriving folds. Training loop mirrors
dat_scan_full_pipeline.ipynb's Path C training exactly (Adam +
ReduceLROnPlateau + early stopping on val log loss + AMP + gradient
clipping + best-checkpoint-only), extended to support per-parameter-group
learning rates for backbone/head fine-tuning.
"""
from __future__ import annotations

import functools
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT))
import preprocessing as pp  # noqa: E402

CACHE_DIR = REPO_ROOT / "preproc_cache"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
VARIANTS_DIR = ARTIFACTS_DIR / "model_variants"
VARIANTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- MedicalNet pretrained weights: not committed to the repo (132MB,
# third-party, MIT license) -- fetched on demand and checksum-verified, so
# this works unmodified on a fresh clone (Colab included), not just on a
# machine that already has the file cached locally. ---
MEDICALNET_WEIGHTS_URL = "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18/resolve/main/resnet_18_23dataset.pth"
MEDICALNET_WEIGHTS_SHA256 = "61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4"
MEDICALNET_WEIGHTS_PATH = ARTIFACTS_DIR / "pretrained_weights" / "medicalnet_resnet18_23dataset.pth"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_medicalnet_weights(path: Path = MEDICALNET_WEIGHTS_PATH, url: str = MEDICALNET_WEIGHTS_URL,
                               expected_sha256: str = MEDICALNET_WEIGHTS_SHA256) -> Path:
    """Downloads the real Tencent/MedicalNet ResNet-18 pretrained weights if
    not already present locally, verifying the checksum before trusting the
    file. Raises rather than silently continuing on a download failure or
    checksum mismatch -- never falls back to random-init without saying so."""
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"MedicalNet weights not found at {path} -- downloading from {url} ...", flush=True)
    import urllib.request
    tmp_path = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp_path)
        actual_sha256 = sha256_of(tmp_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Downloaded MedicalNet weights failed checksum verification: "
                f"expected {expected_sha256}, got {actual_sha256}. Refusing to use an unverified file.")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    print(f"Downloaded and checksum-verified MedicalNet weights -> {path}", flush=True)
    return path

# --- exact same augmentation / training hyperparameters as the existing
# Path C cell in dat_scan_full_pipeline.ipynb (CFG["aug"], CFG["lr_3d"], etc.) ---
AUG_CFG = {
    "translate_vox": 3, "rotate_deg": 8, "scale_range": (0.95, 1.05),
    "gain_range": (0.90, 1.10), "offset_range": (-0.05, 0.05),
    "noise_std": 0.02, "flip_lr": False,
}
BATCH_SIZE_3D = 16
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 6
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cv_folds() -> pd.DataFrame:
    """The existing, already-computed StratifiedGroupKFold(groups=scanner_proxy)
    split -- loaded verbatim, never re-derived."""
    return pd.read_csv(ARTIFACTS_DIR / "cv_folds.csv")


def cache_path_for(uid: str) -> Path:
    return CACHE_DIR / f"{uid}_{pp.CONFIG_HASH}.npy"


def load_cached_volume(uid: str) -> np.ndarray:
    return np.load(cache_path_for(uid)).astype(np.float32)


# ---------------------------------------------------------------------------
# Augmentation + dataset (verbatim port of the notebook's Path C versions)
# ---------------------------------------------------------------------------


def _center_crop_or_pad_3d(arr: np.ndarray, target_shape) -> np.ndarray:
    out = np.zeros(target_shape, dtype=arr.dtype)
    s = [max(0, (a - t) // 2) for a, t in zip(arr.shape, target_shape)]
    c = [min(a, t) for a, t in zip(arr.shape, target_shape)]
    d = [max(0, (t - a) // 2) for a, t in zip(arr.shape, target_shape)]
    src = arr[s[0]:s[0] + c[0], s[1]:s[1] + c[1], s[2]:s[2] + c[2]]
    out[d[0]:d[0] + src.shape[0], d[1]:d[1] + src.shape[1], d[2]:d[2] + src.shape[2]] = src
    return out


def augment_volume_3d(vol: np.ndarray, aug_cfg: dict) -> np.ndarray:
    from scipy.ndimage import shift as nd_shift, rotate as nd_rotate, zoom as nd_zoom
    out = vol
    if aug_cfg.get("flip_lr", False) and np.random.rand() < 0.5:
        out = out[::-1, :, :].copy()
    tvox = aug_cfg.get("translate_vox", 0)
    if tvox:
        d = np.random.uniform(-tvox, tvox, size=3)
        out = nd_shift(out, shift=d, order=1, mode="constant", cval=0.0)
    deg = aug_cfg.get("rotate_deg", 0)
    if deg:
        axes = random.choice([(0, 1), (0, 2), (1, 2)])
        angle = np.random.uniform(-deg, deg)
        out = nd_rotate(out, angle, axes=axes, reshape=False, order=1, mode="constant", cval=0.0)
    lo, hi = aug_cfg.get("scale_range", (1.0, 1.0))
    if lo != 1.0 or hi != 1.0:
        s = np.random.uniform(lo, hi)
        zoomed = nd_zoom(out, zoom=s, order=1)
        out = _center_crop_or_pad_3d(zoomed, vol.shape)
    glo, ghi = aug_cfg.get("gain_range", (1.0, 1.0))
    olo, ohi = aug_cfg.get("offset_range", (0.0, 0.0))
    out = out * np.random.uniform(glo, ghi) + np.random.uniform(olo, ohi)
    noise_std = aug_cfg.get("noise_std", 0.0)
    if noise_std:
        out = out + np.random.normal(0, noise_std, size=out.shape)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class Volume3DDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False, aug_cfg: Optional[dict] = None):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.aug_cfg = aug_cfg or {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vol = load_cached_volume(row["uid"])
        if self.augment:
            vol = augment_volume_3d(vol, self.aug_cfg)
        x = torch.from_numpy(vol[None, ...].astype(np.float32))
        y = torch.tensor(float(row["is_pathologic"]), dtype=torch.float32)
        return x, y, row["uid"]


def build_loaders_3d(train_df, val_df, batch_size, num_workers=0, aug_cfg=None):
    train_ds = Volume3DDataset(train_df, augment=True, aug_cfg=aug_cfg or AUG_CFG)
    val_ds = Volume3DDataset(val_df, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=(len(train_ds) > batch_size))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training loop -- same stabilization as Path C (Adam, ReduceLROnPlateau,
# early stopping on val log loss, AMP, grad clipping, best-checkpoint-only),
# extended to accept pre-built param groups (for differential LR).
# ---------------------------------------------------------------------------


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, grad_clip=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, n_seen = 0.0, 0
    all_probs, all_targets = [], []
    if train_mode:
        optimizer.zero_grad()
    for x, y, _uid in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        use_amp = scaler is not None and device == "cuda"
        with torch.set_grad_enabled(train_mode):
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = criterion(logits, y)
                if train_mode:
                    scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                if train_mode:
                    loss.backward()
        if train_mode:
            if use_amp:
                if grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * x.size(0)
        n_seen += x.size(0)
        all_probs.append(torch.sigmoid(logits).detach().float().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())
    avg_loss = total_loss / max(n_seen, 1)
    probs = np.concatenate(all_probs) if all_probs else np.array([])
    targets = np.concatenate(all_targets) if all_targets else np.array([])
    return avg_loss, probs, targets


def make_grad_scaler(device: str):
    if device != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler()


def train_with_early_stopping(model, train_loader, val_loader, param_groups, max_epochs,
                               patience, grad_clip, ckpt_path, device, class_weight_pos=1.0,
                               model_tag="", log_fn=print):
    """param_groups: list of dicts, e.g. [{"params": ..., "lr": 1e-4}, {"params": ..., "lr": 1e-3}]"""
    model = model.to(device)
    optimizer = torch.optim.Adam(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    pos_weight = torch.tensor([class_weight_pos], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = make_grad_scaler(device)

    best_val_loss = float("inf")
    best_epoch = -1
    patience_left = patience
    history = []

    for epoch in range(max_epochs):
        t0 = time.time()
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler, grad_clip)
        val_loss, val_probs, val_targets = run_epoch(model, val_loader, criterion, device)
        val_auc = roc_auc_score(val_targets, val_probs) if len(set(val_targets.tolist())) > 1 else float("nan")
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "val_auroc": val_auc, "epoch_seconds": elapsed})
        scheduler.step(val_loss)

        improved = val_loss < best_val_loss - 1e-5
        if improved:
            best_val_loss, best_epoch, patience_left = val_loss, epoch, patience
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_left -= 1

        log_fn(f"  [{model_tag}] epoch {epoch:02d}  train_loss={train_loss:.4f}  "
               f"val_loss={val_loss:.4f}  val_auroc={val_auc:.4f}  ({elapsed:.1f}s)"
               f"{'  *best*' if improved else ''}")
        if patience_left <= 0:
            log_fn(f"  [{model_tag}] early stopping at epoch {epoch} (best={best_epoch})")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return model, pd.DataFrame(history), best_epoch, best_val_loss


def train_fold_with_oom_backoff(build_loaders_fn, model_builder_fn, param_groups_fn, batch_size,
                                 min_batch_size=2, **train_kwargs):
    bs = batch_size
    while True:
        try:
            train_loader, val_loader = build_loaders_fn(bs)
            model = model_builder_fn()
            param_groups = param_groups_fn(model)
            return train_with_early_stopping(model, train_loader, val_loader, param_groups, **train_kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > min_batch_size:
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                new_bs = max(min_batch_size, bs // 2)
                print(f"  CUDA OOM at batch_size={bs}; retrying at {new_bs}")
                bs = new_bs
                continue
            raise


def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    n_classes = len(set(y_true.tolist()))
    return {
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "auroc": roc_auc_score(y_true, y_prob) if n_classes > 1 else np.nan,
        "brier": brier_score_loss(y_true, y_prob),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


# ---------------------------------------------------------------------------
# MedicalNet ResNet-18 3D (faithful port of Tencent/MedicalNet's resnet.py,
# shortcut_type='A', verified to load the real pretrained checkpoint with
# zero unexpected keys and only fc.weight/fc.bias missing).
# ---------------------------------------------------------------------------


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, dilation=dilation,
                      stride=stride, padding=dilation, bias=False)


def downsample_basic_block(x, planes, stride):
    out = torch.nn.functional.avg_pool3d(x, kernel_size=1, stride=stride)
    pad_shape = (out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4))
    zero_pads = torch.zeros(pad_shape, dtype=out.dtype, device=out.device)
    return torch.cat([out, zero_pads], dim=1)


class BasicBlock3D(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class MedicalNetResNet3D(nn.Module):
    def __init__(self, layers=(2, 2, 2, 2), num_classes: int = 1, dropout: float = 0.3):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], stride=1, dilation=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(512, layers[3], stride=1, dilation=4)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = functools.partial(downsample_basic_block, planes=planes, stride=stride)
        layers = [BasicBlock3D(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(self.dropout(x)).squeeze(-1)

    def backbone_named_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if not n.startswith("fc.")]

    def head_parameters(self):
        return list(self.fc.parameters())


def load_medicalnet_pretrained(model: MedicalNetResNet3D, ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = ckpt.get("state_dict", ckpt)
    stripped = {k.replace("module.", "", 1): v for k, v in raw_sd.items()}
    result = model.load_state_dict(stripped, strict=False)
    return {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}


# ---------------------------------------------------------------------------
# Model Genesis: pretext transforms + 3D encoder-decoder
# ---------------------------------------------------------------------------


def local_pixel_shuffle(vol, n_blocks=8, block_size=8, rng=None):
    rng = rng or np.random.RandomState(0)
    out = vol.copy()
    for _ in range(n_blocks):
        x, y, z = [rng.randint(0, max(1, s - block_size)) for s in vol.shape]
        block = out[x:x + block_size, y:y + block_size, z:z + block_size].copy()
        flat = block.reshape(-1)
        rng.shuffle(flat)
        out[x:x + block_size, y:y + block_size, z:z + block_size] = flat.reshape(block.shape)
    return out


def nonlinear_intensity_transform(vol, rng=None):
    rng = rng or np.random.RandomState(0)
    points = np.sort(rng.uniform(0, 1, size=4))
    xp = np.linspace(0, 1, len(points))
    return np.interp(vol, xp, points).astype(np.float32)


def inpaint_random_region(vol, size=16, rng=None):
    rng = rng or np.random.RandomState(0)
    out = vol.copy()
    x, y, z = [rng.randint(0, max(1, s - size)) for s in vol.shape]
    out[x:x + size, y:y + size, z:z + size] = rng.uniform(0, 1)
    return out


def outpaint_border(vol, margin=10):
    out = np.zeros_like(vol)
    s = vol.shape
    out[margin:s[0] - margin, margin:s[1] - margin, margin:s[2] - margin] = \
        vol[margin:s[0] - margin, margin:s[1] - margin, margin:s[2] - margin]
    return out


def model_genesis_transform(vol, rng=None):
    rng = rng or np.random.RandomState(0)
    out = vol
    if rng.rand() < 0.6:
        out = local_pixel_shuffle(out, rng=rng)
    if rng.rand() < 0.6:
        out = nonlinear_intensity_transform(out, rng=rng)
    if rng.rand() < 0.3:
        out = inpaint_random_region(out, rng=rng)
    if rng.rand() < 0.3:
        out = outpaint_border(out)
    return out


class ReconstructionDataset(Dataset):
    """Pretext-task dataset: input is a corrupted volume, target is the
    original (uncorrupted) volume. `uids` must come only from the
    permitted subset (fold-training-only during CV; all training uids for
    the final model) -- enforced by the caller, not this class."""
    def __init__(self, uids: list[str], seed: int = 0):
        self.uids = list(uids)
        self.seed = seed

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        vol = load_cached_volume(uid)
        rng = np.random.RandomState(hash((self.seed, uid)) % (2**31))
        corrupted = model_genesis_transform(vol, rng=rng)
        x = torch.from_numpy(corrupted[None, ...].astype(np.float32))
        target = torch.from_numpy(vol[None, ...].astype(np.float32))
        return x, target


class GenesisEncoder(nn.Module):
    """Same depth/receptive-field philosophy as Path C's Compact3DCNN
    encoder, sized to be a genuine self-supervised-pretraining target (not
    a toy): 4 downsampling stages, 64->32->16->8->4 spatial."""
    def __init__(self, base_channels: int = 24):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(nn.Conv3d(1, c, 3, padding=1), nn.BatchNorm3d(c), nn.ReLU(inplace=True))
        self.down1 = nn.Sequential(nn.Conv3d(c, c * 2, 3, stride=2, padding=1), nn.BatchNorm3d(c * 2), nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.Conv3d(c * 2, c * 4, 3, stride=2, padding=1), nn.BatchNorm3d(c * 4), nn.ReLU(inplace=True))
        self.down3 = nn.Sequential(nn.Conv3d(c * 4, c * 8, 3, stride=2, padding=1), nn.BatchNorm3d(c * 8), nn.ReLU(inplace=True))
        self.down4 = nn.Sequential(nn.Conv3d(c * 8, c * 8, 3, stride=2, padding=1), nn.BatchNorm3d(c * 8), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x); x = self.down2(x); x = self.down3(x); x = self.down4(x)
        return x


class GenesisDecoder(nn.Module):
    def __init__(self, base_channels: int = 24):
        super().__init__()
        c = base_channels
        self.up1 = nn.Sequential(nn.ConvTranspose3d(c * 8, c * 8, 2, stride=2), nn.BatchNorm3d(c * 8), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose3d(c * 8, c * 4, 2, stride=2), nn.BatchNorm3d(c * 4), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose3d(c * 4, c * 2, 2, stride=2), nn.BatchNorm3d(c * 2), nn.ReLU(inplace=True))
        self.up4 = nn.ConvTranspose3d(c * 2, c, 2, stride=2)
        self.out = nn.Conv3d(c, 1, 3, padding=1)

    def forward(self, x):
        x = self.up1(x); x = self.up2(x); x = self.up3(x); x = self.up4(x)
        return torch.sigmoid(self.out(x))


class GenesisAutoencoder(nn.Module):
    def __init__(self, base_channels: int = 24):
        super().__init__()
        self.encoder = GenesisEncoder(base_channels)
        self.decoder = GenesisDecoder(base_channels)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class GenesisClassifier(nn.Module):
    """Encoder (optionally initialized from Model Genesis pretraining) +
    global-average-pool + a binary classification head."""
    def __init__(self, base_channels: int = 24, dropout: float = 0.3):
        super().__init__()
        self.encoder = GenesisEncoder(base_channels)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 8, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x)).squeeze(-1)

    def backbone_named_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if not n.startswith("fc.")]

    def head_parameters(self):
        return list(self.fc.parameters())
