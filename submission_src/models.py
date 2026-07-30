"""Model architectures and small inference-time helpers for Path B (2.5D CNN)
and Path C (3D CNN), ported verbatim from dat_scan_full_pipeline.ipynb
sections 8-9 and 12 (ensemble stacker). Architectures must match the trained
checkpoints exactly -- any change here invalidates ``final_25d_model.pt`` /
``final_3d_model.pt``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

# ---------------------------------------------------------------------------
# Shared: peak-axial-slice heuristic (used to center the Path B slice stack)
# ---------------------------------------------------------------------------


def peak_axial_slice(vol: np.ndarray, smooth: bool = True) -> int:
    """Axial index with the greatest (optionally smoothed) total uptake --
    used as the center of the 2.5D slice stack."""
    axial_signal = vol.sum(axis=(0, 1))
    if smooth and axial_signal.size >= 5:
        kernel = np.ones(3) / 3.0
        axial_signal = np.convolve(axial_signal, kernel, mode="same")
    return int(np.argmax(axial_signal))


# ---------------------------------------------------------------------------
# Path B: 2.5D CNN (ResNet18 / EfficientNet-B0 backbone, first conv adapted
# to accept n_slices input channels)
# ---------------------------------------------------------------------------


def _adapt_first_conv(old_conv: nn.Conv2d, n_in: int) -> nn.Conv2d:
    new_conv = nn.Conv2d(n_in, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                          stride=old_conv.stride, padding=old_conv.padding,
                          bias=old_conv.bias is not None)
    with torch.no_grad():
        mean_w = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight.copy_(mean_w.repeat(1, n_in, 1, 1) * (old_conv.in_channels / n_in))
    return new_conv


def build_25d_model(n_slices: int, backbone: str = "resnet18", pretrained: bool = False) -> nn.Module:
    """Construct the Path B architecture. ``pretrained`` defaults to False at
    inference time: the trained checkpoint's ``state_dict`` fully overwrites
    the weights immediately after construction, so downloading ImageNet
    weights here would be wasted work and would require network access we
    don't have in the competition container.
    """
    weights = None
    model = tvm.resnet18(weights=weights) if backbone == "resnet18" else tvm.efficientnet_b0(weights=weights)

    if backbone == "resnet18":
        model.conv1 = _adapt_first_conv(model.conv1, n_slices)
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        old_conv = model.features[0][0]
        model.features[0][0] = _adapt_first_conv(old_conv, n_slices)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    return model


# ---------------------------------------------------------------------------
# Path C: compact 3D CNN
# ---------------------------------------------------------------------------


class ResBlock3D(nn.Module):
    """Basic 3D residual block: conv-BN-ReLU x2 + projection shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class Compact3DCNN(nn.Module):
    """Small 3D ResNet for 64^3 single-channel input: stem + 4 downsampling
    residual stages (64->32->16->8->4) + global average pool + dropout + 1 logit."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, dropout: float = 0.3):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResBlock3D(c, c * 2, stride=2)
        self.layer2 = ResBlock3D(c * 2, c * 4, stride=2)
        self.layer3 = ResBlock3D(c * 4, c * 8, stride=2)
        self.layer4 = ResBlock3D(c * 8, c * 8, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c * 8, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))


# ---------------------------------------------------------------------------
# Ensemble combination (only the branches actually used by the trained
# manifest need to be exercised, but all are implemented for completeness /
# future model updates)
# ---------------------------------------------------------------------------


def predict_logreg_stack(model, probs: np.ndarray) -> np.ndarray:
    """Apply a fitted logistic-regression stacker to base-model probabilities
    (operates on logits of the probabilities, matching how it was fit)."""
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    x = np.log(p / (1 - p))
    return model.predict_proba(x)[:, 1]


def combine_ensemble(ensemble_type: str, prob_vec: np.ndarray, ensemble_obj: dict) -> float:
    """``prob_vec`` is a (1, n_base_models) array ordered per
    ``config.BASE_MODEL_COLS``. Mirrors section 12 of the training notebook."""
    if ensemble_type == "equal_weight":
        return float(prob_vec.mean())
    if ensemble_type == "logit_average":
        p = np.clip(prob_vec, 1e-7, 1 - 1e-7)
        lg = np.log(p / (1 - p))
        return float(1 / (1 + np.exp(-lg.mean())))
    if ensemble_type == "weighted_average":
        return float((prob_vec @ np.array(ensemble_obj["weights"]))[0])
    if ensemble_type == "logistic_stacker":
        return float(predict_logreg_stack(ensemble_obj["model"], prob_vec)[0])
    raise ValueError(f"Unknown ensemble type: {ensemble_type!r}")
