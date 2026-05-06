"""
Feature-analysis utilities: extract features from a backbone, run kNN or
linear probes on them, and compute patch-level PCA visualizations.

These utilities are used both by the feature-exploration notebook (Chapter 3
of the report) and by the evaluation callbacks in the training script.

All functions are deliberately backbone-agnostic: they take a callable
``feature_extractor(x) -> Tensor`` so they work with our custom ViT, with
HuggingFace DINOv2 models, and with timm backbones.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    feature_extractor: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    l2_normalize: bool = True,
    use_amp: bool = True,
    desc: str = "Extracting features",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract (features, labels) from a DataLoader.

    Parameters
    ----------
    feature_extractor : Callable
        Any callable that maps an input batch ``(B, 3, H, W)`` to a feature
        tensor ``(B, d)``.
    loader : DataLoader
        Must yield ``(images, labels)`` tuples.
    device : torch.device
        Device to run the extractor on.
    l2_normalize : bool
        If True, L2-normalize features along the last dim (convention for
        cosine-similarity kNN).
    use_amp : bool
        Use mixed-precision autocast during extraction (faster, barely-detectable
        precision loss for kNN).
    desc : str
        Tqdm progress description.

    Returns
    -------
    features : torch.Tensor, shape (N, d), on CPU
    labels   : torch.Tensor, shape (N,), on CPU, dtype long
    """
    feats, labels = [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        imgs, y = batch
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            z = feature_extractor(imgs)
        if l2_normalize:
            z = F.normalize(z.float(), p=2, dim=-1)
        feats.append(z.detach().cpu())
        labels.append(y.detach().cpu().long())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


# -----------------------------------------------------------------------------
# kNN classification
# -----------------------------------------------------------------------------

@torch.no_grad()
def knn_classify(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    k: int = 20,
    temperature: float = 0.07,
    chunk_size: int = 1024,
) -> dict:
    """Weighted kNN classification, following the DINO linear/kNN protocol.

    The predicted class score for a test sample is
    ``sum_{i in top-k} w_i * onehot(y_i)`` with
    ``w_i = exp(cos_sim(test, train_i) / T)`` and ``T = temperature``. The
    argmax gives the prediction. Features are assumed L2-normalized.

    Returns a dict with ``top1``, ``top5`` accuracies.
    """
    train_features = train_features.float()
    test_features = test_features.float()
    train_labels = train_labels.long()
    test_labels = test_labels.long()

    top1 = top5 = total = 0
    # One-hot for efficient weighted vote
    retrieval_one_hot = torch.zeros(k, num_classes)

    for start in range(0, len(test_features), chunk_size):
        end = min(start + chunk_size, len(test_features))
        feats = test_features[start:end]                       # (b, d)
        targets = test_labels[start:end]                       # (b,)

        sim = feats @ train_features.T                         # cosine sim, (b, N)
        distances, indices = sim.topk(k, dim=1)                # top-k similar train feats
        candidates = train_labels[indices]                     # (b, k)

        # Weighted vote
        b = feats.size(0)
        retrieval_one_hot = retrieval_one_hot.new_zeros(b * k, num_classes)
        retrieval_one_hot.scatter_(1, candidates.view(-1, 1), 1)
        weights = (distances / temperature).exp().view(b, k, 1)
        probs = (retrieval_one_hot.view(b, k, num_classes) * weights).sum(dim=1)  # (b, C)

        preds = probs.argsort(dim=1, descending=True)
        correct = preds.eq(targets.view(-1, 1))
        top1 += correct[:, :1].sum().item()
        top5 += correct[:, :5].sum().item()
        total += b

    return {
        "top1": 100.0 * top1 / total,
        "top5": 100.0 * top5 / total,
        "k": k,
        "temperature": temperature,
        "n_test": total,
    }


# -----------------------------------------------------------------------------
# Linear probe
# -----------------------------------------------------------------------------

def linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
) -> dict:
    """Train a single linear classifier on frozen features and report val accuracy.

    Features are expected already L2-normalized.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dim = train_features.size(-1)

    classifier = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    N = train_features.size(0)
    for epoch in range(epochs):
        classifier.train()
        perm = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            x, y = train_features[idx], train_labels[idx]
            logits = classifier(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

    classifier.eval()
    with torch.no_grad():
        logits = classifier(test_features)
        preds = logits.argsort(dim=1, descending=True)
        correct = preds.eq(test_labels.view(-1, 1))
        top1 = 100.0 * correct[:, :1].sum().item() / test_features.size(0)
        top5 = 100.0 * correct[:, :5].sum().item() / test_features.size(0)
    return {"top1": top1, "top5": top5, "epochs": epochs, "lr": lr}


# -----------------------------------------------------------------------------
# Patch-level PCA visualization
# -----------------------------------------------------------------------------

@torch.no_grad()
def patch_pca(
    patch_features: torch.Tensor,
    n_components: int = 3,
    foreground_mask: Optional[torch.Tensor] = None,
    normalize_to_unit_range: bool = True,
) -> np.ndarray:
    """Run PCA on a single image's patch features and return an RGB visualization.

    Parameters
    ----------
    patch_features : Tensor, shape (H_p, W_p, d)
        Per-patch features (spatial-reshaped). ``H_p = W_p = sqrt(N_patches)``.
    n_components : int
        Number of PCA components to keep (usually 3 for RGB).
    foreground_mask : Tensor or None
        Optional (H_p, W_p) boolean mask. If given, PCA is fit only on
        foreground patches (background patches are rendered black).
    normalize_to_unit_range : bool
        If True, scale each channel of the output to [0, 1] for visualization.

    Returns
    -------
    rgb : np.ndarray of shape (H_p, W_p, n_components), dtype float32
    """
    Hp, Wp, d = patch_features.shape
    feats = patch_features.reshape(-1, d).float()                # (N, d)

    if foreground_mask is not None:
        mask_flat = foreground_mask.reshape(-1).bool()
        fit_feats = feats[mask_flat]
    else:
        fit_feats = feats

    # Center and SVD
    fit_feats = fit_feats - fit_feats.mean(dim=0, keepdim=True)
    # Use reduced SVD for numerical stability with small sample sizes
    U, S, Vh = torch.linalg.svd(fit_feats, full_matrices=False)
    # Project all patches onto top-n components
    components = Vh[:n_components]                                # (k, d)
    projected = (feats - feats.mean(dim=0, keepdim=True)) @ components.T  # (N, k)

    projected = projected.reshape(Hp, Wp, n_components)

    if foreground_mask is not None:
        projected = projected * foreground_mask.unsqueeze(-1).float()

    rgb = projected.cpu().numpy()
    if normalize_to_unit_range:
        # Per-channel min-max to [0, 1]
        mn = rgb.reshape(-1, n_components).min(axis=0)
        mx = rgb.reshape(-1, n_components).max(axis=0)
        rng = np.maximum(mx - mn, 1e-8)
        rgb = (rgb - mn) / rng
    return rgb.astype(np.float32)
