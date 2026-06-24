"""ReFTA: Reconstruction-Free Tensor-Based Adaptation.

A parameter-efficient fine-tuning (PEFT) method built on the tensor SVD (t-SVD).
ReFTA stacks the per-layer weight matrices into a 3rd-order tensor, keeps only the
principal tensor components, and adapts them in a *reconstruction-free* way: the
mode-3 invertible transform (e.g. DCT / LSM-3) is folded into a shared low-rank
projection plus a per-layer diagonal reweighting, so no weight tensor is ever
reconstructed during training. See the CVPR 2026 paper for details.

Quick start
-----------
    from transformers import ViTForImageClassification
    from refta import apply_refta, mark_only_refta_trainable

    model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224-in21k")
    apply_refta(model, model_name="vit", num_layers=24, hidden_dim=1024, rank=15)
    mark_only_refta_trainable(model, also_train=["classifier"])
"""
import types
import numpy as np

from .ReFTAModel import (
    Loading,
    ReFTALinear,
    ReFTA_Combined,
    ReFTA_B1,
    ReFTA_B2,
    modified_Linearlayer,
)

__all__ = ["apply_refta", "mark_only_refta_trainable", "Loading"]


def apply_refta(model, model_name, num_layers, hidden_dim, rank,
                target_modules=("query", "value"), transform="DCT",
                train_transform=False, freeze_base=True, device=None):
    """Insert ReFTA adapters into a HuggingFace model, in place.

    Args:
        model:        a HF model (currently wired for ``"vit"`` and ``"roberta"``).
        model_name:   ``"vit"`` or ``"roberta"`` (selects the encoder path).
        num_layers:   number of stacked attention layers (e.g. 24 for *-large).
        hidden_dim:   q/v projection dimension (square; e.g. 1024 for *-large).
        rank:         total number of principal tensor components R to keep/train.
        target_modules: attention sub-modules to adapt (default query & value).
        transform:    invertible mode-3 transform: ``"DCT"``, ``"LSM-3"``,
                      ``"HOSVD"``, ``"DFT"`` or ``"None"`` (identity).
        train_transform: if True, the transform U is learned (adds an
                      orthogonality penalty term, see ``ReFTAloss``); default False.
        freeze_base:  freeze every base-model parameter before inserting adapters
                      (recommended). The adapter factors are added as trainable.
        device:       defaults to the model's current device.

    Returns:
        the same ``model`` with q/v projections replaced by ReFTA layers.
    """
    import torch
    if device is None:
        device = next(model.parameters()).device
    if freeze_base:
        for p in model.parameters():
            p.requires_grad_(False)
    args = types.SimpleNamespace(
        model_name=model_name,
        FoDs_size=np.array([num_layers, hidden_dim, hidden_dim]),
        target_layers=list(target_modules),
        lenU=train_transform,
        r=rank,
        transform_type=transform,
    )
    return Loading(args, model, device)


def mark_only_refta_trainable(model, also_train=("classifier",)):
    """Freeze everything except the ReFTA adapter factors and any module whose
    name contains one of ``also_train`` (e.g. the task head)."""
    for n, p in model.named_parameters():
        train = ("ReFTA_A" in n) or ("ReFTA_B" in n) or ("ReFTA_U" in n) \
            or any(t in n for t in also_train)
        p.requires_grad_(train)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ReFTA] trainable parameters: {n_train:,}")
    return model
