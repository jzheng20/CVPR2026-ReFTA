"""Minimal ViT example: apply ReFTA and run one training step.

    python examples/vit_example.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import ViTForImageClassification, ViTConfig
from refta import apply_refta, mark_only_refta_trainable

# A small randomly-initialised ViT keeps the example dependency-free; swap in
# `ViTForImageClassification.from_pretrained("google/vit-large-patch16-224-in21k", ...)`
# for a real run (then use num_layers=24, hidden_dim=1024).
cfg = ViTConfig(hidden_size=384, num_hidden_layers=6, num_attention_heads=6,
                intermediate_size=1536, image_size=224, patch_size=16, num_labels=10)
model = ViTForImageClassification(cfg)

apply_refta(model, model_name="vit", num_layers=cfg.num_hidden_layers,
            hidden_dim=cfg.hidden_size, rank=8, transform="DCT")
mark_only_refta_trainable(model, also_train=["classifier"])

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
px = torch.randn(4, 3, 224, 224)
lbl = torch.randint(0, 10, (4,))
for step in range(3):
    opt.zero_grad()
    out = model(pixel_values=px, labels=lbl)
    out.loss.backward()
    opt.step()
    print(f"step {step}  loss={out.loss.item():.4f}")
print("OK")
