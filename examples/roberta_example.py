"""Minimal RoBERTa example: apply ReFTA and run one training step.

    python examples/roberta_example.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import RobertaForSequenceClassification, RobertaConfig
from refta import apply_refta, mark_only_refta_trainable

# Small randomly-initialised RoBERTa for a dependency-free demo; for a real run use
# `RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=2)`
# with num_layers=24, hidden_dim=1024.
cfg = RobertaConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=4,
                    intermediate_size=512, max_position_embeddings=64,
                    vocab_size=1000, num_labels=2)
model = RobertaForSequenceClassification(cfg)

apply_refta(model, model_name="roberta", num_layers=cfg.num_hidden_layers,
            hidden_dim=cfg.hidden_size, rank=5, transform="LSM-3")
mark_only_refta_trainable(model, also_train=["classifier"])

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
ids = torch.randint(0, 1000, (8, 16))
lbl = torch.randint(0, 2, (8,))
for step in range(3):
    opt.zero_grad()
    out = model(input_ids=ids, labels=lbl)
    out.loss.backward()
    opt.step()
    print(f"step {step}  loss={out.loss.item():.4f}")
print("OK")
