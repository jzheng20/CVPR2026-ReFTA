# ReFTA: Reconstruction-Free Tensor-Based Adaptation

Official code for **ReFTA**, a parameter-efficient fine-tuning (PEFT) method built on
the tensor SVD (t-SVD). ReFTA stacks the per-layer weight matrices (e.g. the query /
value projections across all attention layers) into a 3rd-order tensor, keeps only the
**principal tensor components**, and fine-tunes them in a **reconstruction-free** way.

Compared with matrix low-rank methods (LoRA, PiSSA) and other tensor methods
(LoRETTA, Tucker/TT), ReFTA offers:

- **Extreme parameter efficiency** — a single tensor-rank `R` controls the whole model
  (e.g. on `ViT-Large`, `R=15` ≈ 61K trainable params, ~16× fewer than LoRA `r=16`).
- **Lower initialization quantization error** — only principal components are tuned.
- **A single rank hyperparameter** — no per-mode rank tuning.
- **Reconstruction-free training** — the mode-3 invertible transform `U` is folded into
  a *shared* low-rank projection plus a per-layer diagonal reweighting, so the weight
  tensor is never reconstructed. The result runs at **LoRA-level speed and memory**
  while keeping ReFTA's parameter/accuracy advantage.

## Install

```bash
git clone https://github.com/jzheng20/ReFTA.git
cd ReFTA
pip install -r requirements.txt          # torch, transformers, numpy, scipy, tqdm
```

The `refta` package is import-only (no build step). Add the repo root to your
`PYTHONPATH`, or `pip install -e .` if you add a `pyproject.toml`.

## Quick start

### Vision (ViT)

```python
from transformers import ViTForImageClassification
from refta import apply_refta, mark_only_refta_trainable

model = ViTForImageClassification.from_pretrained(
    "google/vit-large-patch16-224-in21k", num_labels=37, ignore_mismatched_sizes=True)

# Insert ReFTA adapters on the query & value projections (24 layers, dim 1024).
apply_refta(model, model_name="vit", num_layers=24, hidden_dim=1024,
            rank=15, transform="DCT")
mark_only_refta_trainable(model, also_train=["classifier"])   # -> ~61K trainable

# ...then train `model` with your usual HuggingFace / PyTorch loop.
```

### Language (RoBERTa)

```python
from transformers import RobertaForSequenceClassification
from refta import apply_refta, mark_only_refta_trainable

model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=2)
apply_refta(model, model_name="roberta", num_layers=24, hidden_dim=1024,
            rank=5, transform="LSM-3")
mark_only_refta_trainable(model, also_train=["classifier"])
```

See [`examples/`](examples/) for runnable end-to-end scripts.

## API

`apply_refta(model, model_name, num_layers, hidden_dim, rank, target_modules=("query","value"), transform="DCT", train_transform=False, freeze_base=True, device=None)`
inserts the adapters in place. `transform` selects the invertible mode-3 transform:
`"DCT"`, `"LSM-3"`, `"HOSVD"`, `"DFT"`, or `"None"` (identity).

`mark_only_refta_trainable(model, also_train=("classifier",))` freezes everything except
the ReFTA factors (and any module whose name matches `also_train`, e.g. the task head).

## Correctness

`tests/test_equivalence.py` checks that the reconstruction-free forward used in training
is numerically identical (to floating-point precision) to the explicit per-slice form.

## Citation

If you use this code, please cite the ReFTA paper:

```bibtex
@inproceedings{refta,
  title     = {Reconstruction-Free Tensor-Based Adaptation},
  author    = {Zheng, Jingjing and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
<!-- TODO: update author list / venue details to match the final paper. -->
