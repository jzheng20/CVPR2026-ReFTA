"""Correctness test for ReFTA's reconstruction-free adapter.

Checks that ReFTA_Combined's fused, reconstruction-free forward (used in training)
is numerically identical -- to floating-point precision -- to the explicit
per-slice definition  out_k = sum_j U[j,k] * ([x|1] @ B_j^T @ A_j^T),  for ragged
per-slice ranks (including empty slices). Run:  python tests/test_equivalence.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from refta.ReFTAModel import ReFTA_Combined


def reference(AAs, BBs, U, x, col):
    """Explicit per-slice reconstruction (the definition ReFTA optimises)."""
    ones = torch.ones(*x.shape[:-1], 1, dtype=x.dtype)
    newx = torch.cat((x, ones), dim=-1)
    ind = [i for i, A in enumerate(AAs) if A is not None]
    out = 0.0
    for j in ind:
        yj = newx @ BBs[j].t() @ AAs[j].t()    # (..., n)
        out = out + U[j, col] * yj
    return out


def main():
    torch.manual_seed(0)
    K, n, d = 12, 32, 24
    ranks = [torch.randint(0, 3, (1,)).item() for _ in range(K)]
    AAs = [torch.randn(n, r).double() if r > 0 else None for r in ranks]
    BBs = [torch.randn(r, d + 1).double() if r > 0 else None for r in ranks]
    U = torch.randn(K, K).double()

    adapter = ReFTA_Combined(AAs, BBs, U, K, lenU=False).double()
    x = torch.randn(4, 7, d).double()

    max_diff = 0.0
    for col in range(K):
        got = adapter(x, col)[..., 0]
        ref = reference(AAs, BBs, U, x, col)
        max_diff = max(max_diff, (got - ref).abs().max().item())
    print(f"max |reconstruction-free - explicit| over {K} columns = {max_diff:.2e}")
    assert max_diff < 1e-10, "ReFTA adapter does not match its definition!"
    print("PASSED")


if __name__ == "__main__":
    main()
