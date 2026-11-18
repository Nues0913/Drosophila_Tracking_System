"""Sparse x dense product shared by the frozen front end and the central core.

The deterministic (gather plus segmented sum) CUDA path was removed on v2 (#131): v2 does not
require bitwise-reproducible reruns, and that mode made GPU evaluation about 2x slower.
"""

import torch
from torch import Tensor


def spmm(matrix: Tensor, dense: Tensor) -> Tensor:
    """``matrix`` [M, N] (sparse COO or CSR) times ``dense`` [N, K]."""
    return torch.sparse.mm(matrix, dense)
