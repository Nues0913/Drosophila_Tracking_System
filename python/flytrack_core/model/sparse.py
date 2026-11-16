"""Sparse x dense products that stay bitwise reproducible on CUDA on request.

cuSPARSE (``torch.sparse.mm`` on CUDA) ignores
``torch.use_deterministic_algorithms`` without raising, so GPU evaluations differed
run to run in every episode (#3). Under deterministic mode the CUDA product is a
gather plus a segmented row sum, which uses no atomics. It fixes
results, it does not make them more accurate: deterministic GPU, default GPU and
CPU still differ from one another at float32 rounding level.
"""

import os

import torch
from torch import Tensor


def enable_deterministic() -> None:
    """Deterministic kernels, a fixed cuBLAS workspace and no cuSPARSE (see spmm).

    Meant for evaluation and regression checks (about 2x slower on GPU); call it
    before the first CUDA matrix product so cuBLAS picks up the workspace setting.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def spmm(matrix: Tensor, dense: Tensor) -> Tensor:
    """``matrix`` [M, N] (sparse COO or CSR) times ``dense`` [N, K]."""
    if not (dense.is_cuda and torch.are_deterministic_algorithms_enabled()):
        return torch.sparse.mm(matrix, dense)
    if matrix.layout == torch.sparse_csr:
        lengths = matrix.crow_indices().diff()
        columns = matrix.col_indices()
    else:
        matrix = matrix.coalesce()  # row-major order, like CSR
        rows, columns = matrix.indices()
        lengths = torch.bincount(rows, minlength=matrix.shape[0])
    # index_select gathers the same rows as dense[columns] (bitwise), and takes int32 indices
    # without first converting them to int64 (the frontend's CSR is int32, #35 G3 latency).
    product = dense.index_select(0, columns) * matrix.values().unsqueeze(1).to(dense.dtype)
    # Each row's entries are contiguous, so a segmented sum replaces scattered adds
    # (index_add_ under deterministic mode was about 2x slower on the frontend).
    return torch.segment_reduce(product, "sum", lengths=lengths, axis=0, unsafe=True)
