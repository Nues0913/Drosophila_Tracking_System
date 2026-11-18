"""Int32 CSR SpMM with a bounded-memory first-order backward on CPU and CUDA."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


def _row_pointer(rows: np.ndarray, n: int) -> Tensor:
    lengths = np.bincount(rows, minlength=n)
    return torch.from_numpy(np.concatenate(([0], lengths.cumsum())).astype(np.int32))


def _matrix(crow: Tensor, col: Tensor, values: Tensor, n: int) -> Tensor:
    return torch.sparse_csr_tensor(crow, col, values, size=(n, n), check_invariants=False)


class _CSRMultiply(torch.autograd.Function):
    """Differentiate CSR SpMM with weight gradients in CSR (post, pre) values order."""

    @staticmethod
    def forward(ctx, matrix, values, rates, layout):
        ctx.layout = layout
        ctx.save_for_backward(values, rates)
        return torch.sparse.mm(matrix, rates)

    @staticmethod
    def backward(ctx, grad_output):
        values, rates = ctx.saved_tensors
        layout = ctx.layout
        grad_output = grad_output.contiguous()
        grad_values = grad_rates = None
        if ctx.needs_input_grad[1]:
            # CPU SDDMM widens CSR indices. CUDA SDDMM omits the half types SpMM accepts.
            if rates.is_cuda and rates.dtype not in (torch.float16, torch.bfloat16):
                grad_values = layout._sddmm_weight_grad(grad_output, rates)
            else:
                grad_values = layout._gather_weight_grad(values, grad_output, rates)
        if ctx.needs_input_grad[2]:
            transpose = _matrix(
                layout.transpose_crow,
                layout.transpose_columns,
                values.index_select(0, layout.transpose_order),
                layout.n_nodes,
            )
            grad_rates = torch.sparse.mm(transpose, grad_output)
        return None, grad_values, grad_rates, None


class CSRLayout(nn.Module):
    """Structure of W[post, pre] and its transpose, sorted once at construction."""

    def __init__(self, pre: np.ndarray, post: np.ndarray, n_nodes: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.order = np.lexsort((pre, post))
        pre, post = pre[self.order], post[self.order]
        transpose = np.lexsort((post, pre))
        for name, tensor in (
            ("crow", _row_pointer(post, n_nodes)),
            ("columns", torch.as_tensor(pre, dtype=torch.int32)),
            ("rows", torch.as_tensor(post, dtype=torch.int32)),
            ("transpose_crow", _row_pointer(pre[transpose], n_nodes)),
            ("transpose_columns", torch.as_tensor(post[transpose], dtype=torch.int32)),
            ("transpose_order", torch.as_tensor(transpose, dtype=torch.int32)),
        ):
            self.register_buffer(name, tensor, persistent=False)
        self.register_buffer("_pattern_values", torch.zeros(pre.size), persistent=False)

    def _sddmm_weight_grad(self, grad_output: Tensor, rates: Tensor) -> Tensor:
        """Return sampled dot products in the CSR pattern's (post, pre) values order."""
        pattern = _matrix(
            self.crow,
            self.columns,
            self._pattern_values.to(dtype=grad_output.dtype),
            self.n_nodes,
        )
        return torch.sparse.sampled_addmm(
            pattern, grad_output, rates.T.contiguous(), beta=0.0
        ).values()

    def _gather_weight_grad(self, values: Tensor, grad_output: Tensor, rates: Tensor) -> Tensor:
        # Each float32 gather holds about 64 MiB at the target size.
        chunk = max(2048, 2**24 // max(1, rates.size(1)))
        grad_values = torch.empty_like(values)
        for start in range(0, grad_values.numel(), chunk):
            end = start + chunk
            post_grad = grad_output.index_select(0, self.rows[start:end])
            pre_rates = rates.index_select(0, self.columns[start:end])
            grad_values[start:end] = (post_grad * pre_rates).sum(dim=1)
        return grad_values

    def matrix(self, values: Tensor) -> Tensor:
        return _matrix(self.crow, self.columns, values.detach(), self.n_nodes)

    def multiply(self, matrix: Tensor, values: Tensor, rates: Tensor) -> Tensor:
        return _CSRMultiply.apply(matrix, values, rates, self)
