"""Counter-based random numbers for the batched world (#132, interfaces.md).

Every draw is a pure function of (episode seed, stream, counter), computed with a SplitMix64
finaliser in int64 torch ops: no generator state, no host sync, and no dependence on which
other rows reset or what the agent did. Paired comparisons therefore share the same world.

torch's int64 multiply wraps like uint64 does, and ``>>`` is arithmetic on negative values, so
the shifts mask off the sign-extended bits to act as logical shifts.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_GOLDEN = -7046029254386353131  # 0x9E3779B97F4A7C15 as int64
_MIX1 = -4658895280553007687  # 0xBF58476D1CE4E5B9
_MIX2 = -7723592293110705685  # 0x94D049BB133111EB
_MASK = (1 << 64) - 1


def _shr(x: Tensor, bits: int) -> Tensor:
    return (x >> bits) & ((1 << (64 - bits)) - 1)


def mix(x: Tensor) -> Tensor:
    """SplitMix64 finaliser on int64 (bijective, good avalanche)."""
    x = x ^ _shr(x, 30)
    x = x * _MIX1
    x = x ^ _shr(x, 27)
    x = x * _MIX2
    return x ^ _shr(x, 31)


def _as_int64(value: int) -> int:
    value &= _MASK
    return value - (1 << 64) if value >= 1 << 63 else value


def key(*parts: Tensor | int) -> Tensor:
    """Hash several integers (tensors broadcast) into one int64 key."""
    out: Tensor | None = None
    for part in parts:
        value = part if isinstance(part, Tensor) else torch.tensor(_as_int64(int(part)))
        value = value.to(torch.int64)
        if out is None:
            out = mix(value + _GOLDEN)
        else:
            out = mix(out ^ (value + _GOLDEN))
    assert out is not None
    return out


def uniform(k: Tensor, dtype: torch.dtype = torch.float64) -> Tensor:
    """[0, 1) from the top 53 bits of a key."""
    return _shr(k, 11).to(dtype) * (1.0 / (1 << 53))


def normal(k: Tensor, dtype: torch.dtype = torch.float64) -> Tensor:
    """Standard normal by Box-Muller from two keys derived from k."""
    u1 = uniform(mix(k ^ 0x1), torch.float64)
    u2 = uniform(mix(k ^ 0x2), torch.float64)
    radius = torch.sqrt(-2.0 * torch.log1p(-u1))  # 1 - u1 is in (0, 1]
    return (radius * torch.cos(2 * math.pi * u2)).to(dtype)


def episode_seed(seed: int, env_index: Tensor, episode_index: Tensor) -> Tensor:
    """Nonnegative int64 seed of episode k of environment i (interfaces.md)."""
    return key(seed, env_index, episode_index) & ((1 << 62) - 1)


def draw(
    seeds: Tensor,
    stream: int,
    counter: Tensor,
    sub: Tensor | int,
    gaussian: bool,
    fast: bool = False,
) -> Tensor:
    """uniform or normal of key(seed, stream, counter, sub): the draw every world stream uses.

    ``counter`` has the batch axis first ([B,...] or [1,...]); ``sub`` broadcasts against it,
    and ``seeds`` [B] gets trailing axes to match. ``fast`` lets CUDA compute a normal in
    float32 maths (for bulk draws such as image noise). On CUDA this runs as one fused kernel
    (kernels.draw); elsewhere as torch int64 ops. The keys are identical; the float maths may
    differ in the last bit.
    """
    if not isinstance(sub, Tensor):
        sub = torch.full((), sub, dtype=torch.int64, device=seeds.device)  # no host copy
    shape = torch.broadcast_shapes(counter.shape, sub.shape)
    seeds = seeds.reshape(seeds.shape + (1,) * (len(shape) - 1))
    if seeds.is_cuda:
        from flytrack_v2.world import kernels

        return kernels.draw(seeds, stream, counter, sub, gaussian, fast)
    k = key(seeds, stream, counter, sub)
    return normal(k) if gaussian else uniform(k)
