"""Fused CUDA kernels for the v1compat world's hot loops (#132).

Each kernel is one jiterator function (NVRTC, no host compiler) that does per element what
the eager torch code in v1compat.py does with many small ops. The eager code stays the
reference: tests run both on CUDA and compare them.

The plume kernel draws its turbulence with the same SplitMix64 key chain as rng.py, so it
reproduces HashDraws.turbulence: the row key (seed, stream, tick) comes in as two exact 32-bit
halves in doubles, and the kernel continues the chain with the puff's emission number and the
axis.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# jiterator's parser wants the code to open with a template; the last function is the entry.
_HASH = """
template <typename T> T ft_identity(T x) { return x; }
typedef unsigned long long u64;
__device__ __forceinline__ u64 ft_mix(u64 x) {
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 27; x *= 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
__device__ __forceinline__ u64 ft_next(u64 k, long long part) {
  return ft_mix(k ^ ((u64)part + 0x9E3779B97F4A7C15ULL));
}
__device__ __forceinline__ double ft_uniform(u64 k) {
  return (double)(k >> 11) * (1.0 / 9007199254740992.0);
}
__device__ __forceinline__ double ft_normal(u64 k) {
  double u1 = ft_uniform(ft_mix(k ^ 1ULL));
  double u2 = ft_uniform(ft_mix(k ^ 2ULL));
  return sqrt(-2.0 * log1p(-u1)) * cos(6.283185307179586 * u2);
}
// The same normal in float32 maths (plume turbulence: sigma is 0.017 m, so float suffices).
__device__ __forceinline__ double ft_normal_f(u64 k) {
  float u1 = (float)ft_uniform(ft_mix(k ^ 1ULL));
  float u2 = (float)ft_uniform(ft_mix(k ^ 2ULL));
  return (double)(sqrtf(-2.0f * log1pf(-u1)) * cosf(6.2831853f * u2));
}
__device__ __forceinline__ u64 ft_seed(double high, double low) {
  return ((u64)(long long)high << 32) | (u64)(long long)low;
}
// rng.key(seed, a, b, c): mix(seed + G), then mix(k ^ (part + G)) for each further part.
__device__ __forceinline__ u64 ft_key4(u64 seed, long long a, long long b, long long c) {
  return ft_next(ft_next(ft_next(ft_mix(seed + 0x9E3779B97F4A7C15ULL), a), b), c);
}
__device__ __forceinline__ double ft_wrap(double a) {
  const double two_pi = 6.283185307179586;
  double m = fmod(a + 3.141592653589793, two_pi);
  if (m != 0.0 && (m < 0.0) != (two_pi < 0.0)) m += two_pi;
  return m - 3.141592653589793;
}
"""

_PLUME = (
    _HASH
    + """
template <typename T> void ft_plume_tick(T x, T y, T age, T seq, T tick, T seed_hi, T seed_lo,
    T tick_index, T dt, T wind_x, T wind_y, T sigma, T lifetime, T& ox, T& oy, T& oage, T& oseq) {
  ox = x; oy = y; oage = age; oseq = seq;
  if (tick != T(0) && seq >= T(0)) {
    oage = age + dt;
    ox = x + wind_x * dt;
    oy = y + wind_y * dt;
    if (sigma != T(0)) {
      // rng.key(rng.key(seed, stream, tick, seq), axis): a new chain starts with mix(k + G).
      u64 key = ft_key4(ft_seed(seed_hi, seed_lo), PLUME_STREAM, (long long)tick_index,
                        (long long)seq);
      u64 k = ft_mix(key + 0x9E3779B97F4A7C15ULL);
      ox = ox + sigma * ft_normal_f(ft_next(k, 0));
      oy = oy + sigma * ft_normal_f(ft_next(k, 1));
    }
    if (oage > lifetime) oseq = T(-1);
  }
}
"""
)

_DRAW = (
    _HASH
    + """
template <typename T> T ft_draw(T seed_hi, T seed_lo, T stream, T counter, T sub, T kind) {
  u64 k = ft_key4(ft_seed(seed_hi, seed_lo), (long long)stream, (long long)counter,
                  (long long)sub);
  // kind 0: uniform, 1: normal, 2: normal in float32 maths (image noise, [B,2,721] per step).
  return kind == T(2) ? (T)ft_normal_f(k) : (kind != T(0) ? ft_normal(k) : ft_uniform(k));
}
"""
)

_CLOCK = """
template <typename T> void ft_clock(T t, T nv, T no, T end, T neural_dt, T vision_dt,
    T odor_dt, T& ot, T& onv, T& ono, T& live, T& vision, T& odor) {
  bool on = t < end - T(1e-10);
  T nxt = fmin(fmin(end, t + neural_dt), fmin(nv, no));
  ot = on ? nxt : t;
  bool v = on && nv <= ot + T(1e-9);
  bool o = on && no <= ot + T(1e-9);
  onv = v ? nv + vision_dt : nv;
  ono = o ? no + odor_dt : no;
  live = on ? T(1) : T(0);
  vision = v ? T(1) : T(0);
  odor = o ? T(1) : T(0);
}
"""

_MOTION = (
    _HASH
    + """
template <typename T> void ft_motion(T x, T y, T yaw, T t0, T t1, T v, T omega, T ratio,
    T& ox, T& oy, T& oyaw) {
  ox = x; oy = y; oyaw = yaw;
  T dt = t1 - t0;
  if (!(dt > T(0))) return;
  // v1's numpy: omega * dt and dt * v are float32 products (float32 scalar x Python float).
  float dt32 = (float)dt;
  double turn = (double)((float)omega * dt32);
  double travel = (double)(dt32 * (float)v);
  bool straight = fabsf((float)omega) < 1e-8f;
  double end_yaw = yaw + turn;
  ox = x + (straight ? travel * cos(yaw) : ratio * (sin(end_yaw) - sin(yaw)));
  oy = y + (straight ? travel * sin(yaw) : ratio * (cos(yaw) - cos(end_yaw)));
  oyaw = ft_wrap(yaw + turn);
}
"""
)

# float32 maths: the result is cast to float32 like v1's, and FP64 exp runs at 1/64 rate on
# consumer GPUs. The eager float64 path is the reference (tests compare within 1e-5).
_CONCENTRATION = """
template <typename T> T ft_puff_term(T px, T py, T x, T y, T age, T seq,
    T sigma0_sq, T diffusion, T mass, T decay_tau) {
  if (seq < T(0)) return T(0);
  float variance = (float)(sigma0_sq + T(2) * diffusion * age);
  float density = (float)mass * expf(-(float)(age / decay_tau)) / (6.2831853f * variance);
  float dx = (float)(px - x);
  float dy = (float)(py - y);
  return (T)(expf(-(dx * dx + dy * dy) / (2.0f * variance)) * density);
}
"""

_WANDER = (
    _HASH
    + """
template <typename T> void ft_wander(T x, T y, T heading, T rate, T draw,
    T decay, T noise, T max_rate, T margin, T gain, T xmin, T ymin, T xmax, T ymax,
    T speed, T dt, T& ox, T& oy, T& oh, T& orate) {
  T cx = (xmin + xmax) / T(2);
  T cy = (ymin + ymax) / T(2);
  T r = decay * rate + noise * draw;
  r = r < -max_rate ? -max_rate : (r > max_rate ? max_rate : r);
  T wall = fmin(fmin(x - xmin, xmax - x), fmin(y - ymin, ymax - y));
  T proximity = (margin - wall) / margin;
  proximity = proximity < T(0) ? T(0) : (proximity > T(1) ? T(1) : proximity);
  T inward = ft_wrap(atan2(cy - y, cx - x) - heading);
  T turn = r + gain * proximity * inward;
  turn = turn < -max_rate ? -max_rate : (turn > max_rate ? max_rate : turn);
  T h = ft_wrap(heading + turn * dt);
  T nx = x + speed * cos(h) * dt;
  T ny = y + speed * sin(h) * dt;
  bool outside = nx < xmin || nx > xmax || ny < ymin || ny > ymax;
  nx = nx < xmin ? xmin : (nx > xmax ? xmax : nx);
  ny = ny < ymin ? ymin : (ny > ymax ? ymax : ny);
  if (outside) h = atan2(cy - ny, cx - nx);
  ox = nx; oy = ny; oh = h; orate = r;
}
"""
)

_CACHE: dict[str, object] = {}


def _get(name: str):
    if name not in _CACHE:
        from torch.cuda import jiterator

        if name == "plume":
            from flytrack_v2.world.draws import PLUME_STREAM

            code = _PLUME.replace("PLUME_STREAM", f"{PLUME_STREAM}LL")
            _CACHE[name] = jiterator._create_multi_output_jit_fn(
                code, num_outputs=4, dt=0.0, wind_x=0.0, wind_y=0.0, sigma=0.0, lifetime=0.0
            )
        elif name == "clock":
            _CACHE[name] = jiterator._create_multi_output_jit_fn(
                _CLOCK, num_outputs=6, neural_dt=0.0, vision_dt=0.0, odor_dt=0.0
            )
        elif name == "motion":
            _CACHE[name] = jiterator._create_multi_output_jit_fn(_MOTION, num_outputs=3)
        elif name == "draw":
            _CACHE[name] = jiterator._create_jit_fn(_DRAW)
        elif name == "concentration":
            _CACHE[name] = jiterator._create_jit_fn(
                _CONCENTRATION, sigma0_sq=0.0, diffusion=0.0, mass=0.0, decay_tau=0.0
            )
        elif name == "wander":
            scalars = ("decay", "noise", "max_rate", "margin", "gain", "xmin", "ymin")
            scalars += ("xmax", "ymax", "speed", "dt")
            _CACHE[name] = jiterator._create_multi_output_jit_fn(
                _WANDER, num_outputs=4, **{key: 0.0 for key in scalars}
            )
    return _CACHE[name]


def split_key(key: Tensor) -> tuple[Tensor, Tensor]:
    """An int64 key as two exact float64 halves (high and low 32 bits)."""
    high = (key >> 32) & 0xFFFFFFFF
    return high.to(torch.float64), (key & 0xFFFFFFFF).to(torch.float64)


def plume_tick(x, y, age, seq, tick, seeds, tick_index, plume) -> tuple[Tensor, ...]:
    """Age, advect, diffuse and expire the puffs of rows in ``tick`` ([B,C] each).

    The turbulence is HashDraws.turbulence: key(seed, PLUME_STREAM, tick, puff number), then
    one sub-key per axis.
    """
    high, low = split_key(seeds)
    sigma = math.sqrt(2 * plume.turbulent_diffusion_m2_s * plume.integration_dt)
    ox, oy, oage, oseq = _get("plume")(
        x, y, age, seq.to(torch.float64), tick.to(torch.float64)[:, None],
        high[:, None], low[:, None], tick_index.to(torch.float64)[:, None],
        dt=plume.integration_dt, wind_x=plume.wind_x, wind_y=plume.wind_y,
        sigma=sigma, lifetime=plume.lifetime_s,
    )  # fmt: skip
    return ox, oy, oage, oseq.to(torch.int64)


def draw(
    seeds: Tensor, stream: int, counter: Tensor, sub: Tensor, gaussian: bool, fast: bool = False
) -> Tensor:
    """rng.draw in one kernel; ``seeds`` already has trailing axes to broadcast. ``fast``
    computes a normal in float32 maths (same key, last bits differ)."""
    high, low = split_key(seeds)
    return _get("draw")(
        high, low, torch.full_like(high, float(stream)), counter.to(torch.float64),
        sub.to(torch.float64), torch.full_like(high, (2.0 if fast else 1.0) if gaussian else 0.0),
    )  # fmt: skip


def concentration(points: Tensor, x, y, age, seq, plume) -> Tensor:
    """[B,2] float32 concentration at [B,2,2] sensor points."""
    terms = _get("concentration")(
        points[..., 0, None], points[..., 1, None], x[:, None], y[:, None], age[:, None],
        seq.to(torch.float64)[:, None],
        sigma0_sq=plume.initial_sigma_m**2, diffusion=plume.diffusion_m2_s,
        mass=plume.puff_mass, decay_tau=plume.decay_tau_s,
    )  # fmt: skip
    return (plume.background + terms.sum(-1)).to(torch.float32)


def wander_step(x, y, heading, rate, draw, config, dt: float) -> tuple[Tensor, ...]:
    c = config
    decay = math.exp(-dt / c.wander_turn_tau_s)
    return _get("wander")(
        x, y, heading, rate, draw,
        decay=decay, noise=c.wander_turn_rate_std * math.sqrt(1 - decay**2),
        max_rate=c.wander_max_turn_rate, margin=c.wander_wall_margin_m, gain=c.wander_wall_gain,
        xmin=c.xmin, ymin=c.ymin, xmax=c.xmax, ymax=c.ymax, speed=c.speed_m_s, dt=dt,
    )  # fmt: skip


def clock(t, nv, no, end, contract) -> tuple[Tensor, ...]:
    """One substep of v1's sensor clock: (t, next vision, next odor, live, vision, odor)."""
    t, nv, no, live, vision, odor = _get("clock")(
        t, nv, no, end, neural_dt=contract.neural_dt,
        vision_dt=1 / contract.vision_hz, odor_dt=1 / contract.odor_hz,
    )  # fmt: skip
    return t, nv, no, live > 0, vision > 0, odor > 0


def motion(pose: Tensor, t0: Tensor, t1: Tensor, applied: Tensor, ratio: Tensor) -> Tensor:
    """The agent pose after moving from t0 to t1 with the applied [B,2] action."""
    action = applied.to(torch.float64)
    x, y, yaw = _get("motion")(
        pose[:, 0], pose[:, 1], pose[:, 2], t0, t1, action[:, 0], action[:, 1], ratio
    )
    return torch.stack((x, y, yaw), -1)
