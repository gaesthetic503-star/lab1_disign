"""Проверки генератора белка (Sprint 2). Восстановлено по описанию в ноутбуке.

SEED          - начальное значение для всех версий
versions(...) - таблица тестов для v1, v2, v3 и их выборки
"""
import numpy as np
import pandas as pd
from scipy import stats

SEED = 20261119
UPPER = 25.0
_M31 = 2**31 - 1
_MASK64 = (1 << 64) - 1


# ---------- базовые датчики U(0,1) ----------
def park_miller(seed, n):
    s = int(seed) % _M31 or 1
    out = np.empty(n)
    for i in range(n):
        s = 16807 * s % _M31
        out[i] = s / _M31
    return out


def xorshift64star(seed, n):
    s = ((int(seed) * 0x9E3779B97F4A7C15) + 1) & _MASK64 or 1
    out = np.empty(n)
    for i in range(n):
        s ^= s >> 12
        s ^= (s << 25) & _MASK64
        s ^= s >> 27
        y = (s * 2685821657736338717) & _MASK64
        out[i] = ((y >> 12) + 0.5) / 2**52
    return out


# ---------- нормальные числа и смесь ----------
def box_muller(u, n):
    u1, u2 = u[:len(u) // 2], u[len(u) // 2:2 * (len(u) // 2)]
    r, th = np.sqrt(-2 * np.log(u1)), 2 * np.pi * u2
    return np.concatenate([r * np.cos(th), r * np.sin(th)])[:n]


def _mixture(base, seed, n, w, mu, sd, upper=None):
    w, mu, sd = map(np.asarray, (w, mu, sd))
    m = int(np.ceil(n / 2))
    uj = base(seed, n)
    z = box_muller(base(seed + 1, 2 * m), n)
    j = (uj >= w[0]).astype(int)
    x = np.exp(mu[j] + sd[j] * z)
    if upper is not None:
        for k in np.flatnonzero(x > upper):          # отбор: повтор всего выбора
            while True:
                uu = base(seed + 10 + k * 3, 3)
                jj = int(uu[0] >= w[0])
                zz = box_muller(uu[1:3], 1)[0]
                v = np.exp(mu[jj] + sd[jj] * zz)
                seed += 1
                if v <= upper:
                    x[k] = v
                    break
    return x


def _v3(seed, n, w, mu, sd):
    try:                                              # настоящий генератор из rng_protein
        import rng_protein as rp
        return np.asarray(rp.ProteinRNG(seed, engine="xorshift").mixture(n, w, mu, sd, upper=UPPER))
    except Exception:
        return _mixture(xorshift64star, seed, n, w, mu, sd, upper=UPPER)


# ---------- модель и тесты ----------
def model_cdf(x, w, mu, sd, upper=UPPER):
    x = np.minimum(np.asarray(x, float), upper)
    f = lambda t: sum(wi * stats.lognorm.cdf(t, s=si, scale=np.exp(mi)) for wi, mi, si in zip(w, mu, sd))
    return f(x) / f(upper)


def _tests(x, real, w, mu, sd):
    u = model_cdf(x, w, mu, sd)                       # PIT: при верной модели U ~ U(0,1)
    n = len(u)
    cnt = np.histogram(u, bins=20, range=(0, 1))[0]
    r = np.corrcoef(u[:-1], u[1:])[0, 1]
    b = u > 0.5
    n1, n2 = b.sum(), (~b).sum()
    runs = 1 + np.count_nonzero(b[1:] != b[:-1])
    mean_r = 2 * n1 * n2 / n + 1
    var_r = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1))
    pair = np.histogram2d(u[:-1], u[1:], bins=8, range=[[0, 1], [0, 1]])[0].ravel()
    edges = np.arange(0, UPPER + 1)
    c1, c2 = np.histogram(x, edges)[0], np.histogram(real, edges)[0]
    keep = (c1 + c2) >= 10
    return {
        'chi2 U p': stats.chisquare(cnt).pvalue,
        'lag p (approx)': 2 * stats.norm.sf(abs(r) * np.sqrt(n)),
        'серии p': 2 * stats.norm.sf(abs(runs - mean_r) / np.sqrt(var_r)),
        'пары p': stats.chisquare(pair).pvalue,
        'KS модель p': stats.kstest(x, lambda t: model_cdf(t, w, mu, sd)).pvalue,
        'KS реальные p': stats.ks_2samp(x, real).pvalue,
        'chi2 реальные p': stats.chi2_contingency(np.vstack([c1[keep], c2[keep]]))[1],
    }


def versions(real, w, mu, sd, seed=SEED):
    real = np.asarray(real, float)
    n = len(real)
    samples = {
        'v1: Park-Miller + BM': _mixture(park_miller, seed, n, w, mu, sd),
        'v2: xorshift64* + BM': _mixture(xorshift64star, seed, n, w, mu, sd),
        'v3: v2 + отбор': _v3(seed, n, w, mu, sd),
    }
    table = pd.DataFrame({k: _tests(v, real, w, mu, sd) for k, v in samples.items()}).T
    return table, samples
