"""Calibrate the 2023 potential-revenue distribution so that 2024 declared revenues of flat-tax users
match the Corte dei conti (June 2026) band counts for tax year 2024."""
import numpy as np
from scipy.optimize import minimize
from sim import draw_population, choose_declared, potential

# Corte dei conti, Relazione sul rendiconto (June 2026): flat-tax users by revenue band, tax year 2024
TARGET_COUNTS = np.array([936_000, 733_830, 276_758, 81_077])   # 0-20k, 20-50k, 50-80k, 80-85k
TARGET = TARGET_COUNTS / TARGET_COUNTS.sum()
EDGES = [0, 20, 50, 80, 85.0001]


def shares(mix, n=60_000, seed=1):
    pop, rng = draw_population(n, mix, seed)
    Z = potential(pop.z0.values, rng, 2, pop.sector.values)      # 2023 and 2024
    d23 = choose_declared(Z[:, 0], pop, 2023, reform=False, prev_d=Z[:, 0])
    d = choose_declared(Z[:, 1], pop, 2024, reform=False, prev_d=d23)
    d = d[d <= 85]
    h, _ = np.histogram(d, bins=EDGES)
    return h / h.sum(), len(d) / n


TAIL = (0.0846, 4.639, 0.35)   # established self-employed above the thresholds: kept fixed (no official count exists)


def to_mix(x):
    """5 free parameters: bulk mixture (w1, mu1, sd1, mu2, sd2); established component fixed at TAIL."""
    return (1 / (1 + np.exp(-x[0])), x[1], abs(x[2]), x[3], abs(x[4])) + TAIL


def loss(x):
    mix = to_mix(x)
    sh, _ = shares(mix)
    return np.sum(((sh - TARGET) / TARGET) ** 2)


if __name__ == "__main__":
    x0 = np.array([0.23, 2.35, 1.21, 3.63, 0.60])
    res = minimize(loss, x0, method="Nelder-Mead", options=dict(maxiter=2000, xatol=1e-3, fatol=1e-6))
    x = res.x
    mix = to_mix(x)
    sh, frac = shares(mix, n=200_000, seed=3)
    print("mix =", tuple(round(v, 4) for v in mix))
    print("simulated shares:", np.round(sh, 4))
    print("target shares:   ", np.round(TARGET, 4))
    print("share of sample declaring <= 85k:", round(frac, 4))
