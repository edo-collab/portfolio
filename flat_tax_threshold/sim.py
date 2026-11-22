"""
Synthetic panel of Italian individual self-employed (partite IVA), 2023-2028.
Units: revenues in EUR thousands. One simulated person = W real people.

Policy:
  Old rules (all years in the no-reform world, and 2023-2026 in the reform world):
     flat tax (15% x coefficient x revenue) and no VAT if revenue <= 85
     ordinary IRPEF on profit + VAT if revenue > 85
  New rules (2027-2028 in the reform world, "double track"):
     no VAT if revenue <= 85; VAT above 85
     flat tax if revenue <= 100; ordinary IRPEF above 100
"""
import numpy as np
import pandas as pd

W = 10                      # each simulated person represents 10 real people
YEARS = list(range(2023, 2029))
REFORM_YEAR = 2027

# ---------- tax rules (simplified, documented) ----------
FLAT_RATE = 0.15
VAT_RATE = 0.22
VAT_SHARE = VAT_RATE / (1 + VAT_RATE)      # share of a VAT-inclusive price that is VAT (~18%)
ORDINARY_FIXED_COST = 1.0   # extra accountant/VAT compliance cost in the ordinary regime, EUR k/yr.
                            # Fiscozen (Aug 2026): ~EUR 700 flat tax vs ~1,500 ordinary (+800);
                            # Alosa (2023) uses +1,300 from Harju et al. (2019). We take ~1,000.


ADDIZIONALI = 0.025   # regional + municipal add-ons on profit. Regional top bracket 1.23%-3.33% (money.it 2025),
                      # municipal up to 0.8% by law; ~2.5% is a mid value. Replaced by the flat tax.


def irpef(profit):
    """IRPEF 2026 brackets: 23% to 28k, 33% 28-50k, 43% above, plus ~2.5% add-ons. No deductions."""
    p = np.maximum(profit, 0)
    return (0.23 * np.minimum(p, 28)
            + 0.33 * np.clip(p - 28, 0, 22)
            + 0.43 * np.maximum(p - 50, 0)
            + ADDIZIONALI * p)


# ---------- population: sectors and client types ----------
# Each person has a SECTOR (fixes the flat-tax coefficient and cost share) and a CLIENT MIX (B2B share s,
# observed in e-invoices). The two are drawn separately, so within a sector there are both B2B and B2C sellers.
# Coefficients: 78% professionals, 67% other activities (law 190/2014, Annex 4; Alosa 2023 Table 4).
# Client mix anchored to Alosa (2023): share of VAT borne by consumers = 0.34 for professionals, 0.52 for
# "other activities", 1.0 for retail -> professionals mostly but not only B2B; other activities mostly B2C.
#   client mix shares: (pure B2B s~U(.85,1), mixed s~U(.3,.7), pure B2C s~U(0,.15))
SECTORS = {
    "Professionals":    dict(share=0.45, coef=0.78, cs=0.22, mix=(0.50, 0.30, 0.20)),   # avg B2C share ~0.37
    "Other activities": dict(share=0.55, coef=0.67, cs=0.33, mix=(0.15, 0.35, 0.50)),   # avg B2C share ~0.65
}
# Overall: ~31% pure B2B, ~33% mixed, ~36% pure B2C.
# Hiding: share of an under-declaration that is hidden (rest = real cut). Sales to consumers can be hidden;
# sales to businesses mostly cannot (paper trail: Pomeranz 2015; Kleven et al. 2011 find evasion on
# third-party-reported income is tiny but not zero) -> hide = (1 - s) + B2B_HIDE * s.
B2B_HIDE = 0.10
# How much each person CAN adjust (cut, delay or hide revenue).
# Kleven & Waseem (2013): 51-86% of self-employed in dominated ranges do not respond (frictions). We target the
# middle of that range (~70% non-response near the cliff).
# Alosa (2023): Italian professionals cut turnover by UP TO ~EUR 19k (~29% of the EUR 65k threshold) - a maximum.
# So: a share of people cannot adjust at all; the others can adjust up to a personal limit a_i (share of true
# revenue) that is SKEWED: most can move only a little (delay a few invoices), few can move a lot (refuse large
# contracts, hide large amounts). a_i ~ Exponential(mean A_MEAN), capped at A_CAP. B2C sellers are more flexible
# (cash, delaying jobs), B2B less (contracts, invoices recorded by the client).
NO_ADJ_B2B, NO_ADJ_B2C = 0.55, 0.45       # share who cannot adjust at all (pure B2B / pure B2C; linear in between)
A_MEAN_B2B, A_MEAN_B2C = 0.12, 0.17       # typical adjustable share of true revenue
A_CAP_B2B, A_CAP_B2C = 0.25, 0.30         # hard maximum (Alosa: up to ~29%)
ADJ_MULT = 1.0                            # scales flexibility in sensitivity checks
# VAT incidence on sales to consumers: share of the VAT the SELLER absorbs (the rest raises prices).
# Kosonen (2015, JPubE): Finnish hairdressers passed only ~half of a VAT cut to prices; Benzarti et al. (2020, JPE):
# price responses to VAT increases are ~2x those to cuts. Crossing the threshold is an increase, but competitors
# below the threshold charge no VAT -> we use 50% absorbed (Alosa 2023 uses 100% as an upper bound).
VAT_ABSORB = 0.50
HIDE_COST = 0.30    # expected penalty + risk cost per euro hidden (Allingham-Sandmo style), assumption
HIDE_CONVEX = 0.004 # cost rises with the amount hidden: audit risk grows as declared falls (Lecture 14, p'(d)<0)


def hide_cost(gap):
    """Expected cost of hiding `gap` (EUR k): ~0.32/EUR for 5k hidden, ~0.44 for 35k, ~0.54 for 60k."""
    g = np.maximum(gap, 0)
    return HIDE_COST * g + HIDE_CONVEX * g ** 2


# Entry rule (law 190/2014, comma 54; EU VAT directive): the flat tax and the VAT exemption are available only if
# revenue was below the threshold in the PREVIOUS year. Someone coming from above who cuts revenue this year gets
# the benefit only from next year: over the 2027-28 horizon they must hold back for two years to gain one.
# We approximate this by valuing the benefit at ENTRY_WEIGHT for people who were above the threshold last year.
ENTRY_WEIGHT = 0.5


def net_income(d, z, typ, s, coef, cs, hide, regime_flat, vat_on, prev_flat_ok=None, prev_vat_ok=None):
    """Net income of declaring d when true potential revenue is z.
    B2B part: declaring less = actually working/collecting less (real response).
    B2C part: declaring less can be hiding (keep true income, pay a convex expected cost of hiding)."""
    gap = z - d
    hidden_part = hide              # share of the gap that is hidden revenue (rest is a real reduction)
    gross = (1 - cs) * d + hidden_part * ((1 - cs) * gap - hide_cost(gap))
    w_flat = 1.0 if prev_flat_ok is None else np.where(prev_flat_ok, 1.0, ENTRY_WEIGHT)
    w_vat = 1.0 if prev_vat_ok is None else np.where(prev_vat_ok, 1.0, ENTRY_WEIGHT)
    ordinary = irpef((1 - cs) * d) + ORDINARY_FIXED_COST
    flat = FLAT_RATE * coef * d
    tax = np.where(regime_flat, w_flat * flat + (1 - w_flat) * ordinary, ordinary)
    vat_full = VAT_ABSORB * VAT_SHARE * (1 - s) * d                          # part of VAT on B2C sales absorbed
    vat_burden = np.where(vat_on, vat_full, (1 - w_vat) * vat_full)
    return gross - tax - vat_burden


def rules(year, reform):
    """Return (vat threshold, flat-tax threshold)."""
    if reform and year >= REFORM_YEAR:
        return 85.0, 100.0
    return 85.0, 85.0


def choose_declared(z, df, year, reform, prev_d=None):
    """Each person picks the best of {true z, just below VAT threshold, just below flat-tax threshold},
    but can only go as low as z * (1 - adj), their personal adjustment limit. prev_d: last year's declared
    revenue (entry rule)."""
    t_vat, t_flat = rules(year, reform)
    prev_flat_ok = None if prev_d is None else prev_d <= t_flat
    prev_vat_ok = None if prev_d is None else prev_d <= t_vat
    floor = z * (1 - df.adj.values)
    cands = [z.copy()]
    for t in [t_vat - 0.01, t_flat - 0.01]:
        target = np.minimum(z, t)
        cands.append(np.where(target >= floor, target, z))      # unreachable target -> stay at z
    best_val = None
    best_d = None
    for d in cands:
        val = net_income(d, z, None, df.s.values, df.coef.values, df.cs.values, df.hide.values,
                         regime_flat=d <= t_flat, vat_on=d > t_vat, prev_flat_ok=prev_flat_ok, prev_vat_ok=prev_vat_ok)
        if best_val is None:
            best_val, best_d = val, d
        else:
            better = val > best_val + 1e-9
            best_val = np.where(better, val, best_val)
            best_d = np.where(better, d, best_d)
    return best_d


def draw_population(n, mix, seed):
    """mix = (w1, mu1, sd1, mu2, sd2[, w3, mu3, sd3]): lognormal mixture for 2023 potential revenue.
    Components 1-2: the bulk of small self-employed. Optional component 3 (share w3 of everyone):
    established self-employed with higher revenues, mostly in the ordinary regime today."""
    rng = np.random.default_rng(seed)
    w1, mu1, sd1, mu2, sd2 = mix[:5]
    w3, mu3, sd3 = mix[5:] if len(mix) == 8 else (0.0, 0.0, 1.0)
    u = rng.random(n)
    z0 = np.where(u < w3, rng.lognormal(mu3, sd3, n),
                  np.where(u < w3 + (1 - w3) * w1, rng.lognormal(mu1, sd1, n), rng.lognormal(mu2, sd2, n)))
    sec_names = list(SECTORS)
    sector = rng.choice(sec_names, size=n, p=[SECTORS[k]["share"] for k in sec_names])
    df = pd.DataFrame({"id": np.arange(n), "sector": sector, "z0": z0})
    s_val = np.empty(n)
    for k, v in SECTORS.items():
        m = (df.sector == k).values
        kind = rng.choice(3, size=m.sum(), p=v["mix"])
        draws = np.select([kind == 0, kind == 1], [rng.uniform(0.85, 1.0, m.sum()), rng.uniform(0.3, 0.7, m.sum())],
                          rng.uniform(0.0, 0.15, m.sum()))
        s_val[m] = draws
        df.loc[m, "coef"] = v["coef"]
        df.loc[m, "cs"] = v["cs"]
    df["s"] = s_val
    df["type"] = np.where(df.s >= 0.8, "B2B", np.where(df.s <= 0.2, "B2C", "Mixed"))
    df["hide"] = (1 - df.s) + B2B_HIDE * df.s
    b2c = 1 - df.s
    no_adj = NO_ADJ_B2B + (NO_ADJ_B2C - NO_ADJ_B2B) * b2c
    a_mean = (A_MEAN_B2B + (A_MEAN_B2C - A_MEAN_B2B) * b2c) * ADJ_MULT
    a_cap = (A_CAP_B2B + (A_CAP_B2C - A_CAP_B2B) * b2c) * ADJ_MULT
    can = rng.random(n) >= no_adj
    df["adj"] = np.where(can, np.minimum(rng.exponential(1.0, n) * a_mean, a_cap), 0.0)
    df["responder"] = df.adj > 0
    return df, rng


# Revenue dynamics. GROWTH: nominal trend. Italian inflation: 2026 carry-over 2.9%, August 2026 3.3% (Istat);
# real GDP 2026 ~0.9% (MEF revision) -> nominal ~3.5-4%; partite IVA average income grew 8.6% in tax year 2024
# (MEF). We use 3.5%/yr. Shocks: permanent (random walk) + transitory, as in standard income-process models.
# Self-employed incomes are ~2.5x more dispersed year-to-year than employees' (NBER, Levine & Rubinstein-type
# evidence summarised in NBER Bulletin 2025). Permanent sd 8% + transitory sd 15% -> sd of a yearly change ~23%.
GROWTH, PERM_SD, TRANS_SD = 0.035, 0.08, 0.15


# Sector-specific conditions: each sector has its own yearly shock (SECTOR_SD), and consumer-facing activities
# ("Other activities": retail, crafts, personal services) take an extra permanent hit from the 2026 energy-price
# shock (Hormuz crisis; Coldiretti/ANSA 2026 report energy, fuel and input cost surges). These make B2B and B2C
# trends differ for reasons unrelated to the reform -> controlled with sector-by-year fixed effects.
SECTOR_SD, ENERGY_HIT_2026 = 0.012, -0.015


def potential(z0, rng, n_years, sector=None):
    """Potential (true) revenue path: trend + random-walk permanent shocks + transitory shocks + sector shocks."""
    n = len(z0)
    perm = np.cumsum(rng.normal(0, PERM_SD, size=(n, n_years)), axis=1)
    perm[:, 0] = 0.0                                  # 2023 is the starting point
    trans = rng.normal(0, TRANS_SD, size=(n, n_years))
    trend = GROWTH * np.arange(n_years)
    sec = np.zeros((n, n_years))
    if sector is not None:
        srng = np.random.default_rng(12345)           # same sector shocks in every run
        for k in SECTORS:
            shock = srng.normal(0, SECTOR_SD, len(YEARS))     # always drawn for all years -> same shocks
            shock[0] = 0.0
            if k == "Other activities":
                shock[[YEARS.index(y) for y in YEARS if y >= 2026]] += ENERGY_HIT_2026
            sec[np.asarray(sector) == k] = shock[:n_years]
    return z0[:, None] * np.exp(trend + perm + trans + sec)


def simulate(mix=(0.55, 2.55, 0.85, 3.75, 0.55), n=230_000, seed=7, reform=True):
    pop, rng = draw_population(n, mix, seed)
    rows = []
    Z = potential(pop.z0.values, rng, len(YEARS), pop.sector.values)   # same draws in both worlds (same seed)
    prev_d = Z[:, 0].copy()                            # before 2023: assume people declared their true revenue
    for j, y in enumerate(YEARS):
        z = Z[:, j]
        d = choose_declared(z, pop, y, reform, prev_d)
        prev_d = d
        t_vat, t_flat = rules(y, reform)
        rows.append(pd.DataFrame({"id": pop.id, "year": y, "z": z, "declared": d,
                                  "flat": d <= t_flat, "vat": d > t_vat}))
    panel = pd.concat(rows, ignore_index=True).merge(pop, on="id")
    # real activity actually carried out: B2B under-declaration is real, hidden B2C revenue still produced
    panel["activity"] = panel.declared + panel.hide * (panel.z - panel.declared)
    # purchases observed in received e-invoices: proportional to real activity, with noise (stock changes, timing)
    noise = np.random.default_rng(seed + 99).normal(0, 0.12, len(panel))
    panel["purchases"] = panel.cs * panel.activity * np.exp(noise)
    return panel


def taxes(panel):
    """Revenue to the state per person-year: income tax (flat or IRPEF) + net VAT on B2C value added."""
    d, cs, coef, s = panel.declared, panel.cs, panel.coef, panel.s
    income_tax = np.where(panel.flat, FLAT_RATE * coef * d, irpef((1 - cs) * d))
    net_vat = np.where(panel.vat, VAT_SHARE * (1 - s) * (1 - cs) * d, 0.0)  # B2B VAT is deducted down the chain
    return income_tax + net_vat


if __name__ == "__main__":
    p = simulate()
    print(p.head())
