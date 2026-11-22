"""Security checks: (1) simulated aggregates vs official numbers, (2) hand-checked tax arithmetic,
(3) does the bunching estimator recover the true answer?, (4) sensitivity of the fiscal result to assumptions."""
import numpy as np
import pandas as pd
import sim
from sim import W, irpef, simulate, taxes

ref = pd.read_parquet("data/panel_reform.parquet")
cf = pd.read_parquet("data/panel_noreform.parquet")
out = []

# (1) aggregates vs sources -------------------------------------------------------------
d24 = ref.loc[ref.year == 2024, "declared"]
official = {"0-20k": 936_000, "20-50k": 733_830, "50-80k": 276_758, "80-85k": 81_077}
sim_counts = (pd.cut(d24[d24 <= 85], [0, 20, 50, 80, 85.0001], labels=list(official)).value_counts() * W)
out.append("(1) Flat-tax users by band, tax year 2024: simulated vs Corte dei conti (June 2026)")
for k, v in official.items():
    out.append(f"   {k:7s} sim {int(sim_counts[k]):>9,}   official {v:>9,}   diff {100*(sim_counts[k]/v-1):+.1f}%")
out.append(f"   total   sim {int(sim_counts.sum()):>9,}   official {sum(official.values()):>9,}")

# (2) hand-checked tax arithmetic ---------------------------------------------------------
out.append("\n(2) Tax arithmetic, checked by hand")
# professional, revenue 92k, costs 22% -> profit 71.76k
p = 0.78 * 92
hand = 0.23 * 28 + 0.33 * 22 + 0.43 * (p - 50) + 0.025 * p
out.append(f"   IRPEF+add-ons on profit {p:.2f}k: code {float(irpef(p)):.3f}k, by hand {hand:.3f}k")
out.append(f"   Flat tax on 92k revenue (coef 78%): {0.15*0.78*92:.3f}k -> saving {hand-0.15*0.78*92:.2f}k")
assert abs(float(irpef(p)) - hand) < 1e-9

# (3) estimator check: true excess vs estimated excess (2026, old rules) -----------------
out.append("\n(3) Does the bunching estimate recover the truth? (2026, all groups)")
y26 = ref[ref.year == 2026]
true_cf = ((y26.z >= 80) & (y26.z < 85)).sum()          # people whose true potential is 80-85k
actual = ((y26.declared >= 80) & (y26.declared < 85)).sum()
out.append(f"   people declaring 80-85k: {actual*W:,};  people truly at 80-85k: {true_cf*W:,};  "
           f"true excess ratio {actual/true_cf:.2f}")

out.append("\n(3b) Share NOT reacting among people truly at EUR 85-100k in 2026 (Kleven & Waseem: 51-86%)")
dom = y26[(y26.z > 85) & (y26.z <= 100)]
for g in ["B2B", "B2C"]:
    dg = dom[dom.type == g]
    out.append(f"   {g}: {100*(dg.declared == dg.z).mean():.0f}% declare their true revenue (do not bunch)")
out.append(f"   all: {100*(dom.declared == dom.z).mean():.0f}%")

# (4) sensitivity of the fiscal result -------------------------------------------------
out.append("\n(4) Sensitivity of 2028 fiscal effect (EUR million; not recalibrated, so band counts move slightly)")
MIX = (0.5574, 2.3504, 1.2056, 3.6272, 0.6047, 0.0846, 4.639, 0.35)


def fiscal(scen_name, **changes):
    saved = {k: getattr(sim, k) for k in changes if hasattr(sim, k)}
    for k, v in changes.items():
        setattr(sim, k, v)
    a = simulate(mix=MIX, n=80_000, reform=True)
    b = simulate(mix=MIX, n=80_000, reform=False)
    a = a[a.year == 2028].set_index("id"); b = b[b.year == 2028].set_index("id")
    dt = np.asarray(taxes(a.reset_index())) - np.asarray(taxes(b.reset_index()))
    dd = (a.declared - b.declared).values
    wind = (np.abs(dd) < 1e-6) & (b.declared.values > 85) & (b.declared.values <= 100)
    scale = W * 229_600 / 80_000 / 1000
    res = dict(scenario=scen_name, windfall=dt[wind].sum() * scale, risers=dt[dd > 1e-6].sum() * scale,
               fallers=dt[dd < -1e-6].sum() * scale, net_excl_reclass=dt.sum() * scale,
               net_declared_revenue=dd.sum() * scale)
    for k, v in saved.items():
        setattr(sim, k, v)
    return res


rows = [fiscal("baseline"),
        fiscal("B2B sellers hide nothing", B2B_HIDE=0.0),
        fiscal("B2B sellers hide more (25%)", B2B_HIDE=0.25),
        fiscal("stricter enforcement (hiding cost 0.40/EUR)", HIDE_COST=0.40),
        fiscal("looser enforcement (hiding cost 0.20/EUR)", HIDE_COST=0.20),
        fiscal("less flexible (limits x0.7)", ADJ_MULT=0.7),
        fiscal("more flexible (limits x1.3)", ADJ_MULT=1.3),
        fiscal("B2C absorb 25% of VAT", VAT_ABSORB=0.25),
        fiscal("B2C absorb 100% of VAT", VAT_ABSORB=1.0)]
sens = pd.DataFrame(rows).round(0)
out.append(sens.to_string(index=False))

txt = "\n".join(out)
open("output/checks.txt", "w").write(txt)
print(txt)
