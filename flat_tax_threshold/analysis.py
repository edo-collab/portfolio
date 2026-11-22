"""Analysis on the synthetic panel: bunching, difference-in-bunching, event study, fiscal table, robustness.
Uses ONLY what a researcher would observe (declared revenue, e-invoice B2B share, purchases, taxes paid),
except the fiscal table, which compares with the simulated no-reform world (in the paper: 'estimated counterfactual')."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sim import W, irpef, FLAT_RATE, ADDIZIONALI

OUT = "output/"
ref = pd.read_parquet("data/panel_reform.parquet")
cf = pd.read_parquet("data/panel_noreform.parquet")

BLUE, ORANGE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})


def group_of(s, hi=0.8, lo=0.2):
    """Classification from e-invoices 2024-2026 (pre-reform): share of revenue invoiced to businesses."""
    return np.where(s >= hi, "B2B", np.where(s <= lo, "B2C", "Mixed"))


ref["group"] = group_of(ref.s)
cf["group"] = group_of(cf.s)

# ------------------------------------------------------------------ bunching
BINS = np.arange(40, 170, 1.0)
MID = BINS[:-1] + 0.5


def hist(p, year, grp):
    d = p.loc[(p.year == year) & (p.group == grp), "declared"]
    return np.histogram(d, bins=BINS)[0] * W


def counterfactual(counts, excluded, deg=4):
    """Fit a polynomial to log counts outside the excluded windows (Chetty et al. 2011 style, in logs)."""
    keep = np.ones_like(MID, dtype=bool)
    for lo, hi in excluded:
        keep &= ~((MID >= lo) & (MID < hi))
    x = (MID - 100) / 50
    coefs = np.polyfit(x[keep], np.log(np.maximum(counts[keep], 1)), deg)
    return np.exp(np.polyval(coefs, x))


def excess_ratio(counts, cfit, T, width=5):
    m = (MID >= T - width) & (MID < T)
    return counts[m].sum() / cfit[m].sum()


def windows(year, grp, reform=True):
    if reform and year >= 2027:
        return [(80, 105), (95, 140)]          # bunching + missing mass at 85 (VAT) and 100 (flat tax)
    return [(80, 140)]                          # old single notch at 85 (missing mass up to ~135)


results = {}
for grp in ["B2B", "B2C"]:
    for yr in [2026, 2028]:
        c = hist(ref, yr, grp)
        fit = counterfactual(c, windows(yr, grp))
        results[(grp, yr)] = dict(counts=c, fit=fit, b85=excess_ratio(c, fit, 85), b100=excess_ratio(c, fit, 100))

pooled = {}
for yr in [2024, 2026]:
    d = ref.loc[ref.year == yr, "declared"]
    c = np.histogram(d, bins=BINS)[0] * W
    fit = counterfactual(c, [(80, 140)])
    pooled[yr] = round(excess_ratio(c, fit, 85), 2)

bunch_tab = pd.DataFrame([{"group": g, "year": y, "ratio_85": round(r["b85"], 2), "ratio_100": round(r["b100"], 2)}
                          for (g, y), r in results.items()])
# difference-in-bunching: change at 100 (and at 85) for B2B minus change for B2C
dib = {}
for T in ["ratio_85", "ratio_100"]:
    d = bunch_tab.set_index(["group", "year"])[T]
    dib[T] = round((d["B2B", 2028] - d["B2B", 2026]) - (d["B2C", 2028] - d["B2C", 2026]), 2)

# ------------------------------------------------------------------ Exhibit 1
fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True)
for i, grp in enumerate(["B2B", "B2C"]):
    col = BLUE if grp == "B2B" else ORANGE
    for j, yr in enumerate([2026, 2028]):
        ax = axes[i, j]
        r = results[(grp, yr)]
        sel = (MID >= 60) & (MID < 150)
        ax.bar(MID[sel], r["counts"][sel] / 1000, width=0.8, color=col, alpha=0.85, linewidth=0)
        ax.plot(MID[sel], r["fit"][sel] / 1000, color=INK2, lw=1.4, ls="--")
        for T in ([85] if yr == 2026 else [85, 100]):
            ax.axvline(T, color=INK, lw=0.8, ls=":")
        title = f"{grp} sellers, {yr}" + (" (before reform)" if yr == 2026 else " (after reform)")
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        txt = f"excess at 85k: {r['b85']:.1f}x"
        if yr == 2028:
            txt += f"\nexcess at 100k: {r['b100']:.1f}x"
        ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        if j == 0:
            ax.set_ylabel("people (thousands)\nper EUR 1k bin")
        if i == 1:
            ax.set_xlabel("declared revenue (EUR thousands)")
# placebo difference-in-bunching: same calculation 2024 -> 2026, when nothing changed
plac85 = {}
for grp in ["B2B", "B2C"]:
    for yr in [2024, 2026]:
        c = hist(ref, yr, grp)
        plac85[(grp, yr)] = excess_ratio(c, counterfactual(c, [(80, 140)]), 85)
dib_placebo = (plac85["B2B", 2026] - plac85["B2B", 2024]) - (plac85["B2C", 2026] - plac85["B2C", 2024])
r_ = {k: v["b85"] for k, v in results.items()}
dib85 = (r_["B2B", 2028] - r_["B2B", 2026]) - (r_["B2C", 2028] - r_["B2C", 2026])
fig.text(0.01, 0.07,
         f"Difference-in-bunching at EUR 85k = (B2B change) - (B2C change) = "
         f"({r_['B2B', 2028]:.1f} - {r_['B2B', 2026]:.1f}) - ({r_['B2C', 2028]:.1f} - {r_['B2C', 2026]:.1f}) = "
         f"{dib85:+.1f}\nPlacebo (same calculation 2024 -> 2026, when nothing changed): {dib_placebo:+.1f}",
         fontsize=8.5, color=INK, weight="bold")
fig.text(0.01, 0.005, "Bars: synthetic counts. Dashed line: estimated distribution without the cliffs (polynomial fit "
         "outside EUR 80-140k).\nExcess = people in the EUR 5k just below the threshold / people expected there. "
         "Dotted lines: thresholds.", fontsize=7.5, color=INK2)
fig.suptitle("Exhibit 1. Where self-employed declare their revenue, before and after the reform",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0.14, 1, 0.96))
fig.savefig(OUT + "exhibit1_bunching.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ Exhibit 2 (cohort comparison)
def cohort_growth(p, grp, start, band=(82, 85), var="declared", n_boot=300, seed=1):
    """People declaring within `band` in year `start`: mean 2-year growth (log points) of `var`."""
    base = p[(p.year == start) & (p.group == grp) & (p.declared >= band[0]) & (p.declared < band[1])].id.values
    w = p[p.id.isin(base) & p.year.isin([start, start + 2])].pivot(index="id", columns="year", values=var)
    g = np.log(w[start + 2]) - np.log(w[start])
    rng = np.random.default_rng(seed)
    boots = [g.values[rng.integers(0, len(g), len(g))].mean() for _ in range(n_boot)]
    return g.mean(), np.std(boots), len(g)


coh = []
for grp in ["B2B", "B2C"]:
    for var in ["declared", "purchases"]:
        for band, label in [((82, 85), "Stuck just below 85k"), ((45, 50), "Placebo: 45-50k")]:
            pre = cohort_growth(ref, grp, 2024, band, var)
            post = cohort_growth(ref, grp, 2026, band, var)
            coh.append(dict(group=grp, outcome=var, band=label, pre=pre[0], pre_se=pre[1], post=post[0], post_se=post[1],
                            effect=post[0] - pre[0], effect_se=np.hypot(pre[1], post[1]), n_post=post[2] * W))
coh = pd.DataFrame(coh)

def crossing(p, grp_mask_fn, start, band, above):
    """Share of people declaring within `band` in `start` who declare more than `above` two years later."""
    w = p[p.year.isin([start, start + 2])].pivot(index="id", columns="year", values="declared")
    meta = p[p.year == start].set_index("id")
    sel = (w[start] >= band[0]) & (w[start] < band[1]) & grp_mask_fn(meta)
    x = (w.loc[sel, start + 2] > above)
    return x.mean(), np.sqrt(x.mean() * (1 - x.mean()) / sel.sum()), int(sel.sum()) * W


is_b2b = lambda m: m.group == "B2B"
is_b2c = lambda m: m.group == "B2C"
is_any = lambda m: m.group.isin(["B2B", "B2C"])
cross = []
for label, fn, band, thr, panel in [("B2B", is_b2b, (82, 85), 85, "A"), ("B2C", is_b2c, (82, 85), 85, "A"),
                                    ("Placebo\n(45-50k, cross 50k)", is_any, (45, 50), 50, "A"),
                                    ("B2B", is_b2b, (82, 85), 100, "B"), ("B2C", is_b2c, (82, 85), 100, "B")]:
    pre = crossing(ref, fn, 2024, band, thr)
    post = crossing(ref, fn, 2026, band, thr)
    cross.append(dict(panel=panel, label=label, threshold=thr, pre=pre[0], pre_se=pre[1], post=post[0],
                      post_se=post[1], n_post=post[2]))
cross = pd.DataFrame(cross)

# ------------------------------------------------------------------ Exhibit 2: panel DiD (event study)
import statsmodels.api as sm


def panel_did(p, band, base_year=2023, years=range(2024, 2029), ref_year=2026):
    """Event study: log declared revenue on person FE + year FE + B2B x year, people selected on base_year
    revenue within `band`; B2B = treated, B2C = control. Within (demeaning) estimator, SE clustered by person."""
    meta = p[p.year == base_year].set_index("id")
    ids = meta.index[(meta.declared >= band[0]) & (meta.declared < band[1]) & meta.group.isin(["B2B", "B2C"])]
    d = p[p.id.isin(ids) & p.year.isin(list(years))][["id", "year", "declared"]].copy()
    d["b2b"] = (meta.loc[d.id, "group"].values == "B2B").astype(float)
    d["y"] = np.log(d.declared)
    ks = [k for k in years if k != ref_year]
    X = pd.DataFrame({**{f"yr{k}": (d.year == k).astype(float) for k in ks},
                      **{f"b2b_x_{k}": d.b2b * (d.year == k) for k in ks}}, index=d.index)
    Z = pd.concat([d[["y"]], X], axis=1)
    Z = Z - Z.groupby(d.id.values).transform("mean")          # removes person fixed effects
    m = sm.OLS(Z.y, Z.drop(columns="y")).fit(cov_type="cluster", cov_kwds={"groups": d.id.values})
    out = pd.DataFrame({"beta": [m.params.get(f"b2b_x_{k}", 0.0) if k != ref_year else 0.0 for k in years],
                        "se": [m.bse.get(f"b2b_x_{k}", 0.0) if k != ref_year else 0.0 for k in years]},
                       index=list(years))
    n_b2b = int(d[d.year == ref_year].b2b.sum()) * W
    n_b2c = int((d.year == ref_year).sum()) * W - n_b2b
    return out, n_b2b, n_b2c


did_main, nb_main, nc_main = panel_did(ref, (82, 85))
did_plac, nb_plac, nc_plac = panel_did(ref, (45, 50))

fig, ax = plt.subplots(figsize=(9, 3.9))
for res_, lab, col, off in [(did_main, f"Stuck just below EUR 85k (B2B {nb_main:,} vs B2C {nc_main:,} people)", BLUE, -0.06),
                            (did_plac, f"Placebo: EUR 45-50k, far from any cliff (B2B {nb_plac:,} vs B2C {nc_plac:,})",
                             GRAY, 0.06)]:
    x = np.array(res_.index) + off
    ax.errorbar(x, 100 * res_.beta, yerr=196 * res_.se, color=col, marker="o", ms=6, lw=2, capsize=3, label=lab)
for yr_, v in zip(did_main.index, did_main.beta):
    if yr_ >= 2027:
        ax.annotate(f"{100*v:+.1f}%", (yr_ - 0.06, 100 * v), xytext=(-9, 9), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8.5, color=INK)
ax.axhline(0, color=INK2, lw=0.8)
ax.axvline(2026.5, color=INK, lw=0.8, ls=":")
ax.text(2026.55, ax.get_ylim()[1] * 0.92, "reform takes effect", fontsize=8, color=INK2, va="top")
ax.set_xticks(list(did_main.index))
ax.set_xlim(2023.6, 2028.5)
ax.set_ylabel("B2B minus B2C, % difference in\ndeclared revenue (2026 = 0)")
ax.grid(axis="y", color="#e6e5e0", lw=0.6)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.suptitle("Exhibit 2. Panel difference-in-differences: B2B vs B2C sellers, followed year by year",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.text(0.01, 0.01, "Each dot: coefficient on B2B x year from a regression of log declared revenue on person and year "
         "fixed effects. Groups fixed on 2023 revenue (before the reform\nwas announced). Bars: 95% confidence "
         "intervals, clustered by person. Flat dots before 2027 = parallel trends.", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.1, 1, 0.93))
fig.savefig(OUT + "exhibit2_panel_did.png", dpi=200)
plt.close(fig)

# (old crossing-shares chart removed; the numbers stay in results.txt for the text)

# ------------------------------------------------------------------ fiscal table (2028)
y = 2028
a = ref[ref.year == y].set_index("id")
b = cf[cf.year == y].set_index("id")
dtax = (a.tax - b.tax)                     # + = state gains from the reform
dd = a.declared - b.declared
cat = np.select(
    [(dd.abs() < 1e-6) & (b.declared > 85) & (b.declared <= 100.0),
     (dd > 1e-6) & (b.declared <= 85),
     (dd < -1e-6)],
    ["Already at 85-100k (windfall)", "Risers: from below 85k", "Fallers: came down from above 100k"],
    default="Unaffected")
fiscal = (pd.DataFrame({"cat": cat, "dtax": dtax, "dd": dd})
          .groupby("cat").agg(people=("dtax", "size"), change_m=("dtax", "sum"), extra_rev_m=("dd", "sum")))
fiscal["people"] *= W
fiscal["change_m"] = fiscal.change_m * W / 1000        # EUR thousands -> millions
fiscal["extra_rev_m"] = fiscal.extra_rev_m * W / 1000

# Reclassification of employees is NOT quantified (no data); discussed as a limitation in the text.
fiscal = fiscal.drop(index="Unaffected", errors="ignore")
direct_cost = -fiscal.loc["Already at 85-100k (windfall)", "change_m"]
recovered = fiscal.loc["Risers: from below 85k", "change_m"]
lost = -fiscal.loc["Fallers: came down from above 100k", "change_m"]
net = fiscal.change_m.sum()
recovery_share = (recovered - lost) / direct_cost

# ------------------------------------------------------------------ Exhibit 3 (table as figure)
rows = [
    ("Already at EUR 85-100k (windfall)", "Pay 15% instead of IRPEF, no change"),
    ("Risers: from below EUR 85k", "Declare more; pay flat tax + VAT on it"),
    ("Fallers: from above EUR 100k", "Declare less; switch from IRPEF to flat tax"),
]
keymap = ["Already at 85-100k (windfall)", "Risers: from below 85k", "Fallers: came down from above 100k"]
fig, ax = plt.subplots(figsize=(9, 2.8))
ax.axis("off")
header = ["Group", "What happens", "People", "Effect on revenue\n(EUR million, 2028)"]
cells = [[r[0], r[1], f"{int(fiscal.loc[k,'people']):,}", f"{fiscal.loc[k,'change_m']:+,.0f}"] for r, k in zip(rows, keymap)]
cells.append(["Net effect", "", "", f"{net:+,.0f}"])
cells.append(["Behavioural effects vs direct cost", "(recovered - lost) / direct cost", "", f"{100*recovery_share:+.0f}%"])
tbl = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="left", colLoc="left",
               colWidths=[0.33, 0.39, 0.09, 0.19])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.55)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#e6e5e0")
    if r == 0:
        cell.set_text_props(color=INK2, weight="bold")
    if r >= len(cells) - 1:
        cell.set_text_props(weight="bold")
    if c >= 2:
        cell.get_text().set_horizontalalignment("right")
        cell.PAD = 0.04
ax.set_title("Exhibit 3. Who gains and what it costs the state (2028)", loc="left", fontsize=11, color=INK)
fig.text(0.01, 0.02, "Revenue = income tax (flat tax or IRPEF + add-ons) + net VAT on sales to consumers. INPS contributions "
         "excluded. Employees reclassified as\nfreelancers are not quantified (would add to the cost). Synthetic data.",
         fontsize=7.5, color=INK2)
fig.tight_layout()
fig.savefig(OUT + "exhibit3_fiscal.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ robustness: bunching sensitivity
rob = []
for deg in [3, 4, 5, 6]:
    for grp in ["B2B", "B2C"]:
        c = hist(ref, 2028, grp)
        fit = counterfactual(c, windows(2028, grp), deg=deg)
        rob.append({"degree": deg, "group": grp, "ratio_85": round(excess_ratio(c, fit, 85), 2),
                    "ratio_100": round(excess_ratio(c, fit, 100), 2)})
rob = pd.DataFrame(rob)
# placebo thresholds (no cliff there): excess ratio around 70k and 120k in 2028
plac_thr = []
for grp in ["B2B", "B2C"]:
    c = hist(ref, 2028, grp)
    for T in [70]:
        fit = counterfactual(c, windows(2028, grp) + [(T - 5, T + 5)])
        plac_thr.append({"group": grp, "threshold": T, "ratio": round(excess_ratio(c, fit, T), 2)})
# alternative B2B/B2C cutoffs
alt = []
for hi, lo in [(0.7, 0.3), (0.9, 0.1)]:
    p = ref.copy()
    p["group"] = group_of(p.s, hi, lo)
    for grp in ["B2B", "B2C"]:
        d = p.loc[(p.year == 2028) & (p.group == grp), "declared"]
        c = np.histogram(d, bins=BINS)[0] * W
        fit = counterfactual(c, windows(2028, grp))
        alt.append({"cutoffs": f"{hi}/{lo}", "group": grp, "ratio_85": round(excess_ratio(c, fit, 85), 2),
                    "ratio_100": round(excess_ratio(c, fit, 100), 2)})
alt = pd.DataFrame(alt)

# transition table: where 2028 bunchers at 100k were in 2026
p26 = ref[ref.year == 2026].set_index("id").declared
p28 = ref[ref.year == 2028].set_index("id").declared
b100 = p28[(p28 >= 95) & (p28 < 100)].index
origin = pd.cut(p26.loc[b100], [0, 80, 85, 100, 135, np.inf],
                labels=["below 80k", "80-85k (old bunchers)", "85-100k", "100-135k", "above 135k"])
transition = (origin.value_counts(normalize=True).sort_index() * 100).round(1)

# ------------------------------------------------------------------ fix for limit 2: within-sector comparison
within = []
for sec in ["Professionals", "Other activities"]:
    for grp in ["B2B", "B2C"]:
        for yr in [2026, 2028]:
            d = ref.loc[(ref.year == yr) & (ref.group == grp) & (ref.sector == sec), "declared"]
            c = np.histogram(d, bins=BINS)[0] * W
            fit = counterfactual(c, windows(yr, grp))
            within.append(dict(sector=sec, group=grp, year=yr, people=int(len(d)) * W,
                               ratio_85=round(excess_ratio(c, fit, 85), 2), ratio_100=round(excess_ratio(c, fit, 100), 2)))
within = pd.DataFrame(within)
w_idx = within.set_index(["sector", "group", "year"])
dib_within = {sec: round((w_idx.loc[(sec, "B2B", 2028), "ratio_100"] - w_idx.loc[(sec, "B2B", 2026), "ratio_100"])
                         - (w_idx.loc[(sec, "B2C", 2028), "ratio_100"] - w_idx.loc[(sec, "B2C", 2026), "ratio_100"]), 2)
              for sec in ["Professionals", "Other activities"]}

# dose-response: among professionals, share of B2B in 5 bins -> bunching at 100k in 2028
dose = []
for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
    d = ref.loc[(ref.year == 2028) & (ref.sector == "Professionals") & (ref.s >= lo) & (ref.s < hi), "declared"]
    c = np.histogram(d, bins=BINS)[0] * W
    fit = counterfactual(c, windows(2028, "B2B"))
    dose.append(dict(b2b_share=f"{lo:.1f}-{min(hi,1):.1f}", ratio_85=round(excess_ratio(c, fit, 85), 2),
                     ratio_100=round(excess_ratio(c, fit, 100), 2)))
dose = pd.DataFrame(dose)


# ------------------------------------------------------------------ fix for limit 1: purchases-to-revenue test
def ratio_change(p, grp, start, band=(82, 85), n_boot=300, seed=2):
    """2-year change in log(purchases / declared revenue) for people in `band` in `start`."""
    ids = p[(p.year == start) & (p.group == grp) & (p.declared >= band[0]) & (p.declared < band[1])].id.values
    q = p[p.id.isin(ids) & p.year.isin([start, start + 2])]
    lr = np.log(q.pivot(index="id", columns="year", values="purchases")) - \
         np.log(q.pivot(index="id", columns="year", values="declared"))
    g = (lr[start + 2] - lr[start]).values
    rng = np.random.default_rng(seed)
    se = np.std([g[rng.integers(0, len(g), len(g))].mean() for _ in range(n_boot)])
    return g.mean(), se


ratio_test = []
for grp in ["B2B", "B2C"]:
    pre = ratio_change(ref, grp, 2024)
    post = ratio_change(ref, grp, 2026)
    ratio_test.append(dict(group=grp, pre=round(pre[0], 3), post=round(post[0], 3), effect=round(post[0] - pre[0], 3),
                           se=round(np.hypot(pre[1], post[1]), 3)))
ratio_test = pd.DataFrame(ratio_test)

# validation only possible with synthetic data: of the EXTRA revenue declared because of the reform
# (2028, reform vs no-reform world), how much is new activity and how much is previously hidden income?
a28 = ref[ref.year == 2028].set_index("id")
b28 = cf[cf.year == 2028].set_index("id")
truth = []
for grp in ["B2B", "B2C"]:
    m = (a28.group == grp) & (a28.declared > b28.declared + 1e-6)
    extra_decl = (a28.declared - b28.declared)[m].sum()
    extra_act = (a28.activity - b28.activity)[m].sum()
    truth.append(dict(group=grp, extra_declared_m=round(extra_decl * W / 1000, 1),
                      of_which_new_activity_pct=round(100 * extra_act / extra_decl, 1) if extra_decl else np.nan))
truth = pd.DataFrame(truth)


trans_group = {}
g26 = ref[ref.year == 2026].set_index("id")
for grp in ["B2B", "B2C"]:
    ids = p28[(p28 >= 95) & (p28 < 100)].index
    ids = ids[g26.loc[ids, "group"].values == grp]
    o = pd.cut(p26.loc[ids], [0, 80, 85, 100, 135, np.inf],
               labels=["below 80k", "80-85k (old bunchers)", "85-100k", "100-135k", "above 135k"])
    trans_group[grp] = (o.value_counts(normalize=True).sort_index() * 100).round(1)
    trans_group[grp].loc["people in 95-100k, 2028"] = len(ids) * W
with open(OUT + "results.txt", "w") as f:
    f.write("BUNCHING (excess ratio = people just below threshold / expected)\n" + bunch_tab.to_string(index=False))
    f.write(f"\n\nDiB at 85k: {dib85:+.2f}; placebo DiB 2024->2026: {dib_placebo:+.2f}")
    f.write("\n\nPANEL DID (B2B vs B2C, groups fixed on 2023 revenue)\nStuck 82-85k:\n" + (100*did_main).round(2).to_string()
            + "\nPlacebo 45-50k:\n" + (100*did_plac).round(2).to_string())
    f.write(f"\n\nPooled excess ratio at 85k (all groups): {pooled}")
    f.write(f"\n\nDifference-in-bunching (B2B change - B2C change): {dib}\n")
    f.write("\nCOHORT GROWTH (log points over 2 years; effect = post cohort 2026-28 minus pre cohort 2024-26)\n")
    f.write(coh.round(3).to_string(index=False))
    f.write("\n\nCROSSING SHARES (Exhibit 2)\n" + cross.round(3).to_string(index=False))
    f.write("\n\nFISCAL TABLE 2028\n" + fiscal.round(1).to_string())
    f.write(f"\n\nDirect cost {direct_cost:.0f}m; recovered {recovered:.0f}m; lost {lost:.0f}m; net {net:.0f}m; "
            f"recovery share {recovery_share:.2%}\n")
    f.write("\nROBUSTNESS: polynomial degree\n" + rob.to_string(index=False))
    f.write("\n\nROBUSTNESS: placebo threshold\n" + pd.DataFrame(plac_thr).to_string(index=False))
    f.write("\n\nROBUSTNESS: B2B/B2C cutoffs\n" + alt.to_string(index=False))
    f.write("\n\nWITHIN-SECTOR BUNCHING (limit 2)\n" + within.to_string(index=False))
    f.write(f"\nDifference-in-bunching at 100k within sector: {dib_within}")
    f.write("\n\nDOSE-RESPONSE among professionals, 2028\n" + dose.to_string(index=False))
    f.write("\n\nPURCHASES/REVENUE TEST (limit 1): change in log ratio; effect = post cohort - pre cohort\n"
            + ratio_test.to_string(index=False))
    f.write("\n\nVALIDATION (synthetic truth): share of extra declared revenue that is new activity\n"
            + truth.to_string(index=False))
    f.write("\n\nTRANSITIONS: 2028 bunchers at 95-100k, where they were in 2026 (%)\n" + transition.to_string())
    for g, t in trans_group.items():
        f.write(f"\n\nTRANSITIONS {g}\n" + t.to_string())
print(open(OUT + "results.txt").read())
