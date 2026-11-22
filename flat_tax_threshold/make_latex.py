"""Produce LaTeX exhibits (pgfplots figure + booktabs tables) from the synthetic panel."""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sim import W, irpef, FLAT_RATE, VAT_SHARE

OUT = "latex/"
os.makedirs(OUT + "data", exist_ok=True)
ref = pd.read_parquet("data/panel_reform.parquet")
cf = pd.read_parquet("data/panel_noreform.parquet")
grp_of = lambda s: np.where(s >= 0.8, "B2B", np.where(s <= 0.2, "B2C", "Mixed"))
ref["group"] = grp_of(ref.s)
cf["group"] = grp_of(cf.s)

# ============================================================== Exhibit 1: data for pgfplots
BINS = np.arange(40, 170, 1.0)
MID = BINS[:-1] + 0.5


def hist(p, year, grp):
    return np.histogram(p.loc[(p.year == year) & (p.group == grp), "declared"], bins=BINS)[0] * W


def cfit(c, excl=((80, 140),), deg=4):
    keep = np.ones_like(MID, bool)
    for lo, hi in excl:
        keep &= ~((MID >= lo) & (MID < hi))
    x = (MID - 100) / 50
    return np.exp(np.polyval(np.polyfit(x[keep], np.log(np.maximum(c[keep], 1)), deg), x))


def excess(c, f, T):
    m = (MID >= T - 5) & (MID < T)
    return c[m].sum() / f[m].sum()


ex = {}
sel = (MID >= 60) & (MID < 150)
for grp in ["B2B", "B2C"]:
    for yr in [2024, 2026, 2028]:
        c = hist(ref, yr, grp)
        f = cfit(c)
        ex[(grp, yr, 85)] = excess(c, f, 85)
        ex[(grp, yr, 100)] = excess(c, f, 100)
        if yr in (2026, 2028):
            pd.DataFrame({"x": MID[sel], "count": c[sel] / 1000, "fit": f[sel] / 1000}).round(4).to_csv(
                OUT + f"data/hist_{grp}_{yr}.csv", index=False)
dib = (ex["B2B", 2028, 85] - ex["B2B", 2026, 85]) - (ex["B2C", 2028, 85] - ex["B2C", 2026, 85])
dib_pl = (ex["B2B", 2026, 85] - ex["B2B", 2024, 85]) - (ex["B2C", 2026, 85] - ex["B2C", 2024, 85])

# bootstrap standard errors (resampling people, 200 draws)
_w = ref[ref.group.isin(["B2B", "B2C"])].pivot(index="id", columns="year", values="declared")
_g = ref[ref.year == 2023].set_index("id").group.loc[_w.index].values
_cols = list(_w.columns)


def _ex85(v):
    c = np.histogram(v, bins=BINS)[0]
    return excess(c, cfit(c), 85)


def _dib(idx, y0, y1):
    g, v = _g[idx], _w.values[idx]
    e = lambda gg, yy: _ex85(v[g == gg, _cols.index(yy)])
    return (e("B2B", y1) - e("B2B", y0)) - (e("B2C", y1) - e("B2C", y0))


_rng = np.random.default_rng(0)
_bm, _bp = [], []
for _ in range(200):
    _idx = _rng.integers(0, len(_w), len(_w))
    _bm.append(_dib(_idx, 2026, 2028))
    _bp.append(_dib(_idx, 2024, 2026))
dib_se, dib_pl_se = float(np.std(_bm)), float(np.std(_bp))


def panel(grp, yr):
    label = {"B2B": "B2B sellers", "B2C": "B2C sellers"}[grp]
    when = "2026 (before reform)" if yr == 2026 else "2028 (after reform)"
    col = "b2b" if grp == "B2B" else "b2c"
    lines = r"\draw[thr] (axis cs:85,0) -- (axis cs:85,\pgfkeysvalueof{/pgfplots/ymax});" + "\n"
    note = f"Excess at 85k: {ex[(grp, yr, 85)]:.1f}$\\times$"
    if yr == 2028:
        lines += r"\draw[thr] (axis cs:100,0) -- (axis cs:100,\pgfkeysvalueof{/pgfplots/ymax});" + "\n"
        note += f"\\\\Excess at 100k: {ex[(grp, yr, 100)]:.1f}$\\times$"
    return (f"\\nextgroupplot[title={{{label}, {when}}}]\n"
            f"\\addplot[ybar interval, fill={col}, draw=white, line width=0.1pt] table[x=x, y=count, col sep=comma, "
            f"x expr=\\thisrow{{x}}-0.5] {{data/hist_{grp}_{yr}.csv}};\n"
            f"\\addplot[fitline] table[x=x, y=fit, col sep=comma] {{data/hist_{grp}_{yr}.csv}};\n"
            f"{lines}"
            f"\\node[anchor=north east, align=right, font=\\scriptsize] at (rel axis cs:0.98,0.96) {{{note}}};\n")


fig1 = r"""% Exhibit 1 -- requires \usepackage{pgfplots} \usepgfplotslibrary{groupplots} \pgfplotsset{compat=1.17}
\definecolor{b2b}{HTML}{2A78D6}
\definecolor{b2c}{HTML}{EB6834}
\begin{figure}[t]
\centering
\caption{Where the self-employed declare their revenue, before and after the reform}
\label{fig:bunching}
\begin{tikzpicture}[thr/.style={black, densely dotted, line width=0.6pt},
                    fitline/.style={black!70, dashed, line width=0.9pt, no markers}]
\begin{groupplot}[group style={group size=2 by 2, horizontal sep=1.0cm, vertical sep=1.5cm,
                               xlabels at=edge bottom, ylabels at=edge left},
                  width=0.46\textwidth, height=4.1cm, xmin=60, xmax=150, ymin=0,
                  xlabel={Declared revenue (EUR thousand)}, ylabel={People (thousand)},
                  title style={font=\small, yshift=-1.5ex}, tick label style={font=\scriptsize},
                  label style={font=\scriptsize}, axis lines=left, enlarge y limits={upper, value=0.08},
                  ytick distance=5, xtick={60,80,85,100,120,140}, xticklabels={60,80,,100,120,140}]
""" + panel("B2B", 2026) + panel("B2B", 2028) + panel("B2C", 2026) + panel("B2C", 2028) + r"""\end{groupplot}
\end{tikzpicture}

\vspace{0.4ex}
\begin{minipage}{0.97\textwidth}\footnotesize
\textit{Difference-in-bunching at EUR 85k} $=(""" + f"{ex['B2B',2028,85]:.2f}-{ex['B2B',2026,85]:.2f})-({ex['B2C',2028,85]:.2f}-{ex['B2C',2026,85]:.2f})={dib:+.2f}" + r"""$
(s.e.\ """ + f"{dib_se:.2f}" + r"""). Placebo (same calculation, 2024 to 2026, no reform): $""" + f"{dib_pl:+.2f}" + r"""$
(s.e.\ """ + f"{dib_pl_se:.2f}" + r"""). Bootstrap standard errors, 200 resamples of people.\\
\textit{Notes:} Bars: number of people per EUR 1,000 revenue bin. Dashed line: counterfactual distribution without
cliffs, a fourth-degree polynomial in log counts fitted outside EUR 80--140k. Excess = people in the EUR 5,000 just
below a threshold divided by the number predicted by the dashed line. Dotted lines: thresholds (VAT at EUR 85k;
flat tax at EUR 100k from 2027). B2B: at least 80\% of 2024--2026 revenue invoiced to businesses; B2C: at most 20\%.
Synthetic data.
\end{minipage}
\end{figure}
"""
open(OUT + "exhibit1_bunching.tex", "w").write(fig1)

# ============================================================== Exhibit 2: classic regression table
YEARS = [2024, 2025, 2026, 2027, 2028]


def build(p, band):
    meta = p[p.year == 2023].set_index("id")
    ids = meta.index[(meta.declared >= band[0]) & (meta.declared < band[1]) & meta.group.isin(["B2B", "B2C"])]
    d = p[p.id.isin(ids) & p.year.isin(YEARS)][["id", "year", "declared"]].copy()
    d["b2b"] = (meta.loc[d.id, "group"].values == "B2B").astype(float)
    d["prof"] = (meta.loc[d.id, "sector"].values == "Professionals").astype(float)
    d["y"] = np.log(d.declared)
    return d


def fe_ols(d, X):
    Z = pd.concat([d[["y"]], X], axis=1)
    Z = Z - Z.groupby(d.id.values).transform("mean")          # person fixed effects (within transformation)
    return sm.OLS(Z.y, Z.drop(columns="y")).fit(cov_type="cluster", cov_kwds={"groups": d.id.values})


def year_terms(d, ctrl):
    """Year fixed effects; with ctrl=True, sector-by-year fixed effects (two sectors -> sector x year dummies)."""
    ks = [k for k in YEARS if k != 2026]
    X = {f"yr{k}": (d.year == k).astype(float) for k in ks}
    if ctrl:
        X.update({f"prof_yr{k}": d.prof * (d.year == k) for k in ks})
    return X


def pooled(d, ctrl):
    X = pd.DataFrame({**year_terms(d, ctrl), "post": d.b2b * (d.year >= 2027)}, index=d.index)
    return fe_ols(d, X)


def event(d, ctrl):
    ks = [k for k in YEARS if k != 2026]
    X = pd.DataFrame({**year_terms(d, ctrl), **{f"es{k}": d.b2b * (d.year == k) for k in ks}}, index=d.index)
    m = fe_ols(d, X)
    return m, float(m.f_test("es2024 = 0, es2025 = 0").pvalue)


d_main, d_pl = build(ref, (82, 85)), build(ref, (45, 50))
m1, p1 = event(d_main, ctrl=False)       # (1) raw event study
m2, p2 = event(d_main, ctrl=True)        # (2) with sector-by-year FE
m3 = pooled(d_main, ctrl=True)           # (3) pooled, headline estimate
m4, p4 = event(d_pl, ctrl=True)          # (4) placebo
models = (m1, m2, m3, m4)


def stars(p):
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.1 else ""


def cell(m, k):
    if k not in m.params:
        return "", ""
    return f"${m.params[k]:.3f}{stars(m.pvalues[k])}$", f"$({m.bse[k]:.3f})$"


rows = [("B2B $\\times$ 2024", "es2024"), ("B2B $\\times$ 2025", "es2025"), ("B2B $\\times$ 2027", "es2027"),
        ("B2B $\\times$ 2028", "es2028"), ("B2B $\\times$ Post (2027--2028)", "post")]
body = ""
for lab, k in rows:
    c = [cell(m, k) for m in models]
    body += (f"{lab} & " + " & ".join(x[0] for x in c) + " \\\\\n & " + " & ".join(x[1] for x in c) + " \\\\[2pt]\n")
npers = lambda d: d.id.nunique() * W
nb2b = lambda d: int(d[d.year == 2026].b2b.sum()) * W
tab2 = r"""% Exhibit 2 -- requires \usepackage{booktabs, threeparttable}
\begin{table}[t]
\centering
\begin{threeparttable}
\caption{Panel difference-in-differences: B2B vs B2C sellers}
\label{tab:did}
\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{3}{c}{Stuck at EUR 82--85k} & Placebo: 45--50k \\
\cmidrule(lr){2-4}\cmidrule(lr){5-5}
 & (1) & (2) & (3) & (4) \\
\midrule
""" + body + r"""\midrule
Pre-trend test, $p$-value (2024 = 2025 = 0) & """ + f"{p1:.2f} & {p2:.2f} & & {p4:.2f}" + r""" \\
Person fixed effects & Yes & Yes & Yes & Yes \\
Year fixed effects & Yes & Yes & Yes & Yes \\
Sector $\times$ year fixed effects & No & Yes & Yes & Yes \\
People (B2B / B2C) & \multicolumn{3}{c}{""" + f"{nb2b(d_main):,} / {npers(d_main)-nb2b(d_main):,}" + r"""} & """ + \
       f"{nb2b(d_pl):,} / {npers(d_pl)-nb2b(d_pl):,}" + r""" \\
Observations (person-years) & \multicolumn{3}{c}{""" + f"{len(d_main)*W:,}" + r"""} & """ + f"{len(d_pl)*W:,}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Notes:} Dependent variable: log declared revenue, 2024--2028. Sample: self-employed who declared
EUR 82--85k (columns 1--3) or EUR 45--50k (column 4) in 2023, before the reform was announced. Treated: B2B sellers
(at least 80\% of revenue invoiced to businesses); control: B2C sellers (at most 20\%). Event-study coefficients are
relative to 2026. Sector $\times$ year fixed effects absorb shocks specific to professionals vs.\ other activities
(e.g.\ the 2026 energy-price shock on consumer-facing businesses). Standard errors clustered by person in parentheses,
computed on the 1-in-10 synthetic sample (conservative). $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$. Synthetic data.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
open(OUT + "exhibit2_did.tex", "w").write(tab2)

# ============================================================== Exhibit 3: fiscal table
def parts(p):
    d, cs, coef, s = p.declared.values, p.cs.values, p.coef.values, p.s.values
    it = np.where(p.flat.values, FLAT_RATE * coef * d, irpef((1 - cs) * d))
    vat = np.where(p.vat.values, VAT_SHARE * (1 - s) * (1 - cs) * d, 0.0)
    return pd.DataFrame({"it": it, "vat": vat, "decl": d}, index=p.id.values)


a, b = parts(ref[ref.year == 2028]), parts(cf[cf.year == 2028])
dd = a.decl - b.decl
cat = np.select([(dd.abs() < 1e-6) & (b.decl > 85) & (b.decl <= 100), (dd > 1e-6) & (b.decl <= 85), dd < -1e-6],
                ["wind", "rise", "fall"], "none")
f = pd.DataFrame({"cat": cat, "it": a.it - b.it, "vat": a.vat - b.vat, "before": b.decl, "after": a.decl})
f = f[f.cat != "none"]
g = f.groupby("cat").agg(n=("it", "size"), it=("it", "sum"), vat=("vat", "sum"), before=("before", "mean"),
                         after=("after", "mean"))
g["n"] *= W
g[["it", "vat"]] *= W / 1000
g["tot"] = g.it + g.vat
names = {"wind": ("Windfall: already at 85--100k", "Same revenue; flat tax instead of IRPEF"),
         "rise": ("Risers: from below 85k", "Declare more; flat tax and VAT on it"),
         "fall": ("Fallers: from above 100k", "Declare less; flat tax instead of IRPEF")}
body = ""
for k in ["wind", "rise", "fall"]:
    r = g.loc[k]
    fmt = lambda v: "$0$" if abs(v) < 0.5 else f"${v:+,.0f}$"
    body += (f"{names[k][0]} & {int(r.n):,} & {r.before:.1f} & {r.after:.1f} & {fmt(r.it)} & {fmt(r.vat)} & "
             f"$\\mathbf{{{r.tot:+,.0f}}}$ \\\\\n")
net = g.tot.sum()
share = (g.loc["rise", "tot"] + g.loc["fall", "tot"]) / -g.loc["wind", "tot"]
tab3 = r"""% Exhibit 3 -- requires \usepackage{booktabs, threeparttable}
\begin{table}[t]
\centering
\begin{threeparttable}
\caption{Who gains and what it costs the state, 2028}
\label{tab:fiscal}
\small\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{lrrrrrr}
\toprule
 & & \multicolumn{2}{c}{Avg.\ revenue (EUR k)} & \multicolumn{3}{c}{Change in tax revenue (EUR m)} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-7}
Group & People & No reform & Reform & Income tax & VAT & Total \\
\midrule
""" + body + r"""\midrule
Net effect & & & & """ + f"${g.it.sum():+,.0f}$ & ${g.vat.sum():+,.0f}$ & $\\mathbf{{{net:+,.0f}}}$" + r""" \\
Behavioural effects as \% of direct cost & & & & & & """ + f"${100*share:+.0f}\\%$" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Notes:} Change in tax revenue in 2028 relative to the same people under the old rules. Income tax:
flat tax, or IRPEF plus regional and municipal add-ons. VAT: net VAT on sales to consumers (VAT on sales to
businesses is deducted by the buyer). Direct cost = windfall; behavioural effects = risers + fallers. INPS
contributions excluded; employees reclassified as freelancers not quantified (would add to the cost). Synthetic data.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
import re
tab3 = re.sub(r"(\d),(\d{3})", r"\1{,}\2", tab3)
open(OUT + "exhibit3_fiscal.tex", "w").write(tab3)

print(f"DiB {dib:+.2f}, placebo {dib_pl:+.2f}")
for i, m in enumerate(models, 1):
    print(f"({i})", {k: (round(m.params[k], 4), round(m.pvalues[k], 3)) for k in m.params.index if k.startswith(("es", "post"))})
print("pretrend p:", round(p1, 3), round(p2, 3), round(p4, 3))
print(g.round(1).to_string(), "\nnet", round(net, 1), "share", round(share, 3))
