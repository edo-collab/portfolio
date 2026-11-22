"""Build both worlds, save the dataset, and print aggregate checks."""
import numpy as np
import pandas as pd
from sim import simulate, taxes, W

MIX = (0.5574, 2.3504, 1.2056, 3.6272, 0.6047, 0.0846, 4.639, 0.35)   # from calibrate.py
N = 229_600                                       # x10; 88.3% declare <= 85k -> ~2.03M flat-tax users (Corte dei conti)

ref = simulate(mix=MIX, n=N, reform=True)
cf = simulate(mix=MIX, n=N, reform=False)        # same people, same shocks, old rules
ref["tax"] = taxes(ref)
cf["tax"] = taxes(cf)

ref.to_parquet("data/panel_reform.parquet")
cf.to_parquet("data/panel_noreform.parquet")


def band_counts(p, year):
    d = p.loc[p.year == year, "declared"]
    bins = [0, 20, 50, 80, 85, 100, 135, np.inf]
    return (pd.cut(d, bins, right=True).value_counts().sort_index() * W).astype(int)


print("2024 real-people counts by declared band (old rules):")
print(band_counts(ref, 2024))
print("\n2028 counts, reform world:")
print(band_counts(ref, 2028))
print("\n2028 counts, no-reform world:")
print(band_counts(cf, 2028))

y = 2028
diff = (cf.loc[cf.year == y, "tax"].values - ref.loc[ref.year == y, "tax"].values).sum() * W
print(f"\nNet revenue cost of reform in {y}: EUR {diff/1e3:,.0f} million")
