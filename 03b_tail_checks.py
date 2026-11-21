"""Robustness of the tail estimates (no bootstrap, fast).
- Hill (1975) and Dekkers-Einmahl-de Haan (1989) moment estimates at k = number of exceedances of the chosen u.
- HHS: reports still under investigation removed (their counts can be revised, e.g. filed with a 500 placeholder);
  reports at exactly 500 or 501 removed; per-year blocks dropped one at a time (jackknife by submission year).
Output: results_tail_checks.json
"""
import json, os, warnings, numpy as np, pandas as pd
from common_evt import gpd_fit, profile_ci_xi
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(f"{HERE}/results_tail.json")); h = pd.read_pickle(f"{HERE}/data/hhs_clean.pkl"); v = pd.read_pickle(f"{HERE}/data/vcdb_clean.pkl")
v = v[v.records >= 500]

def hill_moment(x, k):
    s = np.sort(x)[::-1]; l = np.log(s[:k]) - np.log(s[k]); m1 = l.mean(); m2 = (l ** 2).mean()
    return float(m1), float(m1 + 1 - 0.5 / (1 - m1 ** 2 / m2))

def fit(x, u):
    y = x[x > u] - u; xi, b = gpd_fit(y); lo, hi = profile_ci_xi(y, xi, b); return {"n_exc": int(len(y)), "xi": xi, "ci": [lo, hi]}

S = {"HHS all": h.individuals.values.astype(float), "HHS hacking/IT": h.loc[h.hacking, "individuals"].values.astype(float),
     "VCDB independent of HHS (records >= 500)": v.loc[(v.sub_source != "phidbr") & ~v.hhs_reference & ~v.ai_encoded, "records"].values.astype(float)}
R = {"estimators": {}, "hhs_checks": {}}
for k, x in S.items():
    m = T["main"][k]; hl, mo = hill_moment(x, m["n_exc"]); R["estimators"][k] = {"k": m["n_exc"], "gpd_ml": m["xi"], "hill": hl, "moment": mo}
u = T["main"]["HHS all"]["u"]; x = h.individuals.values.astype(float)
R["hhs_checks"]["baseline"] = fit(x, u)
R["hhs_checks"]["archive only (closed investigations)"] = fit(h.loc[h.list == "archive", "individuals"].values.astype(float), u)
R["hhs_checks"]["without reports of exactly 500 or 501"] = fit(h.loc[~h.individuals.isin([500, 501]), "individuals"].values.astype(float), u)
jk = {}
for yr in sorted(h.year.unique()):
    jk[int(yr)] = fit(h.loc[h.year != yr, "individuals"].values.astype(float), u)["xi"]
R["hhs_checks"]["drop one submission year"] = {"min": min(jk.values()), "max": max(jk.values()), "by_year": jk}
json.dump(R, open(f"{HERE}/results_tail_checks.json", "w"), indent=1)
print(json.dumps(R["estimators"], indent=1)); print({k: (v if k != "drop one submission year" else (round(v["min"], 3), round(v["max"], 3))) for k, v in R["hhs_checks"].items()})
