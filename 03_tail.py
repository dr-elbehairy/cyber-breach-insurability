"""RQ2: how heavy is the severity tail (individuals / records affected), and is the mean finite?

For each sample: GPD fitted by ML to exceedances over thresholds on a log grid (1,000 to 1,000,000; at least
100 exceedances); threshold chosen by ForwardStop on bootstrap Anderson-Darling p-values (Bader, Yan & Zhang 2018)
for the three main samples; profile-likelihood 95% CI for xi; likelihood-ratio test of xi = 1 (one-sided);
nonparametric bootstrap probability that xi >= 1; Hill estimates.
Bounded-support check: dual (log) transformation of Cirillo & Taleb (2016) with lower bound L = 500 and upper bound H
= the U.S. resident population (value and source in config.json); shadow mean computed by numerical integration.
Output: results_tail.json
"""
import json, os, warnings, numpy as np, pandas as pd
from scipy import stats, optimize, integrate
from common_evt import gpd_fit, gpd_loglik, profile_ci_xi, ad_pvalue, forward_stop
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(1121)
CFG = json.load(open(f"{HERE}/config.json"))
h = pd.read_pickle(f"{HERE}/data/hhs_clean.pkl"); v = pd.read_pickle(f"{HERE}/data/vcdb_clean.pkl")
v = v[v.records >= 500].copy()
GRID = np.unique(np.round(np.logspace(3, 6, 19), -2))

def lr_xi1(y, xi, beta):
    """LR statistic for H0: xi = 1 against xi < 1 (one-sided: zero when xi_hat >= 1)."""
    if xi >= 1: return 0.0, 1.0
    r = optimize.minimize_scalar(lambda b: -gpd_loglik(y, 1.0, b), bounds=(1e-6, 100 * beta), method="bounded")
    lr = 2 * (gpd_loglik(y, xi, beta) + r.fun); return float(lr), float(0.5 * stats.chi2.sf(lr, 1))

def fit_at(x, u, boot=0):
    y = x[x > u] - u; xi, beta = gpd_fit(y); lo, hi = profile_ci_xi(y, xi, beta)
    lr, p = lr_xi1(y, xi, beta)
    out = {"u": float(u), "n_exc": int(len(y)), "xi": xi, "beta": beta, "xi_ci95": [lo, hi], "lr_xi1": lr, "p_xi_lt_1": p}
    if boot:
        bs = [gpd_fit(rng.choice(y, len(y)))[0] for _ in range(boot)]; out["boot_P_xi_ge_1"] = float(np.mean(np.array(bs) >= 1))
        out["boot_xi_p05_p95"] = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
    return out

def hill(x, ks):
    s = np.sort(x)[::-1]; return {int(k): float(np.mean(np.log(s[:k])) - np.log(s[k])) for k in ks if k < len(s)}

def select_threshold(x, grid=GRID, B=200):
    """ForwardStop over the grid. Returns (thresholds, p-values, chosen u, accepted flag). If every threshold is
    rejected, the highest grid threshold is returned with accepted = False and must be reported as such."""
    us = [u for u in grid if (x > u).sum() >= 100]; pv = []
    for u in us:
        y = x[x > u] - u; xi, beta = gpd_fit(y); pv.append(ad_pvalue(y, xi, beta, rng, B))
    k = forward_stop(pv)
    return (us, pv, us[k], True) if k < len(us) else (us, pv, us[-1], False)

def dual_shadow(x, L, H, u_x):
    """Cirillo-Taleb dual transform z = L - H log((H - x)/(H - L)); GPD on z above u_z = z(u_x);
    shadow mean of x above u_x by integrating x(z) against the fitted GPD density."""
    z = L - H * np.log((H - x) / (H - L)); uz = L - H * np.log((H - u_x) / (H - L))
    y = z[z > uz] - uz; xi, beta = gpd_fit(y)
    # E[x | z > uz] = H - (H - L) exp((L - uz)/H) E[exp(-Y/H)], Y ~ GPD(xi, beta); integrate on the quantile scale
    g = lambda p: np.exp(-stats.genpareto.ppf(p, xi, 0, beta) / H)
    eg, err = integrate.quad(g, 0, 1, limit=400)
    m = H - (H - L) * np.exp((L - uz) / H) * eg
    emp = float(x[x > u_x].mean())
    return {"xi_dual": xi, "beta_dual": beta, "n_exc": int(len(y)), "shadow_mean_above_u": float(m), "sample_mean_above_u": emp, "uz": float(uz), "z_minus_x_at_max": float(z.max() - x.max()),
            "ratio_shadow_to_sample": float(m / emp)}

if __name__ == "__main__":
    S = {"HHS all": h.individuals.values.astype(float), "HHS hacking/IT": h.loc[h.hacking, "individuals"].values.astype(float),
         "VCDB all (records >= 500)": v.records.values.astype(float),
         "VCDB independent of HHS (records >= 500)": v.loc[(v.sub_source != "phidbr") & ~v.hhs_reference & ~v.ai_encoded, "records"].values.astype(float)}
    R = {"grid": GRID.tolist(), "main": {}, "subsets": {}, "hill": {}, "dual": {}}
    for k, x in S.items():
        grid = GRID if k.startswith("HHS") else np.unique(np.round(np.logspace(3, 8, 26), -2))
        us, pv, u, acc = select_threshold(x, grid); f = fit_at(x, u, boot=1000); f["threshold_accepted"] = acc
        f["threshold_path"] = [{"u": float(a), "ad_p": float(b), **{kk: fit_at(x, a)[kk] for kk in ["n_exc", "xi", "xi_ci95"]}} for a, b in zip(us, pv)]
        R["main"][k] = f; R["hill"][k] = hill(x, [50, 100, 200, 400, 800, 1600])
        print(k, "accepted", acc, "u", u, "n", f["n_exc"], "xi %.3f [%.3f, %.3f]" % (f["xi"], *f["xi_ci95"]), "P(xi>=1) boot", f["boot_P_xi_ge_1"], "p(xi<1)", round(f["p_xi_lt_1"], 4))
    u0 = R["main"]["HHS all"]["u"]
    subs = {"HHS non-hacking": ~h.hacking, "HHS business associate involved": h.ba, "HHS no business associate": ~h.ba,
            "HHS submitted 2009-2017": h.year <= 2017, "HHS submitted 2018-2026": h.year >= 2018,
            "HHS healthcare providers": h.entity_type == "Healthcare Provider", "HHS health plans": h.entity_type == "Health Plan",
            "HHS business associates (reporting entity)": h.entity_type == "Business Associate",
            "HHS excluding the largest event": h.individuals < h.individuals.max()}
    for k, m in subs.items():
        x = h.loc[m, "individuals"].values.astype(float)
        R["subsets"][k] = {f"u={int(uu)}": fit_at(x, uu) for uu in [u0, 10000, 100000] if (x > uu).sum() >= 50}
    vh = v[v.industry.str.startswith("62")].records.values.astype(float)
    R["subsets"]["VCDB healthcare (NAICS 62)"] = {f"u={int(uu)}": fit_at(vh, uu) for uu in [u0, 10000, 100000] if (vh > uu).sum() >= 50}
    vus = v[v.country.str.contains("US")].records.values.astype(float)
    R["subsets"]["VCDB United States"] = {f"u={int(uu)}": fit_at(vus, uu) for uu in [u0, 10000, 100000] if (vus > uu).sum() >= 50}
    H = float(CFG["us_population"]["value"])
    for k in ["HHS all", "HHS hacking/IT"]:
        R["dual"][k] = dual_shadow(S[k], 500.0, H, R["main"][k]["u"])
        R["dual"][k]["sensitivity_H"] = {lab: dual_shadow(S[k], 500.0, hh, R["main"][k]["u"])["shadow_mean_above_u"]
                                          for lab, hh in [("H=max observed x 1.01", 1.01 * S[k].max()), ("H=population/2", H / 2), ("H=population", H), ("H=2 x population", 2 * H), ("H=10 x population", 10 * H)]}
    R["dual_H"] = H
    json.dump(R, open(f"{HERE}/results_tail.json", "w"), indent=1)
    for k, d in R["subsets"].items(): print(k, {kk: (vv["n_exc"], round(vv["xi"], 3), [round(c, 3) for c in vv["xi_ci95"]]) for kk, vv in d.items()})
    print("dual", R["dual"])
