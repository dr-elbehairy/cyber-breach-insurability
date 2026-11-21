"""RQ1: frequency dynamics of HHS-OCR breach reports.

Monthly counts Jan 2010 - Dec 2025 (complete calendar years; Oct-Dec 2009 and 2026 excluded from fitting).
Models: Poisson and negative binomial (NB2) GLMs with a log-linear trend and month-of-year effects;
single-break piecewise trend chosen by profile likelihood; Ljung-Box test on Pearson residuals;
out-of-time test: fit 2010-2021, predict annual totals 2022-2025 with simulation intervals.
Output: results_frequency.json
"""
import json, os, warnings, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(20261121)
h = pd.read_pickle(f"{HERE}/data/hhs_clean.pkl")

def monthly(mask=None):
    d = h if mask is None else h[mask]
    idx = pd.period_range("2010-01", "2025-12", freq="M")
    c = d.submitted.dt.to_period("M").value_counts().reindex(idx, fill_value=0)
    return pd.DataFrame({"y": c.values, "t": np.arange(len(idx)) / 12.0, "m": idx.month, "year": idx.year}, index=idx)

def design(df, brk=None):
    X = pd.get_dummies(df["m"].astype("category"), prefix="m", drop_first=True).astype(float)
    X.insert(0, "t", df["t"].values)
    if brk is not None: X.insert(1, "t_after", np.maximum(df["t"].values - (brk - 2010), 0.0))
    return sm.add_constant(X, has_constant="add")

def fit_nb(df, brk=None):
    X = design(df, brk); return sm.NegativeBinomial(df["y"].values, X).fit(disp=0, maxiter=500), X

def summarise(label, mask):
    df = monthly(mask); X = design(df)
    pois = sm.GLM(df.y.values, X, family=sm.families.Poisson()).fit()
    nb, _ = fit_nb(df)
    lr = 2 * (nb.llf - pois.llf); p_lr = 0.5 * stats.chi2.sf(lr, 1)  # boundary: half chi-square(1)
    alpha = float(nb.params["alpha"]); g = float(np.exp(nb.params["t"]) - 1)
    ci_t = nb.conf_int().loc["t"].values
    # single break search
    brks = list(range(2012, 2024)); ll = []
    for b in brks: ll.append(fit_nb(df, b)[0].llf)
    b_hat = brks[int(np.argmax(ll))]; nbb, _ = fit_nb(df, b_hat)
    g_before = float(np.exp(nbb.params["t"]) - 1); g_after = float(np.exp(nbb.params["t"] + nbb.params["t_after"]) - 1)
    lr_break = float(2 * (max(ll) - nb.llf))
    # a-priori break: HIPAA Omnibus Rule compliance date 23 Sep 2013 (presumption of breach replaces the harm standard)
    X0 = design(df); step = (df.index >= pd.Period("2013-10", "M")).astype(float)
    Xo = X0.copy(); Xo.insert(1, "omni_step", step); Xo.insert(2, "omni_slope", np.maximum(df["t"].values - (2013 + 9 / 12 - 2010), 0.0) * step)
    nbo = sm.NegativeBinomial(df.y.values, Xo).fit(disp=0, maxiter=1000)
    lr_omni = float(2 * (nbo.llf - nb.llf)); p_omni = float(stats.chi2.sf(lr_omni, 2))
    omni = {"lr": lr_omni, "p": p_omni, "step_ratio": float(np.exp(nbo.params["omni_step"])),
            "growth_before": float(np.exp(nbo.params["t"]) - 1), "growth_after": float(np.exp(nbo.params["t"] + nbo.params["omni_slope"]) - 1)}
    # residual dependence after the break model
    mu = nbb.predict(); a = float(nbb.params["alpha"]); pr = (df.y.values - mu) / np.sqrt(mu + a * mu ** 2)
    nres = len(pr); acf = [float(np.corrcoef(pr[:-l], pr[l:])[0, 1]) for l in range(1, 13)]
    # dispersion of raw monthly counts
    disp = float(df.y.var(ddof=1) / df.y.mean())
    return df, {"label": label, "months": int(len(df)), "total": int(df.y.sum()), "mean_month": float(df.y.mean()),
                "dispersion_index_raw": disp, "pois_llf": float(pois.llf), "nb_llf": float(nb.llf), "lr_overdisp": float(lr),
                "p_overdisp": float(p_lr), "nb_alpha": alpha, "growth_pa": g, "growth_pa_ci": [float(np.exp(ci_t[0]) - 1), float(np.exp(ci_t[1]) - 1)],
                "break_year": int(b_hat), "break_profile": dict(zip(brks, [float(x) for x in ll])), "lr_break_vs_linear": lr_break,
                "omnibus_2013": omni, "growth_before_break": g_before, "growth_after_break": g_after, "nb_alpha_break": a,
                "resid_acf_1_12": acf, "acf_band": float(2 / np.sqrt(nres)), "acf_outside_band": int(sum(abs(a) > 2 / np.sqrt(nres) for a in acf)),
                "annual": df.groupby("year").y.sum().astype(int).to_dict()}

FALLBACKS = []

def nb_fit_robust(y, X):
    """NB2 by ML (newton, then bfgs). If ML or its Hessian fails (alpha near zero), use the Poisson GLM
    for the mean coefficients and a moment estimate of alpha; each fallback is recorded."""
    for meth in ["newton", "bfgs"]:
        try:
            m = sm.NegativeBinomial(y, X).fit(disp=0, maxiter=1000, method=meth)
            cov = m.cov_params(); b = m.params
            if m.mle_retvals.get("converged", True) and np.all(np.isfinite(cov.values)):
                return b.drop("alpha"), cov.drop("alpha").drop("alpha", axis=1), float(b["alpha"]), "nb"
        except Exception:
            pass
    g = sm.GLM(y, X, family=sm.families.Poisson()).fit(); mu = g.predict()
    a = max(float(np.sum(((y - mu) ** 2 - mu) / mu ** 2) / max(len(y) - X.shape[1], 1)), 1e-8)
    FALLBACKS.append(int(len(y))); return g.params, g.cov_params(), a, "poisson_moment"

SPECS = ["trend", "break", "recent3"]

def predict_next_year(df, yr, spec, B=10000):
    """Simulated distribution of the year-`yr` total from a model fitted on all months before `yr`.
    trend: NB with log-linear trend + month effects on the whole history;
    break: as trend plus one slope change at the profile-likelihood break year (searched up to yr-2);
    recent3: NB with month effects only, fitted on the last 36 months (flat level)."""
    tr = df[df.year < yr]; te = df[df.year == yr]
    if spec == "trend":
        m = nb_fit_robust(tr.y.values, design(tr)); Xte = design(df).loc[te.index]
    elif spec == "break":
        brks = list(range(2012, yr - 1)); lls = [fit_nb(tr, b)[0].llf for b in brks]; b = brks[int(np.argmax(lls))]
        m = nb_fit_robust(tr.y.values, design(tr, b)); Xte = design(df, b).loc[te.index]
    else:
        tr3 = tr[tr.year >= yr - 3]; X3 = design(tr3).drop(columns="t")
        m = nb_fit_robust(tr3.y.values, X3); Xte = design(df).loc[te.index].drop(columns="t")
    beta, cov, a, how = m
    draws = rng.multivariate_normal(beta.values, cov.values, size=B)
    mu = np.exp(np.clip(draws @ Xte[beta.index].values.T, -20, 20)); a = max(a, 1e-8); r = 1 / a
    s = rng.negative_binomial(r, r / (r + mu)).sum(1); obs = int(te.y.sum())
    lo, md, hi = np.percentile(s, [5, 50, 95])
    return {"observed": obs, "median": float(md), "p05": float(lo), "p95": float(hi), "inside90": bool(lo <= obs <= hi),
            "pit": float((s < obs).mean() + 0.5 * (s == obs).mean()), "fit": how, "ape": float(abs(md - obs) / obs)}

def out_of_time(mask, label, years=range(2014, 2026)):
    df = monthly(mask); res = {}
    for spec in SPECS:
        out = {int(y): predict_next_year(df, y, spec) for y in years}
        res[spec] = {"by_year": out, "coverage90": float(np.mean([v["inside90"] for v in out.values()])),
                     "n_forecasts": len(out), "below_p05": int(sum(v["observed"] < v["p05"] for v in out.values())),
                     "above_p95": int(sum(v["observed"] > v["p95"] for v in out.values())),
                     "mape": float(np.mean([v["ape"] for v in out.values()]))}
    return res

if __name__ == "__main__":
    groups = {"all": None, "hacking": h.hacking, "non_hacking": ~h.hacking, "ba": h.ba, "non_ba": ~h.ba}
    R = {"summary": {}, "out_of_time": {}}
    for k, m in groups.items():
        _, s = summarise(k, m); R["summary"][k] = s
        R["out_of_time"][k] = out_of_time(m, k)
    # partial 2026 (Jan-Sep, months fully past at download) vs. 2010-2025 break model, for the record only
    df = monthly(None); nbb, _ = fit_nb(df, R["summary"]["all"]["break_year"])
    R["note_2026_jan_sep_observed"] = int(((h.submitted >= "2026-01-01") & (h.submitted < "2026-10-01")).sum())
    R["fallback_fits"] = len(FALLBACKS)
    json.dump(R, open(f"{HERE}/results_frequency.json", "w"), indent=1)
    print("fallback fits", len(FALLBACKS))
    for k, s in R["summary"].items():
        print(k, s["total"], "disp", round(s["dispersion_index_raw"], 2), "alpha", round(s["nb_alpha"], 3), "g", round(s["growth_pa"], 3),
              "break", s["break_year"], round(s["growth_before_break"], 3), round(s["growth_after_break"], 3), "LRb", round(s["lr_break_vs_linear"], 1), "omni", {k: round(v, 3) for k, v in s["omnibus_2013"].items()}, "ACF out", s["acf_outside_band"], [round(a, 2) for a in s["resid_acf_1_12"][:3]])
        print("   OOT", {sp: (R["out_of_time"][k][sp]["coverage90"], R["out_of_time"][k][sp]["below_p05"], R["out_of_time"][k][sp]["above_p95"], round(R["out_of_time"][k][sp]["mape"], 3)) for sp in SPECS})
