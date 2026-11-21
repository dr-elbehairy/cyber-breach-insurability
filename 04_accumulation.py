"""RQ3: concentration, third-party (business-associate) accumulation, diversification and policy limits.

1. Concentration: share of all individuals affected in the largest 1, 10 and top-1% reports; per-year share of the
   largest report in the annual total; max-to-sum ratio as the sample grows (chronological order).
2. Business associates: share of reports and of individuals for reports with a business associate involved.
3. Diversification: VaR(99.5%) of the average severity of N independent breaches, N = 1..100, from the spliced
   severity model (empirical below u, fitted GPD above), unlimited and with per-breach limits; relative to N = 1.
   Repeated at the lower and upper ends of the 95% CI for xi.
4. Limited expected value E[min(X, M)] per breach for limits M = 1e4 ... 1e8 and its elasticity in M.
Output: results_accumulation.json
"""
import json, os, warnings, numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(20261122)
h = pd.read_pickle(f"{HERE}/data/hhs_clean.pkl"); T = json.load(open(f"{HERE}/results_tail.json"))
x = h.individuals.values.astype(float)

def spliced_sampler(x, u, xi, beta):
    body = x[x <= u]; q = (x > u).mean()
    def draw(size):
        s = rng.choice(body, size); m = rng.random(size) < q
        s[m] = u + stats.genpareto.rvs(xi, 0, beta, size=int(m.sum()), random_state=rng); return s
    return draw

def lev(x, u, xi, beta, M, nmc=2_000_000):
    """E[min(X, M)] under the spliced model, by exact body average plus GPD integral for the tail part."""
    body = x[x <= u]; q = (x > u).mean(); eb = np.minimum(body, M).mean()
    if M <= u: return float((1 - q) * eb + q * M)
    # E[min(u + Y, M)] = u + E[min(Y, M - u)] = u + int_0^{M-u} S(y) dy
    from scipy import integrate
    S = lambda y: stats.genpareto.sf(y, xi, 0, beta)
    et, _ = integrate.quad(S, 0, M - u, limit=400)
    return float((1 - q) * eb + q * (u + et))

if __name__ == "__main__":
    R = {}
    s = np.sort(x)[::-1]; tot = s.sum()
    R["concentration"] = {"total_individuals": float(tot), "n_reports": int(len(s)), "share_top1": float(s[0] / tot),
                          "share_top10": float(s[:10].sum() / tot), "share_top1pct": float(s[: int(np.ceil(0.01 * len(s)))].sum() / tot),
                          "n_top1pct": int(np.ceil(0.01 * len(s))), "median": float(np.median(x)), "mean": float(x.mean()),
                          "share_reports_ge_1m": float((x >= 1e6).mean()), "n_ge_1m": int((x >= 1e6).sum()),
                          "share_individuals_ge_1m": float(x[x >= 1e6].sum() / tot)}
    yr = h.groupby("year").individuals.agg(["sum", "max", "count"]); yr["max_share"] = yr["max"] / yr["sum"]
    R["by_year"] = {int(k): {"total": float(r["sum"]), "max": float(r["max"]), "count": int(r["count"]), "max_share": float(r["max_share"]),
                             "largest_entity": h.loc[(h.year == k)].sort_values("individuals").entity.iloc[-1]} for k, r in yr.iterrows()}
    hs = h.sort_values("submitted"); c = hs.individuals.values.astype(float)
    ratio = np.maximum.accumulate(c) / np.cumsum(c); idx = np.unique(np.round(np.logspace(1, np.log10(len(c)), 40)).astype(int)) - 1
    R["max_to_sum_path"] = {int(i + 1): float(ratio[i]) for i in idx}
    ba = h.ba.values
    R["business_associate"] = {"share_reports": float(ba.mean()), "share_individuals": float(x[ba].sum() / tot),
                               "share_individuals_excl_largest": float(np.sort(x[ba])[:-1].sum() / (tot - x.max())),
                               "median_ba": float(np.median(x[ba])), "median_non_ba": float(np.median(x[~ba])),
                               "top10_ba_count": int(h.sort_values("individuals").tail(10).ba.sum()),
                               "top10": h.sort_values("individuals").tail(10)[["entity", "individuals", "submitted", "ba", "entity_type", "breach_type"]].assign(submitted=lambda d: d.submitted.astype(str)).to_dict("records"),
                               "by_year_share_individuals": {int(k): float(g.loc[g.ba, "individuals"].sum() / g.individuals.sum()) for k, g in h.groupby("year")}}
    m = T["main"]["HHS all"]; u, xi, beta = m["u"], m["xi"], m["beta"]; lo, hi = m["xi_ci95"]
    Ns = [1, 2, 5, 10, 25, 50, 100]; nsim = 200_000; div = {}
    for lab, xx in [("xi_hat", xi), ("xi_ci_low", lo), ("xi_ci_high", hi)]:
        bb = beta if lab == "xi_hat" else beta  # scale held at the ML value; sensitivity to xi only
        draw = spliced_sampler(x, u, xx, bb); out = {}
        for L in [None, 1e6, 1e7]:
            v = []
            for N in Ns:
                sev = draw(nsim * N).reshape(nsim, N)
                if L is not None: sev = np.minimum(sev, L)
                v.append(float(np.quantile(sev.mean(1), 0.995)))
            out["unlimited" if L is None else f"limit_{int(L)}"] = {"VaR995_mean": dict(zip(Ns, v)), "relative_to_N1": dict(zip(Ns, [a / v[0] for a in v]))}
        div[lab] = {"xi": float(xx), **out}
    R["diversification"] = div
    Ms = [1e4, 1e5, 1e6, 1e7, 1e8]; L = [lev(x, u, xi, beta, M) for M in Ms]
    R["lev"] = {"limits": Ms, "lev": L, "elasticity": [float(np.log(L[i + 1] / L[i]) / np.log(Ms[i + 1] / Ms[i])) for i in range(len(Ms) - 1)],
                "lev_ci_low": [lev(x, u, lo, beta, M) for M in Ms], "lev_ci_high": [lev(x, u, hi, beta, M) for M in Ms]}
    json.dump(R, open(f"{HERE}/results_accumulation.json", "w"), indent=1, default=str)
    print(json.dumps({k: R[k] for k in ["concentration"]}, indent=1))
    print({k: v for k, v in R["business_associate"].items() if k not in ["top10", "by_year_share_individuals"]})
    for lab, d in div.items(): print(lab, round(d["xi"], 3), {k: {n: round(r, 3) for n, r in d[k]["relative_to_N1"].items()} for k in ["unlimited", "limit_1000000", "limit_10000000"]})
    print("LEV", [round(a) for a in L], [round(e, 3) for e in R["lev"]["elasticity"]])
