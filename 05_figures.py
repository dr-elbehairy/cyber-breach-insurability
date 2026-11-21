"""Figures 1-5 from the results files. Output: ../figures/fig1..fig5 (.png, 300 dpi)."""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figures") if os.path.exists(os.path.join(HERE, "run_all.py")) else os.path.join(os.path.dirname(HERE), "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
NAVY, PETROL, GOLD, GREY = "#1C4076", "#007A88", "#B8860B", "#8C8C8C"
F = json.load(open(f"{HERE}/results_frequency.json")); T = json.load(open(f"{HERE}/results_tail.json")); A = json.load(open(f"{HERE}/results_accumulation.json"))
h = pd.read_pickle(f"{HERE}/data/hhs_clean.pkl"); v = pd.read_pickle(f"{HERE}/data/vcdb_clean.pkl")

# Fig 1: annual reports by type and business-associate share of individuals
yrs = list(range(2010, 2026)); hk = h[h.year.isin(yrs)]
a = hk.groupby(["year", "hacking"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax[0].bar(a.index, a[True], color=NAVY, label="Hacking/IT incident"); ax[0].bar(a.index, a[False], bottom=a[True], color=GOLD, label="Other types")
ax[0].set_title("(a) Reports per year, HHS-OCR", loc="left"); ax[0].set_ylabel("Reports"); ax[0].legend(frameon=False, fontsize=8)
bs = A["business_associate"]["by_year_share_individuals"]
ax[1].plot(yrs, [100 * bs[str(y)] for y in yrs], "o-", color=PETROL, ms=3)
ax[1].set_title("(b) Share of individuals in reports\nwith a business associate (%)", loc="left"); ax[1].set_ylim(0, 100)
for a_ in ax: a_.set_xticks([2010, 2013, 2016, 2019, 2022, 2025])
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_frequency.png", dpi=300); plt.close(fig)

# Fig 2: rolling one-year-ahead 90% intervals
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)
for j, g in enumerate(["all", "hacking"]):
    for k, (sp, c, off) in enumerate([("trend", NAVY, -0.22), ("break", PETROL, 0.0), ("recent3", GOLD, 0.22)]):
        by = F["out_of_time"][g][sp]["by_year"]; ys = sorted(int(y) for y in by)
        lo = [by[str(y)]["p05"] for y in ys]; hi = [by[str(y)]["p95"] for y in ys]
        ax[j].vlines(np.array(ys) + off, lo, hi, color=c, lw=2, label={"trend": "Log-linear trend", "break": "Trend with one break", "recent3": "Last 3 years, flat"}[sp])
    by = F["out_of_time"][g]["trend"]["by_year"]; ys = sorted(int(y) for y in by)
    ax[j].plot(ys, [by[str(y)]["observed"] for y in ys], "k_", ms=10, mew=2, label="Observed")
    ax[j].set_title({"all": "(a) All reports", "hacking": "(b) Hacking/IT incidents"}[g], loc="left"); ax[j].set_yscale("log")
ax[0].set_ylabel("Reports in the year (log scale)"); ax[1].legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_forecasts.png", dpi=300); plt.close(fig)

# Fig 3: xi against threshold
keys = ["HHS all", "HHS hacking/IT", "VCDB independent of HHS (records >= 500)"]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for j, k in enumerate(keys):
    m = T["main"][k]; p = m["threshold_path"]; u = [t["u"] for t in p]
    ax[j].fill_between(u, [t["xi_ci95"][0] for t in p], [t["xi_ci95"][1] for t in p], color=PETROL, alpha=0.25, lw=0)
    ax[j].plot(u, [t["xi"] for t in p], "-", color=NAVY); ax[j].axvline(m["u"], color=GOLD, ls="--", lw=1)
    ax[j].axhline(1, color="k", lw=0.7); ax[j].axhline(0.5, color=GREY, lw=0.7, ls=":")
    ax[j].set_xscale("log"); ax[j].set_title({"HHS all": "(a) HHS-OCR, all", "HHS hacking/IT": "(b) HHS-OCR, hacking/IT"}.get(k, "(c) VCDB, not HHS-sourced"), loc="left")
    ax[j].set_xlabel("Threshold u (individuals / records)")
ax[0].set_ylabel("GPD shape ξ (95% profile CI)"); ax[0].set_ylim(0, 4)
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_xi_threshold.png", dpi=300); plt.close(fig)

# Fig 4: empirical survival, log-log, with fitted GPD tails
fig, ax = plt.subplots(figsize=(4.6, 3.2))
vi = v[(v.records >= 500) & (v.sub_source != "phidbr") & ~v.hhs_reference & ~v.ai_encoded].records.values.astype(float)
for x, lab, c, k in [(h.individuals.values.astype(float), "HHS-OCR (individuals)", NAVY, "HHS all"), (vi, "VCDB, not HHS-sourced (records)", GOLD, "VCDB independent of HHS (records >= 500)")]:
    s = np.sort(x); sv = 1 - np.arange(len(s)) / len(s); ax.loglog(s, sv, ".", ms=1.5, color=c, label=lab)
    m = T["main"][k]; q = (x > m["u"]).mean(); g = np.logspace(np.log10(m["u"]), np.log10(s[-1]), 50)
    ax.loglog(g, q * stats.genpareto.sf(g - m["u"], m["xi"], 0, m["beta"]), "-", color=c, lw=1)
ax.set_xlabel("Size of breach"); ax.set_ylabel("P(size > x | size ≥ 500)"); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_survival.png", dpi=300); plt.close(fig)

# Fig 5: diversification and limited expected value
D = A["diversification"]; Ns = [1, 2, 5, 10, 25, 50, 100]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for lab, c in [("unlimited", NAVY), ("limit_10000000", PETROL), ("limit_1000000", GOLD)]:
    ax[0].plot(Ns, [D["xi_hat"][lab]["relative_to_N1"][str(n)] for n in Ns], "o-", color=c, ms=3, label={"unlimited": "No limit", "limit_10000000": "Limit 10 million", "limit_1000000": "Limit 1 million"}[lab])
    ax[0].fill_between(Ns, [D["xi_ci_low"][lab]["relative_to_N1"][str(n)] for n in Ns], [D["xi_ci_high"][lab]["relative_to_N1"][str(n)] for n in Ns], color=c, alpha=0.15, lw=0)
ax[0].set_xscale("log"); ax[0].set_xlabel("Number of independent breaches N"); ax[0].set_ylabel("VaR99.5(mean of N) / VaR99.5(N=1)")
ax[0].set_title("(a) Diversification", loc="left"); ax[0].legend(frameon=False, fontsize=7)
L = A["lev"]; ax[1].loglog(L["limits"], L["lev"], "o-", color=NAVY, ms=3)
ax[1].fill_between(L["limits"], L["lev_ci_low"], L["lev_ci_high"], color=NAVY, alpha=0.15, lw=0)
ax[1].set_xlabel("Per-breach limit M (individuals)"); ax[1].set_ylabel("E[min(X, M)]"); ax[1].set_title("(b) Limited expected value", loc="left")
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_diversification.png", dpi=300); plt.close(fig)
print("figures written to", FIG)
