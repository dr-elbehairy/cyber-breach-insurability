# Is cyber risk insurable? Data-breach records under mandatory notification

Replication code and data for:

> Elbehairy, M. (2026). *Is Cyber Risk Insurable? Frequency Dynamics, Tail Heaviness and Accumulation in Data-Breach Records under Mandatory Notification Regimes.* Working paper, Department of Statistics and Insurance, Suez Canal University.

## Data
| Source | What | In this repository |
|---|---|---|
| HHS-OCR breach portal (ocrportal.hhs.gov) | All HIPAA breaches affecting 500+ individuals, both lists exported 2026-11-21 (8,087 reports after cleaning) | Yes, `data_raw/hhs_*.csv` (U.S. federal government data) |
| VERIS Community Database (github.com/vz-risk/VCDB) | Publicly reported incidents, joined JSON at commit 5ce87b9 | No; `00_download_data.py` fetches it and checks the SHA-256 |
| FRED POPTHM (U.S. Bureau of Economic Analysis) | U.S. population, upper bound for the dual transformation | Yes, `data_raw/FRED_POPTHM.csv` |

## Run
```
pip install -r requirements.txt
python run_all.py
```
Scripts: `01_data.py` (cleaning) → `02_frequency.py` (negative binomial models, rolling one-year-ahead forecasts) → `03_tail.py` (GPD, ForwardStop threshold on bootstrap Anderson–Darling p-values, profile CIs, dual transformation) → `03b_tail_checks.py` (Hill, moment, sub-sample checks) → `04_accumulation.py` (concentration, business associates, diversification, limited expected value) → `05_figures.py`. Results are written to `results_*.json`; the versions used in the paper are in `results/`.

## Main results
- Nominal 90% one-year-ahead intervals for annual report counts cover 25–83% of out-of-sample years (2014–2025), across three forecasting rules and five groups.
- GPD shape above 146,800 individuals: 0.897 (95% CI 0.76 to 1.05); hacking 0.967; VCDB (not HHS-sourced) 1.189.
- Shadow mean above the threshold with the population bound: 1.59 million vs sample mean 1.56 million.
- Largest 1% of reports carry 64.4% of individuals affected; reports with a business associate 55.8%.

Seeds are fixed in every script. Runtime is about 30–60 minutes, dominated by the bootstrap goodness-of-fit tests. Results from a refreshed HHS export will differ slightly because the portal is updated.

## Licence
Code: MIT. HHS-OCR and FRED data: public U.S. government data. VCDB: see its repository.
