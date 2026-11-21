"""Build the two analysis files from the raw downloads.

HHS-OCR breach portal (downloaded 2026-11-21, 'Under Investigation' and 'Archive' CSV exports):
every breach of unsecured protected health information affecting 500 or more individuals.
VCDB (vz-risk/VCDB, data/joined/vcdb.json.zip): publicly reported incidents coded in VERIS.
Output: data/hhs_clean.csv, data/vcdb_clean.csv, data/data_summary.json
"""
import json, os, zipfile, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RAW = f"{HERE}/data_raw"; OUT = f"{HERE}/data"
os.makedirs(OUT, exist_ok=True)
COLS = ["entity", "state", "entity_type", "individuals", "submitted", "breach_type", "location", "ba_present", "description"]

def load_hhs():
    parts = []
    for f, src in [("hhs_archive_2026-11-21.csv", "archive"), ("hhs_open_2026-11-21.csv", "open")]:
        d = pd.read_csv(f"{RAW}/{f}"); d = d.set_axis(COLS, axis=1); d["list"] = src; parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    n0 = len(d)
    d["submitted"] = pd.to_datetime(d["submitted"], format="%m/%d/%Y")
    dup = d.duplicated(["entity", "individuals", "submitted", "breach_type"])
    d = d[~dup].copy()
    miss = d["individuals"].isna()
    d = d[~miss].copy()
    d["individuals"] = d["individuals"].astype(int)
    # type groups: any report whose type includes hacking is 'Hacking/IT'
    d["hacking"] = d["breach_type"].fillna("").str.contains("Hacking", case=False).astype(bool)
    d["ba"] = (d["ba_present"].fillna("").str.strip().str.lower() == "yes") | (d["entity_type"] == "Business Associate")
    d["year"] = d["submitted"].dt.year; d["month"] = d["submitted"].dt.to_period("M").astype(str)
    info = {"rows_raw": int(n0), "exact_duplicates_dropped": int(dup.sum()), "missing_count_dropped": int(miss.sum()),
            "rows": int(len(d)), "first": str(d.submitted.min().date()), "last": str(d.submitted.max().date()),
            "archive_rows": int((d.list == "archive").sum()), "open_rows": int((d.list == "open").sum()),
            "placeholder_500": int((d.individuals == 500).sum()), "placeholder_501": int((d.individuals == 501).sum()),
            "min_individuals": int(d.individuals.min()), "max_individuals": int(d.individuals.max()),
            "max_entity": d.loc[d.individuals.idxmax(), "entity"], "sum_individuals": int(d.individuals.sum()),
            "hacking_share": float(d.hacking.mean()), "ba_share": float(d.ba.mean()),
            "entity_types": d.entity_type.value_counts().to_dict(), "breach_types": d.breach_type.value_counts().to_dict()}
    return d, info

def load_vcdb():
    with zipfile.ZipFile(f"{RAW}/vcdb.json.zip") as z:
        name = [n for n in z.namelist() if n.endswith(".json")][0]; recs = json.loads(z.read(name))
    rows = []
    for r in recs:
        conf = (r.get("attribute") or {}).get("confidentiality") or {}
        vic = r.get("victim") or {}; plus = r.get("plus") or {}
        refs = " ".join(str(r.get(k) or "") for k in ["reference"]) + " " + str(plus.get("analyst_notes") or "")
        rows.append({"incident_id": r.get("incident_id"), "year": (r.get("timeline") or {}).get("incident", {}).get("year"),
                     "records": conf.get("data_total"), "disclosure": conf.get("data_disclosure"),
                     "industry": str(vic.get("industry") or ""), "country": ",".join(vic.get("country") or []),
                     "actions": ",".join(sorted((r.get("action") or {}).keys())),
                     "actors": ",".join(sorted((r.get("actor") or {}).keys())),
                     "sub_source": str(plus.get("sub_source") or ""), "ai_encoded": "encoded by ai" in str(plus.get("analyst_notes") or "").lower(),
                     "hhs_reference": "hhs.gov" in refs.lower() or "ocrportal" in refs.lower()})
    v = pd.DataFrame(rows)
    info = {"incidents": int(len(v)), "with_records": int(v.records.notna().sum()),
            "with_records_ge500": int((v.records >= 500).sum()),
            "healthcare_naics62_ge500": int(((v.records >= 500) & v.industry.str.startswith("62")).sum()),
            "us_ge500": int(((v.records >= 500) & v.country.str.contains("US")).sum()),
            "phidbr": int((v.sub_source == "phidbr").sum()), "ai_encoded": int(v.ai_encoded.sum()), "hhs_reference": int(v.hhs_reference.sum()),
            "independent_ge500": int(((v.records >= 500) & (v.sub_source != "phidbr") & ~v.hhs_reference & ~v.ai_encoded).sum()),
            "year_min": int(v.year.min()), "year_max": int(v.year.max()), "schema_file": name}
    return v, info

if __name__ == "__main__":
    h, hi = load_hhs(); v, vi = load_vcdb()
    h.to_csv(f"{OUT}/hhs_clean.csv", index=False); v.to_csv(f"{OUT}/vcdb_clean.csv", index=False); h.to_pickle(f"{OUT}/hhs_clean.pkl"); v.to_pickle(f"{OUT}/vcdb_clean.pkl")
    json.dump({"hhs": hi, "vcdb": vi}, open(f"{OUT}/data_summary.json", "w"), indent=1, default=str)
    print(json.dumps({"hhs": hi, "vcdb": vi}, indent=1, default=str))
