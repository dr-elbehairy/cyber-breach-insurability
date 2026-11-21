"""Fetch the inputs that are not stored in this repository.

- VCDB joined JSON, pinned to commit 5ce87b9 of github.com/vz-risk/VCDB (SHA-256 checked).
- The HHS-OCR exports (data_raw/hhs_*.csv) ARE included: they are U.S. federal government data. To refresh them,
  open https://ocrportal.hhs.gov/ocr/breach/breach_report.jsf, export the 'Under Investigation' list and the
  'Archive' list as CSV, and save them as data_raw/hhs_open_<date>.csv and data_raw/hhs_archive_<date>.csv
  (then update the file names in 01_data.py). Results will differ slightly because the portal is updated.
- FRED series POPTHM (U.S. population, BEA) is included as data_raw/FRED_POPTHM.csv.
"""
import hashlib, os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "data_raw"); os.makedirs(RAW, exist_ok=True)
URL = "https://github.com/vz-risk/VCDB/raw/5ce87b920697e493e697364a972c799f78add489/data/joined/vcdb.json.zip"
SHA = "e4be5dd432ccfad16520a6b60dd83e9d47c63b0f3352c26c4d43a5dd774c32c0"
dst = os.path.join(RAW, "vcdb.json.zip")
if not os.path.exists(dst):
    subprocess.run(["curl", "-sSL", "-o", dst, URL], check=True)
h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
print("vcdb.json.zip sha256", h, "OK" if h == SHA else "MISMATCH")
