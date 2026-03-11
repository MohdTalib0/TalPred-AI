"""Run feature validation suite against live DB data."""

import json
from datetime import date

from dotenv import load_dotenv

load_dotenv()

from src.db import SessionLocal
from src.features.validation import validate_features

db = SessionLocal()
report = validate_features(db, target_date=date(2026, 3, 10))

for key, val in report.items():
    if isinstance(val, dict):
        passed = val.get("passed")
        issues = val.get("issues", [])
        status = "PASS" if passed else "FAIL"
        print(f"  {key}: {status}")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  {key}: {val}")

db.close()
