"""scripts/run_many.py with outputs in production/experiments (skip finished runs, resume, walltime split).

Usage: python production/scripts/run_prod.py --parallel 2 --walltime_min 1430 --script scripts/train_prod.py -- ENTRY ...
(ENTRY = <cfg>::<run_id>::key=val,...; cfg paths relative to production/).
"""
import os
import sys

PROD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(PROD), "scripts"))
import run_many  # noqa: E402

if __name__ == "__main__":
    run_many.W = PROD
    run_many.main()
