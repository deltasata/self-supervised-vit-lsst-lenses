"""Production fine-tuning: scripts/train.py unchanged, but on all 200k objects and writing to production/experiments.

Every production run must be a k-fold run (train.fold set): the model trains on 9 folds and early-stops on the 10th,
and nothing else. Usage (via run_prod.py): train_prod.py --cfg production/configs/<fam>.json --run_id <id>
--set train.fold=<k> seed=<k> ...
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_all  # noqa: E402

sys.path.insert(0, os.path.join(data_all.W, "scripts"))
import train  # noqa: E402

if __name__ == "__main__":
    sets = sys.argv[sys.argv.index("--set") + 1:] if "--set" in sys.argv else []
    assert any(s.startswith("train.fold=") for s in sets), "production runs must set train.fold"
    train.W = data_all.PROD                 # outputs -> production/experiments/<run_id>
    train.load_trainval = data_all.load_all  # 200k rows; train.py then uses kfold_assign on these keys
    train.main()
