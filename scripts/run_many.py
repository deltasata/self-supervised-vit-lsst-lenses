"""Run several training configs inside ONE Slurm job (one GPU), `--parallel` at a time.

The queue wait on the apu partition is long, so a job works through a list of runs.
Each entry: "<cfg.json>[::run_id][::key=val,key=val]". Runs whose metrics.json exists are skipped,
runs with last.pt resume. A run gets the remaining walltime as its time limit.
Usage: python scripts/run_many.py --parallel 2 --walltime_min 1430 -- cfgA.json cfgB.json::myid::train.lr=3e-4
"""
import argparse
import os
import subprocess
import sys
import time

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse(entry):
    parts = entry.split("::")
    cfg = parts[0]
    rid = parts[1] if len(parts) > 1 and parts[1] else os.path.splitext(os.path.basename(cfg))[0]
    sets = parts[2].split(",") if len(parts) > 2 and parts[2] else []
    return cfg, rid, sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--walltime_min", type=float, default=1430)
    ap.add_argument("--script", default="scripts/train.py")
    ap.add_argument("--min_start_min", type=float, default=15)
    ap.add_argument("entries", nargs="+")
    a = ap.parse_args()
    t0 = time.time()
    todo = [parse(e) for e in a.entries if e != "--"]
    running = []
    while todo or running:
        running = [(p, r) for p, r in running if p.poll() is None]
        while todo and len(running) < a.parallel:
            cfg, rid, sets = todo.pop(0)
            if os.path.exists(os.path.join(W, "experiments", rid, "metrics.json")):
                print(f"skip {rid} (done)", flush=True)
                continue
            left = a.walltime_min - (time.time() - t0) / 60 - 3
            if left < a.min_start_min:
                print(f"not starting {rid}: only {left:.0f} min left", flush=True)
                todo = []
                break
            cmd = [sys.executable, "-u", os.path.join(W, a.script), "--cfg", cfg, "--run_id", rid,
                   "--set", f"time_limit_min={left:.0f}"] + sets
            os.makedirs(os.path.join(W, "experiments", rid), exist_ok=True)
            log = open(os.path.join(W, "experiments", rid, "train.log"), "a")
            print(f"start {rid}: {' '.join(cmd)}", flush=True)
            running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=W), rid))
        time.sleep(10)
        for p, r in running:
            if p.poll() is not None:
                print(f"finished {r} rc={p.returncode} at {(time.time() - t0) / 60:.0f} min", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
