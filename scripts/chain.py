"""Run several python commands one after another inside ONE Slurm job; stop at the first failure.
Usage: python scripts/chain.py "scripts/a.py --x 1" "scripts/run_many.py --parallel 3 -- ..."
   or: python scripts/chain.py jobs/chain_x.txt   (one command per line)
"""
import shlex
import subprocess
import sys

cmds = sys.argv[1:]
if len(cmds) == 1 and cmds[0].endswith(".txt"):
    cmds = [l.strip() for l in open(cmds[0]) if l.strip()]
for c in cmds:
    print("chain:", c, flush=True)
    rc = subprocess.call([sys.executable, "-u"] + shlex.split(c))
    if rc:
        sys.exit(rc)
