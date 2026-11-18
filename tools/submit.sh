#!/bin/bash
# Slurm submission helper used by the queue scripts.
# Usage: tools/submit.sh --name <short> [--test-only] [--burst] <script.sbatch> -- <args...>
# Submits with job name lagn_<short> and prints the job id. --burst is accepted and ignored.
set -e
name=""; extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    --name) name="$2"; shift 2 ;;
    --test-only) extra+=(--test-only); shift ;;
    --burst) shift ;;
    *) break ;;
  esac
done
script="$1"; shift
[ "$1" = "--" ] && shift
[ -n "$name" ] || { echo "usage: $0 --name <short> <script.sbatch> -- <args>"; exit 2; }
sbatch --parsable "${extra[@]}" --job-name "lagn_$name" --chdir "${LAGN_WORKDIR:-$PWD}" "$script" "$@"
