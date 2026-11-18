#!/bin/bash
# Resubmit a continuation job after <jobid> has left the queue (so two jobs never resume the same run),
# once fewer than 10 of your jobs are queued. Usage: resubmit_after.sh <jobid> <name> <sbatch> <args...>
cd "${LAGN_WORKDIR:-$PWD}"
jid=$1; name=$2; sb=$3; shift 3
while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 120; done
while true; do   # retry if the submission is refused
  until [ "$(squeue -u "$USER" -h -t PD,R,CG,CF | wc -l)" -lt 10 ]; do sleep 60; done
  echo "$(date '+%F %T') $jid gone; submitting $name"
  tools/submit.sh --name "$name" "$sb" -- "$@" && break
  sleep 90
done
