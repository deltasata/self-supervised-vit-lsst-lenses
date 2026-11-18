#!/bin/bash
# Submit queued job specs as soon as fewer than 10 of your jobs are queued.
# Each line of the spec file: <name>|<sbatch>|<args...>
cd "${LAGN_WORKDIR:-$PWD}"
spec=$1
while IFS='|' read -r name sb args; do
  [ -z "$name" ] && continue
  until [ "$(squeue -u "$USER" -h -t PD,R,CG,CF | wc -l)" -lt 10 ]; do sleep 60; done
  echo "$(date '+%F %T') submitting $name"
  tools/submit.sh --name "$name" "$sb" -- $args
  sleep 5
done < "$spec"
echo "$(date '+%F %T') all submitted"
