#!/bin/bash
# Chain:
#  - resubmit the swE1600 / swE800s1 MIM pretraining (+ its classification FT) each time its job leaves the queue,
#    until that run's fine-tune has finished (metrics.json of <run>_ft);
#  - as soon as experiments/mimF_v2b_e1600/encoder.pt exists (and is >5 min old, i.e. fully written), submit the two
#    100-epoch regression seeds from it (once).
# All submissions go through tools/submit.sh. Run from the working dir; log lines go to stdout.
cd "${LAGN_WORKDIR:-$PWD}"
declare -A SPEC=(
  [swE1600]="configs/wild/mimF_cnxtv2b_e1600.json::mimF_v2b_e1600"
  [swE800s1]="configs/wild/mimF_cnxtv2b_e800_s1.json::mimF_v2b_e800_s1"
)
ENC=experiments/mimF_v2b_e1600/encoder.pt
reg_done=0
while true; do
  # skip the round when squeue fails (an empty answer would look like "job not queued")
  if ! q=$(squeue -u "$USER" -h -o %j 2>/dev/null); then
    echo "$(date '+%F %T') squeue failed; skipping this round"; sleep 300; continue
  fi
  for name in "${!SPEC[@]}"; do
    run=${SPEC[$name]##*::}
    [ -f "experiments/${run}_ft/metrics.json" ] && continue
    if ! grep -qx "lagn_$name" <<< "$q"; then
      echo "$(date '+%F %T') resubmitting $name"
      tools/submit.sh --name "$name" jobs/train.sbatch -- scripts/run_many.py --walltime_min 1430 \
        --script scripts/pretrain_mim_cnn.py --parallel 1 -- "${SPEC[$name]}"
    fi
  done
  if [ $reg_done -eq 0 ] && [ -f "$ENC" ] && [ -n "$(find "$ENC" -mmin +5)" ]; then
    # submit each seed until the submission is accepted; a seed counts as done once its run
    # dir exists or its job is queued
    reg_done=1
    for s in 0 1; do
      id="reg_cnxtv2b_e1600_ep100_s$s"
      { [ -f "regression/experiments/$id/config.json" ] || grep -qx "lagn_reg_c1600_$s" <<< "$q"; } && continue
      echo "$(date '+%F %T') submitting regression e1600 seed $s"
      tools/submit.sh --name "reg_c1600_$s" jobs/train.sbatch -- regression/scripts/train_reg.py \
        --cfg regression/configs/reg_cnxtv2b_e1600.json --run_id "$id" --set "seed=$s" || reg_done=0
    done
  fi
  if [ $reg_done -eq 1 ] && [ -f experiments/mimF_v2b_e1600_ft/metrics.json ] && [ -f experiments/mimF_v2b_e800_s1_ft/metrics.json ]; then
    echo "$(date '+%F %T') chain finished"; exit 0
  fi
  sleep 300
done
