#!/bin/bash
# Chain: resubmit the continued MIM pretraining swE2400 (mimF_v2b_e1600p800, skip_finetune) whenever its job
# leaves the queue, until experiments/mimF_v2b_e1600p800/encoder.pt exists; then submit two regression seeds from it,
# retrying while the submission is refused. Run from the working dir; log lines go to stdout.
cd "${LAGN_WORKDIR:-$PWD}"
SPEC="configs/wild/mimF_cnxtv2b_e1600p800.json::mimF_v2b_e1600p800"
ENC=experiments/mimF_v2b_e1600p800/encoder.pt
while true; do
  if ! q=$(squeue -u "$USER" -h -o %j 2>/dev/null); then
    echo "$(date '+%F %T') squeue failed; skipping this round"; sleep 300; continue
  fi
  if [ ! -f "$ENC" ]; then
    if ! grep -qx "lagn_swE2400" <<< "$q"; then
      echo "$(date '+%F %T') resubmitting swE2400"
      tools/submit.sh --name swE2400 jobs/train.sbatch -- scripts/run_many.py --walltime_min 1430 \
        --script scripts/pretrain_mim_cnn.py --parallel 1 -- "$SPEC"
    fi
  elif [ -n "$(find "$ENC" -mmin +5)" ]; then
    done_all=1
    for s in 0 1; do
      id="reg_cnxtv2b_e2400_ep150_s$s"
      { [ -f "regression/experiments/$id/config.json" ] || grep -qx "lagn_reg_c2400_$s" <<< "$q"; } && continue
      echo "$(date '+%F %T') submitting regression e2400 seed $s"
      tools/submit.sh --name "reg_c2400_$s" jobs/train.sbatch -- regression/scripts/train_reg.py \
        --cfg regression/configs/reg_cnxtv2b_e2400.json --run_id "$id" --set "seed=$s" || done_all=0
    done
    if [ $done_all -eq 1 ]; then echo "$(date '+%F %T') chain finished"; exit 0; fi
  fi
  sleep 300
done
