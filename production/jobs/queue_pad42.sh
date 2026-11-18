#!/bin/bash
# Production family A2 queue: the 10 ViT fold models of family A rebuilt with pad42 input
# handling (production/configs/vit_pad42.json). Same structure as queue.sh: 5 jobs x 2 folds per GPU, fold k seed k,
# resubmit unfinished jobs (run_prod resumes from last.pt), give up after MAXSUB submissions. Run from the working dir.
cd "${LAGN_WORKDIR:-$PWD}"
PCAP=${PCAP:-5}; MAXSUB=4
P=production; STATE=$P/jobs/queue_pad42_state.txt; touch $STATE
units() {
  for a in "0 1" "2 3" "4 5" "6 7" "8 9"; do set -- $a
    echo "P$1$2|2|configs/vit_pad42.json::pA2_vitpad42_f$1::train.fold=$1,seed=$1 configs/vit_pad42.json::pA2_vitpad42_f$2::train.fold=$2,seed=$2"
  done
}
while true; do
  if ! q=$(squeue -u "$USER" -h -o %j 2>/dev/null); then sleep 300; continue; fi
  n_run=$(grep -c "^lagn_p_P" <<< "$q")
  all_done=1
  while IFS='|' read -r u par entries; do
    done_u=1
    for e in $entries; do rid=$(cut -d: -f3 <<< "$e"); [ -f "$P/experiments/$rid/metrics.json" ] || done_u=0; done
    [ $done_u -eq 1 ] && continue
    all_done=0
    grep -qx "lagn_p_$u" <<< "$q" && continue
    nsub=$(grep -c "^$u$" $STATE)
    if [ "$nsub" -ge $MAXSUB ]; then echo "$(date '+%F %T') GIVING UP on $u after $nsub submissions"; continue; fi
    [ "$n_run" -ge "$PCAP" ] && continue
    if tools/submit.sh --name "p_$u" jobs/train.sbatch -- $P/scripts/run_prod.py --walltime_min 1430 \
        --parallel "$par" --script scripts/train_prod.py -- $entries; then
      echo "$u" >> $STATE; n_run=$((n_run + 1)); echo "$(date '+%F %T') submitted $u"
    else
      echo "$(date '+%F %T') submission refused $u"; break
    fi
  done < <(units)
  if [ $all_done -eq 1 ]; then echo "$(date '+%F %T') all A2 (pad42) fold models done"; exit 0; fi
  sleep 600
done
