#!/bin/bash
# Production queue: submits the 25 fold-model jobs of production/PLAN.md in priority order, at most PCAP
# production jobs at a time, and resubmits a job that left the queue unfinished (run_prod resumes from last.pt).
# A unit submitted MAXSUB times without finishing is skipped and reported. Run from the working dir (background).
cd "${LAGN_WORKDIR:-$PWD}"
PCAP=${PCAP:-6}; MAXSUB=4
P=production; STATE=$P/jobs/queue_state.txt; touch $STATE
# unit name | parallel | entries (cfg::run_id::sets)
units() {
  for a in "0 1" "2 3" "4 5" "6 7" "8 9"; do set -- $a
    echo "A$1$2|2|configs/vit.json::pA_vit_f$1::train.fold=$1,seed=$1 configs/vit.json::pA_vit_f$2::train.fold=$2,seed=$2"
    echo "B$1|1|configs/cnn800.json::pB_cnn800_f$1::train.fold=$1,seed=$1"
  done
  for k in 1 3 5 7 9; do echo "B$k|1|configs/cnn800.json::pB_cnn800_f$k::train.fold=$k,seed=$k"; done
  for k in 0 1 2 3 4 5 6 7 8 9; do echo "C$k|1|configs/cnn1600.json::pC_cnn1600_f$k::train.fold=$k,seed=$k"; done
}
while true; do
  if ! q=$(squeue -u "$USER" -h -o %j 2>/dev/null); then sleep 300; continue; fi
  n_run=$(grep -c "^lagn_p_[ABC]" <<< "$q")
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
  done < <(units | awk -F'|' '!seen[$1]++')
  if [ $all_done -eq 1 ]; then echo "$(date '+%F %T') all production fold models done"; exit 0; fi
  sleep 600
done
