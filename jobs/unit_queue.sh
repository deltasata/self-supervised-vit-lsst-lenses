#!/bin/bash
# Job queue. One line per unit in jobs/units.txt:
#   name|parallel|script|needs|entry1 entry2 ...   (entries as for scripts/run_many.py: cfg::run_id::k=v,k=v)
# needs: a file that must exist before the unit is submitted ("-" = none), e.g. the encoder of a MAE unit.
# Done when every run_id has experiments/<id>/metrics.json (for scripts/pretrain_mae.py: experiments/<id>_ft/metrics.json).
# Earlier lines have priority: a refused submission ends the round. Resubmits after time-limit exits
# (train.py / pretrain_mae.py resume); gives up after MAXSUB submissions per unit. Run detached from the working dir.
cd "${LAGN_WORKDIR:-$PWD}"
MAXSUB=4; STATE=jobs/unit_state.txt; UNITS=jobs/units.txt; touch $STATE
while true; do
  if ! q=$(squeue -u "$USER" -h -o %j 2>/dev/null); then sleep 300; continue; fi
  all_done=1
  while IFS='|' read -r u par script needs entries; do
    [ -z "$u" ] || [[ "$u" == \#* ]] && continue
    suf=""; [[ "$script" == *pretrain_mae* ]] && suf="_ft"
    d=1; for e in $entries; do rid=$(awk -F'::' '{print $2}' <<< "$e"); [ -f "experiments/${rid}${suf}/metrics.json" ] || d=0; done
    [ $d -eq 1 ] && continue
    all_done=0
    grep -qx "lagn_r_$u" <<< "$q" && continue
    [ "$needs" != "-" ] && [ ! -f "$needs" ] && continue
    n=$(grep -c "^$u$" $STATE)
    if [ "$n" -ge $MAXSUB ]; then echo "$(date '+%F %T') GIVING UP on $u"; continue; fi
    if tools/submit.sh --name "r_$u" jobs/train.sbatch -- scripts/run_many.py --walltime_min 1430 \
         --script "$script" --parallel "$par" -- $entries; then
      echo "$u" >> $STATE; echo "$(date '+%F %T') submitted $u"
    else
      echo "$(date '+%F %T') submission refused $u"; break
    fi
  done < $UNITS
  [ $all_done -eq 1 ] && { echo "$(date '+%F %T') all units done"; exit 0; }
  sleep 600
done
