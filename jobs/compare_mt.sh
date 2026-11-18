#!/bin/bash
# Pre-registered multi-task comparison: 6 control runs vs 6 multi-task (aux_w 0.5) runs =
# 2 MIM e800 encoders x FT seeds 0,1,2; same rule as the input-geometry test (better only if delta < -2 SE).
cd "${LAGN_WORKDIR:-$PWD}"
python scripts/compare_inputgeom.py --base control --ref_old none --out results/multitask \
  --title "# Multi-task (aux heads, aux_w 0.5) vs control, ConvNeXtV2-B MIM e800 encoders (pre-registered)" \
  --arms '{"control":["mimF_v2b_e800_ft","ctrl_e800_s1","ctrl_e800_s2","mimF_v2b_e800_s1_ft","ctrl_e800s1_s1","ctrl_e800s1_s2"],
           "mt05":["mt_e800_w05","mt_e800_w05_s1","mt_e800_w05_s2","mt_e800s1_w05","mt_e800s1_w05_s1","mt_e800s1_w05_s2"]}'
