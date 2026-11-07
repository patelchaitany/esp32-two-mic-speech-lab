#!/bin/zsh
# chain B (1 worker): bus combos, meeting (measured), bus core (measured) on the L1 scenes, then the blind variant
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=".venv/bin/python -I tmp/night_realrun/realrun.py run -j 1 --skip-existing"
L=tmp/night_realrun/logs
eval $R --methods x_combo_cdr_gtcrn,x_combo_endfire,x_combo_endfire_cdr_gtcrn --variant measured --set bus --scenes L1_ >> $L/B_combo.log 2>&1
eval $R --methods m2_meeting_beam,m10_wpe,m11_level,m6_ns,m8_gtcrn,m8_dpdfnet,m4_cdr,x_gnfix_m4_cdr,x_combo_cdr_gtcrn,x_combo_meet_chain,x_combo_meet_ns_agc,x_combo_endfire --variant measured --set meeting >> $L/B_meet.log 2>&1
eval $R --methods m4_cdr,m6_ns,m8_gtcrn,m5_fence,m1_das,m2_bus_beam,x_gnfix_m4_cdr,m3_rtf_mvdr,m8_dpdfnet,m7_cacgmm,x_gnfix_m2_bus_beam,x_gnfix_m3_rtf_mvdr --variant measured --set bus --scenes L1_ >> $L/B_core.log 2>&1
eval $R --methods m1_das,m2_bus_beam,m3_rtf_mvdr,m4_cdr,m5_fence,m7_cacgmm --variant blind --set bus --scenes L1_snr-5_recorded,L1_snr+0_recorded,L1_snr+5_recorded >> $L/B_blind.log 2>&1
eval $R --methods m2_meeting_beam,m11_level,m4_cdr --variant blind --set meeting >> $L/B_blind.log 2>&1
echo CHAIN_B_DONE >> $L/B_blind.log
