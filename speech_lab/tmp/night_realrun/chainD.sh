#!/bin/zsh
# chain D (1 worker, starts when chain A is done): blind/wrong4cm endfire + S1-beam stacks, then the L3 extension
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
L=tmp/night_realrun/logs
while ! grep -q CHAIN_A_DONE $L/A_wrong.log 2>/dev/null; do sleep 10; done
R=".venv/bin/python -I tmp/night_realrun/realrun.py run -j 1 --skip-existing"
eval $R --methods x_combo_endfire_dpdf --variant measured --set meeting >> $L/D.log 2>&1
eval $R --methods x_combo_endfire_auto,x_combo_endfire_cdr_gtcrn_auto,x_combo_s1beam,x_combo_s1beam_cdr_gtcrn,x_combo_endfire --variant wrong4cm --set bus --scenes L1_snr-5_recorded,L1_snr+0_recorded,L1_snr+5_recorded >> $L/D.log 2>&1
eval $R --methods x_combo_endfire,x_combo_endfire_cdr_gtcrn,m8_gtcrn,m8_dpdfnet --variant measured --set bus --scenes L3_ >> $L/D.log 2>&1
echo CHAIN_D_DONE >> $L/D.log
