#!/bin/zsh
# chain C (1 worker, starts when chain B is done): endfire+DPDFNet stack (meeting, bus L1), blind S1-steered beams (bus L1),
# then the L2 extension
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
L=tmp/night_realrun/logs
while ! grep -q CHAIN_B_DONE $L/B_blind.log 2>/dev/null; do sleep 10; done
R=".venv/bin/python -I tmp/night_realrun/realrun.py run -j 1 --skip-existing"
eval $R --methods x_combo_s1beam,x_combo_s1beam_cdr_gtcrn --variant measured --set bus --scenes L1_ >> $L/C.log 2>&1
eval $R --methods x_combo_endfire_dpdf --variant measured --set bus --scenes L1_ >> $L/C.log 2>&1
eval $R --methods x_combo_endfire,x_combo_endfire_cdr_gtcrn,m8_gtcrn,m8_dpdfnet --variant measured --set bus --scenes L2_ >> $L/C.log 2>&1
echo CHAIN_C_DONE >> $L/C.log
