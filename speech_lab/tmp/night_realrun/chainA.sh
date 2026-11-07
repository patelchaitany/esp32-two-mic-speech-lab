#!/bin/zsh
# chain A (1 worker): M9 family, then the wrong4cm variant for the S0-dependent methods
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=".venv/bin/python -I tmp/night_realrun/realrun.py run -j 1 --skip-existing"
L=tmp/night_realrun/logs
eval $R --methods m9_voxlock2 --variant measured --set bus --scenes L1_snr+0_recorded,L1_snr+5_quiet >> $L/A_m9.log 2>&1
eval $R --methods m9_voxlock2 --variant measured --set bus --enroll device --scenes L1_snr+0_recorded >> $L/A_m9.log 2>&1
eval $R --methods m9_voxlock2_self,x_combo_m9_cdr --variant measured --set bus --scenes L1_snr+0_recorded >> $L/A_m9.log 2>&1
eval $R --methods m1_das,m2_bus_beam,m3_rtf_mvdr,m4_cdr,m5_fence,m7_cacgmm,x_auto_m1_das,x_auto_m2_bus_beam,x_auto_m3_rtf_mvdr,x_auto_m4_cdr,x_auto_m5_fence,x_auto_m7_cacgmm --variant wrong4cm --set bus --scenes L1_snr-5_recorded,L1_snr+0_recorded,L1_snr+5_recorded >> $L/A_wrong.log 2>&1
eval $R --methods m2_meeting_beam,m11_level,m4_cdr,x_auto_m2_meeting_beam,x_auto_m4_cdr --variant wrong4cm --set meeting >> $L/A_wrong.log 2>&1
echo CHAIN_A_DONE >> $L/A_wrong.log
