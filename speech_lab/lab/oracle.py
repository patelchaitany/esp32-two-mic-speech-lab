"""Diagnostic upper bounds (cheat: uses the true components). Not a deployable method."""
import sys, numpy as np
from .scenes import load_scene
from .bench import scene_names, HERE
from .metrics import score_bus, score_meeting
from .dsp import stft, istft
import os


def oracle_irm(sc, beta=0.5):
    x0 = sc["mix"][0].astype(np.float64)
    ref = sc["ref_early"].astype(np.float64)
    S = stft(ref); X = stft(x0)
    N = X - S  # everything that is not the early target (noise + late reverb)
    m = (np.abs(S) ** 2 / (np.abs(S) ** 2 + np.abs(N) ** 2 + 1e-20)) ** beta
    return istft(m * X, len(x0))


if __name__ == "__main__":
    rows = []
    for n in scene_names(sys.argv[1] if len(sys.argv) > 1 else "dev"):
        sc = load_scene(os.path.join(HERE, "scenes", n + ".npz"))
        y = oracle_irm(sc)
        r = score_bus(y, sc) if n.startswith("bus") else score_meeting(y, sc)
        rows.append((n, r))
    bus = [r for n, r in rows if n.startswith("bus")]
    mtg = [r for n, r in rows if n.startswith("meeting")]
    print("ORACLE IRM (Mic A, cheating upper bound for single-channel masking)")
    print(f"bus: dSISDR {np.mean([r['si_sdr_imp'] for r in bus]):.2f}  ESTOI {np.mean([r['estoi'] for r in bus]):.3f} "
          f"(in {np.mean([r['estoi_in'] for r in bus]):.3f})  PESQ {np.nanmean([r['pesq'] for r in bus]):.2f}  BR {np.mean([r['bg_reduction_db'] for r in bus]):.1f}")
    print(f"mtg: ESTOI {np.mean([r['estoi_mean'] for r in mtg]):.3f} worst {np.mean([r['estoi_worst'] for r in mtg]):.3f} "
          f"dSISDR {np.mean([r['si_sdr_imp'] for r in mtg]):.2f} PESQ {np.nanmean([r['pesq'] for r in mtg]):.2f}")
    for n, r in rows:
        if n.startswith("bus") and "d40mm" in n:
            print(f"  {n}: ESTOI in {r['estoi_in']:.3f} -> oracle {r['estoi']:.3f}")
