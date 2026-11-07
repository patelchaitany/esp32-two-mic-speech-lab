"""V2: 2-D figures under the alternative design convention (J0-designed vs 3-D-designed weights), 1.2 vs 4 cm."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_physics import single_target, aidi
for d in (0.012, 0.04):
    for fl in (-10, -3):
        ax3 = aidi(single_target(d, 0, fl, "3d", "2d")[0]); axj = aidi(single_target(d, 0, fl, "2d", "2d")[0])
        az3 = np.mean([aidi(single_target(d, t, fl, "3d", "2d")[0]) for t in range(0, 91, 5)])
        azj = np.mean([aidi(single_target(d, t, fl, "2d", "2d")[0]) for t in range(0, 91, 5)])
        print(f"{d*100:g} cm floor {fl}: on-axis 2-D  3D-designed {ax3:.2f}  J0-designed {axj:.2f} | az-avg 2-D  3D-designed {az3:.2f}  J0-designed {azj:.2f}")
