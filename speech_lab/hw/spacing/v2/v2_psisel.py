"""V2: does choosing psi by the J0 (2-D) DF (DESIGN/B1 convention, oracle on the evaluation model) vs by the
3-D design model change the two-talker 2-D AI-DI?  Own soft-LCMV, strict -10 dB floor, +-1 dB balance."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_physics import C, FE, gamma, steer, G2, aidi

def tt(d, floor_db, sel, ev, lam=1e3):
    W = 10 ** (floor_db / 10); psis = np.radians(np.arange(0, 360, 2.5))
    aA = steer(FE, d / C); aB = steer(FE, -d / C)
    Gd = G2(gamma(FE, d, "3d")); Gs = G2(gamma(FE, d, sel)); Ge = G2(gamma(FE, d, ev))
    A = np.stack([aA, aB], -1); AAH = A @ np.conj(np.swapaxes(A, -1, -2)); out = []
    for i in range(len(FE)):
        h = np.stack([np.ones_like(psis), np.exp(1j * psis)], -1).astype(complex)
        rhs = lam * (A[i] @ h.conj().T).T
        w_of = lambda mu: np.linalg.solve(Gd[i][None] + mu[:, None, None] * np.eye(2) + lam * AAH[i][None], rhs[..., None])[..., 0]
        wn = lambda w: np.abs(np.einsum("pi,i->p", w.conj(), aA[i])) ** 2 / np.real(np.einsum("pi,pi->p", w.conj(), w))
        l = np.full(len(psis), -9.0); hg = np.full(len(psis), 6.0); ok0 = wn(w_of(10 ** l)) >= W
        for _ in range(50):
            m = 0.5 * (l + hg); ok = wn(w_of(10 ** m)) >= W; hg = np.where(ok, m, hg); l = np.where(ok, l, m)
        w = w_of(np.where(ok0, 1e-9, 10 ** hg))
        rA = np.einsum("pi,i->p", w.conj(), aA[i]); rB = np.einsum("pi,i->p", w.conj(), aB[i])
        ok = (wn(w) >= W * (1 - 1e-9)) & (np.abs(20 * np.log10(np.abs(rA) / np.abs(rB))) <= 1)
        num = 0.5 * (abs(rA) ** 2 + abs(rB) ** 2)
        dfs = num / np.real(np.einsum("pi,ij,pj->p", w.conj(), Gs[i], w))
        dfe = num / np.real(np.einsum("pi,ij,pj->p", w.conj(), Ge[i], w))
        out.append(10 * np.log10(dfe[np.argmax(np.where(ok, dfs, -np.inf))]) if ok.any() else 0.0)
    return aidi(np.array(out))

for d in (0.012, 0.025, 0.04, 0.06):
    print(f"{d*100:g} cm  2-D AI-DI, psi by J0 DF: {tt(d, -10, '2d', '2d'):.3f}   psi by 3-D DF: {tt(d, -10, '3d', '2d'):.3f}")
