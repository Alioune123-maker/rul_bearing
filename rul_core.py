"""Coeur du pronostic : indicateur de sante -> modele exponentiel -> RUL avec incertitude.
Choix volontairement simples (la validation prime sur la complexite)."""
import numpy as np, pandas as pd
from dataclasses import dataclass

@dataclass
class Params:
    ft: float = 8.0        # seuil de fin de vie
    win: int = 15          # fenetre de lissage (mediane glissante causale)
    n0: int = 100          # points sains pour estimer la ligne de base
    n_sigma: float = 4.0   # sensibilite de detection de debut de degradation
    persist: int = 10      # points consecutifs au-dessus du seuil pour valider l'onset
    min_pts: int = 40      # points post-onset mini avant de faire confiance au modele
    n_boot: int = 300      # reechantillonnages bootstrap
    cap: int = 4000        # RUL maximale renvoyee
    scale: float = 1.0     # gonflement du bruit bootstrap, calibre sur le train pour atteindre la couverture cible

def health_indicator(x, win):
    """HI = RMS lisse (mediane glissante causale : robuste aux chocs, sans regarder le futur)."""
    return pd.Series(x).rolling(win, min_periods=1).median().to_numpy()

def baseline(hi, p):
    h = hi[:p.n0]
    b = np.median(h)
    sig = 1.4826 * np.median(np.abs(h - b)) + 1e-6
    return b, sig

def detect_onset(hi, b, sig, p):
    """Premier instant ou HI > b + n_sigma*sig pendant `persist` points consecutifs (causal)."""
    above = hi > b + p.n_sigma * sig
    run = 0
    for i in range(p.n0, len(hi)):
        run = run + 1 if above[i] else 0
        if run >= p.persist:
            return i - p.persist + 1
    return None

def prior_rul(age, train_lives, rng, n=300, cap=4000):
    """Quand le modele n'est pas identifiable : distribution des (duree de vie - age) des roulements d'apprentissage."""
    rem = np.array([L - age for L in train_lives if L > age], float)
    if len(rem) == 0:
        rem = np.array([30.0])
    return np.clip(rng.choice(rem, n) , 1, cap)

def rul_distribution(x_hist, train_lives, p: Params, rng):
    """Retourne (echantillons de RUL, mode) a l'instant t = len(x_hist)."""
    t = len(x_hist)
    if t <= p.n0:
        return prior_rul(t, train_lives, rng, p.n_boot, p.cap), "prior"
    hi = health_indicator(x_hist, p.win)
    b, sig = baseline(hi, p)
    t0 = detect_onset(hi, b, sig, p)
    if t0 is None or t - t0 < p.min_pts:
        return prior_rul(t, train_lives, rng, p.n_boot, p.cap), "prior"
    tau = np.arange(t0, t) - t0
    y = hi[t0:t]
    K = np.logspace(-4, np.log10(2e-2), 70)                    # grille des taux de croissance k
    E = np.exp(np.outer(K, tau))                               # (nk, n)
    def fit_all(Y):
        """Pour chaque ligne de Y (B,n) : meilleur k de la grille, modele y = c + A*exp(k*tau), A>0 (c,A en forme fermee)."""
        B = Y.shape[0]
        best_sse = np.full(B, np.inf); bk = np.zeros(B); bc = np.zeros(B); bA = np.zeros(B)
        for j, k in enumerate(K):
            X = np.column_stack([np.ones_like(tau, float), E[j]])
            C = Y @ np.linalg.pinv(X).T                         # (B,2)
            sse = ((Y - C @ X.T) ** 2).sum(1)
            ok = (C[:, 1] > 0) & (sse < best_sse)
            best_sse = np.where(ok, sse, best_sse); bk = np.where(ok, k, bk)
            bc = np.where(ok, C[:, 0], bc); bA = np.where(ok, C[:, 1], bA)
        return bk, bc, bA, best_sse
    k0, c0, A0, sse0 = fit_all(y[None, :])
    if not np.isfinite(sse0[0]):
        return prior_rul(t, train_lives, rng, p.n_boot, p.cap), "prior"
    fit0 = c0[0] + A0[0] * np.exp(k0[0] * tau)
    res = y - fit0
    blk = max(5, p.win)                                         # bootstrap par blocs (bruit autocorrele)
    nb = int(np.ceil(len(res) / blk))
    starts = rng.integers(0, max(1, len(res) - blk + 1), size=(p.n_boot, nb))
    idx = (starts[:, :, None] + np.arange(blk)).reshape(p.n_boot, -1)[:, :len(res)]
    Yb = fit0[None, :] + p.scale * res[idx]      # scale>1 : bruit gonfle pour couvrir l'erreur de structure du modele
    kb, cb, Ab, sb = fit_all(Yb)
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = (p.ft - cb) / Ab
        tau_star = np.where(arg > 1, np.log(arg) / kb, 0.0)    # deja au-dessus du seuil -> 0
    rul = t0 + tau_star - t
    rul = np.where(np.isfinite(sb) & (kb > 0) & np.isfinite(rul), rul, p.cap)
    return np.clip(rul, 0, p.cap), "model"

def rul_summary(samples, p: Params, qs=(0.1, 0.5, 0.9), mode="model"):
    """Intervalle central (10 %-90 % => 80 %) et mediane de la distribution de RUL."""
    q = np.quantile(samples, qs)
    return float(q[0]), float(q[1]), float(q[2])
