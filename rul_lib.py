"""Fonctions partagees par le demonstrateur (demo_app.py) : chargement, detection, prediction de RUL.
Memes methodes que rul_cv17.py : M0c = population de meme condition ; M1c = deux regimes (detection + vie residuelle), meme condition."""
import numpy as np, pandas as pd
from pathlib import Path
F = Path("features"); WIN, B0, B1 = 15, 150, 250
STEPS_PER_MIN = 6                                       # 1 snapshot = 10 s
KEYS = {("rms_tot", "M1c"): (3.0, 5), ("bh6", "M1c"): (1.5, 3)}   # (seuil, persistance) les plus frequents de la validation croisee imbriquee
METHODS = {"M0c": "Population de même condition (sans capteur)", "M1c": "Deux régimes : détection capteur + condition"}

def load_units(hi="rms_tot"):
    n_test = {p.stem.split("__")[1]: len(pd.read_csv(p)) for p in F.glob("Test_set__*.csv")}; units = []
    for split in ("Training_set", "Validation_Set"):
        for p in sorted(F.glob(f"{split}__*.csv")):
            d = pd.read_csv(p); sp = Path("features_spec") / p.name
            if sp.exists(): d = d.merge(pd.read_csv(sp), on="t")
            d["rms_tot"] = np.sqrt(d["rms_h"] ** 2 + d["rms_v"] ** 2)
            if hi not in d: raise ValueError(f"Indicateur {hi} absent : lance d'abord features_spec.py")
            raw = d[hi].to_numpy(float); s = pd.Series(raw).rolling(WIN, min_periods=1).median().to_numpy(); name = p.stem.split("__")[1]
            u = dict(name=name, cond=int(name[7]), L=len(raw), split=split, n_test=n_test.get(name), h=None, _td={})
            if len(raw) >= B1: u["h"] = s / np.median(s[B0:B1])
            units.append(u)
    return units

def td(u, key):
    """Instant de detection (premier depassement persistant), connu a t > td."""
    if key not in u["_td"]:
        thr, per = key; out = None
        if u["h"] is not None:
            run = 0
            for i in range(B1, u["L"]):
                run = run + 1 if u["h"][i] > thr else 0
                if run >= per: out = i; break
        u["_td"][key] = out
    return u["_td"][key]

def _prior(P, t):
    rem = [p["L"] - t for p in P if p["L"] > t]; return np.array(rem if rem else [5.0])

def samples(method, units, b, t, hi):
    """Echantillons de RUL (en snapshots) du roulement b a l'instant t, appris sur les AUTRES roulements uniquement."""
    pool = [p for p in units if p["name"] != b["name"]]; P = [p for p in pool if p["cond"] == b["cond"]]
    P = P if (len(P) >= 3 and any(p["L"] > t for p in P)) else pool
    if method == "M0c" or t < B1 or b["h"] is None: return _prior(P, t)
    key = KEYS[(hi, "M1c")]; tb = td(b, key); tdet = tb if (tb is not None and tb < t) else None
    if tdet is None:
        rem = [p["L"] - t for p in P if p["L"] > t and (td(p, key) is None or td(p, key) > t)]
        return np.array(rem) if rem else _prior(P, t)
    s = t - tdet; resid = [p["L"] - td(p, key) for p in P if td(p, key) is not None]; rem = [r - s for r in resid if r > s]
    return np.array(rem if rem else [5.0])

def decision_rows(smp, t, L, D_steps, quantiles=(0.5, 0.2, 0.1, 0.05)):
    """Pour chaque regle (quantile de la RUL) : debut et fin de l'intervention, et issue par rapport a la panne reelle."""
    rows = []
    for q in quantiles:
        rq = float(np.quantile(smp, q)); start = max(0.0, rq - D_steps); end = start + D_steps
        rows.append(dict(regle=f"q{int(q*100)}" + (" (médiane)" if q == 0.5 else ""), rul_q=rq, debut=start, fin=end, vrai=L - t, ok=end <= (L - t), marge=(L - t) - end))
    return pd.DataFrame(rows)
