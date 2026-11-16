"""ETAPE 5 : comparaison de 3 methodes de RUL sur les vraies donnees PRONOSTIA.
  M0 = prior de population (aucune info capteur)  <- point de comparaison indispensable
  M1 = modele a deux regimes non parametrique (detection du debut d'acceleration + vie residuelle empirique)
  M2 = extrapolation exponentielle (ancien modele rul_core.py)
Indicateur : rms_tot = sqrt(rms_h^2 + rms_v^2), lisse (mediane glissante causale), normalise par sa base (snapshots 150-250).
Reglage (seuil/persistance de detection) par validation croisee 'un roulement laisse de cote' sur le TRAIN uniquement.
Evaluation : (a) protocole officiel : 1 prediction par roulement de test, a son point d'arret ;
             (b) multi-ages sur le Validation_Set (vie complete des memes 11 roulements).
"""
import itertools
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rul_core import Params, rul_distribution

F = Path("features"); WIN = 15; B0, B1 = 150, 250
FR = np.round(np.arange(0.10, 0.951, 0.05), 2)
rng = np.random.default_rng(0)
METHODS = ["M0 prior", "M1 deux régimes", "M2 exponentiel"]

def load(split): return {p.stem.split("__")[1]: pd.read_csv(p) for p in sorted(F.glob(f"{split}__*.csv"))}
def prep(name, d, L=None):
    raw = np.sqrt(d["rms_h"].to_numpy() ** 2 + d["rms_v"].to_numpy() ** 2)
    s = pd.Series(raw).rolling(WIN, min_periods=1).median().to_numpy()
    u = dict(name=name, raw=raw, n=len(raw), L=L if L is not None else len(raw), h=None, xn=None)
    if len(raw) >= B1:
        b = np.median(s[B0:B1]); u["h"], u["xn"] = s / b, raw / b
    return u

def knee(h, t, thr, persist):
    """Instant (causal) ou h depasse thr pendant `persist` points consecutifs, en ne regardant que h[:t]."""
    run = 0
    for i in range(B1, t):
        run = run + 1 if h[i] > thr else 0
        if run >= persist: return i
    return None

def fit(train, thr, persist):
    lives = np.array([u["L"] for u in train]); td = [knee(u["h"], u["L"], thr, persist) for u in train]
    resid = np.array([L - t for L, t in zip(lives, td) if t is not None])
    ft = float(np.median([np.median(u["h"][-5:]) for u in train]))
    return dict(lives=lives, td=td, resid=resid, ft=ft, thr=thr, persist=persist)

def m0(M, t):
    rem = [L - t for L in M["lives"] if L > t]; return np.array(rem if rem else [5.0])
def m1(M, u, t):
    if t < B1 or u["h"] is None: return m0(M, t)
    tdet = knee(u["h"], t, M["thr"], M["persist"])
    if tdet is None:
        rem = [L - t for L, td in zip(M["lives"], M["td"]) if L > t and (td is None or td > t)]
        return np.array(rem) if rem else m0(M, t)
    s = t - tdet; rem = [r - s for r in M["resid"] if r > s]
    return np.array(rem if rem else [5.0])
def m2(M, u, t):
    if u["xn"] is None or t < B1: return m0(M, t)
    p = Params(ft=M["ft"], win=WIN, n0=100, n_sigma=4.0, persist=10, min_pts=80, n_boot=100, cap=4000, scale=1.0)
    s, _ = rul_distribution(u["xn"][B0:t], M["lives"] - B0, p, rng); return s

def predict(M, u, t, which):
    s = {"M0 prior": lambda: m0(M, t), "M1 deux régimes": lambda: m1(M, u, t), "M2 exponentiel": lambda: m2(M, u, t)}[which]()
    q = np.quantile(s, [0.1, 0.5, 0.9]); return q

def evaluate(M, units, points, methods=METHODS):
    rows = []
    for u in units:
        for t in points(u):
            true = u["L"] - t
            for m in methods:
                lo, med, hi = predict(M, u, t, m)
                rows.append(dict(unit=u["name"], t=t, frac=t / u["L"], true=true, method=m, lo=lo, med=med, hi=hi))
    d = pd.DataFrame(rows); d["frac"] = d["frac"].round(2); d["err"] = d["med"] - d["true"]; d["er_pct"] = 100 * (d["true"] - d["med"]) / d["true"]
    d["in"] = (d["true"] >= d["lo"]) & (d["true"] <= d["hi"]); return d

def summarize(d):
    g = d.groupby("method")
    return pd.DataFrame({"MAE (pas)": g["err"].apply(lambda s: s.abs().mean()), "|err| médiane (pas)": g["err"].apply(lambda s: s.abs().median()),
                         "|Er| médian (%)": g["er_pct"].apply(lambda s: s.abs().median()), "% prédictions tardives": g["err"].apply(lambda s: (s > 0).mean() * 100),
                         "couverture IC80 (%)": g["in"].mean() * 100, "largeur IC médiane": g.apply(lambda s: (s["hi"] - s["lo"]).median())}).loc[METHODS].round(1)

tr = {n: prep(n, d) for n, d in load("Training_set").items()}; train = list(tr.values())
te = load("Test_set"); va = load("Validation_Set")

# ---- controle d'integrite Test / Validation -------------------------------------------------
print("\n=== Contrôle d'intégrité (début de Validation == Test ?) et RUL vraie ===")
test_units, val_units, bad = [], [], []
for n in sorted(te):
    k = len(te[n]); L = len(va[n]); diff = np.abs(te[n]["rms_h"].to_numpy() - va[n]["rms_h"].to_numpy()[:k]).max()
    ok = diff < 1e-6; print(f"  {n}: n_test={k:5d}  L_val={L:5d}  RUL vraie={L-k:4d} pas ({(L-k)*10:5d} s)  écart={diff:.1e}  {'OK' if ok else '<-- INCOHÉRENT : exclu des évaluations'}")
    if ok: test_units.append(prep(n, te[n], L)); val_units.append(prep(n, va[n]))
    else: bad.append(n)

# ---- reglage par validation croisee 'un roulement laisse de cote' (train seul) ----------------
print("\n=== Réglage de M1 par validation croisée sur le train (MAE, âges 50-95 % de vie) ===")
best, bmae = None, 1e18
for thr, per in itertools.product([1.3, 1.5, 2.0, 3.0], [5, 10, 20, 40]):
    errs = []
    for i, u in enumerate(train):
        M = fit(train[:i] + train[i + 1:], thr, per)
        errs.append(evaluate(M, [u], lambda x: [int(f * x["L"]) for f in np.arange(0.5, 0.951, 0.05)], ["M1 deux régimes"])["err"].abs().mean())
    mae = np.mean(errs); print(f"  seuil={thr:3.1f}  persistance={per:2d}  MAE={mae:7.1f}")
    if mae < bmae: best, bmae = (thr, per), mae
print("  -> choix :", best)
M = fit(train, *best)
print(f"  Seuil de fin de vie moyen (indicateur normalisé, pour M2) = {M['ft']:.2f} ; détections sur le train : {M['td']} (vies : {[int(v) for v in M['lives']]})")

# ---- (a) protocole officiel : 1 prediction par roulement de test, a son point d'arret -----------
da = evaluate(M, test_units, lambda u: [u["n"]])
print("\n=== (a) Protocole officiel : prédiction au point d'arrêt de chaque roulement de test ===")
print(summarize(da).to_string())
pv = da.pivot(index="unit", columns="method", values="med").round(0); pv.insert(0, "RUL vraie", da.groupby("unit")["true"].first())
print("\nRUL médiane prédite (pas de 10 s) par roulement :\n", pv.to_string()); da.to_csv("results_official.csv", index=False)

# ---- (b) multi-ages sur le Validation_Set ---------------------------------------------------------
db = evaluate(M, val_units, lambda u: [int(f * u["L"]) for f in FR])
print("\n=== (b) Multi-âges sur la vie complète des roulements de test (Validation_Set) ===")
print(summarize(db).to_string())
byf = db.groupby(["frac", "method"])["err"].apply(lambda s: s.abs().mean()).unstack()[METHODS].round(0)
print("\nMAE selon l'avancement en vie :\n", byf.loc[[0.2, 0.4, 0.6, 0.8, 0.9, 0.95]].to_string()); db.to_csv("results_multiage.csv", index=False)

# ---- figures --------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8)); col = dict(zip(METHODS, ["#7f7f7f", "#d62728", "#1f77b4"]))
for k, m in enumerate(METHODS):
    s = da[da.method == m]; x = s["true"] * (1 + 0.05 * (k - 1))
    ax[0].errorbar(x, s["med"].clip(lower=1), yerr=[(s["med"] - s["lo"]).clip(lower=0), (s["hi"] - s["med"]).clip(lower=0)], fmt="o", ms=4, lw=.8, color=col[m], label=m, alpha=.8)
mx = max(da["true"].max(), da["hi"].max()) * 1.2; ax[0].plot([5, mx], [5, mx], "k--", lw=.8); ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("RUL vraie (pas de 10 s)"); ax[0].set_ylabel("RUL prédite (médiane, IC 10-90 %)"); ax[0].set_title("(a) Protocole officiel : 11 roulements de test"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
for m in METHODS: ax[1].plot(byf.index * 100, byf[m], "o-", ms=3, label=m, color=col[m])
ax[1].set_xlabel("% de la vie consommée"); ax[1].set_ylabel("MAE de RUL (pas)"); ax[1].set_title("(b) Erreur selon l'avancement (Validation_Set)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig_real_results.png", dpi=140); print("\nFigure écrite : fig_real_results.png ; CSV : results_official.csv, results_multiage.csv")
