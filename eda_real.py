"""ETAPE 3 : premier regard sur les vraies données (à lancer après features.py).
 - table des RUL de test (= vie complète Validation - longueur Test) + contrôle d'intégrité
 - résumé des 6 roulements d'entraînement
 - fig_real_train.png : RMS horizontal des 6 roulements d'entraînement
 - fig_real_norm.png  : RMS normalisé par la ligne de base, en fonction du % de vie (train + validation)
"""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

F = Path("features")
def load(split, name): return pd.read_csv(F / f"{split}__{name}.csv")
names = lambda split: sorted(p.stem.split("__")[1] for p in F.glob(f"{split}__*.csv"))
COND = lambda n: int(n[7])                                   # Bearing1_x -> condition 1
def base(x, n0=100):
    b = np.median(x[:n0]); return b, 1.4826 * np.median(np.abs(x[:n0] - b))

# 1) RUL de test + contrôle d'intégrité -------------------------------------------------
print("\n=== RUL de test (snapshots de 10 s) et contrôle d'intégrité ===")
print(f"{'roulement':<11}{'n_test':>7}{'L_val':>7}{'RUL(pas)':>9}{'RUL(s)':>8}{'écart max Test vs début Val':>30}")
for n in names("Test_set"):
    t, v = load("Test_set", n), load("Validation_Set", n)
    k = len(t); diff = np.abs(t["rms_h"].to_numpy() - v["rms_h"].to_numpy()[:k]).max()
    flag = "" if diff < 1e-6 else "   <-- ATTENTION : le début de Validation ≠ Test"
    print(f"{n:<11}{k:>7}{len(v):>7}{len(v)-k:>9}{(len(v)-k)*10:>8}{diff:>30.2e}{flag}")

# 2) Résumé de l'entraînement ------------------------------------------------------------
print("\n=== Roulements d'entraînement (RMS horizontal) ===")
print(f"{'roulement':<11}{'cond':>5}{'L':>6}{'base':>7}{'final':>8}{'max':>8}{'final/base':>11}{'kurt_fin':>9}")
tr = {n: load("Training_set", n) for n in names("Training_set")}
for n, d in tr.items():
    x = d["rms_h"].to_numpy(); b, _ = base(x); fin = np.median(x[-10:])
    print(f"{n:<11}{COND(n):>5}{len(x):>6}{b:>7.2f}{fin:>8.2f}{x.max():>8.2f}{fin/b:>11.1f}{np.median(d['kurt_h'].to_numpy()[-10:]):>9.1f}")

# 3) Figures --------------------------------------------------------------------------------
plt.rcParams.update({"axes.grid": True, "grid.alpha": .3, "font.size": 9})
fig, ax = plt.subplots(2, 3, figsize=(14, 7))
for a, (n, d) in zip(ax.ravel(), tr.items()):
    x = d["rms_h"]; a.plot(x, color="#bbb", lw=.7, label="RMS brut"); a.plot(x.rolling(15, min_periods=1).median(), color="#1f77b4", lw=1.3, label="médiane glissante")
    a.set_title(f"{n} (cond. {COND(n)}, L={len(x)})"); a.set_xlabel("snapshot (10 s)"); a.set_ylabel("RMS horizontal (g)")
ax[0, 0].legend(); plt.tight_layout(); plt.savefig("fig_real_train.png", dpi=140); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
cols = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}
for a, split in zip(ax, ["Training_set", "Validation_Set"]):
    for n in names(split):
        x = load(split, n)["rms_h"]; b, _ = base(x.to_numpy()); s = x.rolling(15, min_periods=1).median() / b
        a.plot(np.linspace(0, 100, len(s)), s, color=cols[COND(n)], lw=1.1, alpha=.8, label=f"{n}")
    a.set_yscale("log"); a.set_xlabel("% de la vie consommée"); a.set_title(f"{split} : RMS / ligne de base (échelle log)")
    a.legend(fontsize=6, ncol=2)
ax[0].set_ylabel("RMS lissé / base"); plt.tight_layout(); plt.savefig("fig_real_norm.png", dpi=140); plt.close()
print("\nFigures écrites : fig_real_train.png, fig_real_norm.png")
