"""ETAPE 8a : choix de l'indicateur parmi RMS / pic / 14 bandes de frequence, sur les 6 roulements d'ENTRAINEMENT seulement.
Criteres : monotonie, tendance (min des correlations), prognosabilite, et PREAVIS SOUTENU =
nombre de snapshots restant apres la derniere fois ou l'indicateur normalise est redescendu sous 1,5 (robuste aux fausses alertes).
Classement : rang moyen sur (monotonie, tendance, prognosabilite, preavis median). Ecrit hi_choice.json et fig_hi_spec.png.
Les centroides spectraux sont exclus (la logique de detection suppose un indicateur qui AUGMENTE)."""
import json
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

F, S = Path("features"), Path("features_spec"); WIN, B0, B1, THR = 15, 150, 250, 1.5
CAND = ["rms_tot", "rms_h", "rms_v", "peak_h"] + [f"bh{k}" for k in range(7)] + [f"bv{k}" for k in range(7)]
BANDS = ["0-500 Hz", "0,5-1 kHz", "1-2 kHz", "2-3,5 kHz", "3,5-5,5 kHz", "5,5-8 kHz", "8-12,8 kHz"]
tr = {}
for p in sorted(F.glob("Training_set__*.csv")):
    d = pd.read_csv(p).merge(pd.read_csv(S / p.name), on="t"); d["rms_tot"] = np.sqrt(d.rms_h ** 2 + d.rms_v ** 2); tr[p.stem.split("__")[1]] = d

def norm(x):
    s = pd.Series(x).rolling(WIN, min_periods=1).median().to_numpy(); return s / np.median(s[B0:B1])
mono = lambda x: abs((np.diff(x) > 0).sum() - (np.diff(x) < 0).sum()) / (len(x) - 1)
res = lambda x, n=200: np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)
def lead(h):
    below = np.where(h[B1:] <= THR)[0]
    return len(h) - (B1 + below[-1] + 1) if len(below) else len(h) - B1

rows, curves = [], {}
for f in CAND:
    H = [norm(d[f].to_numpy(float)) for d in tr.values()]; curves[f] = H
    R = np.array([res(h) for h in H]); C = np.corrcoef(R); fin = np.array([np.median(h[-5:]) for h in H])
    ld = np.array([lead(h) for h in H]); lp = np.array([100 * l / len(h) for l, h in zip(ld, H)])
    rows.append(dict(indicateur=f, monotonie=np.mean([mono(h) for h in H]), tendance_min=np.min(np.abs(C[np.triu_indices(len(R), 1)])),
                     prognosabilite=float(np.exp(-np.std(fin) / np.mean(np.abs(fin - 1)))), preavis_med_pas=np.median(ld), preavis_min_pas=ld.min(), preavis_med_pct=np.median(lp)))
T = pd.DataFrame(rows)
for c in ["monotonie", "tendance_min", "prognosabilite", "preavis_med_pas"]: T["r_" + c] = T[c].rank(ascending=False)
T["rang_moyen"] = T[[c for c in T if c.startswith("r_")]].mean(axis=1); T = T.sort_values("rang_moyen").reset_index(drop=True)
show = T[["indicateur", "monotonie", "tendance_min", "prognosabilite", "preavis_med_pas", "preavis_min_pas", "preavis_med_pct", "rang_moyen"]].round(2)
print("\n=== Classement des indicateurs (6 roulements d'entraînement) ===")
print("préavis = snapshots (10 s) restants après la dernière redescente sous 1,5 x base ; 'min' = pire roulement\n")
print(show.to_string(index=False))
ref = T[T.indicateur == "rms_tot"].index[0] + 1; best = T.loc[0, "indicateur"]
print(f"\nRMS total (référence) classé {ref}e sur {len(T)}. Meilleur indicateur selon ce critère : {best}")
for k, n in enumerate(BANDS): print(f"   bh{k}/bv{k} = bande {n}")
json.dump({"hi": best}, open("hi_choice.json", "w")); print("-> écrit hi_choice.json")

top = list(T["indicateur"][:5]) + (["rms_tot"] if "rms_tot" not in list(T["indicateur"][:5]) else [])
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
for a, f in zip(ax.ravel(), top):
    for n, h in zip(tr, curves[f]): a.plot(np.linspace(0, 100, len(h)), h, lw=.9, label=n)
    a.set_yscale("log"); a.set_title(f"{f} / base" + ("  (référence)" if f == "rms_tot" else "")); a.set_xlabel("% de vie consommée"); a.grid(alpha=.3)
ax.ravel()[0].legend(fontsize=7); plt.tight_layout(); plt.savefig("fig_hi_spec.png", dpi=140); print("Figure écrite : fig_hi_spec.png")
