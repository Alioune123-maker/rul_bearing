"""ETAPE 6 : validation croisee 'un roulement laisse de cote' sur les 17 roulements menes a la panne
(6 Training + 11 Validation = vies completes). Pour chaque roulement : apprentissage sur les 16 autres, evaluation sur lui.
Le reglage (seuil, persistance de detection) est refait A L'INTERIEUR de chaque pli (validation croisee imbriquee) : pas de fuite.
Methodes : M0 prior (tous) | M0c prior (meme condition) | M1 deux regimes | M1c deux regimes (meme condition)
Evaluation : (i) multi-ages (10-95 % de vie) sur les 17 ; (ii) point d'arret officiel des 11 roulements de test.
"""
import itertools, collections, sys, json
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

F = Path("features"); WIN = 15; B0, B1 = 150, 250
HI = sys.argv[1] if len(sys.argv) > 1 else "rms_tot"          # python rul_cv17.py [rms_tot | auto | bh3 | ...]
if HI == "auto": HI = json.load(open("hi_choice.json"))["hi"]
print("Indicateur de santé utilisé :", HI)
KEYS = [(thr, per) for thr in (1.5, 2.0, 3.0) for per in (3, 5, 10, 20)]
FR = np.round(np.arange(0.10, 0.951, 0.05), 2); FR_TUNE = np.arange(0.5, 0.951, 0.05)
METH = {"M0 prior": (False, False), "M0c prior même condition": (True, False), "M1 deux régimes": (False, True), "M1c deux régimes même cond.": (True, True)}

def load(split):
    out = {}
    for p in sorted(F.glob(f"{split}__*.csv")):
        d = pd.read_csv(p); sp = Path("features_spec") / p.name
        if sp.exists(): d = d.merge(pd.read_csv(sp), on="t")
        d["rms_tot"] = np.sqrt(d["rms_h"] ** 2 + d["rms_v"] ** 2); out[p.stem.split("__")[1]] = d
    return out
def prep(name, d):
    raw = d[HI].to_numpy(float); s = pd.Series(raw).rolling(WIN, min_periods=1).median().to_numpy()
    u = dict(name=name, cond=int(name[7]), L=len(raw), h=None)
    if len(raw) >= B1: u["h"] = s / np.median(s[B0:B1])
    return u

units = [prep(n, d) for n, d in load("Training_set").items()]
units += [prep(n, d) for n, d in load("Validation_Set").items()]
n_test = {n: len(d) for n, d in load("Test_set").items()}
print(f"{len(units)} roulements menés à la panne ; conditions : {dict(collections.Counter(u['cond'] for u in units))}")

def knee_full(u, key):
    thr, per = key
    if u["h"] is None: return None
    run = 0
    for i in range(B1, u["L"]):
        run = run + 1 if u["h"][i] > thr else 0
        if run >= per: return i
    return None
TD = {u["name"]: {k: knee_full(u, k) for k in KEYS} for u in units}       # detection causale : connue a t > TD

def prior(P, t):
    rem = [p["L"] - t for p in P if p["L"] > t]; return np.array(rem if rem else [5.0])
def samples(method, pool, u, t, key):
    use_c, use_r = METH[method]; P = pool
    if use_c:
        Pc = [p for p in pool if p["cond"] == u["cond"]]
        if len(Pc) >= 3 and any(p["L"] > t for p in Pc): P = Pc
    if not use_r or t < B1 or u["h"] is None: return prior(P, t)
    td = TD[u["name"]][key]; tdet = td if (td is not None and td < t) else None
    if tdet is None:
        rem = [p["L"] - t for p in P if p["L"] > t and (TD[p["name"]][key] is None or TD[p["name"]][key] > t)]
        return np.array(rem) if rem else prior(P, t)
    s = t - tdet; resid = [p["L"] - TD[p["name"]][key] for p in P if TD[p["name"]][key] is not None]
    rem = [r - s for r in resid if r > s]; return np.array(rem if rem else [5.0])

def tune(pool, method):
    best, bm = None, 1e18
    for key in KEYS:
        errs = []
        for j in range(len(pool)):
            inner = pool[:j] + pool[j + 1:]; h = pool[j]
            errs += [abs(np.median(samples(method, inner, h, int(f * h["L"]), key)) - (h["L"] - int(f * h["L"]))) for f in FR_TUNE]
        m = np.mean(errs)
        if m < bm: best, bm = key, m
    return best

rows, chosen = [], collections.defaultdict(list)
for b in units:
    pool = [p for p in units if p["name"] != b["name"]]
    for m, (use_c, use_r) in METH.items():
        key = tune(pool, m) if use_r else KEYS[0]; chosen[m].append(key) if use_r else None
        pts = [("age", int(f * b["L"])) for f in FR] + ([("trunc", n_test[b["name"]])] if b["name"] in n_test else [])
        for kind, t in pts:
            lo, med, hi = np.quantile(samples(m, pool, b, t, key), [0.1, 0.5, 0.9])
            rows.append(dict(unit=b["name"], cond=b["cond"], kind=kind, frac=round(t / b["L"], 2), t=t, true=b["L"] - t, method=m, lo=lo, med=med, hi=hi))
D = pd.DataFrame(rows); D["err"] = D["med"] - D["true"]; D["er_pct"] = 100 * (D["true"] - D["med"]) / D["true"]; D["in"] = (D["true"] >= D["lo"]) & (D["true"] <= D["hi"])
ML = list(METH)
def summ(d):
    g = d.groupby("method"); return pd.DataFrame({"MAE (pas)": g["err"].apply(lambda s: s.abs().mean()), "|err| médiane": g["err"].apply(lambda s: s.abs().median()),
        "|Er| médian %": g["er_pct"].apply(lambda s: s.abs().median()), "% tardives": g["err"].apply(lambda s: (s > 0).mean() * 100),
        "couverture IC80 %": g["in"].mean() * 100, "largeur IC méd.": g.apply(lambda s: (s["hi"] - s["lo"]).median())}).loc[ML].round(1)

print("\n=== Réglages choisis par pli (nested CV) : (seuil, persistance) les plus fréquents ===")
for m, ks in chosen.items(): print(f"  {m}: {collections.Counter(ks).most_common(3)}")

age = D[D.kind == "age"]; tr = D[D.kind == "trunc"]
print("\n=== (i) Multi-âges, validation croisée sur 17 roulements ===\n", summ(age).to_string())
byf = age.groupby(["frac", "method"])["err"].apply(lambda s: s.abs().mean()).unstack()[ML].round(0)
print("\nMAE selon l'avancement en vie :\n", byf.loc[[0.2, 0.4, 0.6, 0.8, 0.9, 0.95]].to_string())
print("\n=== (ii) Point d'arrêt officiel des 11 roulements de test (modèle appris sur les 16 autres) ===\n", summ(tr).to_string())
pv = tr.pivot(index="unit", columns="method", values="med").round(0)[ML]; pv.insert(0, "RUL vraie", tr.groupby("unit")["true"].first())
print("\nRUL médiane prédite :\n", pv.to_string())

key = collections.Counter(chosen["M1 deux régimes"]).most_common(1)[0][0]
print(f"\n=== Les deux modes de défaillance (détection avec seuil={key[0]}, persistance={key[1]}) ===")
print(f"{'roulement':<12}{'cond':>5}{'L':>6}{'détection':>10}{'vie restante':>13}{'à % de vie':>11}")
for u in sorted(units, key=lambda x: (x["cond"], x["name"])):
    td = TD[u["name"]][key]; print(f"{u['name']:<12}{u['cond']:>5}{u['L']:>6}{(td if td is not None else '—'):>10}{(u['L']-td if td is not None else '—'):>13}{(round(100*td/u['L'],1) if td is not None else '—'):>11}")
D.to_csv(f"results_cv17_{HI}.csv", index=False)

plt.rcParams.update({"axes.grid": True, "grid.alpha": .3, "font.size": 9}); col = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for u in units:
    if u["h"] is None: continue
    ttf = u["L"] - np.arange(u["L"]); ax[0].plot(ttf, u["h"], color=col[u["cond"]], lw=.9, alpha=.75, label=f"cond. {u['cond']}")
h_, l_ = ax[0].get_legend_handles_labels(); seen = dict(zip(l_, h_)); ax[0].legend(seen.values(), seen.keys(), fontsize=8)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].invert_xaxis(); ax[0].set_xlabel("snapshots avant la panne (échelle log, panne à droite)"); ax[0].set_ylabel("indicateur / base"); ax[0].set_title("17 roulements alignés sur l'instant de panne")
mc = dict(zip(ML, ["#7f7f7f", "#2ca02c", "#d62728", "#1f77b4"]))
for m in ML: ax[1].plot(byf.index * 100, byf[m], "o-", ms=3, label=m, color=mc[m])
ax[1].set_xlabel("% de la vie consommée"); ax[1].set_ylabel("MAE de RUL (pas)"); ax[1].set_title("(i) Erreur selon l'avancement (CV 17 roulements)"); ax[1].legend(fontsize=7)
for k, m in enumerate(ML):
    s = tr[tr.method == m]; ax[2].plot(s["true"] * (1 + .04 * (k - 1.5)), s["med"].clip(lower=1), "o", ms=5, color=mc[m], label=m, alpha=.85)
mx = max(tr["true"].max(), tr["med"].max()) * 1.2; ax[2].plot([5, mx], [5, mx], "k--", lw=.8); ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_xlabel("RUL vraie (pas)"); ax[2].set_ylabel("RUL prédite (médiane)"); ax[2].set_title("(ii) 11 roulements de test, point d'arrêt"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"fig_cv17_{HI}.png", dpi=140); print(f"\nFigure écrite : fig_cv17_{HI}.png ; CSV : results_cv17_{HI}.csv")
