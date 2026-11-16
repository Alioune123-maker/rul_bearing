"""Validation temporelle par unites + intervalles + decision. Reproductible (graine fixe)."""
import numpy as np, pandas as pd, json, itertools
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulate import make_fleet, FT
from rul_core import Params, rul_distribution, rul_summary, health_indicator, baseline, detect_onset

SEED = 7
fleet = make_fleet(14, SEED)
for u in fleet: u["L"] = len(u["x"])

# ---- 1. Separation PAR UNITES (aucun roulement du test n'est vu a l'apprentissage) ----
test_ids = ["B02", "B03", "B07", "B09", "B13"]   # conditions 2,3,1,3,1 : test équilibré
train = [u for u in fleet if u["unit"] not in test_ids]
test  = [u for u in fleet if u["unit"] in test_ids]
print("train:", [u["unit"] for u in train], "| test:", test_ids)

FRACS = np.round(np.arange(0.10, 0.951, 0.05), 2)

def evaluate(units, p, pool, seed=0, fracs=FRACS):
    """pool: durees de vie connues ; leave-one-out si l'unite en fait partie."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in units:
        lives = [l for (n, l) in pool if n != u["unit"]]
        for f in fracs:
            t = int(f * u["L"])
            s, mode = rul_distribution(u["x"][:t], lives, p, rng)
            lo, med, hi = rul_summary(s, p, mode=mode)
            rows.append(dict(unit=u["unit"], frac=f, t=t, true=u["L"] - t, lo=lo, med=med, hi=hi, mode=mode))
    d = pd.DataFrame(rows)
    d["err"] = d["med"] - d["true"]                 # >0 : prediction trop optimiste (tardive)
    d["in"] = (d["true"] >= d["lo"]) & (d["true"] <= d["hi"])
    return d

pool_train = [(u["unit"], u["L"]) for u in train]

# ---- 2. Reglage des hyperparametres SUR LE TRAIN UNIQUEMENT ----
best, best_mae = None, 1e18
for win, ns, mp in itertools.product([15, 25], [4.0, 6.0], [40, 80, 150]):
    p = Params(ft=FT, win=win, n_sigma=ns, min_pts=mp)
    d = evaluate(train, p, pool_train, fracs=np.array([0.5, 0.6, 0.7, 0.8, 0.9]))
    mae = d["err"].abs().mean()
    print(f"win={win:2d} n_sigma={ns} min_pts={mp:3d}  MAE train={mae:6.1f}")
    if mae < best_mae: best, best_mae = (win, ns, mp), mae
p = Params(ft=FT, win=best[0], n_sigma=best[1], min_pts=best[2])
print("choix:", best)

# ---- 3. Calibration de l'intervalle (80 %) sur le TRAIN ----
target = 0.80
for sc in np.arange(1.0, 6.01, 0.5):
    p.scale = float(sc)
    cov = evaluate(train, p, pool_train)["in"].mean()
    if cov >= target: break
print(f"facteur de calibration = {p.scale:.1f} (couverture train {cov:.2f})")

# ---- 4. Evaluation sur le TEST (jamais vu) ----
pool_all_train = pool_train
d = evaluate(test, p, pool_all_train)
p_raw = Params(**{**p.__dict__, "scale": 1.0})
d_raw = evaluate(test, p_raw, pool_all_train)

def tbl(d):
    g = d.groupby("frac")
    return pd.DataFrame({
        "MAE_RUL": g["err"].apply(lambda s: s.abs().mean()),
        "biais_moyen": g["err"].mean(),
        "%_pred_tardive": g["err"].apply(lambda s: (s > 0).mean() * 100),
        "couverture_80": g["in"].mean() * 100,
        "largeur_IC": g.apply(lambda s: (s["hi"] - s["lo"]).mean()),
    }).round(1)
T = tbl(d)
sel = T.loc[[0.2, 0.4, 0.6, 0.8, 0.9]] if all(x in T.index for x in [0.2, 0.4, 0.6, 0.8, 0.9]) else T
print("\nPerformance TEST selon l'avancement en vie :\n", sel)
glob = dict(
    MAE=d["err"].abs().mean(), RMSE=np.sqrt((d["err"] ** 2).mean()), biais=d["err"].mean(),
    pct_tardif=(d["err"] > 0).mean() * 100, couverture_calibree=d["in"].mean() * 100,
    couverture_brute=d_raw["in"].mean() * 100,
    modele_actif_pct=(d["mode"] == "model").mean() * 100,
    MAE_modele_actif=d[d["mode"] == "model"]["err"].abs().mean(),
    couverture_modele_actif=d[d["mode"] == "model"]["in"].mean() * 100,
    MAE_prior_seul=d[d["mode"] == "prior"]["err"].abs().mean())
print("\nGlobal TEST:", {k: round(float(v), 1) for k, v in glob.items()})
T.to_csv("metrics_by_life_fraction.csv"); d.to_csv("test_predictions.csv", index=False)

# alpha-lambda (+-20 %) : precision relative sur les points ou le modele est actif
d["rel"] = d["err"].abs() / d["true"].clip(lower=1)
print("Part des predictions a +-20 % de la RUL vraie, par avancement :")
print((d.groupby("frac")["rel"].apply(lambda s: (s <= 0.2).mean() * 100)).round(0).loc[[0.4, 0.6, 0.8, 0.9]].to_dict())

# ---- 5. Decision : quand intervenir ? (delai logistique DELTA) ----
DELTA = 120   # pas de 10 s necessaires pour preparer l'intervention (hypothese a justifier)
def decision(units, p, quantiles=(0.5, 0.2, 0.1, 0.05), step=10):
    rng = np.random.default_rng(1)
    out = {q: [] for q in quantiles}
    for u in units:
        lives = [l for (n, l) in pool_all_train if n != u["unit"]]
        qt = {q: None for q in quantiles}
        for t in range(p.n0 + 1, u["L"], step):
            s, mode = rul_distribution(u["x"][:t], lives, p, rng)
            if mode != "model": continue
            med = np.median(s)
            for q in quantiles:
                if qt[q] is None:
                    val = np.quantile(s, q)
                    if val <= DELTA: qt[q] = t
        for q in quantiles:
            ti = qt[q]
            out[q].append(dict(unit=u["unit"], L=u["L"], t_int=ti,
                               failure=(ti is None), lost=(u["L"] - ti) / u["L"] * 100 if ti else 0.0))
    rows = []
    for q, lst in out.items():
        df = pd.DataFrame(lst); ok = df[~df["failure"]]
        rows.append(dict(regle=f"intervenir si quantile {int(q*100)}% de RUL <= {DELTA}",
                         pannes_non_evitees=int(df["failure"].sum()), sur_=len(df),
                         vie_perdue_moy_pct=round(ok["lost"].mean(), 1) if len(ok) else np.nan))
    return pd.DataFrame(rows), out
dec, dec_raw = decision(test, p)
print("\nDecision (test) :\n", dec.to_string(index=False)); dec.to_csv("decision_policies.csv", index=False)

# ---- 6. Figures ----
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3})
cols = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for u in test:
    hi = health_indicator(u["x"], p.win); ax[0].plot(hi, lw=1.2, label=f'{u["unit"]} (cond.{u["cond"]})', color=cols[u["cond"]], alpha=.8)
ax[0].axhline(FT, color="k", ls="--"); ax[0].text(20, FT + .15, "seuil de fin de vie FT", fontsize=9)
ax[0].set_xlabel("snapshot (10 s)"); ax[0].set_ylabel("indicateur de santé : RMS lissé (g)"); ax[0].set_title("Trajectoires de dégradation (test)"); ax[0].legend(fontsize=8)
u = test[2]; h = health_indicator(u["x"], p.win); b, sg = baseline(h, p); t0 = detect_onset(h, b, sg, p)
ax[1].plot(u["x"], color="#bbb", lw=.7, label="RMS brut"); ax[1].plot(h, color="#1f77b4", lw=1.5, label="HI (médiane glissante)")
ax[1].axhline(FT, color="k", ls="--"); ax[1].axvline(t0, color="orange", ls=":", label=f"onset détecté (t={t0})")
ax[1].axvline(u["onset_true"], color="gray", ls=":", label=f'onset réel ({u["onset_true"]})')
ax[1].set_title(f'Construction de l\'indicateur – {u["unit"]}'); ax[1].set_xlabel("snapshot"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig1_indicateur_sante.png", dpi=150); plt.close()

fig, axs = plt.subplots(1, 5, figsize=(17, 3.6), sharey=False)
for a, u in zip(axs, test):
    s = d[d.unit == u["unit"]]
    a.fill_between(s.t, s.lo, s.hi, alpha=.3, color="#1f77b4", label="IC 80 % calibré")
    a.plot(s.t, s.med, "o-", ms=3, color="#1f77b4", label="RUL estimée (médiane)")
    a.plot(s.t, s.true, "k--", label="RUL vraie")
    a.set_title(f'{u["unit"]} (L={u["L"]})'); a.set_xlabel("instant t")
axs[0].set_ylabel("RUL (snapshots)"); axs[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig2_rul_intervalles.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(T.index * 100, T["MAE_RUL"], "o-"); ax[0].set_title("Erreur absolue de RUL selon l'avancement"); ax[0].set_xlabel("% de vie consommée"); ax[0].set_ylabel("MAE (snapshots)")
ax[1].plot(T.index * 100, T["biais_moyen"], "o-", color="#d62728"); ax[1].axhline(0, color="k", lw=.8)
ax[1].set_title("Biais (>0 : trop optimiste = tardif)"); ax[1].set_xlabel("% de vie consommée")
ax[2].plot(T.index * 100, T["couverture_80"], "o-", color="#2ca02c"); ax[2].axhline(80, color="k", ls="--")
ax[2].set_title("Couverture empirique de l'IC (cible 80 %)"); ax[2].set_xlabel("% de vie consommée"); ax[2].set_ylim(0, 105)
plt.tight_layout(); plt.savefig("fig3_erreur_biais_couverture.png", dpi=150); plt.close()

json.dump({k: float(v) for k, v in glob.items()} | dict(params=p.__dict__, delta=DELTA, test=test_ids), open("summary.json", "w"), indent=1, default=float)
