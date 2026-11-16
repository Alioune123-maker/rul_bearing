"""DEMONSTRATEUR (Streamlit) : RUL d'un roulement reel PRONOSTIA, a plusieurs instants, avec incertitude et date d'intervention.
Lancer :  streamlit run demo_app.py
Le modele est appris sur les 16 AUTRES roulements (validation 'un roulement laisse de cote') : aucune fuite."""
import numpy as np, pandas as pd
import streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from rul_lib import load_units, samples, decision_rows, METHODS, KEYS, STEPS_PER_MIN as SPM

st.set_page_config(page_title="Pronostic de RUL – roulements PRONOSTIA", layout="wide")
st.title("Pronostic de RUL d'un roulement – données réelles PRONOSTIA / FEMTO-ST")
st.caption("Chaque prédiction est faite par un modèle appris sur les 16 autres roulements. 1 snapshot = 10 s ; durées affichées en minutes.")

@st.cache_data(show_spinner="Chargement des données…")
def get_units(hi): return load_units(hi)

with st.sidebar:
    st.header("Scénario")
    hi_opts = ["rms_tot"] + (["bh6"] if Path("features_spec").exists() else [])
    hi = st.selectbox("Indicateur de santé", hi_opts, format_func=lambda h: {"rms_tot": "RMS total (vibration)", "bh6": "Énergie 8–12,8 kHz (axe horizontal)"}[h])
    method = st.radio("Méthode", list(METHODS), format_func=lambda m: f"{m} – {METHODS[m]}")
    units = get_units(hi); tests = [u for u in units if u["n_test"] is not None]
    name = st.selectbox("Roulement (jeu de test)", [u["name"] for u in tests]); b = next(u for u in units if u["name"] == name)
    t = st.slider("Instant t (snapshot)", 60, b["L"] - 1, int(b["n_test"]), help="Par défaut : le point d'arrêt officiel du jeu de test.")
    D_min = st.slider("Délai logistique d'intervention D (min)", 1, 120, 15)
    truth = st.checkbox("Révéler la panne réelle", value=False)
    st.markdown(f"Condition de fonctionnement : **{b['cond']}** · durée de vie réelle : **{b['L']/SPM:.0f} min**")

smp = samples(method, units, b, t, hi); true_rul = b["L"] - t; D = D_min * SPM
lo, med, hi_q = np.quantile(smp, [0.1, 0.5, 0.9]); rows = decision_rows(smp, t, b["L"], D)
tab1, tab2, tab3 = st.tabs(["① RUL à l'instant t et décision", "② RUL à plusieurs instants", "③ Validation et limites"])

with tab1:
    c = st.columns(3)
    c[0].metric("RUL médiane", f"{med/SPM:.0f} min"); c[1].metric("Intervalle à 80 %", f"{lo/SPM:.0f} – {hi_q/SPM:.0f} min")
    c[2].metric("RUL réelle", f"{true_rul/SPM:.0f} min" if truth else "masquée")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2)); h = b["h"]
    if h is not None:
        ax[0].plot(np.arange(t) / SPM, h[:t], color="#1f77b4", lw=1); thr = KEYS[(hi, "M1c")][0]; ax[0].axhline(thr, color="gray", ls="--", lw=.8, label=f"seuil de détection ({thr}×base)")
        from rul_lib import td; tdt = td(b, KEYS[(hi, "M1c")])
        if tdt is not None and tdt < t: ax[0].axvline(tdt / SPM, color="orange", ls=":", label="détection d'accélération")
        ax[0].set_yscale("log"); ax[0].legend(fontsize=8)
    ax[0].set_xlabel("temps (min)"); ax[0].set_ylabel("indicateur / base"); ax[0].set_title(f"{name} : indicateur de santé jusqu'à t"); ax[0].grid(alpha=.3)
    for i, r in rows.iterrows():
        col = ("#2ca02c" if r["ok"] else "#d62728") if truth else "#1f77b4"
        ax[1].barh(r["regle"], D / SPM, left=r["debut"] / SPM, color=col, alpha=.8)
    ax[1].axvspan(lo / SPM, hi_q / SPM, color="#1f77b4", alpha=.12, label="IC 80 % de la RUL"); ax[1].axvline(med / SPM, color="#1f77b4", ls="--", lw=1, label="RUL médiane")
    if truth: ax[1].axvline(true_rul / SPM, color="red", lw=2, label="panne réelle")
    ax[1].set_xlabel("minutes à partir de maintenant"); ax[1].set_title("Fenêtre d'intervention selon la règle de décision"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3); ax[1].invert_yaxis()
    plt.tight_layout(); st.pyplot(fig)
    show = pd.DataFrame({"Règle": rows["regle"], "RUL retenue (min)": (rows["rul_q"] / SPM).round(1), "Début de l'intervention dans (min)": (rows["debut"] / SPM).round(1),
                         "Fin de l'intervention (min)": (rows["fin"] / SPM).round(1)})
    if truth: show["Issue"] = np.where(rows["ok"], "panne évitée", "PANNE avant la fin"); show["Marge (min)"] = (rows["marge"] / SPM).round(1)
    st.dataframe(show, hide_index=True)
    st.info("Plus la règle est prudente (quantile bas), plus l'intervention est avancée : on évite plus de pannes mais on jette plus de vie utile. C'est l'effet de l'incertitude sur la date d'intervention.")

with tab2:
    fr = np.arange(0.10, 0.951, 0.05); ts = [int(f * b["L"]) for f in fr]; R = []
    for ti in ts:
        s_ = samples(method, units, b, ti, hi); R.append(np.quantile(s_, [0.1, 0.5, 0.9]))
    R = np.array(R) / SPM; tm = np.array(ts) / SPM; fig2, ax2 = plt.subplots(figsize=(9, 4.2))
    ax2.fill_between(tm, R[:, 0], R[:, 2], alpha=.25, color="#1f77b4", label="IC 80 %"); ax2.plot(tm, R[:, 1], "o-", ms=3, color="#1f77b4", label="RUL médiane estimée")
    ax2.plot(tm, (b["L"] - np.array(ts)) / SPM, "k--", label="RUL réelle"); ax2.set_xlabel("instant t (min)"); ax2.set_ylabel("RUL (min)"); ax2.grid(alpha=.3); ax2.legend()
    ax2.set_title(f"{name} : RUL à 18 instants de sa vie (méthode {method})"); st.pyplot(fig2)
    st.caption("Une bande large signifie que le modèle sait qu'il ne sait pas. L'estimation s'affine rarement avant les derniers pourcents de vie.")

with tab3:
    f = Path(f"results_cv17_{hi}.csv")
    if f.exists():
        D_ = pd.read_csv(f); a = D_[D_.kind == "age"].groupby("method").agg(MAE_pas=("err", lambda s: s.abs().mean()), couverture_IC80=("in", lambda s: 100 * s.mean()), predictions_tardives=("err", lambda s: 100 * (s > 0).mean())).round(1)
        st.subheader("Validation croisée sur 17 roulements (multi-âges)"); st.dataframe(a)
    st.markdown("""**Limites à garder en tête**
- 17 roulements, 3 conditions de fonctionnement, un seul type de défaut : les écarts entre méthodes ne sont pas statistiquement établis.
- Banc de vieillissement **accéléré** : les durées en minutes ne se transposent pas telles quelles à un roulement industriel.
- Les intervalles annoncés à 80 % couvrent la vérité bien moins souvent (≈ 10 à 60 % selon la méthode) : l'incertitude est **sous-estimée**.
- Une partie des roulements n'est jamais détectée par le capteur ; l'alarme ne peut pas être la seule protection.""")
