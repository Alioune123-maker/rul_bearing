"""Generateur de trajectoires de degradation de roulements, au format PRONOSTIA simplifie.

Chaque roulement = une serie d'indicateur (RMS vibratoire horizontale, en g), un echantillon
toutes les 10 s (comme PRONOSTIA/FEMTO-ST). Trois conditions de fonctionnement.
Fin de vie reelle = premier instant ou le RMS lisse depasse FT.

!! DONNEES SIMULEES : sert a developper et tester la chaine. A remplacer par les vraies
   donnees PRONOSTIA via load_pronostic() (voir README).
"""
import numpy as np, pandas as pd

FT = 8.0          # seuil de fin de vie sur le RMS lisse (g) -- a justifier dans le dossier
DT_S = 10         # pas d'echantillonnage (s), comme PRONOSTIA
CONDITIONS = {1: dict(base=0.45, amp=0.30), 2: dict(base=0.55, amp=0.35), 3: dict(base=0.50, amp=0.25)}

def simulate_bearing(rng, cond, life):
    c = CONDITIONS[cond]
    t = np.arange(life)
    onset = int(life * rng.uniform(0.35, 0.70))        # debut de la degradation
    tau = np.clip(t - onset, 0, None)
    # k tel que le RMS vrai atteint FT exactement en fin de vie
    k = np.log(1 + (FT - c["base"]) / c["amp"]) / (life - onset)
    clean = c["base"] + c["amp"] * (np.exp(k * tau) - 1)
    # bruit AR(1) multiplicatif + bruit additif + quelques pics (chocs parasites)
    e = np.zeros(life)
    for i in range(1, life):
        e[i] = 0.6 * e[i - 1] + rng.normal(0, 0.06)
    x = clean * np.exp(e) + rng.normal(0, 0.03, life)
    spikes = rng.random(life) < 0.004
    x[spikes] += rng.uniform(0.5, 2.0, spikes.sum())
    return np.clip(x, 0.05, None), onset

def make_fleet(n=14, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        cond = 1 + i % 3
        life = int(rng.uniform(900, 2800))
        x, onset = simulate_bearing(rng, cond, life)
        rows.append(dict(unit=f"B{i+1:02d}", cond=cond, x=x, onset_true=onset))
    return rows

def load_pronostic(csv_dir):
    """A completer avec les vrais fichiers PRONOSTIA : calculer le RMS de acc_*.csv
    (colonne horizontale) pour chaque snapshot -> un vecteur x par roulement."""
    raise NotImplementedError
