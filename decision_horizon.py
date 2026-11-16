"""ETAPE 9 : DECISION - le prevavis est-il compatible avec un delai d'intervention reel ?
Compare, sur les 17 roulements menes a la panne, des politiques de maintenance :
  - laisser aller a la panne ; - remplacement a age fixe (quantile q des durees de vie des 16 autres roulements, leave-one-out) ;
  - alarme capteur (detection + delai logistique D) ; - hybride (le plus tot des deux).
Pour chaque delai logistique D (minutes) : pannes non evitees et vie perdue moyenne (% de la vie du roulement, parmi les sauves).
Usage : python decision_horizon.py [indicateur] [seuil] [persistance]      ex. : python decision_horizon.py rms_tot 3.0 5
1 snapshot = 10 s -> 6 snapshots = 1 minute."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
F = Path("features"); WIN, B0, B1 = 15, 150, 250
HI = sys.argv[1] if len(sys.argv) > 1 else "rms_tot"; THR = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0; PER = int(sys.argv[3]) if len(sys.argv) > 3 else 5
DELAYS_MIN = [5, 15, 30, 60]; QS = [0.05, 0.10, 0.25]

units = []
for split in ("Training_set", "Validation_Set"):
    for p in sorted(F.glob(f"{split}__*.csv")):
        d = pd.read_csv(p); sp = Path("features_spec") / p.name
        if sp.exists(): d = d.merge(pd.read_csv(sp), on="t")
        d["rms_tot"] = np.sqrt(d["rms_h"] ** 2 + d["rms_v"] ** 2); x = d[HI].to_numpy(float)
        s = pd.Series(x).rolling(WIN, min_periods=1).median().to_numpy(); L = len(x); td = None
        if L >= B1:
            h = s / np.median(s[B0:B1]); run = 0
            for i in range(B1, L):
                run = run + 1 if h[i] > THR else 0
                if run >= PER: td = i; break
        units.append(dict(name=p.stem.split("__")[1], L=L, td=td))
lives = np.array([u["L"] for u in units]); n = len(units)
print(f"\nIndicateur = {HI} ; détection : seuil {THR} x base pendant {PER} snapshots ; {n} roulements ; 1 snapshot = 10 s\n")

print("=== A. Préavis offert par l'alarme capteur (temps entre la détection et la panne) ===")
leads = [u["L"] - u["td"] if u["td"] is not None else None for u in units]
print(f"   non détectés : {sum(l is None for l in leads)}/{n}  |  préavis des détectés (minutes) : " + ", ".join(f"{l/6:.1f}" for l in sorted(l for l in leads if l is not None)))
print(f"\n   {'délai logistique D':<22}{'préavis suffisant':>18}{'préavis insuffisant':>21}{'non détecté':>13}")
for D in DELAYS_MIN:
    ok = sum(l is not None and l >= D * 6 for l in leads); ins = sum(l is not None and l < D * 6 for l in leads); nd = sum(l is None for l in leads)
    print(f"   {D:>3} min{'':<16}{ok:>14}/{n}{ins:>18}/{n}{nd:>10}/{n}")

print("\n=== B. Politiques de maintenance (validation 'un roulement laissé de côté' pour l'âge de remplacement) ===")
for D in DELAYS_MIN:
    rows = []
    def add(name, tint):
        fails = sum(1 for u, t in zip(units, tint) if t is None or t >= u["L"])
        saved = [(u["L"] - t) / u["L"] * 100 for u, t in zip(units, tint) if t is not None and t < u["L"]]
        rows.append((name, f"{fails}/{n}", f"{np.mean(saved):.1f} %" if saved else "—", fails, sum(saved) / 100))
    add("laisser aller à la panne", [None] * n)
    for q in QS:
        ages = [np.quantile(np.delete(lives, i), q) for i in range(n)]; add(f"remplacement à l'âge q{int(q*100)} de la population", ages)
    alarm = [(u["td"] + D * 6) if u["td"] is not None else None for u in units]; add(f"alarme capteur (+{D} min de délai)", alarm)
    for q in (0.10,):
        ages = [np.quantile(np.delete(lives, i), q) for i in range(n)]
        add(f"hybride : âge q{int(q*100)} OU alarme", [min(a, al) if al is not None else a for a, al in zip(ages, alarm)])
    print(f"\n   --- délai logistique D = {D} min ---\n   {'politique':<46}{'pannes non évitées':>20}{'vie perdue moy.':>17}")
    for r in rows: print(f"   {r[0]:<46}{r[1]:>20}{r[2]:>17}")
    FL = [0.5, 1, 2, 5, 10]
    print(f"\n   coût moyen par roulement (unité = 1 vie de roulement ; une panne non planifiée coûte F vies) :")
    print(f"   {'politique':<46}" + "".join(f"{'F='+str(f):>9}" for f in FL))
    for r in rows: print(f"   {r[0]:<46}" + "".join(f"{(r[3] * f + r[4]) / n:>9.2f}" for f in FL))
    ref = [r for r in rows if "q10 de la population" in r[0]][0]; al = [r for r in rows if r[0].startswith("alarme")][0]
    if al[3] > ref[3]:
        print(f"   -> seuil de rentabilité : l'alarme coûte moins cher que l'âge q10 seulement si F < {(ref[4] - al[4]) / (al[3] - ref[3]):.2f} vie(s) de roulement")
print("\nLecture : une politique est bonne si elle évite les pannes SANS jeter trop de vie utile. Le remplacement à âge fixe est la référence que le capteur doit battre.")
