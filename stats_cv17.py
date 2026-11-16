"""ETAPE 7a : les ecarts entre methodes sont-ils significatifs ? (a lancer apres rul_cv17.py)
Pour chaque roulement : MAE de chaque methode ; difference par paires ; IC95 % par bootstrap SUR LES ROULEMENTS ; test de Wilcoxon."""
import sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
D = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results_cv17.csv"); rng = np.random.default_rng(0)
PAIRS = [("M0c prior même condition", "M0 prior", "la condition aide-t-elle ?"),
         ("M1 deux régimes", "M0 prior", "le capteur aide-t-il (sans condition) ?"),
         ("M1c deux régimes même cond.", "M0c prior même condition", "le capteur aide-t-il (à condition connue) ?")]
for kind, titre in [("age", "multi-âges (17 roulements)"), ("trunc", "point d'arrêt officiel (11 roulements)")]:
    d = D[D.kind == kind]; mae = d.groupby(["unit", "method"])["err"].apply(lambda s: s.abs().mean()).unstack()
    print(f"\n=== {titre} : MAE par roulement, moyenne et écart-type ===")
    print(mae.agg(["mean", "std"]).round(0).T.to_string())
    print("\n  différence de MAE (A - B) : négatif = A meilleur que B")
    for a, b, q in PAIRS:
        diff = (mae[a] - mae[b]).to_numpy(); n = len(diff)
        boot = [rng.choice(diff, n).mean() for _ in range(10000)]; lo, hi = np.percentile(boot, [2.5, 97.5])
        try: p = wilcoxon(diff).pvalue
        except Exception: p = float("nan")
        verdict = "écart significatif" if (lo > 0 or hi < 0) else "NON significatif (l'IC contient 0)"
        print(f"  {q}\n     {a}  vs  {b}: moyenne={diff.mean():+7.0f} pas  IC95=[{lo:+.0f}; {hi:+.0f}]  A meilleur sur {(diff<0).sum()}/{n} roulements  Wilcoxon p={p:.3f}  -> {verdict}")
