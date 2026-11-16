"""ETAPE 8b : l'indicateur B est-il significativement meilleur que l'indicateur A ? (meme methode, meme roulement, MAE par paires)
Usage : python compare_hi.py results_cv17_rms_tot.csv results_cv17_bh6.csv"""
import sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
if len(sys.argv) < 3: sys.exit("Usage : python compare_hi.py fichier_A.csv fichier_B.csv")
A, B = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2]); rng = np.random.default_rng(0)
nameA, nameB = sys.argv[1].replace("results_cv17_", "").replace(".csv", ""), sys.argv[2].replace("results_cv17_", "").replace(".csv", "")
print(f"\nComparaison : A = {nameA}   vs   B = {nameB}   (différence = MAE(B) - MAE(A) ; négatif = B meilleur)")
for kind, titre in [("age", "multi-âges (17 roulements)"), ("trunc", "point d'arrêt officiel (11 roulements)")]:
    print(f"\n=== {titre} ===")
    ma = A[A.kind == kind].groupby(["unit", "method"])["err"].apply(lambda s: s.abs().mean()).unstack()
    mb = B[B.kind == kind].groupby(["unit", "method"])["err"].apply(lambda s: s.abs().mean()).unstack()
    for m in ma.columns:
        d = (mb[m] - ma[m]).dropna().to_numpy(); n = len(d)
        lo, hi = np.percentile([rng.choice(d, n).mean() for _ in range(10000)], [2.5, 97.5])
        try: p = wilcoxon(d).pvalue
        except Exception: p = float("nan")
        v = "SIGNIFICATIF" if (lo > 0 or hi < 0) else "non significatif"
        print(f"  {m:<30} MAE {nameA}={ma[m].mean():6.0f}  {nameB}={mb[m].mean():6.0f}  diff={d.mean():+6.0f}  IC95=[{lo:+.0f}; {hi:+.0f}]  B meilleur sur {(d<0).sum()}/{n}  p={p:.3f}  {v}")
