"""ETAPE 1 : inspecter le jeu FEMTO/PRONOSTIA décompressé dans ./data
Usage :  python explore_data.py            (ou  python explore_data.py chemin/vers/data)
Colle-moi toute la sortie : je m'en sers pour adapter le découpage train/test."""
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
if not ROOT.exists():
    sys.exit(f"Dossier introuvable : {ROOT.resolve()}  -> décompresse les 3 zips dedans.")

def split_of(p: Path):                       # Training_set / Test_set / Validation_Set
    rel = p.relative_to(ROOT).parts
    return rel[0] if len(rel) > 1 else "?"

bearings = defaultdict(list); others = []
for p in sorted(ROOT.rglob("*")):
    if p.is_file():
        if p.name.startswith("acc_") and p.suffix == ".csv": bearings[(split_of(p), p.parent.name)].append(p)
        elif not p.name.startswith("temp_"): others.append(p)

print(f"\n=== {len(bearings)} dossiers de roulements trouvés dans {ROOT.resolve()} ===\n")
print(f"{'ensemble':<16}{'roulement':<14}{'n_snapshots':>12}{'durée (s)':>11}  séparateur  forme_fichier")
for (split, name), files in sorted(bearings.items()):
    first = files[0]
    sep = ";" if ";" in first.open().readline() else ","
    df = pd.read_csv(first, header=None, sep=sep)
    print(f"{split:<16}{name:<14}{len(files):>12}{len(files)*10:>11}  {sep!r:^10}  {df.shape}")

print("\n=== Autres fichiers (étiquettes de RUL ? documentation ?) ===")
for p in others[:60]: print("  ", p.relative_to(ROOT))
if len(others) > 60: print(f"   ... +{len(others)-60} autres")
