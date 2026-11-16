"""ETAPE 2 : extraire un point d'indicateur par snapshot (fichier acc_XXXXX.csv) et le mettre en cache.
Chaque fichier = 2560 mesures d'accélération (25,6 kHz, 0,1 s), colonnes : h, min, s, µs, horizontale, verticale.
Usage :  python features.py              -> crée ./features/<ensemble>__<roulement>.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kurtosis

ROOT = Path("data"); OUT = Path("features")

def read_acc(path):
    sep = ";" if ";" in path.open().readline() else ","       # certains fichiers utilisent ';'
    a = pd.read_csv(path, header=None, sep=sep).apply(pd.to_numeric, errors="coerce").to_numpy()
    return a[:, 4], a[:, 5]                                   # horizontale, verticale

def snapshot_features(h, v):
    rms = lambda s: float(np.sqrt(np.mean(s ** 2)))
    return dict(rms_h=rms(h), rms_v=rms(v), kurt_h=float(kurtosis(h)), peak_h=float(np.max(np.abs(h))),
                crest_h=float(np.max(np.abs(h)) / rms(h)))

def find_bearings(root=ROOT):
    groups = {}
    for p in root.rglob("acc_*.csv"):
        rel = p.relative_to(root).parts
        groups.setdefault((rel[0], p.parent.name), []).append(p)
    return {k: sorted(v, key=lambda q: int(q.stem.split("_")[1])) for k, v in groups.items()}

def extract_all(root=ROOT, out=OUT):
    out.mkdir(exist_ok=True)
    for (split, name), files in sorted(find_bearings(root).items()):
        dest = out / f"{split}__{name}.csv"
        if dest.exists(): print("déjà fait :", dest.name); continue
        rows = []
        for i, f in enumerate(files):
            rows.append(snapshot_features(*read_acc(f)))
            if i % 300 == 0: print(f"  {split}/{name}: {i}/{len(files)}", flush=True)
        df = pd.DataFrame(rows); df.insert(0, "t", np.arange(len(df)))
        df.to_csv(dest, index=False); print("écrit :", dest.name, len(df), "snapshots")

def load_features(split, name, out=OUT):
    return pd.read_csv(out / f"{split}__{name}.csv")

if __name__ == "__main__":
    extract_all(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT)
