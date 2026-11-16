"""ETAPE 7b : indicateurs FREQUENTIELS (energie par bande de frequence, FFT) -> ./features_spec/
Chaque snapshot : 2560 points a 25,6 kHz. Bandes (Hz) : 0-500, 500-1k, 1k-2k, 2k-3,5k, 3,5k-5,5k, 5,5k-8k, 8k-12,8k.
Colonnes : bh0..bh6 (horizontale), bv0..bv6 (verticale), cent_h, cent_v (centroide spectral, Hz).
Usage :  python features_spec.py     (relit ~39 000 fichiers : plusieurs minutes ; reprend la ou il s'est arrete)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from features import find_bearings, read_acc

FS = 25600; BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 3500), (3500, 5500), (5500, 8000), (8000, 12800)]
OUT = Path("features_spec")

def spectral(sig):
    sig = sig - np.mean(sig); sp = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) / len(sig); f = np.fft.rfftfreq(len(sig), 1 / FS)
    bands = [float(np.sqrt(np.sum(sp[(f >= lo) & (f < hi)] ** 2))) for lo, hi in BANDS]
    cent = float(np.sum(f * sp) / (np.sum(sp) + 1e-12)); return bands, cent

if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data"); OUT.mkdir(exist_ok=True)
    for (split, name), files in sorted(find_bearings(root).items()):
        dest = OUT / f"{split}__{name}.csv"
        if dest.exists(): print("déjà fait :", dest.name); continue
        rows = []
        for i, fpath in enumerate(files):
            h, v = read_acc(fpath); bh, ch = spectral(h); bv, cv = spectral(v)
            rows.append({**{f"bh{k}": x for k, x in enumerate(bh)}, **{f"bv{k}": x for k, x in enumerate(bv)}, "cent_h": ch, "cent_v": cv})
            if i % 300 == 0: print(f"  {split}/{name}: {i}/{len(files)}", flush=True)
        df = pd.DataFrame(rows); df.insert(0, "t", np.arange(len(df))); df.to_csv(dest, index=False); print("écrit :", dest.name, len(df))
