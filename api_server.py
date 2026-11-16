"""Service de pronostic temps reel : Node-RED (capteur simule) -> POST /sample -> RUL + incertitude.
Lancer :  pip install flask numpy pandas  &&  python api_server.py     (ecoute sur http://localhost:5000)
"""
import json, numpy as np
from flask import Flask, request, jsonify
from simulate import make_fleet, FT
from rul_core import Params, rul_distribution, health_indicator

S = json.load(open("summary.json"))
P = Params(**S["params"]); DELTA = S["delta"]
fleet = make_fleet(14, 7)                                   # memes unites que run_validation.py
TRAIN_LIVES = [len(u["x"]) for u in fleet if u["unit"] not in S["test"]]
EVERY = 5                                                   # recalcul de la RUL tous les 5 echantillons
app = Flask(__name__); state = {}

def reset():
    state.update(x=[], last=None, rng=np.random.default_rng(0))
reset()

@app.post("/reset")
def _reset(): reset(); return jsonify(ok=True)

@app.post("/sample")
def sample():
    x = float(request.get_json()["x"]); state["x"].append(x); t = len(state["x"])
    if state["last"] is None or t % EVERY == 0:
        s, mode = rul_distribution(np.array(state["x"]), TRAIN_LIVES, P, state["rng"])
        q = {k: float(np.quantile(s, v)) for k, v in dict(q05=.05, q10=.1, q20=.2, q50=.5, q90=.9).items()}
        state["last"] = dict(mode=mode, **q)
    hi = float(health_indicator(np.array(state["x"][-P.win:]), P.win)[-1])
    out = dict(t=t, x=x, hi=hi, ft=FT, delta=DELTA, **state["last"])
    # decision : date d'intervention = t + quantile - delai logistique (>= t)
    for k in ("q50", "q20", "q10", "q05"):
        out["intervene_at_" + k] = max(t, int(t + out[k] - DELTA))
        out["alarm_" + k] = bool(out[k] <= DELTA and out["mode"] == "model")
    return jsonify(out)

if __name__ == "__main__":
    app.run(port=5000)
