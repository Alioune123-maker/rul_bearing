import json
TAB="tab1"; UITAB="uitab"; G1="g_ctrl"; G2="g_view"; G3="g_dec"

SENSOR = r"""
// Capteur simulé : RMS vibratoire horizontal d'un roulement (g), 1 échantillon par tick.
if (!flow.get('run')) return null;
let t = (flow.get('t') || 0) + 1; flow.set('t', t);
const life = flow.get('life') || 1500, noiseGain = flow.get('noise') || 1;
const drift = flow.get('drift') || false, FT = 8, base = 0.5, amp = 0.3;
const onset = Math.floor(0.5 * life);
const tau = Math.max(0, t - onset);
const k = Math.log(1 + (FT - base) / amp) / (life - onset);
const clean = base + amp * (Math.exp(k * tau) - 1);
function gauss() { return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random()); }
let e = 0.6 * (flow.get('e') || 0) + 0.06 * noiseGain * gauss(); flow.set('e', e);
let x = clean * Math.exp(e) + 0.03 * noiseGain * gauss();
if (Math.random() < 0.004) x += 0.5 + 1.5 * Math.random();     // choc parasite
if (drift) x += 0.0015 * t;                                      // défaut capteur : dérive lente
x = Math.max(0.05, x);
if (t >= life) {                                                  // défaillance réelle
  flow.set('run', false);
  return [null, { payload: "DÉFAILLANCE du roulement à t = " + t }];
}
msg.payload = { x: x }; msg.headers = { 'Content-Type': 'application/json' };
return [msg, null];
"""
FORMAT = r"""
const r = msg.payload; if (!r || r.t === undefined) return null;
const pol = flow.get('policy') || 'q10';
const rul = Math.round(r[pol]), at = r['intervene_at_' + pol], alarm = r['alarm_' + pol];
const ts = Date.now(), r0 = v => Math.round(v * 100) / 100;
const hi  = [{ topic: 'Indicateur de santé (RMS lissé)', payload: r0(r.hi) }, { topic: 'Seuil de fin de vie', payload: r.ft }];
const rulc = [{ topic: 'RUL q10 (prudent)', payload: Math.round(r.q10) }, { topic: 'RUL médiane', payload: Math.round(r.q50) }, { topic: 'RUL q90 (optimiste)', payload: Math.min(3000, Math.round(r.q90)) }];
const status = alarm ? "⚠ INTERVENIR MAINTENANT (règle " + pol + ")"
  : "Intervention conseillée à t = " + at + "  (dans " + (at - r.t) + " pas, règle " + pol + ", délai logistique " + r.delta + ")";
const info = "t = " + r.t + " | " + (r.mode === 'model' ? "modèle ajusté sur la dégradation" : "pas de dégradation détectée → prior de population") +
             " | RUL " + Math.round(r.q10) + " – " + Math.round(r.q50) + " – " + Math.round(Math.min(r.q90, 9999)) + " (q10 – méd. – q90)";
return [hi, rulc, { payload: rul }, { payload: status }, { payload: info }];
"""
PARAMS = r"""
if (msg.topic === 'reset') {
  ['t', 'e'].forEach(k => flow.set(k, 0)); flow.set('run', false);
  return [ { payload: [] }, { payload: {} }, { payload: "Réinitialisé" } ];
}
flow.set(msg.topic, msg.payload); return null;
"""
INIT = r"""
flow.set('life', 1500); flow.set('noise', 1); flow.set('drift', false); flow.set('run', false);
flow.set('policy', 'q10'); flow.set('t', 0); flow.set('e', 0); return null;
"""

def fn(id, name, code, outputs, x, y, wires):
    return dict(id=id, type="function", z=TAB, name=name, func=code.strip(), outputs=outputs, noerr=0, x=x, y=y, wires=wires)
def ui(id, typ, group, order, x, y, wires=None, **kw):
    d = dict(id=id, type=typ, z=TAB, group=group, order=order, x=x, y=y, wires=wires if wires is not None else [])
    d.update(kw); return d
def chart(id, group, order, label, ymin, ymax, x, y):
    return ui(id, "ui_chart", group, order, x, y, [[]], name="", label=label, width="12", height="6", chartType="line", legend="true",
              xformat="HH:mm:ss", interpolate="linear", nodata="en attente…", dot=False, ymin=str(ymin), ymax=str(ymax),
              removeOlder=1, removeOlderPoints="1500", removeOlderUnit="3600", cutout=0, useOneColor=False, useUTC=False,
              colors=["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"], outputs=1, className="")
ctrl = lambda id, typ, order, y, topic, **kw: ui(id, typ, G1, order, 210, y, [["setp"]], name="", topic=topic, topicType="str", **kw)

nodes = [
 dict(id=UITAB, type="ui_tab", name="Pronostic RUL – roulement", icon="dashboard", order=1, disabled=False, hidden=False),
 dict(id=G1, type="ui_group", name="1. Capteur simulé & défauts", tab=UITAB, order=1, disp=True, width="6", collapse=False),
 dict(id=G2, type="ui_group", name="2. Indicateur de santé et RUL", tab=UITAB, order=2, disp=True, width="12", collapse=False),
 dict(id=G3, type="ui_group", name="3. Décision de maintenance", tab=UITAB, order=3, disp=True, width="6", collapse=False),
 dict(id=TAB, type="tab", label="Maintenance 4.0 – RUL roulement", disabled=False, info="Capteur simulé -> API Python /sample -> dashboard"),
 dict(id="init", type="inject", z=TAB, name="init", props=[{"p":"payload"}], repeat="", crontab="", once=True, onceDelay="0.3", topic="", payload="", payloadType="date", x=110, y=40, wires=[["fninit"]]),
 fn("fninit", "valeurs par défaut", INIT, 1, 300, 40, [[]]),
 dict(id="tick", type="inject", z=TAB, name="horloge capteur (0,1 s)", props=[{"p":"payload"}], repeat="0.1", crontab="", once=False, topic="", payload="", payloadType="date", x=150, y=120, wires=[["sensor"]]),
 fn("sensor", "Capteur simulé (RMS)", SENSOR, 2, 380, 120, [["http"], ["status"]]),
 dict(id="http", type="http request", z=TAB, name="API Python /sample", method="POST", ret="obj", paytoqs="ignore", url="http://localhost:5000/sample", tls="", persist=False, proxy="", insecureHTTPParser=False, authType="", senderr=False, headers=[], x=640, y=120, wires=[["fmt"]]),
 fn("fmt", "Mise en forme + décision", FORMAT, 5, 880, 120, [["c1"], ["c2"], ["gauge"], ["status"], ["info"]]),
 chart("c1", G2, 1, "Indicateur de santé vs seuil de fin de vie (g)", 0, 10, 1130, 60),
 chart("c2", G2, 2, "RUL estimée : médiane et bande d'incertitude (pas de 10 s)", 0, 3000, 1130, 120),
 ui("gauge", "ui_gauge", G3, 1, 1130, 180, name="", gtype="gage", title="RUL selon la règle choisie", label="pas", format="{{value}}", min=0, max=3000, colors=["#d62728", "#ff7f0e", "#2ca02c"], seg1="120", seg2="500", diff=False, className=""),
 ui("status", "ui_text", G3, 2, 1130, 240, name="", label="Décision", format="{{msg.payload}}", layout="col-center"),
 ui("info", "ui_text", G3, 3, 1130, 300, name="", label="État du pronostic", format="{{msg.payload}}", layout="col-center"),
 # contrôles
 ctrl("sw_run", "ui_switch", 1, 200, "run", label="Marche / arrêt du capteur", tooltip="", passthru=True, decouple="false", style="", onvalue="true", onvalueType="bool", offvalue="false", offvalueType="bool", animate=False),
 ctrl("sl_life", "ui_slider", 2, 260, "life", label="Durée de vie réelle (pas)", tooltip="", min=800, max=2800, step=50, passthru=True, outs="end"),
 ctrl("sl_noise", "ui_slider", 3, 320, "noise", label="Niveau de bruit capteur", tooltip="", min=0.5, max=3, step=0.1, passthru=True, outs="end"),
 ctrl("sw_drift", "ui_switch", 4, 380, "drift", label="Défaut capteur : dérive", tooltip="", passthru=True, decouple="false", style="", onvalue="true", onvalueType="bool", offvalue="false", offvalueType="bool", animate=False),
 ctrl("dd_pol", "ui_dropdown", 5, 440, "policy", label="Règle de décision", tooltip="", place="", passthru=True, multiple=False, options=[
     {"label": "Médiane (q50) – risqué", "value": "q50", "type": "str"}, {"label": "q20", "value": "q20", "type": "str"},
     {"label": "q10 – prudent", "value": "q10", "type": "str"}, {"label": "q05 – très prudent", "value": "q05", "type": "str"}]),
 ui("btn_reset", "ui_button", G1, 6, 210, 500, [["setp"]], name="", label="Réinitialiser", tooltip="", color="", bgcolor="", icon="", payload="", payloadType="str", topic="reset", topicType="str", className=""),
 fn("setp", "paramètres / reset", PARAMS, 3, 480, 320, [["c1r", "c2r"], ["c1r", "c2r"], ["status"]]),
 dict(id="c1r", type="function", z=TAB, name="vider graphiques", func="msg.payload=[]; return msg;", outputs=1, noerr=0, x=700, y=420, wires=[["c1","c2","rst"]]),
 dict(id="c2r", type="function", z=TAB, name="(inutile)", func="return null;", outputs=1, noerr=0, x=700, y=470, wires=[[]]),
 dict(id="rst", type="http request", z=TAB, name="API /reset", method="POST", ret="txt", paytoqs="ignore", url="http://localhost:5000/reset", tls="", persist=False, proxy="", authType="", senderr=False, headers=[], x=900, y=420, wires=[[]]),
]
# simplification : un seul chemin de reset (setp sortie 1 -> c1r -> c1,c2,rst)
for n in nodes:
    if n["id"] == "setp": n["outputs"] = 2; n["wires"] = [["c1r"], ["status"]]
    if n["id"] == "setp": n["func"] = n["func"].replace("return [ { payload: [] }, { payload: {} }, { payload: \"Réinitialisé\" } ];", "return [ { payload: [] }, { payload: \"Réinitialisé\" } ];")
nodes = [n for n in nodes if n["id"] != "c2r"]
json.dump(nodes, open("nodered_flow.json", "w"), ensure_ascii=False, indent=1)
print(len(nodes), "noeuds")
