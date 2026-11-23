"""Browser UI served from the standard-library http.server.

Run:  python -m soundsculpt.web <project.json> [--port 8000]
The page exposes the real tree (not an EQ cartoon): nominal bands, measured
basis support, edge coefficients, node audition, both stroke kinds, rebase
with retargeting, local subdivision and the two-tone probe measurement.
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .wavio import write_wav
from .basis import (Project, BasisSpec, SubtreeStroke,
                    UnresolvedError, band_stroke_for_node)
from .wpt import all_paths, nominal_band
from .signals import probe_pair
from . import verification as vf


class State:
    def __init__(self, project: Project):
        self.project = project
        self.lock = threading.Lock()


def _node_payload(project: Project, path: str) -> dict:
    engine = project.build_engine()
    info = engine.node_info(path, measure_support=True)
    lo, hi = info.band_hz
    return {
        "path": path or "<root>", "depth": info.depth,
        "n_coeffs": info.n_coeffs,
        "band_hz": [lo, hi],
        "decimation": info.decimation,
        "nominal_step_ms": info.nominal_step_sec * 1e3,
        "nominal_span_ms": info.nominal_span_sec * 1e3,
        "support_samples": list(info.support_samples)
        if info.support_samples else None,
        "support_ms": info.support_sec * 1e3 if info.support_sec else None,
        "edge_coeffs": list(info.edge_coeffs) if info.edge_coeffs else None,
        "edge_support_samples": (list(info.edge_support_samples)
                                 if info.edge_support_samples else None),
        "is_leaf": path in project.basis.leaves,
        "can_subdivide": info.depth < project.basis.max_level,
        "leakage": [round(float(v), 4)
                    for v in engine.leakage_profile(path)]
        if info.depth >= 1 else None,
    }


def tree_payload(project: Project) -> dict:
    engine = project.build_engine()
    nodes = []
    for depth in range(project.basis.max_level + 1):
        for p in all_paths(depth):
            info = engine.node_info(p, measure_support=False)
            lo, hi = info.band_hz
            nodes.append({"path": p or "<root>", "depth": depth,
                          "band_hz": [lo, hi],
                          "n_coeffs": info.n_coeffs,
                          "is_leaf": p in project.basis.leaves,
                          "can_subdivide": depth < project.basis.max_level})
    report = project.dry_run()
    return {
        "wavelet": project.basis.wavelet,
        "max_level": project.basis.max_level,
        "mode": project.basis.mode,
        "sr": project.sr, "n": project.original.size,
        "duration": project.original.size / project.sr,
        "leaves": project.basis.leaves,
        "nodes": nodes,
        "strokes": [s.to_dict() for s in project.strokes],
        "resolution_notes": report.notes,
        "unresolved": [{"stroke_id": u.stroke_id, "reason": u.reason,
                        "candidates": u.candidates}
                       for u in report.unresolved],
    }


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>soundsculpt</title>
<style>
body{font:13px/1.45 ui-monospace,Menlo,Consolas,monospace;margin:0;display:flex;min-height:100vh}
#left{flex:1.4;padding:14px;overflow:auto;border-right:1px solid #ccc}
#right{flex:1;padding:14px;overflow:auto;background:#fafafa}
.row{padding:2px 4px;border-radius:4px;cursor:pointer;white-space:nowrap}
.row:hover{background:#e8f0fe}.row.sel{background:#d2e3fc}.leaf{font-weight:bold}
.band{color:#666;margin-left:8px}.tag{font-size:11px;padding:0 5px;border-radius:8px;background:#eee}
.tag.edge{background:#fde293}.tag.brick{background:#f4cccc}
button{font:inherit;margin:2px;padding:3px 9px;cursor:pointer}
input,select{font:inherit;margin:2px;padding:2px}
.card{border:1px solid #ccc;border-radius:8px;padding:10px;margin:10px 0;background:#fff}
h2{font-size:14px;margin:4px 0}h3{font-size:13px;margin:8px 0 4px}
table{border-collapse:collapse}td,th{padding:2px 8px;border-bottom:1px solid #eee;text-align:right}
th{text-align:left}.warn{color:#b06000}.err{color:#b00020}.ok{color:#188038}
.bar{height:8px;background:#1a73e8;display:inline-block;vertical-align:middle}
.small{font-size:11px;color:#666}
</style></head><body>
<div id="left">
  <h2>decomposition tree (click a node)</h2>
  <div id="tree"></div>
</div>
<div id="right">
  <div class="card">
    <h2>basis</h2>
    wavelet <input id="wv" size="7"> level <input id="lv" size="2" type="number">
    mode <input id="md" size="11">
    <button onclick="rebase()">rebase</button>
    <div id="unresolved"></div>
  </div>
  <div class="card" id="nodecard"><h2>node</h2><div id="node">select a node</div></div>
  <div class="card">
    <h2>strokes</h2>
    <div id="strokes"></div>
    <h3>new stroke on selected node</h3>
    gain <input id="gain" value="0.25" size="5">
    t0 <input id="t0" value="0" size="6"> t1 <input id="t1" value="" size="6" placeholder="end">
    <br>
    <button onclick="addStroke('band')">band circle (fixed Hz band)</button>
    <button onclick="addStroke('subtree')">subtree push (follows node)</button>
  </div>
  <div class="card">
    <h2>probe tones (realised gain after re-open)</h2>
    low <input id="plow" value="300" size="6"> high <input id="phigh" value="7000" size="6">
    <button onclick="probe()">measure</button>
    <div id="probe"></div>
  </div>
  <div class="card">
    <h2>render</h2>
    <button onclick="render(true)">download edited wav</button>
    <button onclick="render(false)">round-trip wav (no edits)</button>
  </div>
</div>
<script>
let STATE=null, SEL="";
async function api(path, opts){ const r=await fetch(path,opts); return r.json(); }
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function load(){
  STATE=await api('/api/state'); SEL=SEL||''; renderTree(); renderStrokes(); renderUnresolved();
  if(SEL) showNode(SEL);
}
function renderTree(){
  const t=document.getElementById('tree'); t.innerHTML='';
  for(const n of STATE.nodes){
    const d=document.createElement('div'); d.className='row'+(n.path===SEL?' sel':'')+(n.is_leaf?' leaf':'');
    d.style.paddingLeft=(n.depth*16+4)+'px';
    d.innerHTML=esc(n.path)+' <span class="band">'+n.band_hz[0].toFixed(0)+'-'+n.band_hz[1].toFixed(0)
      +' Hz, n='+n.n_coeffs+'</span>'+(n.is_leaf?' <span class="tag">leaf</span>':'');
    d.onclick=()=>{SEL=n.path;showNode(n.path);load();};
    t.appendChild(d);
  }
}
async function showNode(p){
  SEL=p; const n=await api('/api/node?path='+encodeURIComponent(p));
  let h='<h3>'+esc(n.path)+' depth='+n.depth+'</h3><table>'
    +'<tr><th>nominal band</th><td>'+n.band_hz[0].toFixed(1)+' - '+n.band_hz[1].toFixed(1)+' Hz</td></tr>'
    +'<tr><th>coefficients</th><td>'+n.n_coeffs+' (decimation '+n.decimation+')</td></tr>'
    +'<tr><th>nominal coeff step</th><td>'+n.nominal_step_ms.toFixed(3)+' ms</td></tr>'
    +'<tr><th>measured basis support</th><td>'+(n.support_samples?n.support_samples[0]+'..'+n.support_samples[1]+' ('+n.support_ms.toFixed(2)+' ms)':'-')+'</td></tr>'
    +'<tr><th>edge coefficients</th><td>'+(n.edge_coeffs?'L'+n.edge_coeffs[0]+' / R'+n.edge_coeffs[1]:'-')+'</td></tr></table>';
  if(n.leakage){
    h+='<h3 class="warn">energy across nominal bands (not a brick wall)</h3>';
    n.leakage.forEach((v,k)=>{h+='<div class="small">k'+k+' <span class="bar" style="width:'+(v*160)+'px"></span> '+v.toFixed(3)+'</div>';});
  }
  h+='<p><button onclick="audition()">audition this node</button> '
    +(n.can_subdivide?'<button onclick="subdivide()">subdivide leaf</button>':'')+'</p>';
  document.getElementById('node').innerHTML=h;
}
function renderStrokes(){
  const el=document.getElementById('strokes'); el.innerHTML='';
  for(const s of STATE.strokes){
    const detail=s.kind==='band'?('band '+s.band_hz[0].toFixed(0)+'-'+s.band_hz[1].toFixed(0)+' Hz'):('node '+s.node_path);
    el.innerHTML+='<div>'+s.kind+' <b>'+esc(s.id)+'</b> gain='+s.gain+' window=['+s.t0+','+(s.t1??'end')+'] '+detail
      +(s.kind==='subtree'?' <button onclick="retargetPrompt(\\''+esc(s.id)+'\\')">retarget</button>':'')
      +' <button onclick="delStroke(\\''+esc(s.id)+'\\')">delete</button></div>';
  }
  el.innerHTML+='<div class="small">'+STATE.resolution_notes.map(esc).join('<br>')+'</div>';
}
function renderUnresolved(){
  const el=document.getElementById('unresolved');
  if(!STATE.unresolved.length){el.innerHTML='<span class="ok">all strokes resolved</span>';return;}
  el.innerHTML=STATE.unresolved.map(u=>'<div class="err">UNRESOLVED '+esc(u.stroke_id)+': '+esc(u.reason)
    +'<br>candidates: '+u.candidates.map(c=>'<button onclick="retarget(\\''+esc(u.stroke_id)+'\\',\\''+esc(c)+'\\')">'+esc(c)+'</button>').join(' ')+'</div>').join('');
}
async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});return r.json();}
async function rebase(){
  const r=await post('/api/rebase',{wavelet:wv.value,max_level:parseInt(lv.value),mode:md.value});
  if(r.error)alert(r.error); await load();
}
async function addStroke(kind){
  const body={kind,gain:parseFloat(gain.value),t0:parseFloat(t0.value)||0,t1:t1.value?parseFloat(t1.value):null,path:SEL};
  const r=await post('/api/stroke',body); if(r.error)alert(r.error); await load();
}
async function delStroke(id){await post('/api/stroke-delete',{id});await load();}
async function retarget(id,node){const r=await post('/api/retarget',{stroke_id:id,node_path:node});if(r.error)alert(r.error);await load();}
function retargetPrompt(id){const n=prompt('new node path, e.g. ada');if(n)retarget(id,n);}
async function subdivide(){const r=await post('/api/subdivide',{node:SEL});if(r.error)alert(r.error);await load();}
async function audition(){
  const r=await fetch('/api/audition?path='+encodeURIComponent(SEL));
  const b=await r.blob(); const a=document.createElement('audio');a.controls=true;
  a.src=URL.createObjectURL(b); document.getElementById('node').appendChild(a);
}
async function probe(){
  const r=await post('/api/probe',{low:parseFloat(plow.value),high:parseFloat(phigh.value)});
  document.getElementById('probe').innerHTML=r.error?('<span class="err">'+r.error+'</span>')
    :('<table><tr><th></th><th>linear</th><th>dB</th></tr>'
      +'<tr><th>low '+r.low_hz+' Hz</th><td>'+r.gain_low.toFixed(4)+'</td><td>'+(20*Math.log10(r.gain_low)).toFixed(2)+'</td></tr>'
      +'<tr><th>high '+r.high_hz+' Hz</th><td>'+r.gain_high.toFixed(4)+'</td><td>'+(20*Math.log10(r.gain_high)).toFixed(2)+'</td></tr></table>'
      +'<div class="small">Measured from FFT bins of the rendered probe; non-brick-wall filters show gain leaking into the other tone.</div>');
}
async function render(edited){
  const r=await fetch('/api/render?edited='+edited);
  if(!r.ok){const j=await r.json();alert(j.error);return;}
  const b=await r.blob(); const a=document.createElement('a');
  a.href=URL.createObjectURL(b); a.download=edited?'edited.wav':'roundtrip.wav'; a.click();
}
wv.value='db4';lv.value=5;md.value='symmetric';
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    state: State = None

    def log_message(self, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, default=float).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _wav(self, signal, sr):
        buf = io.BytesIO()
        write_wav(buf, signal, sr)
        data = buf.getvalue()
        self.send_response(200)
        self.send_header("content-type", "audio/wav")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(parsed.query)
        with self.state.lock:
            proj = self.state.project
            if parsed.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("content-type", "text/html")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/api/state":
                self._json(tree_payload(proj))
            elif parsed.path == "/api/node":
                path = q["path"][0].replace("<root>", "")
                self._json(_node_payload(proj, path))
            elif parsed.path == "/api/audition":
                path = q["path"][0].replace("<root>", "")
                self._wav(proj.build_engine().node_contribution(path), proj.sr)
            elif parsed.path == "/api/render":
                edited = q.get("edited", ["true"])[0] == "true"
                if not edited:
                    engine = proj.build_engine()
                    self._wav(engine.reconstruct(proj.basis.leaves), proj.sr)
                else:
                    try:
                        out, _ = proj.render()
                        self._wav(out, proj.sr)
                    except UnresolvedError as e:
                        self._json({"error": "unresolved strokes: "
                                    + ", ".join(u.stroke_id for u in e.report.unresolved)}, 409)
            else:
                self._json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        with self.state.lock:
            proj = self.state.project
            try:
                if parsed.path == "/api/rebase":
                    proj.rebase(body["wavelet"], int(body["max_level"]), body["mode"])
                    self._json(tree_payload(proj))
                elif parsed.path == "/api/stroke":
                    path = body["path"].replace("<root>", "")
                    t1 = body.get("t1")
                    if body["kind"] == "band":
                        st = band_stroke_for_node(
                            f"b{len(proj.strokes)+1}", path, proj.sr,
                            float(body["gain"]), float(body.get("t0", 0)), t1)
                    else:
                        st = SubtreeStroke(f"s{len(proj.strokes)+1}", path,
                                           float(body["gain"]),
                                           float(body.get("t0", 0)), t1,
                                           nominal_band(path, proj.sr))
                    proj.add_stroke(st)
                    self._json(tree_payload(proj))
                elif parsed.path == "/api/stroke-delete":
                    proj.strokes = [s for s in proj.strokes if s.id != body["id"]]
                    self._json(tree_payload(proj))
                elif parsed.path == "/api/retarget":
                    proj.retarget(body["stroke_id"], body["node_path"])
                    self._json(tree_payload(proj))
                elif parsed.path == "/api/subdivide":
                    proj.basis.subdivide(body["node"])
                    self._json(tree_payload(proj))
                elif parsed.path == "/api/probe":
                    tone, lo, hi = probe_pair(proj.original.size, proj.sr,
                                              float(body["low"]), float(body["high"]))
                    scratch = Project(tone, proj.sr, "probe.wav",
                                      BasisSpec(**proj.basis.to_dict()),
                                      list(proj.strokes))
                    try:
                        edited, _ = scratch.render()
                    except UnresolvedError as e:
                        self._json({"error": "unresolved: "
                                    + ", ".join(u.stroke_id
                                                for u in e.report.unresolved)}, 409)
                        return
                    self._json(vf.probe_gain_table(scratch, lo, hi))
                else:
                    self._json({"error": "not found"}, 404)
            except (ValueError, KeyError) as e:
                self._json({"error": str(e)}, 400)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    project = Project.load(args.project)
    state = State(project)
    Handler.state = state
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"serving http://127.0.0.1:{args.port}  (project {args.project})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
