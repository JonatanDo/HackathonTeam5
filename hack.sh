#!/usr/bin/env bash
set -euo pipefail


# --- project layout ---
APP_DIR="$HOME/rag"
mkdir -p "$APP_DIR"/app/{templates,static} "$APP_DIR"/{qdrant_data,caddy_data,caddy_config}
cd "$APP_DIR"


# --- Python deps ---
cat > requirements.txt <<'REQ'
fastapi==0.115.4
uvicorn[standard]==0.32.0
httpx==0.27.2
qdrant-client==1.12.1
pypdf==5.0.1
python-docx==1.1.2
pandas==2.2.2
jinja2==3.1.4
REQ


# --- FastAPI application ---
cat > app/main.py <<'PY'
import os, io, uuid, json, re, traceback, pandas as pd
from typing import List
from fastapi import FastAPI, UploadFile, Request, HTTPException, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader
from docx import Document as Docx


# --- Config ---
QDRANT_URL = os.getenv("QDRANT_URL","http://localhost:6333")
N2YO_KEY   = os.getenv("N2YO_API_KEY","")


OPENAI_KEY  = os.getenv("OPENAI_API_KEY","")
OPENAI_CHAT = os.getenv("OPENAI_CHAT_MODEL","gpt-4o-mini")
OPENAI_EMB  = os.getenv("OPENAI_EMBED_MODEL","text-embedding-3-small")


MAX_MB = int(os.getenv("MAX_UPLOAD_MB","50"))


# --- App wiring ---
app = FastAPI(title="RAG + N2YO (OpenAI-only)")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
qdr = QdrantClient(url=QDRANT_URL, timeout=30)


def ensure_collections():
    for name, size in (("docs", 1536), ("sat_tracks", 32)):
        try:
            qdr.get_collection(name)
        except Exception:
            try:
                qdr.recreate_collection(name, vectors_config=VectorParams(size=size, distance=Distance.COSINE))
            except Exception:
                pass
ensure_collections()


# --- OpenAI helpers ---
async def embed_texts(texts: List[str]):
    if not OPENAI_KEY: raise HTTPException(status_code=400, detail="OpenAI key not configured.")
    async with httpx.AsyncClient(timeout=45) as cx:
        r = await cx.post("https://api.openai.com/v1/embeddings",
                          headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                          json={"model": OPENAI_EMB, "input": texts})
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"OpenAI embeddings error: {r.text}")
        return [d["embedding"] for d in r.json()["data"]]


def messages_to_prompt(messages: List[dict]) -> str:
    return "\n".join(f"[{m.get('role','user')}] {m.get('content','')}" for m in messages)


async def chat_complete(messages: List[dict]):
    if not OPENAI_KEY: raise HTTPException(status_code=400, detail="OpenAI key not configured.")
    async with httpx.AsyncClient(timeout=60) as cx:
        # Try Chat Completions
        r = await cx.post("https://api.openai.com/v1/chat/completions",
                          headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                          json={"model": OPENAI_CHAT, "messages": messages, "temperature": 0.1})
        if r.status_code < 400:
            j = r.json()
            return j["choices"][0]["message"]["content"]
        # Fallback: Responses API
        prompt = messages_to_prompt(messages)
        r2 = await cx.post("https://api.openai.com/v1/responses",
                           headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                           json={"model": OPENAI_CHAT, "input": prompt, "temperature": 0.1})
        if r2.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"OpenAI chat error: {r.text} | fallback: {r2.text}")
        j2 = r2.json()
        if "output_text" in j2: return j2["output_text"]
        if "output" in j2 and isinstance(j2["output"], list):
            return "".join(str(p.get("content","") if isinstance(p,dict) else p) for p in j2["output"])
        return str(j2)


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="No file uploaded or file is empty.")
    if len(data) > MAX_MB*1024*1024:
        return {"ok": False, "msg": f"File exceeds {MAX_MB} MB"}


    name = (file.filename or "upload").lower()
    text = ""
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    elif name.endswith(".docx"):
        doc = Docx(io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs)
    elif name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data))
        text = df.to_csv(index=False)
    else:
        text = data.decode("utf-8", errors="ignore")


    chunks = [text[i:i+1500] for i in range(0, len(text), 1500) if text[i:i+1500].strip()]
    if not chunks:
        return {"ok": False, "msg": "No text extracted from the file."}


    vecs = await embed_texts(chunks)
    pts = [PointStruct(id=str(uuid.uuid4()), vector=v, payload={"filename": name, "text": c})
           for v,c in zip(vecs, chunks)]
    qdr.upsert("docs", points=pts, wait=True)
    return {"ok": True, "chunks": len(pts)}


class ChatReq(BaseModel):
    query: str


@app.post("/chat")
async def rag_chat(req: ChatReq):
    try:
        qv = (await embed_texts([req.query]))[0]
        hits = qdr.search("docs", query_vector=qv, limit=8, with_payload=True)
        ctx = "\n\n".join(h.payload.get("text","") for h in hits)
        messages = [
            {"role":"system","content":"Use only the provided context. Cite filenames. If unsure, say you cannot verify."},
            {"role":"user","content": f"Context:\n<<<\n{ctx}\n>>>\n\nQuestion: {req.query}"}
        ]
        ans = await chat_complete(messages)
        return {"answer": ans, "used": [{"id": h.id, "file": h.payload.get("filename")} for h in hits]}
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Unhandled: {e}"})


# --- N2YO (parametric kept for API), UI uses NLQ ---
@app.get("/n2yo/track")
async def n2yo_track(norad_id: int, obs_lat: float, obs_lng: float, obs_alt: float=0, seconds: int=60):
    if not N2YO_KEY:
        raise HTTPException(status_code=400, detail="Missing N2YO_API_KEY")
    url = f"https://api.n2yo.com/rest/v1/satellite/positions/{norad_id}/{obs_lat}/{obs_lng}/{obs_alt}/{seconds}/&apiKey={N2YO_KEY}"
    async with httpx.AsyncClient(timeout=30) as cx:
        r = await cx.get(url)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"N2YO error: {r.text}")
        data = r.json()
    pid = str(uuid.uuid4())
    qdr.upsert("sat_tracks", [PointStruct(id=pid, vector=[0.0]*32, payload={"norad": norad_id, "track": data})], wait=True)
    return {"ok": True, "id": pid, "positions": len(data.get("positions", []))}


class N2YONLQReq(BaseModel):
    query: str


def extract_json_block(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m: return m.group(1)
    m = re.search(r"(\{.*\})", text, re.S)
    return m.group(1) if m else ""


@app.post("/n2yo/nlq")
async def n2yo_nlq(req: N2YONLQReq):
    if not N2YO_KEY:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Missing N2YO_API_KEY"})
    sys_prompt = (
        "You convert natural-language satellite tracking requests into a strict JSON object. "
        "Return ONLY one single-line JSON object with EXACT keys: "
        "{norad_id:int, obs_lat:float, obs_lng:float, obs_alt:float, seconds:int}. "
        "Rules: "
        "• Satellite may be given by common name or NORAD ID. If 'ISS' is mentioned, use norad_id 25544. "
        "• Location may be a city name or coordinates. If a city is used, choose reasonable lat/lng for that city "
        "(e.g., Stockholm 59.33, 18.06; New York City 40.7128, -74.0060; Los Angeles 34.05, -118.24; "
        "London 51.5074, -0.1278; Paris 48.8566, 2.3522; Tokyo 35.6762, 139.6503). "
        "• If coordinates include N/S/E/W, convert to signed decimal: N/E positive, S/W negative. "
        "• obs_alt is observer altitude in meters; default 0 if absent. Accept 'm' or 'km' and convert to meters. "
        "• seconds is tracking duration; accept '120s', '2 min', '2 minutes' and convert minutes to seconds. "
        "Clamp seconds to the range 1..3600. "
        "• If something is missing, infer sensible defaults (seconds=60, obs_alt=0). "
        "• Output JUST the JSON. No prose, no code fences, no extra keys, no trailing text. "
        "Examples: "
        "Q: Track the ISS from Stockholm for 120 seconds -> "
        '{"norad_id":25544,"obs_lat":59.33,"obs_lng":18.06,"obs_alt":0,"seconds":120} '
        "Q: Track 43013 from 40.71N 74.01W for 90s -> "
        '{"norad_id":43013,"obs_lat":40.71,"obs_lng":-74.01,"obs_alt":0,"seconds":90} '
        "Q: Track ISS over 34.05, -118.24 at 250 m for 2 min -> "
        '{"norad_id":25544,"obs_lat":34.05,"obs_lng":-118.24,"obs_alt":250,"seconds":120}'
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": req.query}
    ]
    raw = await chat_complete(messages)
    js = extract_json_block(raw)
    try:
        parsed = json.loads(js)
    except Exception:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Could not parse parameters", "raw": raw})


    for k in ["norad_id","obs_lat","obs_lng"]:
        if k not in parsed:
            return JSONResponse(status_code=400, content={"ok": False, "error": f"Missing fields: {k}", "parsed": parsed})


    norad_id = int(parsed["norad_id"])
    obs_lat  = float(parsed["obs_lat"])
    obs_lng  = float(parsed["obs_lng"])
    obs_alt  = float(parsed.get("obs_alt", 0))
    seconds  = int(parsed.get("seconds", 60))
    if seconds <= 0: seconds = 60
    if seconds > 3600: seconds = 3600


    url = f"https://api.n2yo.com/rest/v1/satellite/positions/{norad_id}/{obs_lat}/{obs_lng}/{obs_alt}/{seconds}/&apiKey={N2YO_KEY}"
    async with httpx.AsyncClient(timeout=30) as cx:
        r = await cx.get(url)
        if r.status_code >= 400:
            return JSONResponse(status_code=502, content={"ok": False, "error": f"N2YO error: {r.text}", "parsed": parsed})
        data = r.json()


    pid = str(uuid.uuid4())
    qdr.upsert("sat_tracks", [PointStruct(id=pid, vector=[0.0]*32, payload={
        "norad": norad_id, "track": data, "nlq": req.query, "parsed": parsed
    })], wait=True)
    return {"ok": True, "id": pid, "positions": len(data.get("positions", [])), "parsed": parsed}
PY


# --- HTML template (chat UI + N2YO card) ---
cat > app/templates/index.html <<'HTML'
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>RAG Hackathon</title>
  <link rel="stylesheet" href="/static/style.css">
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
<h1>RAG Hackathon Team</h1>


<section>
  <h3>Upload (PDF, DOCX, CSV)</h3>
  <form id="upForm"><input type="file" id="file" required />
    <button type="submit">Upload and index</button></form>
  <pre id="upOut" class="mono"></pre>
</section>


<section class="chat-card">
  <h3>Chat</h3>
  <div id="chatWindow" class="chat-window"></div>


  <form id="chatForm" class="composer">
    <textarea id="q" rows="2" placeholder="Type your question. Enter to send, Shift+Enter for newline."></textarea>
    <div class="composer-actions">
      <button id="sendBtn" type="submit">Send</button>
      <button id="clearBtn" type="button" class="ghost">Clear</button>
    </div>
  </form>
</section>


<section>
  <h3>N2YO Natural Language</h3>
  <textarea id="nlq" rows="3" style="width:100%;" placeholder="e.g., Track the ISS from Stockholm for 120 seconds"></textarea><br/>
  <button id="nlqBtn">Interpret & Query</button>
  <div id="nlqCard"></div>
</section>


<script>
const chatEl = document.getElementById('chatWindow');
const qEl    = document.getElementById('q');
const sendBtn= document.getElementById('sendBtn');


function esc(s){
  return String(s)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;')
    .replace(/'/g,'&#39;');
}
function toHtml(text){
  const safe = esc(text);
  return safe.split(/\n{2,}/).map(p=>'<p>'+p.replace(/\n/g,'<br>')+'</p>').join('');
}
function addMsg(role, html, refs=[]) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + (role === 'user' ? 'right' : 'left');
  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + (role === 'user' ? 'user' : 'assistant');
  bubble.innerHTML = html;
  wrap.appendChild(bubble);


  if (role === 'assistant' && refs.length) {
    const cite = document.createElement('div');
    cite.className = 'refs';
    cite.textContent = 'sources: ' + refs.map(r => r.file || r.id).join(', ');
    wrap.appendChild(cite);
  }


  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
}


function setSending(on){
  sendBtn.disabled = on;
  qEl.disabled = on;
  if (on) sendBtn.dataset.original = sendBtn.textContent, sendBtn.textContent = 'Sending…';
  else sendBtn.textContent = sendBtn.dataset.original || 'Send';
}


// Upload
(document.getElementById('upForm')).addEventListener('submit', async (e)=>{
  e.preventDefault();
  const fd=new FormData();
  fd.append('file',document.getElementById('file').files[0]);
  const r=await fetch('/upload',{method:'POST',body:fd});
  const t=await r.text();
  document.getElementById('upOut').textContent = r.ok ? t : `Upload error ${r.status}: ${t}`;
});


// Chat
(document.getElementById('chatForm')).addEventListener('submit', async (e)=>{
  e.preventDefault();
  const q = qEl.value.trim();
  if (!q) return;


  addMsg('user', toHtml(q));
  qEl.value = '';
  const id = 'spin-' + Date.now();
  addMsg('assistant', `<div id="${id}" class="spinner">Thinking…</div>`);


  setSending(true);
  try {
    const r = await fetch('/chat', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({query:q}) });
    const respText = await r.text();
    if (r.ok) {
      const j = JSON.parse(respText);
      const spin = document.getElementById(id);
      if (spin) spin.parentElement.parentElement.remove();
      addMsg('assistant', toHtml(j.answer || ''), j.used || []);
    } else {
      const j = JSON.parse(respText);
      const spin = document.getElementById(id);
      if (spin) spin.parentElement.parentElement.remove();
      addMsg('assistant', `<span class="error">Chat error ${r.status}: ${esc(j.error || respText)}</span>`);
    }
  } catch (err) {
    const spin = document.getElementById(id);
    if (spin) spin.parentElement.parentElement.remove();
    addMsg('assistant', `<span class="error">Network or parse error: ${esc(String(err))}</span>`);
  } finally {
    setSending(false);
    qEl.focus();
  }
});


(document.getElementById('clearBtn')).addEventListener('click', ()=>{
  chatEl.innerHTML = '';
  qEl.focus();
});


// NLQ summary card
function fmt(num, digits=2){
  if (num === undefined || num === null || Number.isNaN(num)) return '—';
  return Number(num).toFixed(digits);
}
function renderN2YoCard(j){
  const card = document.getElementById('nlqCard');
  if (!j || j.ok !== true){
    const msg = j && j.error ? esc(j.error) : 'Unknown error';
    card.innerHTML = `<div class="card error"><div class="card-title">N2YO request failed</div><div class="note">${msg}</div></div>`;
    return;
  }
  const p = j.parsed || {};
  const pos = (typeof j.positions === 'number') ? j.positions : '—';
  const lat = p.obs_lat, lng = p.obs_lng, altm = p.obs_alt || 0, secs = p.seconds || 60;
  const norad = p.norad_id;


  card.innerHTML = `
  <div class="card success">
    <div class="card-title">Track stored</div>
    <div class="stats-grid">
      <div class="stat"><div class="stat-label">Satellite (NORAD)</div><div class="stat-value">${esc(norad)}</div></div>
      <div class="stat"><div class="stat-label">Duration</div><div class="stat-value">${esc(secs)} s</div></div>
      <div class="stat"><div class="stat-label">Observer</div><div class="stat-value">${fmt(lat)}, ${fmt(lng)} @ ${Math.round(altm)} m</div></div>
      <div class="stat"><div class="stat-label">Positions stored</div><div class="stat-value">${esc(pos)}</div></div>
    </div>
    <div class="note">Track ID: <code>${esc(j.id)}</code></div>
  </div>`;
}


(document.getElementById('nlqBtn')).addEventListener('click', async (e)=>{
  e.preventDefault();
  const query=document.getElementById('nlq').value.trim();
  if (!query) return;
  const card = document.getElementById('nlqCard');
  card.innerHTML = `<div class="card"><div class="card-title">Processing…</div></div>`;
  try{
    const r=await fetch('/n2yo/nlq',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query})});
    const t=await r.text();
    let j; try { j = JSON.parse(t); } catch { j = {ok:false, error:t}; }
    renderN2YoCard(j);
  }catch(err){
    renderN2YoCard({ok:false, error:String(err)});
  }
});


// Enter to send, Shift+Enter for newline
qEl.addEventListener('keydown', (e)=>{
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});
</script>
</body>
</html>
HTML


# --- CSS ---
cat > app/static/style.css <<'CSS'
:root{
  --bg:#0f1116; --card:#151823; --text:#e6e6e6; --muted:#a8b0bf; --accent:#4f8cff; --accent-2:#23c483; --error:#ff6b6b; --border:#222636;
}
*{box-sizing:border-box}
body{margin:0 auto; max-width:1000px; padding:24px 16px; background:var(--bg); color:var(--text); font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
h1{margin:0 0 20px 0; font-weight:700}
h3{margin:12px 0}
section{background:var(--card); border:1px solid var(--border); border-radius:14px; padding:16px; margin:14px 0; box-shadow:0 4px 16px rgba(0,0,0,.25)}
button{background:var(--accent); color:#fff; border:none; padding:10px 14px; border-radius:12px; cursor:pointer; font-weight:600}
button.ghost{background:transparent; color:var(--muted); border:1px solid var(--border)}
button:disabled{opacity:.6; cursor:not-allowed}
input[type="file"], input, textarea{width:100%; background:#0c0f16; color:var(--text); border:1px solid var(--border); border-radius:10px; padding:10px 12px}
textarea{resize:vertical}
pre.mono{background:#0c0f16; border:1px solid var(--border); border-radius:10px; padding:10px; overflow:auto}
.chat-card{padding-bottom:8px}
.chat-window{height:62vh; min-height:360px; max-height:72vh; overflow-y:auto; background:#0c0f16; border:1px solid var(--border); border-radius:14px; padding:14px}
.msg{display:flex; margin:10px 0}
.msg.left{justify-content:flex-start}
.msg.right{justify-content:flex-end}
.bubble{max-width:75%; padding:12px 14px; border-radius:16px; border:1px solid var(--border); white-space:pre-wrap; word-break:break-word; box-shadow:0 3px 10px rgba(0,0,0,.25)}
.bubble.user{background:#17223a}
.bubble.assistant{background:#121a2a}
.refs{font-size:12px; color:var(--muted); margin:6px 8px}
.error{color:var(--error)}
.composer{display:flex; gap:10px; align-items:flex-end; margin-top:12px}
.composer textarea{flex:1; height:64px; max-height:180px}
.composer-actions{display:flex; gap:10px}
.spinner{display:inline-block; padding-left:28px; position:relative; color:var(--muted)}
.spinner:before{content:""; position:absolute; left:0; top:3px; width:16px; height:16px; border:3px solid var(--border); border-top-color:var(--accent); border-radius:50%; animation:spin 1s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
p{margin:0 0 10px 0}
/* N2YO summary card */
.card{background:#0c0f16; border:1px solid var(--border); border-radius:14px; padding:14px; margin-top:12px}
.card.success{border-color:#1f6644}
.card.error{border-color:#7a2e2e}
.card-title{font-weight:700; margin-bottom:8px}
.stats-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; margin-top:6px}
.stat-label{font-size:12px; color:var(--muted)}
.stat-value{font-size:16px; font-weight:600}
.note{margin-top:8px; color:var(--muted); word-break:break-all}
CSS


# --- Dockerfile ---
cat > Dockerfile <<'DOCKER'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY app /app
ENV PORT=8080
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8080","--workers","2"]
DOCKER


# --- docker-compose ---
cat > docker-compose.yml <<'YML'
version: "3.9"
services:
  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - ./qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"
    healthcheck:
      test: ["CMD","wget","-qO-","http://localhost:6333/collections"]
      interval: 10s
      timeout: 3s
      retries: 10


  app:
    build: .
    environment:
      - QDRANT_URL=http://qdrant:6333
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_CHAT_MODEL=${OPENAI_CHAT_MODEL}
      - OPENAI_EMBED_MODEL=${OPENAI_EMBED_MODEL}
      - N2YO_API_KEY=${N2YO_API_KEY}
      - MAX_UPLOAD_MB=${MAX_UPLOAD_MB}
    depends_on:
      qdrant:
        condition: service_healthy


  caddy:
    image: caddy:2
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - ./caddy_data:/data
      - ./caddy_config:/config
    depends_on:
      - app
YML


# --- env file template ---
cat > .env.app <<'ENV'
# Fill these in Step 3 and 4
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
N2YO_API_KEY=
QDRANT_URL=http://qdrant:6333
MAX_UPLOAD_MB=50
ENV


# --- Basic Auth with Caddy ---
BASIC_USER=${BASIC_USER:-team06}
BASIC_PASS=${BASIC_PASS:-StrongPass123}
HASH=$(docker run --rm caddy caddy hash-password --plaintext "$BASIC_PASS")
cat > Caddyfile <<EOF
:80 {
  encode gzip
  basic_auth {
    $BASIC_USER $HASH
  }
  reverse_proxy app:8080
}
EOF


# --- Export env to compose and launch ---
set -a; . ./.env.app; set +a


echo "Building and starting containers..."
docker compose up -d --build


echo "Done. HTTP is on port 80 behind Basic Auth."
echo "Login: $BASIC_USER / $BASIC_PASS"
echo "Next: set keys in $APP_DIR/.env.app, then run: docker compose up -d --force-recreate app"