import io, numpy as np, librosa
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os

APP_TOKEN = os.getenv("PY_BEARER_TOKEN")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def estimate_adsr(y, sr):
    return 0.01, 0.32, 0.0, 0.18  # simple MVP defaults; refine later

def analyze(y, sr):
    y = librosa.to_mono(y) if y.ndim > 1 else y
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    rolloff  = float(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95).mean())
    zcr      = float(librosa.feature.zero_crossing_rate(y).mean())

    wave   = "Saw" if zcr > 0.1 and centroid > 1500 else ("Square" if zcr > 0.06 else "Sine")
    voices = 6 if rolloff > 4000 else 3
    detune = 0.12 if voices >= 6 else 0.06

    ftype, cutoff, res, drive = "LP24", float(min(12000.0, max(1500.0, rolloff))), 0.22, 0.1
    a, d, s, r = estimate_adsr(y, sr)
    chorus = 0.25 if voices >= 6 else 0.1
    reverb_mix, reverb_decay = 0.3, 1.2

    return {
        "OscA": {"wave": wave, "voices": voices, "detune": round(detune, 3), "octave": 0},
        "OscB": {"wave": "Square" if wave != "Square" else "Saw", "voices": 4, "detune": 0.08, "octave": -1, "mix": 0.6},
        "Noise": {"type": "BrightWhite", "level": 0.2},
        "Env1": {"attack": a, "decay": d, "sustain": s, "release": r},
        "Filter": {"type": ftype, "cutoff": cutoff, "res": res, "drive": drive},
        "FX": {"chorus": chorus, "reverb": {"mix": reverb_mix, "decay": reverb_decay}},
        "confidence": {"osc": 0.7, "filter": 0.65, "env": 0.6}
    }

@app.post("/analyze")
async def analyze_endpoint(request: Request, file: UploadFile = File(...)):
    if APP_TOKEN:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {APP_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        data = await file.read()
        y, sr = librosa.load(io.BytesIO(data), sr=44100, mono=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid audio: {e}")
    return analyze(y, sr)

@app.get("/healthz")
def healthz():
    return {"ok": True}
