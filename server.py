"""
server.py
─────────────────────────────────────────────────────────────────────────────
FastAPI Inference Server untuk GO DIET App
Hanya melakukan PREDIKSI (inference) — tidak ada training di sisi server.

Cara jalankan:
    cd ML
    pip install -r requirements.txt
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Endpoint:
    POST /api/recommend   → Rekomendasi menu berdasarkan profil user
    GET  /api/health      → Cek status server
    GET  /api/goals       → Daftar tujuan diet yang tersedia
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
from pathlib import Path
from contextlib import asynccontextmanager

import io
from io import BytesIO
from PIL import Image

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

# ── Import dari modul DRL yang sudah ada ────────────────────────────────────
from drl_food_agent import (
    ActorCritic,
    MenuEnv,
    hitung_target,
    load_menu_db,
    N_MENU_CHOICES,
    MEALS_PER_DAY,
    GOALS,
    DEVICE,
)

# ── Konfigurasi Path ────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "ppo_food_agent.pt"
JSON_PATH  = BASE_DIR / "nutrition_results.json"
YOLO_MODEL_PATH = BASE_DIR / "best.pt"

# ── Global State ─────────────────────────────────────────────────────────────
model_state = {
    "model": None,
    "db": None,
    "yolo_model": None,
    "ready": False,
}


# ── Lifespan: load model sekali saat server start ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model dan database saat startup, cleanup saat shutdown."""
    print("=" * 60)
    print("  GO DIET — Inference Server Starting...")
    print("=" * 60)

    # Load database menu
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Database menu tidak ditemukan: {JSON_PATH}")
    model_state["db"] = load_menu_db(str(JSON_PATH))
    print(f"[✓] Database: {len(model_state['db'])} menu dimuat")

    # Load model terlatih
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}\n"
            "Jalankan training dulu: python drl_food_agent.py --mode train"
        )

    ckpt = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=False)
    model = ActorCritic(
        ckpt["state_size"],
        ckpt["action_size"],
        ckpt["hidden_size"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model_state["model"] = model
    model_state["ready"] = True
    print(f"[✓] Model PPO dimuat dari {MODEL_PATH}")
    print(f"[✓] Device: {DEVICE}")

    # Load YOLO Model (Optional)
    if HAS_ULTRALYTICS and YOLO_MODEL_PATH.exists():
        try:
            model_state["yolo_model"] = YOLO(str(YOLO_MODEL_PATH))
            print(f"[✓] Model YOLO dimuat dari {YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"[!] Gagal load YOLO model: {e}")
    else:
        print("[!] Model YOLO (best.pt) belum tersedia atau ultralytics belum diinstall.")

    print("=" * 60)
    print("  Server siap menerima request!")
    print("  Docs: http://localhost:8000/docs")
    print("=" * 60)

    yield  # Server berjalan

    # Cleanup
    model_state["model"] = None
    model_state["db"] = None
    model_state["yolo_model"] = None
    model_state["ready"] = False
    print("[INFO] Server shutdown.")


# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="GO DIET - AI Food Recommendation",
    description="Inference-only server. Model sudah dilatih secara lokal.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — izinkan akses dari React Native (Expo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Expo dev biasanya dari berbagai port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────
class UserProfile(BaseModel):
    """Profil pengguna untuk rekomendasi diet."""
    jk: str = Field(..., description="Jenis kelamin: 'l'/'p' atau 'male'/'female'")
    umur: int = Field(..., ge=10, le=80, description="Umur (tahun)")
    tb: float = Field(..., ge=100, le=250, description="Tinggi badan (cm)")
    bb: float = Field(..., ge=30, le=200, description="Berat badan (kg)")
    tujuan: str = Field(..., description="Tujuan diet: turun_berat / tetap_bugar / lebih_kuat / massa_otot")

class MenuRecommendation(BaseModel):
    nama_menu: str
    url: str
    kalori: float
    protein_g: float
    karbohidrat_g: float
    lemak_g: float
    serat_g: float
    bahan: str
    skor_agen: float
    dipilih_kali: int

class NutritionTarget(BaseModel):
    kalori: float
    protein_g: float
    karbohidrat_g: float
    lemak_g: float

class RecommendResponse(BaseModel):
    success: bool
    profile: dict
    target: NutritionTarget
    recommendations: list[MenuRecommendation]
    total_3_meals: NutritionTarget | None = None


# ── Fungsi Prediksi (inference only) ────────────────────────────────────────
def run_inference(profil: dict, top_k: int = 5, n_rollout: int = 200) -> tuple:
    """
    Jalankan inference dengan model terlatih.
    Tidak ada training — hanya forward pass.
    """
    model = model_state["model"]
    db    = model_state["db"]

    env    = MenuEnv(db)
    target = hitung_target(profil)
    scores: dict[str, dict] = {}

    for _ in range(n_rollout):
        state   = env.reset(profil)
        total_r = 0.0

        for _ in range(MEALS_PER_DAY):
            action, _, _ = model.get_action(state)
            menu_row     = env.current_menus.iloc[action]
            nama         = menu_row["nama_menu"]

            next_state, reward, done = env.step_env(action)
            total_r += reward

            if nama not in scores:
                scores[nama] = {
                    "total_reward": 0.0,
                    "count": 0,
                    "row": menu_row,
                }
            scores[nama]["total_reward"] += total_r
            scores[nama]["count"]        += 1

            state = next_state if not done else state

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1]["total_reward"] / max(x[1]["count"], 1),
        reverse=True,
    )

    recs = []
    for nama, data in ranked[:top_k]:
        row = data["row"]
        recs.append({
            "nama_menu"    : nama,
            "url"          : str(row.get("url", "")),
            "kalori"       : round(float(row["kalori"]), 1),
            "protein_g"    : round(float(row["protein_g"]), 1),
            "karbohidrat_g": round(float(row["karbohidrat_g"]), 1),
            "lemak_g"      : round(float(row["lemak_g"]), 1),
            "serat_g"      : round(float(row.get("serat_g", 0)), 1),
            "bahan"        : str(row.get("bahan", "")),
            "skor_agen"    : round(data["total_reward"] / data["count"], 4),
            "dipilih_kali" : data["count"],
        })

    return recs, target


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Cek apakah server dan model siap."""
    return {
        "status": "ok" if model_state["ready"] else "loading",
        "model_loaded": model_state["ready"],
        "yolo_loaded": model_state["yolo_model"] is not None,
        "menu_count": len(model_state["db"]) if model_state["db"] is not None else 0,
        "device": str(DEVICE),
    }


@app.get("/api/goals")
async def get_goals():
    """Daftar tujuan diet yang tersedia."""
    labels = {
        "turun_berat": "Menurunkan Berat Badan",
        "tetap_bugar": "Tetap Bugar",
        "lebih_kuat" : "Menjadi Lebih Kuat",
        "massa_otot" : "Meningkatkan Massa Otot",
    }
    return {
        "goals": [
            {"key": key, "label": labels[key]}
            for key in GOALS
        ]
    }


@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend_menu(profile: UserProfile):
    """
    Endpoint utama: dapatkan rekomendasi menu berdasarkan profil user.
    
    Server HANYA melakukan inference (forward pass) dengan model terlatih.
    Tidak ada training yang terjadi di endpoint ini.
    """
    if not model_state["ready"]:
        raise HTTPException(status_code=503, detail="Model belum siap. Tunggu sebentar...")

    # Validasi tujuan
    if profile.tujuan not in GOALS:
        raise HTTPException(
            status_code=400,
            detail=f"Tujuan '{profile.tujuan}' tidak valid. Pilih: {GOALS}",
        )

    # Konversi ke format profil internal
    profil = {
        "jk"    : profile.jk,
        "umur"  : profile.umur,
        "tb"    : profile.tb,
        "bb"    : profile.bb,
        "tujuan": profile.tujuan,
    }

    try:
        recs, target = run_inference(profil, top_k=5, n_rollout=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # Hitung total nutrisi jika ambil top 3 menu
    total_3 = None
    if len(recs) >= 3:
        total_3 = NutritionTarget(
            kalori=round(sum(r["kalori"] for r in recs[:3]), 1),
            protein_g=round(sum(r["protein_g"] for r in recs[:3]), 1),
            karbohidrat_g=round(sum(r["karbohidrat_g"] for r in recs[:3]), 1),
            lemak_g=round(sum(r["lemak_g"] for r in recs[:3]), 1),
        )

    return RecommendResponse(
        success=True,
        profile=profil,
        target=NutritionTarget(**target),
        recommendations=[MenuRecommendation(**r) for r in recs],
        total_3_meals=total_3,
    )

@app.post("/api/scan")
async def scan_food(file: UploadFile = File(...)):
    """
    Endpoint untuk mendeteksi makanan dari gambar menggunakan YOLOv8.
    """
    if model_state["yolo_model"] is None:
        raise HTTPException(status_code=503, detail="YOLO model (best.pt) belum diload.")
    
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        
        # Prediksi menggunakan YOLO
        results = model_state["yolo_model"].predict(image, conf=0.25)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = result.names[cls_id]
                
                # Koordinat bounding box (opsional jika dibutuhkan di UI)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections.append({
                    "label": label,
                    "confidence": round(conf, 2),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        return {
            "success": True,
            "detected_objects": detections,
            "message": "Deteksi berhasil" if detections else "Tidak ada objek yang terdeteksi"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat scan: {str(e)}")
