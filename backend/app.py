# ====================================================
# app.py — stabil + tolerant shoe filter + logging + /logs
# ====================================================
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input, MobileNetV2, decode_predictions
)
from PIL import Image
import traceback

from ultralytics import YOLO
from PIL import Image
import uuid


app = Flask(__name__)

# longgarin CORS biar frontend Vite (5173) aman
CORS(app, resources={r"/*": {"origins": "*"}})

# === Load model utama dan prefilter sekali di awal ===
MODEL_PATH = "best_model.keras"
model = load_model(MODEL_PATH)
shoe_checker = MobileNetV2(weights="imagenet")

# === Label ===
with open("bahan_labels.json") as f:
    bahan_labels = json.load(f)
with open("kotor_labels.json") as f:
    kotor_labels = json.load(f)

idx_to_bahan = {v: k for k, v in bahan_labels.items()}
idx_to_kotor = {v: k for k, v in kotor_labels.items()}
IMG_SIZE = (224, 224)

# === Helper ===
def safe_softmax(x):
    x = np.clip(x, -500, 500)
    exp_x = np.exp(x - np.max(x))
    softmax = exp_x / np.sum(exp_x)
    return np.nan_to_num(softmax, nan=0.0, posinf=0.0, neginf=0.0)

def log_prediction(data):
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", "log_predictions.txt")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {json.dumps(data, ensure_ascii=False)}\n")

# === Rekomendasi ===
rekom_map = {
    "canvas": {
        "bersih": "Lap dengan kain lembut, hindari air berlebih.",
        "kotor": "Gunakan sabun lembut dan sikat halus, lalu jemur di tempat teduh.",
        "kotorbanget": "Rendam sebentar dengan air sabun lembut, sikat lembut seluruh permukaan."
    },
    "kulit": {
        "bersih": "Lap dengan kain kering, bisa tambahkan conditioner kulit.",
        "kotor": "Gunakan cairan pembersih khusus kulit dan lap lembut.",
        "kotorbanget": "Gunakan pembersih kulit mendalam, lalu semir setelah kering."
    },
    "suede": {
        "bersih": "Sikat lembut dengan sikat suede, hindari air.",
        "kotor": "Gunakan penghapus noda suede dan sikat lembut.",
        "kotorbanget": "Gunakan cairan pembersih suede, lalu keringkan alami."
    }
}

@app.route("/health")
def health():
    return jsonify({"ok": True})

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400

    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        # ===========================================
        # 1. AUTO-CROP YOLO SEPATU
        # ===========================================
        try:
            cropped_path = auto_crop_shoe(file_path)
        except:
            cropped_path = file_path  # fallback

        # ===========================================
        # 2. PREPROCESS GAMBAR UNTUK MODEL
        # ===========================================
        img = Image.open(cropped_path).convert("RGB").resize(IMG_SIZE)
        x = np.expand_dims(np.array(img), axis=0)
        x_pre = preprocess_input(x)

        # ===========================================
        # 3. PREFILTER (apakah gambar mengandung sepatu?)
        # ===========================================
        preds = shoe_checker.predict(x_pre)
        decoded = decode_predictions(preds, top=5)[0]
        labels = [(lbl.lower(), float(score)) for (_, lbl, score) in decoded]

        shoe_keywords = ["shoe", "sneaker", "boot", "footwear", "slipper"]
        banned_keywords = ["building", "mosque", "church", "logo", "screen", "tower"]

        shoe_conf = sum(s for lbl, s in labels if any(k in lbl for k in shoe_keywords))
        banned_conf = sum(s for lbl, s in labels if any(b in lbl for b in banned_keywords))
        avg_conf = np.mean([s for _, s in labels])

        if shoe_conf < 0.10 and (banned_conf > 0.25 or avg_conf < 0.15):
            log_prediction({
                "status": "rejected",
                "filename": file.filename,
                "detected_labels": labels
            })
            return jsonify({"error": "Gambar tidak dikenali sebagai sepatu."}), 400

        # ===========================================
        # 4. PREDIKSI MODEL UTAMA (EFFICIENTNET)
        # ===========================================
        pred_bahan, pred_kotor = model.predict(x_pre)
        pred_bahan = safe_softmax(pred_bahan[0])
        pred_kotor = safe_softmax(pred_kotor[0])

        bahan_idx = int(np.argmax(pred_bahan))
        kotor_idx = int(np.argmax(pred_kotor))

        bahan_name = idx_to_bahan.get(bahan_idx, "unknown")
        kotor_name = idx_to_kotor.get(kotor_idx, "unknown")

        bahan_conf = round(float(np.max(pred_bahan)) * 100, 2)
        kotor_conf = round(float(np.max(pred_kotor)) * 100, 2)

        rekom = rekom_map.get(bahan_name, {}).get(
            kotor_name, "Gunakan metode pembersihan lembut sesuai bahan."
        )

        result = {
            "filename": file.filename,
            "cropped_used": cropped_path != file_path,
            "bahan": bahan_name,
            "tingkat_kotor": kotor_name,
            "confidence_bahan": bahan_conf,
            "confidence_kotor": kotor_conf,
            "rekomendasi": rekom,
            "detected_labels": labels
        }

        log_prediction({"status": "success", **result})

        return jsonify({"filename": file.filename, "prediction": result})

    except Exception as e:
        traceback.print_exc()
        log_prediction({"status": "error", "filename": file.filename, "error": str(e)})
        return jsonify({"error": f"Kesalahan server: {str(e)}"}), 500


@app.route("/logs", methods=["GET"])
def get_logs():
    """Kembalikan isi log_predictions.txt dalam bentuk JSON list."""
    path = os.path.join("logs", "log_predictions.txt")
    if not os.path.exists(path):
        return jsonify([])
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # ambil bagian JSON dari tiap baris
            start = line.find("{")
            if start >= 0:
                try:
                    rows.append(json.loads(line[start:]))
                except Exception:
                    pass
    return jsonify(rows)

#YOLO FUNC
def auto_crop_shoe(input_path, output_dir="uploads/cropped"):
    os.makedirs(output_dir, exist_ok=True)

    results = yolo_model(input_path)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        # kalau tidak ketemu apa-apa, pakai gambar asli
        return input_path

    # ambil bbox dengan confidence tertinggi
    best = boxes[boxes.cls.argmax()] if len(boxes) > 1 else boxes[0]
    x1, y1, x2, y2 = best.xyxy[0].tolist()

    img = Image.open(input_path).convert("RGB")
    w, h = img.size

    # pastikan bounding box masih di dalam gambar
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    cropped = img.crop((x1, y1, x2, y2))

    out_name = f"crop_{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(output_dir, out_name)
    cropped.save(out_path, format="JPEG", quality=95)

    return out_path


if __name__ == "__main__":
    app.run(debug=True)
