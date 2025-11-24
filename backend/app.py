# ====================================================
# app.py — ShoeSense AI (6-Layer Gemini Fallback → Local Model)
# ====================================================
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, numpy as np, base64
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input, MobileNetV2, decode_predictions
)
from PIL import Image
import requests

# ====================================================
# LOAD ENV
# ====================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("Gemini Key Loaded:", GEMINI_API_KEY is not None)

# ====================================================
# MODEL LIST (6 fallback)
# ====================================================
GEMINI_MODELS = [
    "gemini-2.5-flash",         # PRIMARY
    "gemini-flash-latest",      # fallback 1
    "gemini-2.5-flash-lite",    # fallback 2
    "gemini-2.0-flash",         # fallback 3
    "gemini-2.0-flash-lite"     # fallback 4
]
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# ====================================================
# FLASK CONFIG
# ====================================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ====================================================
# LOAD LOCAL MODEL
# ====================================================
MODEL_PATH = "best_shoe_model.h5"
model = load_model(MODEL_PATH)

shoe_checker = MobileNetV2(weights="imagenet")

with open("bahan_labels.json") as f:
    bahan_labels = json.load(f)
with open("kotor_labels.json") as f:
    kotor_labels = json.load(f)

idx_to_bahan = {v: k for k, v in bahan_labels.items()}
idx_to_kotor = {v: k for k, v in kotor_labels.items()}

IMG_SIZE = (224, 224)

# ====================================================
# HELPERS
# ====================================================
def safe_softmax(x):
    x = np.clip(x, -500, 500)
    e = np.exp(x - np.max(x))
    return np.nan_to_num(e / np.sum(e))

def normalize_bahan(text):
    t = (text or "").lower()
    if "canvas" in t: return "canvas"
    if "suede" in t: return "suede"
    if "kulit" in t or "leather" in t: return "kulit"
    return "unknown"

def normalize_kotor(text):
    t = (text or "").lower()
    if "kotor banget" in t: return "kotorbanget"
    if "sangat kotor" in t: return "kotorbanget"
    if "kotor" in t: return "kotor"
    if "bersih" in t: return "bersih"
    return "kotor"

REKOM_MAP = {
    "canvas": {
        "bersih": "Lap dengan kain lembut.",
        "kotor": "Cuci dengan sabun lembut.",
        "kotorbanget": "Rendam sebentar & sikat halus."
    },
    "kulit": {
        "bersih": "Lap dengan kain kering.",
        "kotor": "Gunakan pembersih kulit.",
        "kotorbanget": "Deep cleaner + conditioner kulit."
    },
    "suede": {
        "bersih": "Sikat lembut suede.",
        "kotor": "Gunakan penghapus noda suede.",
        "kotorbanget": "Cairan khusus suede."
    }
}

def log_prediction(data):
    os.makedirs("logs", exist_ok=True)
    with open("logs/log_predictions.txt", "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

# ====================================================
# GEMINI REQUEST
# ====================================================
def send_gemini_request(model_name, img_b64, prompt):
    print(f"[DEBUG] Try Gemini model → {model_name}")

    url = f"{GEMINI_BASE_URL}/{model_name}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
            ]
        }]
    }

    try:
        res = requests.post(url, json=payload, timeout=25)
        data = res.json()

        if "error" in data:
            print(f"[DEBUG] Error @ {model_name}:", data["error"])
            return None

        txt = data["candidates"][0]["content"]["parts"][0]["text"]

        txt = txt.replace("```json", "").replace("```", "").strip()

        if not txt.startswith("{"):
            s = txt.find("{")
            e = txt.rfind("}")
            if s != -1 and e != -1:
                txt = txt[s:e+1]

        return json.loads(txt)

    except Exception as e:
        print(f"[DEBUG] Exception @ {model_name}:", e)
        return None

# ====================================================
# RUN 5 GEMINI MODELS (sekuensial)
# ====================================================
def analyze_with_gemini(path):
    if not GEMINI_API_KEY:
        print("[DEBUG] Tidak ada API Key → skip Gemini.")
        return None

    img_b64 = base64.b64encode(open(path, "rb").read()).decode()

    prompt = """
    You are a professional shoe-cleaning vision AI.
    Your ONLY job is to analyze SHOES.

    If image is not a shoe, ALWAYS return:
      "is_shoe": false

    Shoe may be:
    - bersih, kotor, sangat kotor
    - sepatu tunggal, sepasang
    - lusuh, robek, bekas pakai

    Return STRICT JSON:
    {
      "is_shoe": true/false,
      "bahan": "canvas/kulit/suede/null",
      "tingkat_kotor": "bersih/kotor/kotorbanget/null",
      "rekomendasi": "..."
    }
    """

    for model_name in GEMINI_MODELS:
        result = send_gemini_request(model_name, img_b64, prompt)
        if result:
            print(f"[DEBUG] Gemini OK → {model_name}")
            return result

    print("[DEBUG] Semua Gemini gagal → fallback ke lokal")
    return None

# ====================================================
# LOCAL SHOE CHECK
# ====================================================
def is_shoe_local(path):
    print("[DEBUG] Local shoe check via MobileNet...")

    try:
        img = Image.open(path).convert("RGB").resize((224, 224))
        x = preprocess_input(np.expand_dims(np.array(img), axis=0))

        preds = shoe_checker.predict(x)
        decoded = decode_predictions(preds, top=5)[0]

        shoe_words = ["shoe", "sandal", "boot", "sneaker", "running_shoe"]
        labels = [d[1].lower() for d in decoded]
        scores = [float(d[2]) for d in decoded]

        score = sum(s for lbl, s in zip(labels, scores) if any(w in lbl for w in shoe_words))

        return score >= 0.10

    except:
        return False

# ====================================================
# MAIN ENDPOINT
# ====================================================
@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Tidak ada file"}), 400

    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.filename)
    file.save(path)

    # ---- 1) GEMINI CHAIN ----
    gem = analyze_with_gemini(path)

    if gem is not None and gem.get("is_shoe"):
        bahan = normalize_bahan(gem.get("bahan"))
        tingkat = normalize_kotor(gem.get("tingkat_kotor"))
        rekom = gem.get("rekomendasi") or REKOM_MAP[bahan][tingkat]

        pred = {"bahan": bahan, "tingkat_kotor": tingkat, "rekomendasi": rekom}
        log_prediction({"source": "gemini", **pred})

        return jsonify({"source": "gemini", "prediction": pred})

    # ---- 2) SHOE CHECK LOCAL ----
    if not is_shoe_local(path):
        return jsonify({"error": "Gambar bukan sepatu."}), 400

    # ---- 3) LOCAL MODEL ----
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    x = preprocess_input(np.expand_dims(np.array(img), axis=0))

    pred_bahan, pred_kotor = model.predict(x)
    pb = safe_softmax(pred_bahan[0])
    pk = safe_softmax(pred_kotor[0])

    bahan = idx_to_bahan[int(np.argmax(pb))]
    tingkat = idx_to_kotor[int(np.argmax(pk))]
    rekom = REKOM_MAP[bahan][tingkat]

    pred = {
        "bahan": bahan,
        "tingkat_kotor": tingkat,
        "rekomendasi": rekom,
        "confidence_bahan": float(pb.max() * 100),
        "confidence_kotor": float(pk.max() * 100)
    }

    log_prediction({"source": "local_model", **pred})
    return jsonify({"source": "local_model", "prediction": pred})

# ====================================================
# RUN SERVER
# ====================================================
if __name__ == "__main__":
    app.run(debug=True)
