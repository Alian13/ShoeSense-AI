# ===========================================================
# train_model.py — versi fix (multi-output + stabil)
# ===========================================================
import os, json, numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




# === Load dataset dari folder ===
BASE_DIR = os.path.join(os.getcwd(), "dataset")
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 35
MODEL_OUT = "shoe_model.h5"

# === Ambil data manual ===
image_paths, bahan_labels, kotor_labels = [], [], []
for bahan_class in os.listdir(BASE_DIR):
    bahan_path = os.path.join(BASE_DIR, bahan_class)
    if not os.path.isdir(bahan_path): 
        continue
    for kotor_class in os.listdir(bahan_path):
        kotor_path = os.path.join(bahan_path, kotor_class)
        for img_name in os.listdir(kotor_path):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(kotor_path, img_name))
                bahan_labels.append(bahan_class)
                kotor_labels.append(kotor_class)

print(f"🖼️ Total gambar: {len(image_paths)}")

# === Encode label ===
from sklearn.preprocessing import LabelEncoder
le_bahan = LabelEncoder()
le_kotor = LabelEncoder()
bahan_encoded = le_bahan.fit_transform(bahan_labels)
kotor_encoded = le_kotor.fit_transform(kotor_labels)

with open("bahan_labels.json", "w") as f:
    json.dump(dict(zip(le_bahan.classes_, range(len(le_bahan.classes_)))), f, indent=2)
with open("kotor_labels.json", "w") as f:
    json.dump(dict(zip(le_kotor.classes_, range(len(le_kotor.classes_)))), f, indent=2)

# === Load gambar ===
X = []
for path in image_paths:
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    X.append(tf.keras.utils.img_to_array(img))
X = preprocess_input(np.array(X))
y_bahan = tf.keras.utils.to_categorical(bahan_encoded)
y_kotor = tf.keras.utils.to_categorical(kotor_encoded)

# === Split data ===
X_train, X_val, y_bahan_train, y_bahan_val, y_kotor_train, y_kotor_val = train_test_split(
    X, y_bahan, y_kotor, test_size=0.2, random_state=42, stratify=bahan_encoded
)

# === Base model ===
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)

out_bahan = layers.Dense(len(le_bahan.classes_), activation="softmax", name="bahan")(x)
out_kotor = layers.Dense(len(le_kotor.classes_), activation="softmax", name="kotor")(x)

model = models.Model(inputs, [out_bahan, out_kotor])

# === Compile (fix multi-output) ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={"bahan": "categorical_crossentropy", "kotor": "categorical_crossentropy"},
    metrics={"bahan": "accuracy", "kotor": "accuracy"}
)

# === Callback ===
checkpoint = ModelCheckpoint("best_" + MODEL_OUT, monitor="val_loss", save_best_only=True)
early = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)

# === Training ===
history = model.fit(
    X_train,
    {"bahan": y_bahan_train, "kotor": y_kotor_train},
    validation_data=(X_val, {"bahan": y_bahan_val, "kotor": y_kotor_val}),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early, reduce_lr],
    verbose=1
)

# === Fine-tuning ===
for layer in base_model.layers[-30:]:
    layer.trainable = True

# compile ulang (fix metrics juga)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss={"bahan": "categorical_crossentropy", "kotor": "categorical_crossentropy"},
    metrics={"bahan": "accuracy", "kotor": "accuracy"}
)

print("\n🚀 Fine-tuning tahap akhir...\n")

model.fit(
    X_train,
    {"bahan": y_bahan_train, "kotor": y_kotor_train},
    validation_data=(X_val, {"bahan": y_bahan_val, "kotor": y_kotor_val}),
    epochs=8,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early, reduce_lr]
)

model.save(MODEL_OUT)
print(f"✅ Model disimpan ke {MODEL_OUT}")

# =========================================================
# 📊 DIAGRAM BATANG – PERFORMANSI MODEL (lebih mudah dibaca)
# =========================================================

# Hitung nilai terakhir (epoch terakhir)
final_train_bahan_acc = history.history["bahan_accuracy"][-1]
final_val_bahan_acc   = history.history["val_bahan_accuracy"][-1]
final_train_kotor_acc = history.history["kotor_accuracy"][-1]
final_val_kotor_acc   = history.history["val_kotor_accuracy"][-1]

final_train_bahan_loss = history.history["bahan_loss"][-1]
final_val_bahan_loss   = history.history["val_bahan_loss"][-1]
final_train_kotor_loss = history.history["kotor_loss"][-1]
final_val_kotor_loss   = history.history["val_kotor_loss"][-1]

# -------------------------
# 🔥 Diagram batang ACCURACY
# -------------------------
labels = ["Train Bahan", "Val Bahan", "Train Kotor", "Val Kotor"]
acc_values = [
    final_train_bahan_acc,
    final_val_bahan_acc,
    final_train_kotor_acc,
    final_val_kotor_acc
]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, acc_values, color=["#4caf50", "#2196f3", "#ff9800", "#9c27b0"])
plt.title("Final Accuracy Per Output (Bahan & Kotor)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

# Tampilkan nilai di atas batang
for bar, val in zip(bars, acc_values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha='center')

plt.savefig("evaluation/bar_accuracy.png")
plt.close()

# -------------------------
# 🔥 Diagram batang LOSS
# -------------------------
loss_values = [
    final_train_bahan_loss,
    final_val_bahan_loss,
    final_train_kotor_loss,
    final_val_kotor_loss
]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, loss_values, color=["#e91e63", "#3f51b5", "#ff5722", "#8bc34a"])
plt.title("Final Loss Per Output (Bahan & Kotor)")
plt.ylabel("Loss")

for bar, val in zip(bars, loss_values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha='center')

plt.savefig("evaluation/bar_loss.png")
plt.close()

# =========================================================
# 📉 ERROR RATE & STATISTIK TAMBAHAN
# =========================================================
error_stats = {
    "bahan_train_error": round(1 - final_train_bahan_acc, 4),
    "bahan_val_error": round(1 - final_val_bahan_acc, 4),
    "kotor_train_error": round(1 - final_train_kotor_acc, 4),
    "kotor_val_error": round(1 - final_val_kotor_acc, 4),
    "total_images": len(image_paths),
    "train_images": len(X_train),
    "val_images": len(X_val),
}

# Simpan statistik sebagai JSON
with open("evaluation/error_stats.json", "w") as f:
    json.dump(error_stats, f, indent=2)

# =========================================================
# 📚 STATISTIK PER-KELAS (Canvas / Kulit / Suede)
# =========================================================
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

os.makedirs("evaluation", exist_ok=True)

# =============================
# 🔍 Prediksi ke validation set
# =============================
pred_bahan_val, pred_kotor_val = model.predict(X_val)
y_true = np.argmax(y_bahan_val, axis=1)
y_pred = np.argmax(pred_bahan_val, axis=1)
class_names = le_bahan.classes_

# ==========================================
# 1️⃣ CONFUSION MATRIX (PNG)
# ==========================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Klasifikasi Bahan")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("evaluation/confusion_matrix_bahan.png")
plt.close()

# ==========================================
# 2️⃣ ACCURACY PER KELAS (Bar Chart)
# ==========================================
class_accuracy = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(8, 6))
bars = plt.bar(class_names, class_accuracy, color=["#4caf50", "#2196f3", "#ff5722"])
plt.title("Accuracy Per Kelas (Bahan)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

for bar, acc in zip(bars, class_accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, acc + 0.02, f"{acc:.2f}", ha='center')

plt.savefig("evaluation/class_accuracy_bahan.png")
plt.close()

# ==========================================
# 3️⃣ DETAIL PRECISION / RECALL / F1
# ==========================================
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

precision = [report[c]["precision"] for c in class_names]
recall    = [report[c]["recall"] for c in class_names]
f1        = [report[c]["f1-score"] for c in class_names]

x = np.arange(len(class_names))
w = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - w, precision, width=w, label="Precision", color="#2196f3")
plt.bar(x, recall, width=w, label="Recall", color="#4caf50")
plt.bar(x + w, f1, width=w, label="F1-score", color="#ff9800")

plt.xticks(x, class_names)
plt.title("Precision / Recall / F1-score Per Kelas")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()

plt.savefig("evaluation/class_precision_recall_f1_bahan.png")
plt.close()

# ==========================================
# 4️⃣ MISCLASSIFICATION (Grafik Kesalahan)
# ==========================================
mis = cm.copy()
np.fill_diagonal(mis, 0)   # nolkan yang benar

plt.figure(figsize=(8, 6))
sns.heatmap(mis, annot=True, fmt="d", cmap="Reds",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Jumlah Salah Prediksi Antar Kelas")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("evaluation/class_misclassification_bahan.png")
plt.close()

# ==========================================
# 5️⃣ DISTRIBUSI DATASET PER KELAS (Bar Chart)
# ==========================================
unique, counts = np.unique(bahan_encoded, return_counts=True)

plt.figure(figsize=(8, 6))
bars = plt.bar(class_names, counts, color=["#673ab7", "#009688", "#795548"])
plt.title("Distribusi Dataset Per Kelas (Bahan)")
plt.ylabel("Jumlah Gambar")

for bar, val in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, val + 5, str(val), ha='center')

plt.savefig("evaluation/dataset_distribution.png")
plt.close()

print("📊 Statistik lengkap berhasil dibuat:")
print(" - evaluation/confusion_matrix_bahan.png")
print(" - evaluation/class_accuracy_bahan.png")
print(" - evaluation/class_precision_recall_f1_bahan.png")
print(" - evaluation/class_misclassification_bahan.png")
print(" - evaluation/dataset_distribution.png")

print("\n📌 Statistik kesalahan disimpan di evaluation/error_stats.json")
print("📊 Grafik batang tersimpan di evaluation/bar_accuracy.png dan bar_loss.png")