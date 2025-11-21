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
# ️📈 PLOT TRAINING GRAPH (LOSS & ACCURACY)
# =========================================================
os.makedirs("evaluation", exist_ok=True)

# ----- Loss -----
plt.figure(figsize=(10, 6))
plt.plot(history.history["bahan_loss"], label="Train Bahan Loss")
plt.plot(history.history["kotor_loss"], label="Train Kotor Loss")
plt.plot(history.history["val_bahan_loss"], label="Val Bahan Loss")
plt.plot(history.history["val_kotor_loss"], label="Val Kotor Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("evaluation/loss_curve.png")
plt.close()

# ----- Accuracy -----
plt.figure(figsize=(10, 6))
plt.plot(history.history["bahan_accuracy"], label="Train Bahan Acc")
plt.plot(history.history["kotor_accuracy"], label="Train Kotor Acc")
plt.plot(history.history["val_bahan_accuracy"], label="Val Bahan Acc")
plt.plot(history.history["val_kotor_accuracy"], label="Val Kotor Acc")
plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("evaluation/accuracy_curve.png")
plt.close()

print("📊 Grafik training tersimpan di folder /evaluation/")
