import os, json, numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ===========================
# CONFIG
# ===========================
BASE_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 30
MODEL_OUT = "shoe_model.keras"

tf.random.set_seed(42)
np.random.seed(42)

# ===========================
# DATA AUGMENTATION
# ===========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.12),
    layers.RandomBrightness(0.08),
    layers.GaussianNoise(0.01),
    layers.RandomTranslation(0.05, 0.05),
], name="data_aug")

# ===========================
# LOAD DATASET
# ===========================
image_paths, bahan_labels, kotor_labels = [], [], []

for bahan_class in os.listdir(BASE_DIR):
    bahan_path = os.path.join(BASE_DIR, bahan_class)
    if not os.path.isdir(bahan_path): continue

    for kotor_class in os.listdir(bahan_path):
        kotor_path = os.path.join(bahan_path, kotor_class)
        if not os.path.isdir(kotor_path): continue

        for img in os.listdir(kotor_path):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(kotor_path, img))
                bahan_labels.append(bahan_class)
                kotor_labels.append(kotor_class)

print("Total gambar:", len(image_paths))

# ===========================
# LABEL ENCODER
# ===========================
le_bahan = LabelEncoder()
le_kotor = LabelEncoder()

bahan_encoded = le_bahan.fit_transform(bahan_labels)
kotor_encoded = le_kotor.fit_transform(kotor_labels)

with open("bahan_labels.json", "w") as f:
    json.dump({cls: i for i, cls in enumerate(le_bahan.classes_)}, f, indent=2)

with open("kotor_labels.json", "w") as f:
    json.dump({cls: i for i, cls in enumerate(le_kotor.classes_)}, f, indent=2)

# ===========================
# LOAD IMAGES
# ===========================
X = []
for path in image_paths:
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    img = tf.keras.utils.img_to_array(img)
    X.append(img)

X = preprocess_input(np.array(X))

y_bahan = tf.keras.utils.to_categorical(bahan_encoded)
y_kotor = tf.keras.utils.to_categorical(kotor_encoded)

# ===========================
# CLASS WEIGHTS (untuk BAHAN)
# ===========================
class_weight_bahan = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(bahan_encoded),
    y=bahan_encoded
)

class_weight_bahan = dict(enumerate(class_weight_bahan))

print("Class weight bahan:", class_weight_bahan)

# ===========================
# TRAIN TEST SPLIT
# ===========================
X_train, X_val, y_bahan_train, y_bahan_val, y_kotor_train, y_kotor_val = train_test_split(
    X, y_bahan, y_kotor, test_size=0.2, random_state=42, stratify=bahan_encoded
)

# ===========================
# BUILD MODEL (EfficientNetV2B0)
# ===========================
base_model = EfficientNetV2B0(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)

# Dense stabil
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)

# SE block
se = layers.Dense(64, activation="relu")(x)
se = layers.Dense(256, activation="sigmoid")(se)
x = layers.Multiply()([x, se])

x = layers.Dropout(0.30)(x)

out_bahan = layers.Dense(len(le_bahan.classes_), activation="softmax", name="bahan")(x)
out_kotor = layers.Dense(len(le_kotor.classes_), activation="softmax", name="kotor")(x)

model = models.Model(inputs, [out_bahan, out_kotor])
model.summary()

# ===========================
# COMPILE
# ===========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={"bahan": "categorical_crossentropy", "kotor": "categorical_crossentropy"},
    metrics={"bahan": "accuracy", "kotor": "accuracy"}
)

# ===========================
# CALLBACKS
# ===========================
checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_loss",
    save_best_only=True
)

# ===========================
# TRAIN (PHASE 1)
# ===========================
history = model.fit(
    X_train,
    {"bahan": y_bahan_train, "kotor": y_kotor_train},
    validation_data=(X_val, {"bahan": y_bahan_val, "kotor": y_kotor_val}),
    epochs=10,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint],
    verbose=1
)

# ===========================
# FINE-TUNING (PHASE 2)
# ===========================
for layer in base_model.layers[-60:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-6),
    loss={"bahan": "categorical_crossentropy", "kotor": "categorical_crossentropy"},
    metrics={"bahan": "accuracy", "kotor": "accuracy"}
)

history2 = model.fit(
    X_train,
    {"bahan": y_bahan_train, "kotor": y_kotor_train},
    validation_data=(X_val, {"bahan": y_bahan_val, "kotor": y_kotor_val}),
    epochs=20,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint],
    verbose=1
)

# ===========================
# EVALUASI: CONFUSION MATRIX & REPORT
# ===========================
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

os.makedirs("evaluation", exist_ok=True)

# --- Prediksi Validation ---
pred_bahan_val, pred_kotor_val = model.predict(X_val)

y_true_bahan = np.argmax(y_bahan_val, axis=1)
y_pred_bahan = np.argmax(pred_bahan_val, axis=1)

y_true_kotor = np.argmax(y_kotor_val, axis=1)
y_pred_kotor = np.argmax(pred_kotor_val, axis=1)

labels_bahan = list(le_bahan.classes_)
labels_kotor = list(le_kotor.classes_)

# ===========================
# CONFUSION MATRIX (BAHAN)
# ===========================
cm_bahan = confusion_matrix(y_true_bahan, y_pred_bahan)

plt.figure(figsize=(8,6))
sns.heatmap(cm_bahan, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_bahan, yticklabels=labels_bahan)
plt.title("Confusion Matrix - Bahan")
plt.xlabel("Prediksi")
plt.ylabel("Asli")
plt.savefig("evaluation/confusion_matrix_bahan.png")
plt.close()

# ===========================
# CONFUSION MATRIX (KOTOR)
# ===========================
cm_kotor = confusion_matrix(y_true_kotor, y_pred_kotor)

plt.figure(figsize=(8,6))
sns.heatmap(cm_kotor, annot=True, fmt="d", cmap="Oranges",
            xticklabels=labels_kotor, yticklabels=labels_kotor)
plt.title("Confusion Matrix - Kotor")
plt.xlabel("Prediksi")
plt.ylabel("Asli")
plt.savefig("evaluation/confusion_matrix_kotor.png")
plt.close()

# ===========================
# CLASSIFICATION REPORT (BAHAN)
# ===========================
report_bahan = classification_report(
    y_true_bahan, y_pred_bahan, target_names=labels_bahan, output_dict=True
)

# Convert report to heatmap
import pandas as pd
df_bahan = pd.DataFrame(report_bahan).transpose()

plt.figure(figsize=(8,6))
sns.heatmap(df_bahan.iloc[:-1, :-1], annot=True, cmap="Greens")
plt.title("Classification Report - Bahan (Precision/Recall/F1)")
plt.savefig("evaluation/classification_report_bahan.png")
plt.close()

print("📊 Evaluasi selesai! Hasil disimpan di folder evaluation/")

# ===========================
# SAVE FINAL MODEL
# ===========================
model.save("shoe_model.keras")
print("Model saved:", MODEL_OUT)
