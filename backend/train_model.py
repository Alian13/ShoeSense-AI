import os, json, numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ===========================
# CONFIG
# ===========================
BASE_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 30
MODEL_OUT = "shoe_model.h5"

# ===========================
# DATA AUGMENTATION
# ===========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.25),
    layers.GaussianNoise(12),
    layers.RandomTranslation(0.1, 0.1),
], name="data_aug")

# ===========================
# LOAD DATASET
# ===========================
image_paths, bahan_labels, kotor_labels = [], [], []

for bahan_class in os.listdir(BASE_DIR):
    bahan_path = os.path.join(BASE_DIR, bahan_class)
    if not os.path.isdir(bahan_path):
        continue
    for kotor_class in os.listdir(bahan_path):
        kotor_path = os.path.join(bahan_path, kotor_class)
        if not os.path.isdir(kotor_path):
            continue
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
# TRAIN TEST SPLIT
# ===========================
X_train, X_val, y_bahan_train, y_bahan_val, y_kotor_train, y_kotor_val = train_test_split(
    X, y_bahan, y_kotor, test_size=0.2, random_state=42, stratify=bahan_encoded
)

# ===========================
# BUILD MODEL (EfficientNetB0)
# ===========================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False

inputs = layers.Input(shape=(*IMG_SIZE, 3))

x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)

# Attention (Squeeze-and-Excitation)
se = layers.Dense(256, activation="relu")(x)
se = layers.Dense(256, activation="sigmoid")(se)
x = layers.multiply([x, se])

x = layers.Dropout(0.35)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)

out_bahan = layers.Dense(len(le_bahan.classes_), activation="softmax", name="bahan")(x)
out_kotor = layers.Dense(len(le_kotor.classes_), activation="softmax", name="kotor")(x)

model = models.Model(inputs, [out_bahan, out_kotor])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={"bahan": "categorical_crossentropy", "kotor": "categorical_crossentropy"},
    metrics={"bahan": "accuracy", "kotor": "accuracy"}
)

# ===========================
# CALLBACKS
# ===========================
checkpoint = ModelCheckpoint("best_" + MODEL_OUT, monitor="val_loss", save_best_only=True)
early = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5)

# ===========================
# TRAINING
# ===========================
history = model.fit(
    X_train,
    {"bahan": y_bahan_train, "kotor": y_kotor_train},
    validation_data=(X_val, {"bahan": y_bahan_val, "kotor": y_kotor_val}),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early, reduce_lr],
    verbose=1
)

# ===========================
# FINE TUNING EfficientNetB0
# ===========================
for layer in base_model.layers[-80:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss={"bahan": "categorical_crossentropy", "kotor": "categorical_crossentropy"},
    metrics={"bahan": "accuracy", "kotor": "accuracy"},
)

model.fit(
    X_train,
    {"bahan": y_bahan_train, "kotor": y_kotor_train},
    validation_data=(X_val, {"bahan": y_bahan_val, "kotor": y_kotor_val}),
    epochs=8,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early, reduce_lr]
)

# ===========================
# PLOT CURVE
# ===========================
def plot_curve(history, key, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history[key], label=f"train {key}")
    plt.plot(history.history[f"val_{key}"], label=f"val {key}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

plot_curve(history, "bahan_accuracy", "Akurasi Bahan", "plot_bahan_accuracy.png")
plot_curve(history, "kotor_accuracy", "Akurasi Kotor", "plot_kotor_accuracy.png")
plot_curve(history, "bahan_loss", "Loss Bahan", "plot_bahan_loss.png")
plot_curve(history, "kotor_loss", "Loss Tingkat Kotor", "plot_kotor_loss.png")

print("📈 Grafik training tersimpan.")

# ===========================
# SAVE FINAL MODEL
# ===========================
model.save(MODEL_OUT)
print("Model saved:", MODEL_OUT)