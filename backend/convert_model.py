import tensorflow as tf

print("TF version:", tf.__version__)

# load model lama (.h5)
model = tf.keras.models.load_model(
    "best_shoe_model.h5",
    compile=False
)

# simpan ulang ke format SavedModel
model.save("best_shoe_model_tf215", save_format="tf")

print("✅ Model berhasil dikonversi ke SavedModel")
