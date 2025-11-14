# predict_debug.py
import os, json, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = 'shoe_model.h5'
model = load_model(MODEL_PATH)

with open('labels.json','r') as f:
    class_indices = json.load(f)
idx_to_label = {int(v): k for k, v in class_indices.items()}

def predict_image(path):
    IMG_SIZE = (224,224)
    img = image.load_img(path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)[0]
    for i,score in enumerate(preds):
        print(f"{i:02d} {idx_to_label[i]:25s}  {score:.4f}")
    best = int(np.argmax(preds))
    print("=> Predicted:", idx_to_label[best], " confidence:", preds[best])

if __name__ == '__main__':
    # ganti test_path ke file gambar yang ada di dataset (contoh)
    test_path = 'dataset/canvas_kotor/ck5.png'
    if os.path.exists(test_path):
        predict_image(test_path)
    else:
        print("Ganti test_path ke gambar yang ada di dataset")
