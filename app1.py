from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io, base64
from concurrent.futures import ThreadPoolExecutor
from rembg import remove  # dùng rembg AI

app = Flask(__name__)

IMG_SIZE = (256, 256)

# ================================
# THREADPOOL
# ================================
thread_pool = ThreadPoolExecutor(max_workers=8)

# ================================
# LOAD 1 MODEL DUY NHẤT
# ================================
print("🔻 Loading unified model...")
model = load_model("/Users/nghiphan/Desktop/training/webtran2/model_grape_rose.keras")

# ================================
# LỚP DUY NHẤT (8 CLASS)
# ================================
CLASSES = [
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Rose___Black Spot",
    "Rose___Downy Mildew",
    "Rose___Healthy_Leaf",
    "Rose___Leaf Holes",
    "Rose___Rose_Rust"
]





# ================================
# BACKGROUND REMOVAL (REMBG)
# ================================
def remove_background(image_pil):
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    result_bytes = remove(img_bytes)
    return Image.open(io.BytesIO(result_bytes)).convert("RGBA")

# ================================
# BASE64
# ================================
def pil_to_base64(img_pil):
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

# ================================
# PREDICT WORKER
# ================================
def classify(image):
    # rembg → RGBA → chuyển về RGB
    image = image.convert("RGB")

    img = image.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, 0)

    pred = model.predict(img)
    return CLASSES[np.argmax(pred)]

# ================================
# ROUTE
# ================================
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            img = Image.open(file.stream).convert("RGB")

            # remove background
            no_bg = remove_background(img)
            img_base64 = pil_to_base64(no_bg)

            # async
            future = thread_pool.submit(classify, no_bg)
            prediction = future.result()

            return render_template(
                "index.html",
                prediction=prediction,
                img_base64=img_base64
            )

    return render_template("index.html")


if __name__=="__main__":
    app.run(debug=True, port=5001, threaded=True)
