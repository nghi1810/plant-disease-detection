from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io, base64
import socket
import qrcode
from concurrent.futures import ThreadPoolExecutor
from rembg import remove

app = Flask(__name__)

# Settings
IMG_SIZE = (256, 256)
thread_pool = ThreadPoolExecutor(max_workers=8)

# --- NEW: Auto-detect IP & Generate QR ---
def get_local_ip():
    try:
        # Connect to a public DNS to find the correct local IP (doesn't actually send data)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def generate_qr_base64(url):
    qr = qrcode.QRCode(box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

# Global variables for QR
LOCAL_IP = get_local_ip()
MOBILE_URL = f"http://{LOCAL_IP}:5001"
QR_CODE_B64 = generate_qr_base64(MOBILE_URL)

# --- LOAD MODEL ---
print("🔻 Loading unified model...")
try:
    model = load_model("model_final3_v2.keras")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

CLASSES = ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Lemon___Anthracnose', 'Lemon___Citrus Canker', 'Lemon___Healthy Leaf']

# --- HELPERS ---
def remove_background(image_pil):
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    result_bytes = remove(img_bytes)
    return Image.open(io.BytesIO(result_bytes)).convert("RGBA")

def pil_to_base64(img_pil):
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

def classify(image):
    image = image.convert("RGB")
    img = image.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, 0)
    pred = model.predict(img)
    return CLASSES[np.argmax(pred)]

# --- ROUTES ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            try:
                img = Image.open(file.stream).convert("RGB")
                
                no_bg = remove_background(img)
                img_base64 = pil_to_base64(no_bg)

                future = thread_pool.submit(classify, no_bg)
                prediction = future.result()

                return render_template("index.html", prediction=prediction, img_base64=img_base64, qr_code=QR_CODE_B64, mobile_url=MOBILE_URL)
            except Exception as e:
                print(f"Error: {e}")
                return render_template("index.html", prediction="Error processing image", qr_code=QR_CODE_B64, mobile_url=MOBILE_URL)

    return render_template("index.html", qr_code=QR_CODE_B64, mobile_url=MOBILE_URL)

if __name__=="__main__":
    print(f"\n{'='*40}")
    print(f"📱 MOBILE ACCESS QR CODE")
    print(f"Scan this to open the app on your phone:")
    print(f"URL: {MOBILE_URL}")
    print(f"{'='*40}\n")
    
    # Print QR Code to Terminal (ASCII)
    try:
        qr = qrcode.QRCode()
        qr.add_data(MOBILE_URL)
        qr.print_ascii(invert=True)
    except:
        print("(Could not print ASCII QR code in this terminal)")
    
    print(f"\n{'='*40}")
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)