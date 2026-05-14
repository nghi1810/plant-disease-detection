# Import required libraries
import cv2
import numpy as np
import os
from pathlib import Path
import logging
from tqdm import tqdm
from rembg import remove
from PIL import Image
import io

# ======================================================
#                  SETUP
# ======================================================

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ✅ ROOT input: chứa 5 folder class
INPUT_ROOT = Path("/Users/nghiphan/Desktop/imageprocess/imageprocess_split/valid")

# ✅ ROOT output: sẽ tự tạo 5 folder tương ứng
OUTPUT_ROOT = Path("/Users/nghiphan/Desktop/imageprocess/Lemon aug reszie noback/valid")

# Tạo thư mục output nếu chưa có
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

valid_extensions = {".jpg", ".jpeg"}

# Save first preview image
first_output_image_data = None
first_original_image_path = None


# ======================================================
#          XÓA NỀN BẰNG REMBG (AI)
# ======================================================
def remove_background_with_rembg(img_bgr):

    pil_input = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    buffer = io.BytesIO()
    pil_input.save(buffer, format="PNG")
    input_bytes = buffer.getvalue()

    result_bytes = remove(input_bytes)

    pil_output = Image.open(io.BytesIO(result_bytes)).convert("RGBA")

    return cv2.cvtColor(np.array(pil_output), cv2.COLOR_RGBA2BGRA)


# ======================================================
#              XỬ LÝ MỘT ẢNH
# ======================================================
def process_image(image_path, output_dir):
    global first_output_image_data, first_original_image_path

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            logging.warning(f"Skipping unreadable: {image_path.name}")
            return

        output_rgba = remove_background_with_rembg(img)

        if first_output_image_data is None:
            first_output_image_data = output_rgba.copy()
            first_original_image_path = image_path

        output_path = output_dir / image_path.with_suffix(".png").name
        cv2.imwrite(str(output_path), output_rgba)
        logging.info(f"Saved: {output_path}")

    except Exception as e:
        logging.error(f"Error processing {image_path.name}: {e}")


# ======================================================
#          BATCH PROCESSING 5 FOLDER TRAIN
# ======================================================

if not INPUT_ROOT.exists():
    raise FileNotFoundError(f"❌ Không tìm thấy thư mục: {INPUT_ROOT}")

class_folders = [f for f in INPUT_ROOT.iterdir() if f.is_dir()]

if not class_folders:
    raise RuntimeError("❌ Không tìm thấy folder class nào trong train!")

print("\n📂 Các folder sẽ được xử lý:")
for folder in class_folders:
    print("   -", folder.name)

for class_dir in class_folders:
    print(f"\n🚀 Đang xử lý class: {class_dir.name}")

    output_class_dir = OUTPUT_ROOT / class_dir.name
    output_class_dir.mkdir(parents=True, exist_ok=True)

    image_files_to_process = [
        f for f in class_dir.iterdir()
        if f.suffix.lower() in valid_extensions and f.is_file()
    ]

    if not image_files_to_process:
        logging.warning(f"⚠ Không có ảnh JPG trong: {class_dir.name}")
        continue

    for image_path in tqdm(image_files_to_process, desc=f"Processing {class_dir.name}"):
        process_image(image_path, output_class_dir)


# ======================================================
#        HIỂN THỊ ẢNH PREVIEW ĐẦU TIÊN
# ======================================================
if first_output_image_data is not None:
    bgr = first_output_image_data[:, :, :3]
    alpha = first_output_image_data[:, :, 3]

    result_display = cv2.bitwise_and(bgr, bgr, mask=alpha)
    original_img = cv2.imread(str(first_original_image_path))

    logging.info(f"Previewing result for: {first_original_image_path.name}")

    cv2.imshow("1. Original", original_img)
    cv2.imshow("2. Rembg Result", result_display)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
