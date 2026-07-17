# ==============================
# IMPORT LIBRARIES
# ==============================

import cv2
import numpy as np
import os
from pathlib import Path
import logging
from tqdm import tqdm
from rembg import remove
from PIL import Image
import io

# ==============================
# SETUP LOGGING
# ==============================

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ==============================
# INPUT / OUTPUT PATH
# ==============================

input_dir_str = "/Users/nghiphan/Desktop/imageprocess/imageprocess_split/train/Citrus Canker"
input_dir = Path(input_dir_str)

output_dir_str = "/Users/nghiphan/Desktop/imageprocess/Lemon aug reszie noback/train/Citrus Canker"
output_dir = Path(output_dir_str)

# ==============================
# VALIDATE FOLDER
# ==============================

if not input_dir.exists() or not input_dir.is_dir():
    logging.error(f"Input folder not found at: {input_dir_str}")
    raise FileNotFoundError("Missing input folder.")

output_dir.mkdir(exist_ok=True)

# ==============================
# CONFIG
# ==============================

valid_extensions = {".jpg", ".jpeg"}
first_output_image_data = None
first_original_image_path = None

# ======================================================
#          REMOVE BACKGROUND WITH REMBG
# ======================================================

def remove_background_with_rembg(img_bgr):

    pil_input = Image.fromarray(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    )

    buffer = io.BytesIO()
    pil_input.save(buffer, format="PNG")
    input_bytes = buffer.getvalue()

    result_bytes = remove(input_bytes)

    pil_output = Image.open(io.BytesIO(result_bytes)).convert("RGBA")

    return cv2.cvtColor(
        np.array(pil_output),
        cv2.COLOR_RGBA2BGRA
    )

# ======================================================
#              PROCESS SINGLE IMAGE
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

        logging.info(f"Saved: {output_path.name}")

    except Exception as e:
        logging.error(f"Error processing {image_path.name}: {e}")

# ======================================================
#              BATCH PROCESSING
# ======================================================

image_files_to_process = [
    f for f in input_dir.iterdir()
    if f.suffix.lower() in valid_extensions and f.is_file()
]

if not image_files_to_process:
    logging.warning(f"No JPG/JPEG images in folder {input_dir.name}")

else:
    for image_path in tqdm(image_files_to_process, desc="Processing images"):
        process_image(image_path, output_dir)

    # ======================================================
    #              PREVIEW FIRST IMAGE
    # ======================================================

    if first_output_image_data is not None:
        bgr = first_output_image_data[:, :, :3]
        alpha = first_output_image_data[:, :, 3]

        result_display = cv2.bitwise_and(bgr, bgr, mask=alpha)

        original_img = cv2.imread(str(first_original_image_path))

        logging.info(f"Previewing result for: {first_original_image_path.name}")

        cv2.imshow(
            f"1. Original - {first_original_image_path.name}",
            original_img
        )

        cv2.imshow(
            f"2. Rembg Result - {first_original_image_path.name}",
            result_display
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()
