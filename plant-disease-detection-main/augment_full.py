import cv2
import os
import numpy as np

# 💾 Input / Output
INPUT_DIR = "/Users/nghiphan/Desktop/imageprocess/lemon ori"
OUTPUT_DIR = "/Users/nghiphan/Desktop/imageprocess/Lemon aug"

# ✅ Các góc xoay
ANGLES = [0, 45, 90, 135, 180, 240, 275]

# ✅ Hệ số scale
SCALES = [0.8, 1.2]

# ✅ Tịnh tiến (pixels)
TRANSLATES = [(30, 0), (-30, 0), (0, 30), (0, -30)]

# ✅ Shear
SHEARS = [0.1, -0.1]


def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)


def scale_image(img, scale):
    (h, w) = img.shape[:2]
    resized = cv2.resize(img, None, fx=scale, fy=scale)

    # resize về lại kích thước gốc
    return cv2.resize(resized, (w, h))


def translate_image(img, tx, ty):
    (h, w) = img.shape[:2]
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))


def shear_image(img, shear):
    (h, w) = img.shape[:2]
    M = np.float32([[1, shear, 0],
                    [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h))


def perspective_image(img):
    (h, w) = img.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[20, 30], [w-20, 10], [30, h-20], [w-30, h-30]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h))


def process_dataset():
    count = 0

    for root, _, files in os.walk(INPUT_DIR):
        print("📂 Đang xử lý folder:", root)

        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif")):

                input_path = os.path.join(root, file)

                # Giữ cấu trúc thư mục
                relative = os.path.relpath(root, INPUT_DIR)
                save_dir = os.path.join(OUTPUT_DIR, relative)
                os.makedirs(save_dir, exist_ok=True)

                img = cv2.imread(input_path)
                if img is None:
                    print("⚠ Không đọc được ảnh:", input_path)
                    continue

                name, ext = os.path.splitext(file)

                # ✅ 1. Rotate
                for angle in ANGLES:
                    out = rotate_image(img, angle)
                    cv2.imwrite(os.path.join(save_dir, f"{name}_rot_{angle}{ext}"), out)

                # ✅ 2. Flip
                cv2.imwrite(os.path.join(save_dir, f"{name}_flip_h{ext}"),
                            cv2.flip(img, 1))
                cv2.imwrite(os.path.join(save_dir, f"{name}_flip_v{ext}"),
                            cv2.flip(img, 0))

                # ✅ 3. Scale
                for s in SCALES:
                    out = scale_image(img, s)
                    cv2.imwrite(os.path.join(save_dir, f"{name}_scale_{s}{ext}"), out)

                # ✅ 4. Translate
                for tx, ty in TRANSLATES:
                    out = translate_image(img, tx, ty)
                    cv2.imwrite(os.path.join(save_dir, f"{name}_trans_{tx}_{ty}{ext}"), out)

                # ✅ 5. Shear
                for sh in SHEARS:
                    out = shear_image(img, sh)
                    cv2.imwrite(os.path.join(save_dir, f"{name}_shear_{sh}{ext}"), out)

                # ✅ 6. Perspective
                out = perspective_image(img)
                cv2.imwrite(os.path.join(save_dir, f"{name}_persp{ext}"), out)

                print("✔ Xong:", file)
                count += 1

    print(f"\n🎉 Hoàn tất! Tổng ảnh gốc xử lý: {count}")
    print("📁 Ảnh augment nằm tại:", OUTPUT_DIR)


process_dataset()
