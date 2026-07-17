import cv2
import os

# 💾 Input / Output
INPUT_DIR = "/Users/nghiphan/Desktop/imageprocess/grape resize "
OUTPUT_DIR = "/Users/nghiphan/Desktop/imageprocess/grape resize aug"


# ✅ Các góc cần xoay (đã thêm 45°, 135°, 275°)
ANGLES = [45, 90, 180, 240]


def rotate_image(img, angle):
    """Rotate ảnh theo góc tùy ý, không crop."""
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    return rotated


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

                # Tạo ảnh xoay theo các góc
                for angle in ANGLES:
                    img_rot = rotate_image(img, angle)
                    output_path = os.path.join(save_dir, f"{name}_rot_{angle}{ext}")
                    cv2.imwrite(output_path, img_rot)
                    print("✔ Tạo:", output_path)

                count += 1

    print(f"\n🎉 Xong! Tổng số ảnh gốc xử lý: {count}")
    print("📁 Ảnh augment được lưu trong:", OUTPUT_DIR)


process_dataset()
