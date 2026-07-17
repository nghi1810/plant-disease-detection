import cv2 as cv
import os

input_folder  = "/Users/nghiphan/Desktop/imageprocess/Rose copy aug/train/Rose___Downy Mildew"
output_folder = "/Users/nghiphan/Desktop/imageprocess/blurred/Rose___Downy Mildew"

os.makedirs(output_folder, exist_ok=True)

# Chỉ 1 mức blur
ksize = (3, 3)

count = 0
for file in os.listdir(input_folder):
    if not file.lower().endswith((".jpg", ".png", ".jpeg", ".webp", ".bmp", ".tiff")):
        continue

    path = os.path.join(input_folder, file)
    img = cv.imread(path, cv.IMREAD_UNCHANGED)  # QUAN TRỌNG!

    if img is None:
        print(f"Không đọc được {file}")
        continue

    has_alpha = img.shape[2] == 4 if len(img.shape) == 3 else False

    if has_alpha:
        bgr  = img[:, :, :3]
        alpha = img[:, :, 3]
        bgr_blur = cv.GaussianBlur(bgr, ksize, 0)
        result = cv.merge([bgr_blur[:, :, 0], bgr_blur[:, :, 1], bgr_blur[:, :, 2], alpha])
    else:
        result = cv.GaussianBlur(img, ksize, 0)

    save_path = os.path.join(output_folder, file)
    cv.imwrite(save_path, result)

    count += 1

print(f"Đã xử lý {count} ảnh giữ nguyên nền trong suốt!")
