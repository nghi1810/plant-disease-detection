import os
import random

# ==== THAY ĐƯỜNG DẪN CỦA BẠN ====
root_dir = "/Users/nghiphan/Desktop/imageprocess/grape resize /valid"

# Lặp qua tất cả folder con
for class_name in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Lấy danh sách ảnh
    images = [f for f in os.listdir(class_path)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    total = len(images)
    keep_count = 120 # số ảnh muốn giữ lại

    if total <= keep_count:
        print(f"[{class_name}] Chỉ có {total} ảnh, không xoá gì.")
        continue

    # Chọn ảnh để giữ lại
    images_to_keep = set(random.sample(images, keep_count))
    images_to_delete = [img for img in images if img not in images_to_keep]

    print(f"[{class_name}] Tổng: {total} -> Xoá: {len(images_to_delete)} ảnh để còn {keep_count}")

    # Xoá ảnh
    for img in images_to_delete:
        os.remove(os.path.join(class_path, img))

print("Hoàn tất! Mỗi folder còn đúng 4.000 ảnh (hoặc ít hơn nếu tổng < 4.000).")
