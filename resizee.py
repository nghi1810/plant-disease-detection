import os
from PIL import Image

# Thư mục gốc
root_dir = "/Users/nghiphan/Desktop/imageprocess/imageprocess_split"
# Duyệt tất cả thư mục con
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        # Kiểm tra đuôi file ảnh
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            file_path = os.path.join(subdir, file)
            try:
                with Image.open(file_path) as img:
                    # Resize ảnh về 256x256
                    img_resized = img.resize((256, 256))
                    # Ghi đè lên ảnh cũ (hoặc đổi tên nếu muốn giữ bản gốc)
                    img_resized.save(file_path)
                    print(f"Đã resize: {file_path}")
            except Exception as e:
                print(f"Lỗi với file {file_path}: {e}")
