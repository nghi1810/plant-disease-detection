import os
import shutil
import random

# 📂 Thư mục chứa ảnh gốc
origin_dir = "/Users/nghiphan/Desktop/imageprocess/Lemon aug/Spider Mites"

# 📂 Thư mục lưu kết quả sau khi chia
base_output = "/Users/nghiphan/Desktop/imageprocess/imageprocess_split"
train_dir = os.path.join(base_output, "train", "Spider Mites")
valid_dir = os.path.join(base_output, "valid", "Spider Mites")

# 🔢 Tỉ lệ chia
train_ratio = 0.85  # 80% train, 20% valid

# Tạo thư mục train/valid nếu chưa có
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Lấy danh sách tất cả ảnh trong thư mục gốc
images = [f for f in os.listdir(origin_dir)
          if os.path.isfile(os.path.join(origin_dir, f))]

# Xáo trộn ngẫu nhiên
random.shuffle(images)

# Chia ảnh
train_count = int(len(images) * train_ratio)
train_files = images[:train_count]
valid_files = images[train_count:]

# Copy ảnh sang thư mục train
for file in train_files:
    shutil.copy(os.path.join(origin_dir, file), os.path.join(train_dir, file))

# Copy ảnh sang thư mục valid
for file in valid_files:
    shutil.copy(os.path.join(origin_dir, file), os.path.join(valid_dir, file))

print(f"✅ Đã chia {len(images)} ảnh:")
print(f" - Train: {len(train_files)} ảnh → {train_dir}")
print(f" - Valid: {len(valid_files)} ảnh → {valid_dir}")
print("\n🎉 Train và Valid KHÔNG trùng nhau!")
