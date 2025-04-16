import os

image_path = r"Anh\1.JPG"

if os.path.exists(image_path):
    print("✅ Ảnh tồn tại")
else:
    print("❌ Không tìm thấy ảnh")