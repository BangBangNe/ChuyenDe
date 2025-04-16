import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import load_img, img_to_array # type: ignore

# Load mô hình đã train
model = load_model("traffic_sign_model.h5")

# Load danh sách nhãn từ file labels.json
with open("labels.json", "r") as f:
    labels = json.load(f)

# Đường dẫn ảnh test
image_path = r"Anh\1.JPG"  # Thay bằng đường dẫn ảnh thực tế

# Load ảnh và chuẩn bị dữ liệu
img = load_img(image_path, target_size=(64, 64))
plt.imshow(img)  # Hiển thị ảnh gốc
img = img_to_array(img)
img = img.reshape(1, 64, 64, 3)
img = img.astype("float32") / 255.0

# Dự đoán
prediction = model.predict(img)
predicted_label = np.argmax(prediction)

# Lấy tên biển báo từ labels.json
predicted_name = labels.get(str(predicted_label), "Không rõ")

print(f"✅ Dự đoán: {predicted_name} ({predicted_label})")

# Đường dẫn ảnh dự đoán dựa theo tên biển báo
predicted_folder = f"data/train/{predicted_name}"
predicted_image_path = None

# Tìm ảnh trong thư mục tương ứng
if os.path.exists(predicted_folder) and os.listdir(predicted_folder):
    predicted_image_path = os.path.join(predicted_folder, os.listdir(predicted_folder)[0])

# Nếu không tìm thấy ảnh, báo lỗi
if not predicted_image_path:
    print(f"❌ Không tìm thấy ảnh nào cho nhãn {predicted_name}")
    exit()

# Kiểm tra và hiển thị ảnh dự đoán
predicted_img = load_img(predicted_image_path, target_size=(64, 64))
predicted_img = img_to_array(predicted_img)
predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2RGB)

# Hiển thị ảnh gốc và ảnh dự đoán
predicted_img = load_img(predicted_image_path, target_size=(64, 64))
predicted_img = img_to_array(predicted_img)
predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2RGB)
predicted_img = np.clip(predicted_img, 0, 255).astype(np.uint8)  # Giới hạn giá trị ảnh về [0,255]

# Hiển thị ảnh gốc và ảnh dự đoán
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(load_img(image_path))
plt.title("Ảnh gốc")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(predicted_img)
plt.title(f"Ảnh dự đoán: {predicted_name}")
plt.axis("off")

plt.show()
