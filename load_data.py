import os
import cv2
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

# Đường dẫn đến thư mục chứa dữ liệu
class_path = r"Test"  # Trỏ đến thư mục, không phải file ảnh
  # Thay bằng đường dẫn thực tế

# Danh sách các biển báo (giả sử mỗi loại biển báo có thư mục riêng)
classes = os.listdir(class_path)
num_classes = len(classes)

# Kích thước ảnh chuẩn để huấn luyện
IMG_SIZE = 64  

# Tạo danh sách lưu dữ liệu ảnh và nhãn
X, y = [], []

# Đọc ảnh và gán nhãn
# Duyệt qua từng thư mục biển báo
for label, class_name in enumerate(classes):
    image_folder = os.path.join(class_path, class_name)  # Dùng biến khác, không ghi đè class_path

    if not os.path.isdir(image_folder):  # Kiểm tra xem có đúng là thư mục không
        print(f" Cảnh báo: '{image_folder}' không phải là thư mục, bỏ qua!")
        continue

    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)

        # Kiểm tra file có phải ảnh không
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f" Bỏ qua file không phải ảnh: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:  # Nếu ảnh bị lỗi
            print(f" LỖI: Không thể đọc ảnh {img_path}, bỏ qua!")
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize về kích thước chuẩn
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển về định dạng RGB
        X.append(img)
        y.append(label)


# Chuyển danh sách thành mảng NumPy
X = np.array(X) / 255.0  # Chuẩn hóa ảnh về [0, 1]
y = np.array(y)

# Chia tập train/test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kiểm tra kích thước tập dữ liệu
print(f"Số lượng ảnh huấn luyện: {len(X_train)}")
print(f"Số lượng ảnh kiểm tra: {len(X_test)}")

import pickle

with open("dataset.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)

print(" Đã lưu dữ liệu vào 'dataset.pkl'")


# Hiển thị vài ảnh mẫu
import matplotlib.pyplot as plt  
import numpy as np  

# Số ảnh hiển thị trên mỗi hàng và cột
rows = 5  # Số hàng
cols = 5  # Số cột
num_images = len(X_train)  # Tổng số ảnh

# Chia thành nhiều figure nếu quá nhiều ảnh
for start_idx in range(0, num_images, rows * cols):
    plt.figure(figsize=(15, 10))  # Kích thước figure
    
    for i in range(rows * cols):
        idx = start_idx + i  # Chỉ số ảnh trong dataset
        if idx >= num_images:
            break  # Nếu hết ảnh, thoát vòng lặp
        
        plt.subplot(rows, cols, i + 1)
        plt.imshow(X_train[idx])  # Hiển thị ảnh
        plt.title(f"Label: {y_train[idx]}")
        plt.axis("off")
    
    plt.show()  # Hiển thị figure


# Hiển thị một số ảnh từ tập dữ liệu
# plt.figure(figsize=(10, 5))
# for i in range(10):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(X[i])  # Hiển thị ảnh
#     plt.title(f"Label: {y[i]}")  # Hiển thị nhãn
#     plt.axis("off")
# plt.show()

# # Kiểm tra số lượng ảnh đã tải
# print(f"Tổng số ảnh: {len(X)}")
# print(f"Số lượng nhãn: {len(set(y))}")  # Kiểm tra có bao nhiêu lớp biển báo
