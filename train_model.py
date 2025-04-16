import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Đường dẫn đến dữ liệu
train_dir = "data/train"
val_dir = "data/val"

# Kích thước ảnh
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

# Tạo ImageDataGenerator để tăng cường dữ liệu
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load dữ liệu
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=IMG_SIZE,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=IMG_SIZE,
                                                batch_size=BATCH_SIZE,
                                                class_mode='categorical')

# Số lớp
num_classes = len(train_generator.class_indices)

# Lưu danh sách nhãn vào file labels.json
labels = train_generator.class_indices  # Lấy danh sách nhãn từ train_generator
labels = {v: k for k, v in labels.items()}  # Đảo ngược key-value để tra cứu nhanh

with open("labels.json", "w") as f:
    json.dump(labels, f)

print("✅ Đã lưu danh sách nhãn vào labels.json!")

# Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
EPOCHS = 20
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Lưu mô hình
model.save("traffic_sign_model.h5")
print("✅ Mô hình đã được lưu thành công!")