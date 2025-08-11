import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


img = tf.keras.preprocessing.image.load_img('../data/train/happy/Training_1206.jpg')

# 이미지 사이즈 출력
print(np.array(img).shape)

#  훈련, 테스트 데이터셋 만들기
## 텐서플로로 CNN모델을 설계하여 훈련

train_generator = ImageDataGenerator(rotation_range=10,  # Degree range for random rotations
                                     zoom_range=0.2,  # Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range]
                                     horizontal_flip=True,  # Randomly flip inputs horizontally
                                     rescale=1/255)  # Rescaling by 1/255 to normalize

train_dataset = train_generator.flow_from_directory(directory='../data/train',
                                                    target_size=(48, 48),  # Tuple of integers (height, width), defaults to (256, 256)
                                                    class_mode='categorical',
                                                    batch_size=16,  # Size of the batches of data (default: 32)
                                                    shuffle=True,  # Whether to shuffle the data (default: True) If set to False, sorts the data in alphanumeric order
                                                    seed=10)

# 훈련 데이터셋의 타깃 값 
print(train_dataset.classes)

# 각 타깃 값의 의미
print(train_dataset.class_indices)

# 각 타깃 값별로 데이터 갯수가 몇개인지
print(np.unique(train_dataset.classes, return_counts=True))

test_generator = ImageDataGenerator(rescale=1/255)

test_dataset = test_generator.flow_from_directory(directory='../data/test',
                                                  target_size=(48, 48),
                                                  class_mode='categorical',
                                                  batch_size=1,
                                                  shuffle=False,
                                                  seed=10)

