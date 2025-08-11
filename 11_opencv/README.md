# 학습 목표 : TensorFlow를 활용해 openCV 프로젝트를 진행한다.

# TensorFlow / 

## 목차

1. Tensorflow
   - TensorFlow란?
   - Tensorflow 설치
  
2. Tensorflow 실습 (얼굴 이미지에서 감정 분류)

## 1. Tensorflow

<details>
<summary></summary>
<div markdown="1">

## **1-1. Tensorflow란?**

구글이 개발한 **오픈소스 머신러닝/딥러닝 프레임워크로, 딥러닝 신경망 모델을 만들고 학습**시키는데 많이 쓰인다.

## **1-2. Tensorflow 설치** 

_**학습시에는 vmware를 통해 우분투에서 실행함**_

**[1. vscode 설치]**

[vscode 메인 페이지](https://code.visualstudio.com/download)에서 .deb 다운로드

1. 다운로드 디렉토리로 이동

```terminla
cd Downloads/
```

2. 다운로드한 파일 실행

```terminal
sudo dpkg -i code_1.103.0-1754517494_amd64.deb
```

3. 다운로드 완

<br><br>

**[2. 윈도우 SSH키를 VMware Ubuntu로 복사하기]**

1. 윈도우에서 SSH 키 위치 확인

```
C:\Users\[사용자명]\.ssh\
```

2. 윈도우에서 복사한 뒤에 VMworkstation에 붙여넣기

```
.ssh\
```

3. 필수 권한 설정

```terminal
# 디렉토리 권한
chmod 700 ~/.ssh

# 개인키 권한 (Ed25519 키의 경우)
chmod 600 ~/.ssh/id_ed25519

chmod 644 ~/.ssh/id_ed25519.pub

# 기타 파일들
chmod 600 ~/.ssh/config          # 설정파일 (있다면)
chmod 600 ~/.ssh/known_hosts     # 호스트 정보

# 소유자 확인
chown -R $USER:$USER ~/.ssh
```

4. SSH 연결 테스트

```terminal
# GitHub 테스트
ssh -T git@github.com

# 성공 시 메시지:
# Hi [사용자명]! You've successfully authenticated, but GitHub does not provide shell access
```

<br><br>

**[3. 가상환경 생성]**

1. 텐서플로우 디렉토리 생성

```termianl
mkdir opencv_tf
cd opencv_tf
```

2. 파이썬 가상환경 설치

```terminal
sudo apt install python3.10-venv
```

3. 가상환경 생성

```terminal
# 가상환경 생성
python3 -m venv tfvenv

# 가상환경 진입
source tfvenv/bin/activate
```

4. 가상환경 진입 후 ros2 충돌시 pip list 해결방법

 _가상환경 종료 후_

```terminal
# 백업 파일 생성
cp ~/.bashrc ~/.bashrc.backup

# bashrc에서 ros2 자동 실행 명령어 주석처리
nano ~/.bashrc
# source /opt/ros/humble/setup.bash
```

5. ROS2 환경 변수 제거

```terminal
unset ROS_VERSION

unset ROS_PYTHON_VERSION  

unset ROS_LOCALHOST_ONLY

unset ROS_DISTRO

unset AMENT_PREFIX_PATH

unset PYTHONPATH
```

6. 가상환경 삭제 후 재설치

```terminal
# 가상환경 삭제
rm -rf tfvenv

# 가상환경 재설치
python3 -m venv tfvenv
```

7. 가상환경 진입 후 pip list 확인

```terminal
# 가상환경 실행
source tfvenv/bin/activate

# pip list 확인

# Package    Version
# ---------- -------
# pip        22.0.2
# setuptools 59.6.0
```

<br><br>

**[4. tensorflow 설치]**

```terminal
python3 -m pip install tensorflow
```

</div>
</details>

## 2. Tensorflow 실습 (얼굴 이미지에서 감정 분류)

<details>
<summary></summary>
<div markdown="1">

**[데이터셋 다운로드](https://www.kaggle.com/datasets/msambare/fer2013)**

## **2-1. 훈련, 테스트 데이터셋 만들기**

```python3
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
```

<br><br>

## **2-2. CNN 모델 설계**

```python3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization

num_classes = 7
num_detectors = 32
width, height = 48, 48

network = Sequential()

network.add(Conv2D(filters=num_detectors, kernel_size=3, activation='relu', padding='same', input_shape=(width, height, 3)))
network.add(BatchNormalization())
network.add(Conv2D(filters=num_detectors, kernel_size=3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))

network.add(Flatten())

network.add(Dense(2*2*num_detectors, activation='relu'))
network.add(BatchNormalization())
network.add(Dropout(0.2))

network.add(Dense(2*num_detectors, activation='relu'))
network.add(BatchNormalization())
network.add(Dropout(0.2))

network.add(Dense(num_classes, activation='softmax'))

network.summary()
```

<br><br>

## **2-3. 모델 훈련과 성능 평가**

```python3
# 모델 훈련
network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 3

network.fit(train_dataset, epochs=epochs)

# 모델 성능 평가
network.evaluate(test_dataset)
preds = network.predict(test_dataset)
print(preds)
preds = np.argmax(preds, axis=1)
print(preds)
print(test_dataset.classes)
print(accuracy_score(test_dataset.classes, preds))

# 모델 저장
network.save('../models/emotion_model.h5')
```

<br><br>

## **2-4. 감정 분류**

```python3

```

</div>
</details>



