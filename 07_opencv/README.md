# 머신러닝 / K-평균 클러스터링 / K-최근접 이웃

## 목차

1. 머신러닝

2. K-평균 클러스터링 (K-Means)
   - K-평균 클러스터링이란?
   - 프로세스
   - k-means 기본 코드
   - k-means 랜덤 설정
   - k-means 색상 분류
   - MNIST
   - k-means 손글씨 숫자 군집화
  
3. 개인 프로젝트 (차선 색상 분류)

4. K-최근접 이웃(KNN)
   - K-최근접 이웃이란?
   - Lazy Model
   - 유클리드 거리 계산법
   - 맨해튼 거리 계산법
   - KNN 랜덤 설정
   - KNN MNIST 분류
   - KNN 손글씨 숫자 예제

5. 개인 프로젝트 (옷 색상 kNN 분류)

## 1. 머신러닝

<details>
<summary></summary>
<div markdown="1">

## **1-1. 머신러닝이란?**

컴퓨터가 명시적으로 프로그래밍되지 않아도 **경험(데이터)을 통해 스스로 학습하고 개선하는 기술**

**[대표적인 적용 사례]**

`이미지 분류` : 제품 생산 시 제품의 이미지를 분석해 자동으로 분류하는 시스템

`시맨틱 분할` : 인간의 뇌를 스캔하여 종양 여부의 진단

`텍스트 분류(자연어 처리)` : 자동으로 뉴스, 블로그 등의 게시글 분류

`텍스트 분류` : 토론 또는 사이트 등에서의 부정적인 코멘트를 자동으로 구분

`텍스트 요약` : 긴 문서를 자동으로 요약하여 요점 정리

`자연어 이해` : 챗봇(chatbot) 또는 인공지능 비서 만들기

`회귀 분석` : 회사의 내년도 수익 예측

`음성 인식` : 음성 명령에 반응하는 프로그램

`이상치 탐지` : 신용 카드 부정 거래 감지

`군집 작업` : 구매 이력을 기반으로 고객 분류 후 서로 다른 마케팅 전략 계획

`데이터 시각화` : 고차원의 복잡한 데이터셋을 그래프와 같은 효율적인 시각 표현

`추천 시스템` : 과거 구매이력, 관심 상품, 찜 목록 등을 분석하여 상품 추천

`강화 학습` : 지능형 게임 봇 만들기

<br><br>

## **1-2. 머신러닝 시스템의 분류**

<img width="994" height="541" alt="image" src="https://github.com/user-attachments/assets/63c69391-616f-424e-a464-0e2b6a5ba568" />

`1. 훈련 지도 여부 : 지도 학습, 비지도 학습, 준지도 학습, 강화 학습`

`2. 실시간 훈련 여부 : 온라인 학습, 배치 학습`

`3. 예측 모델 사용 여부 : 사례 기반 학습, 모델 기반 학습`

**훈련 지도 여부 구분]**

1. 지도 학습
   - 훈련 데이터로부터 하나의 함수를 유추해내기 위한 방법
   - 지도 학습에는 훈련 데이터에 레이블(label) 또는 타깃(garget)이라는 정답지가 포함되어 있음

1) 분류(classification)
   
<img width="924" height="364" alt="image" src="https://github.com/user-attachments/assets/d826aded-2184-45b5-881b-c97ac89d1f6e" />

2) 회귀(regression)
   
<img width="753" height="412" alt="image" src="https://github.com/user-attachments/assets/707a4500-3fcd-45e5-b9db-636fe84bcd88" />

3) 지도 학습 알고리즘

- k-최근접 이웃(kNN : k-Nearest Neighbors)
- 선형 회귀(linear regression)
- 로지스틱 회귀(logistic regression)
- 서포트 벡터 머신(SVC : support vector machines)
- 결정 트리(decision trees)
- 랜덤 포레스트(randome forests)
- 신경망(neural networks)

<br><br>

2. 비지도 학습
   - 레이블이 없는 훈련 데이터를 이용하여 시스템이 스스로 학습을 하도록 하는 학습 방법
   - 입력 값에 대한 목표치가 주어지지 않음

<img width="775" height="402" alt="image" src="https://github.com/user-attachments/assets/576168b3-a218-4ae5-8f88-5cc1f8c59d71" />

1) 군집
   - 데이터를 비슷한 특징을 가진 몇 개의 그룹으로 나누는 것

<img width="752" height="406" alt="image" src="https://github.com/user-attachments/assets/f97fd93a-665f-4cc8-95b2-99e1d60f27d5" />

2) 시각화와 차원 축소
   - 레이블이 없는 다차원 특성을 가진 데이터셋을 2D 또는 3D로 표현하는 것
   - 시각화를 하기 위해서는 데이터 특성을 두 가지로 줄여야 한다.

<img width="884" height="589" alt="image" src="https://github.com/user-attachments/assets/ef5e7578-ee54-4988-a832-a93bb568defe" />

3) 이상치 탐지(Outlier detection)와 특이치 탐지(Novelty detection)
   -정상 샘플을 이용하여 훈련 후 입력 샘플의 정상여부를 판단하여 이상치를 추출하거나 자동으로 제거하는 것

<img width="517" height="283" alt="image" src="https://github.com/user-attachments/assets/ef6629b1-c9e8-401e-ab2d-00245b1e8a9c" />

4) 연관 규칙 학습
   - 데이터 특성 간의 흥미로운 관계를 찾는 것


<br><br>

3. 준지도 학습
   - 레이블이 적용된 적은 수의 샘플이 주어졌을 때 유용한 방법
   - 비지도 학습을 통해 군집을 분류한 후 샘플들을 활용해 지도 학습을 실행한다.

<img width="742" height="393" alt="image" src="https://github.com/user-attachments/assets/e0bab86c-4b51-4190-8c73-43beba63873b" />

<br><br>

4. 강화 학습
   - 학습 시스템을 에이전트라 부르며, 에이전트가 취한 행동에 대해 보상 또는 벌점을 주어 가장 큰 보상을 받는 방향으로 유도하는 방법

<img width="542" height="537" alt="image" src="https://github.com/user-attachments/assets/e305efa1-a98a-4b97-9d49-183c44e78951" />

</div>
</details>

## 2.K-평균 클러스터링 (K-means Clustering)

<details>
<summary></summary>
<div markdown="1">

## **2-1. K-평균 클러스터링이란?f**

 **비지도 학습의 클러스터링 모델 중 하나이다.**

 <img width="220" height="147" alt="image" src="https://github.com/user-attachments/assets/294b81c2-b7e0-4e43-a10e-aff4cf383934" />

<br><br>

**클러스터**란 _비슷한 특성을 가진 데이터끼리의 묶음_ 이고, **클러스터링**이란 어떤 데이터들이 주어졌을 때, _그 데이터들을 클러스터로 그루핑 시켜주는 것_ 이다.

각 클러스터의 중심을 **Centroid**라고 한다.

K-means Clustering에서 **K는 클러스터의 갯수**를 뜻하므로 위의 사진 속 K는 총 3개가 된다.

# 📊 클러스터링 개념 정리

| **클러스터 (Cluster)** | 비슷한 특성을 가진 데이터들의 묶음<br>→ 일반적으로 "서로 가까운 위치에 있는 데이터" |
| **클러스터링 (Clustering)** | 주어진 데이터들을 클러스터로 자동 분류하는 작업<br>→ 처음엔 구분이 없던 데이터들을 거리 기반으로 그룹화 |
| **Centroid** | 각 클러스터의 중심에 해당하는 좌표값 |
| **K-means Clustering** | K개의 클러스터를 생성하는 알고리즘<br>`K` = 클러스터 개수<br>`means` = 각 클러스터의 중심 (Centroid) |
| **예시** | 그림에 3개의 클러스터가 있다면 K=3이며, 각 클러스터는 가까운 점들로 구성되고, 중심에는 Centroid가 존재 |

## **K-means Clustering의 목적은 유사한 데이터 포인트끼리 그루핑 하여 패턴을 찾아내는 것**

## **2-2. 프로세스**

1. **K값 결정**  
   - 얼마나 많은 클러스터가 필요한지 결정

2. **초기 Centroid 설정**  
   - 랜덤 설정  
   - 수동 설정  
   - K-means++ 방식 사용 가능

3. **데이터 할당 (Assign)**  
   - 각 데이터를 가장 가까운 Centroid가 속한 클러스터에 할당

4. **Centroid 업데이트**  
   - 각 클러스터의 중심값으로 Centroid를 이동

5. **반복 수행**  
   - 클러스터 할당이 더 이상 바뀌지 않을 때까지  
   - 또는 최대 반복 횟수에 도달할 때까지  
   - Step 3과 4를 반복

> 시각화 시물레이션 사이트 : https://www.naftaliharris.com/blog/visualizing-k-means-clustering/
  
## **2-3. k-means 기본 코드**

```python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X= -2 * np.random.rand(100,2)
X1 = 1 + 2 * np.random.rand(50,2)
X[50:100, :] = X1
plt.scatter(X[ : , 0], X[ :, 1], s = 50, c = 'b')
plt.show()
```

<img width="374" height="252" alt="image" src="https://github.com/user-attachments/assets/ea8d6fc7-47d2-4453-b0b6-20bcf9f49acc" />

<br><br>

```python3
# 두 centroid의 위치 확인
Kmean.cluster_centers_

>>> array([[ 2.02664296,  1.88206121],
          [-1.01085055, -1.03792754]])
```

```python3
# 두 centroid의 위치 함께 출력
plt.scatter(-0.94665068, -0.97138368, s=200, c='g', marker='s')
plt.scatter(2.01559419, 2.02597093, s=200, c='r', marker='s')
plt.show()
```

<img width="374" height="252" alt="image" src="https://github.com/user-attachments/assets/705df97f-9534-490f-a773-e6e275e121cf" />

<br><br>

## **2-4. k-means 랜덤 설정**

```python3
import numpy as np, cv2
import matplotlib.pyplot as plt

# 0~150 임의의 2수, 25개
a = np.random.randint(0,150,(25,2))

# 128~255 임의의 2수, 25개
b = np.random.randint(128, 255,(25,2))

# a, b를 병합
data = np.vstack((a,b)).astype(np.float32)

# 중지 요건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 평균 클러스터링 적용
# data : 처리 대상 데이터
# K : 원하는 묶음 갯수
# 결과 데이터
# 반복 종료 조건
# 매전 다른 초기 레이블로 실행할 횟수
# 초기 중앙점 선정 방법
ret,label,center=cv2.kmeans(data,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# label에 따라 결과 분류
red = data[label.ravel()==0]
blue = data[label.ravel()==1]

# plot에 결과 출력
plt.scatter(red[:,0],red[:,1], c='r')
plt.scatter(blue[:,0],blue[:,1], c='b')

# 각 그룹의 중앙점 출력
plt.scatter(center[0,0],center[0,1], s=100, c='r', marker='s')
plt.scatter(center[1,0],center[1,1], s=100, c='b', marker='s')
plt.show()
```

<img width="640" height="545" alt="image" src="https://github.com/user-attachments/assets/e70f83e8-03fb-4b3f-a42f-4037a1362ab3" />

<br><br>

## **2-4. k-means 색상 분류**

```python3
# 3채널 컬러 영상은 하나의 색상을 위해서 24비트 (8x3)
# 16777216가지 색상 표현 가능

# 모든 색을 다 사용하지 않고 비슷한 색상 그룹 지어서 같은 색상으로 처리
# 처리 용량 간소화

import numpy as np
import cv2

K = 16 # 군집화 갯수
img = cv2.imread('../img/taekwonv1.jpg')
data = img.reshape((-1, 3)).astype(np.float32)
# 데이터 평균을 구할 때 소수점 이하값을 가질 수있으므로 변환
# 반복 중지 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 평균 클러스터링 적용
ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

# 중심값을 정수형으로 변환

center = np.uint8(center)
print(center)

# 각 레이블에 해당하는 중심값으로 픽셀 값 선택
res = center[label.flatten()]
# 원본 영상의 형태로 변환
res = res.reshape((img.shape))

# 결과 출력
merged = np.hstack((img, res))
cv2.imshow('Kmeans color', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img width="797" height="475" alt="image" src="https://github.com/user-attachments/assets/d1ad69ee-2782-4976-a323-67b5dc33bc00" />

<img width="111" height="278" alt="image" src="https://github.com/user-attachments/assets/411d309f-da71-4560-b910-754d2564e674" />

<br><br>

## **2-5. MNIST**

**MNIST란? : Modified National Institute of Standards and Technology database**

**각 이미지의 크기가 28x28픽셀인 그레이스케일의 손글씨 숫자 이미지 7만개 모음 **

<img width="2000" height="1000" alt="image" src="https://github.com/user-attachments/assets/0722ea95-47c8-4b85-96ed-f866f92b8a05" />

<br><br>

[MNIST 데이터 전처리 모듈]

```python3
import numpy as np, cv2

data = None  # 이미지 데이타 셋 
k = list(range(10)) # [0,1,2,3,4,5,6,7,8,9] 레이블 셋

# 이미지 데이타 읽어들이는 함수 ---①
def load():
    global data
    # 0~9 각각 500(5x100)개, 총5000(50x100)개, 한 숫자당 400(20x20)픽셀
    image = cv2.imread('../img/digits.png')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # 숫자 한개(20x20)씩 구분하기 위해 행별(50)로 나누고 열별(100)로 나누기
    cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
    # 리스트를 NumPy 배열로  변환 (50 x 100 x 20 x 20 )
    data = np.array(cells)

# 모든 숫자 데이타 반환 ---②
def getData(reshape=True):
    if data is None: load() # 이미지 읽기 확인
    # 모든 데이타를 N x 400 형태로 변환
    if reshape:
        full = data.reshape(-1, 400).astype(np.float32) # 5000x400
    else:
        full = data
    labels = np.repeat(k,500).reshape(-1,1)  # 각 숫자당 500번 반복(10x500)
    return (full, labels)

# 훈련용 데이타 반환 ---③
def getTrain(reshape=True):
    if data is None: load() # 이미지 읽기 확인
    # 50x100 중에 90열만 훈련 데이타로 사용
    train = data[:,:90]
    if reshape:
        # 훈련 데이타를 N X 400으로 변환
        train = train.reshape(-1,400).astype(np.float32) # 4500x400
    # 레이블 생성
    train_labels = np.repeat(k,450).reshape(-1,1) # 각 숫자당 45번 반복(10x450)
    return (train, train_labels)

# 테스트용 데이타 반환 ---④
def getTest(reshape=True):
    if data is None: load()
    # 50x100 중에 마지막 10열만 훈련 데이타로 사용
    test = data[:,90:100]
    # 테스트 데이타를 N x 400으로 변환
    if reshape:
        test = test.reshape(-1,400).astype(np.float32) # 500x400
    test_labels = np.repeat(k,50).reshape(-1,1)
    return (test, test_labels)


# 손글씨 숫자 한 개를 20x20 로 변환후에 1x400 형태로 변환 ---⑤
def digit2data(src, reshape=True):
    h, w = src.shape[:2]
    square = src
    # 정사각형 형태로 만들기
    if h > w:
        pad = (h - w)//2
        square = np.zeros((h, h), dtype=np.uint8)
        square[:, pad:pad+w] = src
    elif w > h :
        pad = (w - h)//2
        square = np.zeros((w, w), dtype=np.uint8)
        square[pad:pad+h, :] = src
    # 0으로 채워진 20x20 이미지 생성
    px20 = np.zeros((20,20), np.uint8)
    # 원본을 16x16으로 축소해서 테두리 2픽셀 확보
    px20[2:18, 2:18] = cv2.resize(square, (16,16), interpolation=cv2.INTER_AREA)
    if reshape:
        # 1x400형태로 변환
        px20 = px20.reshape((1,400)).astype(np.float32)
    return px20
```

<br><br>

## **2-6. k-means 손글씨 숫자 군집화**

```python3
import cv2, numpy as np
import matplotlib.pyplot as plt
import mnist

# 공통 모듈로 부터 MINST 전체 이미지 데이타 읽기
data, _ = mnist.getData()

# 중지 요건 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 평균 클러스터링 적용, 10개의 그룹으로 묶음
ret,label,center=cv2.kmeans(data,10,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# 중앙점 이미지 출력
for i in range(10):
    # 각 중앙점 값으로 이미지 생성
    cent_img = center[i].reshape(20,20).astype(np.uint8)
    plt.subplot(2,5, i+1)
    plt.imshow(cent_img, 'gray')
    plt.xticks([]);plt.yticks([])
    
plt.show()
```

<img width="639" height="545" alt="image" src="https://github.com/user-attachments/assets/416f7a93-08a0-4f6c-af50-576db3f81d4f" />

_**비지도 학습 모델이기 때문에 누락된 숫자가 발생한다.**_

</div>
</details>

## 3. 개인 프로젝트 (차선 색상 분류)

<details>
<summary></summary>
<div markdown="1">

```python3
'''
1. 이미지를 불러온다.
2. 평균 클러스터링을 사용해 색상을 분류한다.
3. 분류한 이미지를 출력한다.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

K = 8  # 군집화 갯수

img = cv2.imread('../img/load_line.jpg')
# 이미지 사이즈를 1/5로 줄임
img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

data = img.reshape((-1, 3)).astype(np.float32)

# 반복 중지 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 10회 반복, 결과 확인 후 변경

# 평균 클러스터링 적용
ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

# 중심값을 정수형으로 변환

center = np.uint8(center)
print(center)

# 각 레이블에 해당하는 중심값으로 픽셀 값 선택
res = center[label.flatten()]

# 원본 영상의 형태로 변환
res = res.reshape((img.shape))

# 결과 출력
merged = np.hstack((img, res))
cv2.imshow('Load Line', merged)

# --- 색상 팔레트 생성 ---

# 픽셀 수 계산
unique, counts = np.unique(label, return_counts=True)
total_pixels = data.shape[0]

# 픽셀 수 내림차순 정렬 인덱스
sorted_idx = np.argsort(counts)[::-1]

# 상위 3개 클러스터 인덱스와 값들만 선택
top3_idx = sorted_idx[:3]
top3_centers = center[top3_idx]
top3_counts = counts[top3_idx]
top3_ratios = top3_counts / total_pixels

palette = np.zeros((50, 300, 3), dtype=np.uint8)
step = 300 // 3
for i, color in enumerate(top3_centers):
    palette[:, i*step:(i+1)*step, :] = color
cv2.imshow('Top 3 Color Palette', palette)

# --- 색상 분포 차트 및 상세 분석 ---

# 클러스터 별 비율 계산
ratios = counts / total_pixels

# BGR → RGB 변환 (matplotlib는 RGB)
colors_rgb = center[:, ::-1] / 255.0  # 0~1 정규화

# 분포 차트 출력
plt.figure(figsize=(8, 4))
plt.bar(range(K), ratios, color=colors_rgb, tick_label=[f'C{i}' for i in range(K)])
plt.title('Cluster Color Distribution')
plt.xlabel('Cluster')
plt.ylabel('Pixel Ratio')
plt.ylim(0, 1)
plt.show()

# 상세 분석 출력
print("\n클러스터 상세 분석:")
for i in range(K):
    b, g, r = center[i]
    print(f"Cluster {i}: BGR=({b}, {g}, {r}), 픽셀 수={counts[i]}, 비율={ratios[i]:.4f}")

cv2.waitKey(0)
cv2.destroyAllWindows()
```

**[결과 출력]**

<img width="1485" height="518" alt="image" src="https://github.com/user-attachments/assets/707ff264-a2cc-4252-bce2-5c5eec5aea57" />

<br><br>

**[추출된 3가지 대표색상]**

<img width="299" height="79" alt="image" src="https://github.com/user-attachments/assets/0f047eb7-b1ad-4cd6-a666-f86fc8298d22" />

<br><br>

**[색상 분포 차트]**

<img width="799" height="466" alt="image" src="https://github.com/user-attachments/assets/6fe01437-6dba-4260-b274-f79f5b2a286d" />

<br><br>

**[각 색상의 중심값(Centroid) 좌표]**

<img width="109" height="141" alt="image" src="https://github.com/user-attachments/assets/f9cdb40e-769e-468d-a1b5-9324d181d484" />

<br><br>

**[클러스터 분석 표]**

<img width="430" height="160" alt="image" src="https://github.com/user-attachments/assets/c2d30a0b-fe1d-4b4b-9fd5-062936d0ae10" />

</div>
</details>

## 4. K-최근접 이웃(KNN)

<details>
<summary></summary>
<div markdown="1">

## **4-1. K-최근접 이웃(KNN)이란?**

**지도 학습 알고리즘 중 하나이다.**

어떤 데이터가 주어지면 그 _주변(이웃)의 데이터를 살펴본 뒤_ 더 많은 데이터가 포함되어 있는 범주로 분류하는 방식

<img width="753" height="563" alt="image" src="https://github.com/user-attachments/assets/c5f56640-6365-4031-968e-28b271478694" />

[K = 3 일때는 Class B로 분류, K = 6일때는 Class A로 분류]

> 시각화 시물레이션 사이트 : http://vision.stanford.edu/teaching/cs231n-demos/knn/
> 시물레이션 해설 사이트 : https://pangguinland.tistory.com/127

## **4-2. Lazt Model**

KNN은 사전 모델링이 따로 필요 없는 모델이므로 처리 속도가 빠름

## **4-3. 유클리드 거리 계산법 (Euclidean Distance)**

**일반적으로 점과 점 사이의 거리를 구하는 방법**

<img width="792" height="171" alt="image" src="https://github.com/user-attachments/assets/1e1865ad-5d34-4889-93a0-3b01fa2baa1a" />

<br><br>

[3차원에서 유클리드 거리를 구하는 방법]

<img width="778" height="636" alt="image" src="https://github.com/user-attachments/assets/dcad2be9-ccd3-44cd-89b6-e2f8671a6016" />

<img width="539" height="104" alt="image" src="https://github.com/user-attachments/assets/7043e91e-040b-4cfe-b608-60806e0e33d2" />

## **4-4. 맨해튼 거리 계산법 (Manhattan Distance)**

**점과 점사이의 직선거리가 아닌 X축, Y축을 따라 간 거리를 구하는 방법**

<img width="749" height="647" alt="image" src="https://github.com/user-attachments/assets/d0d664df-e605-4f53-8bdf-06be5ca62546" />

## **4-5. KNN 랜덤 설정 **

```python3
import cv2, numpy as np, matplotlib.pyplot as plt

# 0~200 사이의 무작위 수 50x2개 데이타 생성
red = np.random.randint(0, 110, (25,2)).astype(np.float32)
blue = np.random.randint(90, 200, (25, 2)).astype(np.float32)
trainData = np.vstack((red, blue))

# 50x1개 레이블 생성
labels = np.zeros((50,1), dtype=np.float32) # 0:빨강색 삼각형
labels[25:] = 1           # 1:파랑색 사각형

# 레이블 값 0과 같은 자리는 red, 1과 같은 자리는 blue로 분류해서 표시
plt.scatter(red[:,0], red[:,1], s=80, c='r', marker='^') # 빨강색 삼각형
plt.scatter(blue[:,0], blue[:,1], s=80, c='b', marker='s')# 파랑색 사각형

# 0 ~ 200 사이의 1개의 새로운 무작위 수 생성
newcomer = np.random.randint(0,200,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0], newcomer[:,1], s=80, c='g', marker='o') # 초록색 원

# KNearest 알고리즘 객체 생성
knn = cv2.ml.KNearest_create()

# train, 행 단위 샘플
knn.train(trainData, cv2.ml.ROW_SAMPLE, labels)

# 예측
#ret, results = knn.predict(newcomer)
ret, results, neighbours, dist = knn.findNearest(newcomer, 3) #K=3

# 결과 출력
print('ret:%s, result:%s, neighbours:%s, distance:%s' \
        %(ret, results, neighbours, dist))
plt.annotate('red' if ret==0.0 else 'blue', xy=newcomer[0], \
             xytext=(newcomer[0]+1))
plt.show()
```

<img width="640" height="546" alt="image" src="https://github.com/user-attachments/assets/46477e38-ccf8-4a69-9374-bb15d9c9e62d" />

<img width="538" height="22" alt="image" src="https://github.com/user-attachments/assets/cb12f0c9-82f1-4070-8af9-87787278b837" />

| 키워드          | 값                   | 의미                              |
| ------------ | ------------------- | ------------------------------- |
| `ret`        | `1.0`               | 최종 예측 클래스 (여기선 `1`: 파랑 사각형)     |
| `result`     | `[[1.]]`            | 예측 결과 (same as `ret`)           |
| `neighbours` | `[[1. 1. 1.]]`      | 가장 가까운 3개의 이웃의 클래스 (모두 `1`)     |
| `distance`   | `[[49. 409. 436.]]` | newcomer와 각 이웃 간의 거리 (작을수록 가까움) |


## **4-6. KNN MNIST 분류**

```python3
import numpy as np, cv2
import mnist

# 훈련 데이타와 테스트 데이타 가져오기
train, train_labels = mnist.getTrain()
test, test_labels = mnist.getTest()

# kNN 객체 생성 및 훈련
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

# k값을 1~10까지 변경하면서 예측
for k in range(1, 11):
    # 결과 예측
    ret, result, neighbors, distance = knn.findNearest(test, k=k)

    # 정확도 계산 및 출력
    correct = np.sum(result == test_labels)
    accuracy = correct / result.size * 100.0
    print("K:%d, Accuracy :%.2f%%(%d/%d)" % (k, accuracy, correct, result.size))
```

<img width="998" height="562" alt="image" src="https://github.com/user-attachments/assets/01017012-dbde-4b06-84c6-654d6d90fd1a" />


<img width="226" height="173" alt="image" src="https://github.com/user-attachments/assets/249359c5-eb06-4aa2-967e-d2468eda4ec0" />

## **4-7. KNN 손글씨 숫자 예제**

```python3
import numpy as np, cv2
import mnist

# 훈련 데이타 가져오기
train, train_labels = mnist.getData()

# Knn 객체 생성 및 학습
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

# 인식시킬 손글씨 이미지 읽기
image = cv2.imread('../img/4027.png')
cv2.imshow("image", image)
cv2.waitKey(0) 

# 그레이 스케일 변환과 스레시홀드
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
_, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# 최외곽 컨투어만 찾기
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, \
                                        cv2.CHAIN_APPROX_SIMPLE)[-2:]

# 모든 컨투어 순회
for c in contours:
    # 컨투어를 감싸는 외접 사각형으로 숫자 영역 좌표 구하기
    (x, y, w, h) = cv2.boundingRect(c) 

    # 외접 사각형의 크기가 너무 작은것은 제외
    if w >= 5 and h >= 25:
        # 숫자 영역만 roi로 확보하고 사각형 그리기
        roi = gray[y:y + h, x:x + w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # 테스트 데이타 형식으로 변환
        data = mnist.digit2data(roi)
        
        # 결과 예측해서 이미지에 표시
        ret, result, neighbours, dist = knn.findNearest(data, k=1)
        cv2.putText(image, "%d"%ret, (x , y + 155), \
                        cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("image", image)
        cv2.waitKey(0) 

cv2.destroyAllWindows()
```

<img width="597" height="229" alt="image" src="https://github.com/user-attachments/assets/8b76e04a-2229-4056-8b7f-4f4f0c101504" />

**[결과가 틀린 경우]**
<img width="896" height="344" alt="image" src="https://github.com/user-attachments/assets/80584b73-7192-44ac-815a-da5b6c48f1bd" />

</div>
</details>

## 5. 개인 프로젝트 (옷 색상 kNN 분류)

<details>
<summary></summary>
<div markdown="1">

## **5-1. 옷 색상 데이터셋 만들기 (kNN_makeColorDataset_park.py)**

```python3
import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 숫자 키와 색상 이름 매핑 (키보드 ASCII 코드 기준)
color_labels = {
    49: "Red",    # '1'
    50: "Blue",   # '2'
    51: "Green",  # '3'
    52: "Yellow", # '4'
    53: "Black",  # '5'
    54: "White",  # '6'
    55: "Gray"    # '7'
}

# 수집한 색상 샘플 저장 리스트 (B, G, R, 라벨)
samples = []

# ROI(관심 영역) 크기와 초기 위치 설정 (중앙 기준)
roi_size = 100
frame_width, frame_height = 640, 480
cx, cy = frame_width // 2, frame_height // 2

# KNN 모델과 데이터 스케일러 (초기값 None)
knn_model = None
scaler = None

# --- CSV에서 데이터 불러와 KNN 모델 학습 ---
def load_dataset_and_train():
    global knn_model, scaler
    try:
        # 저장된 CSV 불러오기
        df = pd.read_csv('color_dataset.csv')
        X = df[['B', 'G', 'R']].values.astype(float)  # 특징값 (BGR)
        y = df['Label'].values                         # 라벨(문자열)

        # 데이터 정규화 (0~1 사이)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # 학습/검증 데이터 분리 (20% 검증)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)

        # KNN 분류기 생성 및 학습 (k=3)
        knn_model = KNeighborsClassifier(n_neighbors=3)
        knn_model.fit(X_train, y_train)

        # 검증 데이터 정확도 출력
        acc = knn_model.score(X_test, y_test)
        print(f"K-NN 모델 학습 완료, 테스트 정확도: {acc*100:.2f}%")
        return True
    except Exception as e:
        # 데이터 없거나 오류 시 안내 메시지 출력
        print("데이터셋 없음 또는 학습 실패:", e)
        return False

# 웹캠 초기화 및 프레임 크기 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

print("웹캠 실행 중. ESC 키로 종료")
print("마우스 왼쪽 클릭으로 ROI 내 색상 샘플 수집 후 1~7 숫자키로 라벨링")

# 현재 선택된 색상과 클릭 위치, 프레임 저장 변수
current_color = None
click_pos = None
current_frame = None

# --- 마우스 클릭 이벤트 처리 함수 ---
def mouse_callback(event, x, y, flags, param):
    global current_color, click_pos, current_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        # 프레임 없으면 무시
        if current_frame is None:
            return
        # ROI 내부 클릭했는지 확인
        if (cx - roi_size//2 <= x <= cx + roi_size//2) and (cy - roi_size//2 <= y <= cy + roi_size//2):
            # ROI 영역 평균 색상 계산
            roi = current_frame[cy - roi_size//2:cy + roi_size//2, cx - roi_size//2:cx + roi_size//2]
            avg_color = np.mean(roi.reshape(-1,3), axis=0).astype(int)
            current_color = avg_color
            click_pos = (x, y)
            print(f"샘플 색상 추출됨: BGR = {avg_color}")

# OpenCV 윈도우 생성 및 마우스 콜백 연결
cv2.namedWindow("Color Collect & Predict")
cv2.setMouseCallback("Color Collect & Predict", mouse_callback)

# 학습 데이터 불러와 모델 학습 시도
model_ready = load_dataset_and_train()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_frame = frame.copy()  # 원본 프레임 복사 (마우스 이벤트용)

    # ROI 위치에 사각형 그리기 (초록색)
    x1, y1 = cx - roi_size//2, cy - roi_size//2
    x2, y2 = cx + roi_size//2, cy + roi_size//2
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # 클릭해 추출한 샘플 색상과 위치 표시
    if current_color is not None:
        cv2.putText(frame, f"Sampled BGR: {current_color}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.circle(frame, click_pos, 5, (0,255,0), -1)

    # 모델이 준비된 경우 ROI 내 색상 예측 결과 표시
    if model_ready:
        roi = frame[y1:y2, x1:x2]
        avg_color = np.mean(roi.reshape(-1,3), axis=0).astype(int)
        avg_scaled = scaler.transform([avg_color])
        pred_label = knn_model.predict(avg_scaled)[0]
        cv2.putText(frame, f"Predicted Color: {pred_label}", (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Color Collect & Predict", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC 키 누르면 종료
        break
    elif key in color_labels and current_color is not None:
        # 숫자키(1~7) 눌러 라벨링 시 샘플 리스트에 저장
        label = color_labels[key]
        b, g, r = current_color
        samples.append([int(b), int(g), int(r), label])
        print(f"샘플 수집됨: {label} - BGR({b},{g},{r})")
        current_color = None

    elif key == ord('s'):
        # 's' 키 누르면 지금까지 수집한 샘플 CSV로 저장 후 모델 재학습
        if samples:
            df = pd.DataFrame(samples, columns=['B', 'G', 'R', 'Label'])
            df.to_csv('color_dataset.csv', index=False)
            print(f"샘플 {len(samples)}개 저장됨 (color_dataset.csv)")
            model_ready = load_dataset_and_train()
        else:
            print("저장할 샘플이 없습니다")

    # 방향키로 ROI 위치 이동 (좌, 우, 상, 하)
    elif key == 81:  # 왼쪽 화살표
        cx = max(cx - 10, roi_size//2)
    elif key == 83:  # 오른쪽 화살표
        cx = min(cx + 10, frame_width - roi_size//2)
    elif key == 82:  # 위쪽 화살표
        cy = max(cy - 10, roi_size//2)
    elif key == 84:  # 아래쪽 화살표
        cy = min(cy + 10, frame_height - roi_size//2)

# 종료 처리
cap.release()
cv2.destroyAllWindows()
```

<img width="639" height="511" alt="image" src="https://github.com/user-attachments/assets/d65c85de-b14c-48ff-90de-90fc3b9110cb" />

<img width="303" height="36" alt="image" src="https://github.com/user-attachments/assets/9e5c604c-6348-4f41-b326-7757ab9b78a3" />

<img width="109" height="295" alt="image" src="https://github.com/user-attachments/assets/f794509b-e504-4cdd-a7f9-4df5651c5d13" />


**각 색상별로 15개의 데이터 입력**

## **5-2. 옷 색상 KNN 분류 알고리즘 코드**

```python3
import cv2
import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from PIL import ImageFont, ImageDraw, Image

# --- 설정 변수 ---
CSV_PATH = 'color_dataset.csv'    # 학습 데이터 파일명
MODEL_PATH = 'knn_model.pkl'      # 저장할 KNN 모델 파일명
roi_size = 100                   # 관심영역(ROI) 크기 (정사각형 한 변 길이)
mode = 'predict'                 # 프로그램 시작 모드 ('learn' or 'predict')
knn_k = 5                       # KNN의 k값 (이웃 개수)
label_encoder = LabelEncoder()  # 문자열 라벨을 숫자로 변환할 때 사용
color_list = []                 # 최근 색상 기록 저장용 리스트

# 윈도우 기본 한글 폰트 경로 (필요 시 변경)
FONT_PATH = 'C:/Windows/Fonts/malgun.ttf'  

# --- 한글 출력 함수 (OpenCV는 한글 지원이 약해서 PIL 사용) ---
def put_text_korean(img, text, pos, font_path=FONT_PATH, font_size=30, color=(255,255,255)):
    # OpenCV(BGR) 이미지를 PIL(RGB) 이미지로 변환
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)   # 폰트 및 크기 설정
    draw.text(pos, text, font=font, fill=color)       # 텍스트 그리기
    # 다시 PIL 이미지를 OpenCV 형식(BGR)으로 변환 후 반환
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- KNN 예측 함수 ---
def knn_predict(X_train, y_train, x, k):
    distances = np.linalg.norm(X_train - x, axis=1)  # 각 학습 데이터와 거리 계산
    nearest = np.argsort(distances)[:k]              # 가장 가까운 k개 데이터 인덱스
    top_k_labels = y_train[nearest]                   # 그 데이터들의 라벨
    most_common = Counter(top_k_labels).most_common() # 최빈값 찾기
    pred_label = most_common[0][0]                    # 예측 라벨
    confidence = most_common[0][1] / k                 # 신뢰도(빈도/k)
    return pred_label, confidence

# --- 모델 학습 함수 ---
def train_model():
    global label_encoder
    if not os.path.exists(CSV_PATH):  # 학습 데이터 파일이 없으면 종료
        print("❌ 학습 데이터가 없습니다.")
        return None, None, None

    df = pd.read_csv(CSV_PATH)  # CSV에서 데이터 읽기
    X = df[['R', 'G', 'B']].values / 255.0  # RGB 값 0~1로 정규화
    y = df['label'].values                   # 문자열 라벨

    label_encoder.fit(y)             # 문자열 라벨 → 숫자 인코딩 학습
    y_encoded = label_encoder.transform(y)

    # 학습/검증 데이터 분리 (랜덤 시드 고정)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    acc = 0
    best_k = knn_k
    # 여러 k값에 대해 가장 좋은 정확도 찾기
    for k in [3, 5, 7, 9]:
        correct = 0
        for i in range(len(X_test)):
            pred, _ = knn_predict(X_train, y_train, X_test[i], k)
            if pred == y_test[i]:
                correct += 1
        accuracy = correct / len(X_test)
        if accuracy > acc:
            acc = accuracy
            best_k = k

    print(f"✅ 최적 K: {best_k}, 정확도: {acc*100:.2f}%")

    # 학습 결과 저장 (X_train, y_train, best_k)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((X_train, y_train, best_k), f)

    return X_train, y_train, best_k

# --- ROI 위치 조정을 위한 마우스 콜백 함수 ---
cx, cy = 320, 240  # 초기 ROI 중심 위치 (프레임 중간)
def mouse_callback(event, x, y, flags, param):
    global cx, cy
    if event == cv2.EVENT_LBUTTONDOWN:
        cx, cy = x, y   # 클릭 위치로 ROI 중심 이동

# --- 메인 프로그램 시작 ---
cv2.namedWindow("Color Recognizer")
cv2.setMouseCallback("Color Recognizer", mouse_callback)

# 모델 파일이 있으면 불러오고, 없으면 학습 실행
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        X_train, y_train, best_k = pickle.load(f)
    # 학습 데이터에 맞춰 라벨 인코더 다시 학습
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        label_encoder.fit(df['label'])
else:
    X_train, y_train, best_k = train_model()
    if X_train is None:
        # 학습 데이터가 없으면 빈 배열 세팅
        X_train, y_train, best_k = np.empty((0, 3)), np.empty((0,)), knn_k

cap = cv2.VideoCapture(0)
print("🎥 웹캠 실행 중. 'L': 학습 모드, 'P': 예측 모드, 'S': 모델 저장, 'R': 데이터 초기화, 'Q' 또는 ESC: 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    # ROI 좌표 계산 (중심 기준)
    x1, y1 = cx - roi_size // 2, cy - roi_size // 2
    x2, y2 = cx + roi_size // 2, cy + roi_size // 2
    roi = frame[y1:y2, x1:x2]  # ROI 영역 추출

    if roi.shape[0] > 0 and roi.shape[1] > 0:
        avg_color = roi.mean(axis=0).mean(axis=0)  # ROI 내 평균 BGR 색상 계산
        norm_color = avg_color / 255.0             # 0~1 범위로 정규화

        if mode == 'predict' and len(X_train) > 0:
            # 예측 모드일 때 KNN으로 색상 예측
            pred, conf = knn_predict(X_train, y_train, norm_color, best_k)
            try:
                color_name = label_encoder.inverse_transform([int(pred)])[0]  # 숫자→문자 라벨 변환
            except Exception:
                color_name = str(pred)
            frame = put_text_korean(frame, f"{color_name} ({conf*100:.1f}%)", (10, 50), font_size=30, color=(0,255,0))

        elif mode == 'learn':
            # 학습 모드 안내 텍스트 출력 + 키별 색상 안내 표시
            frame = put_text_korean(frame, "학습 모드: 숫자키(1~7)로 라벨링", (10, 50), font_size=30, color=(0,0,255))
            labels_info = ["Red = 1", "Blue = 2", "Green = 3", "Yellow = 4", "Black = 5", "White = 6", "Gray = 7"]
            for i, text in enumerate(labels_info):
                y_pos = 50 + 35 * (i + 1)
                frame = put_text_korean(frame, text, (10, y_pos), font_size=25, color=(0,0,255))
        
        # 최근 색상 히스토리 사각형으로 시각화
        color_list.append(avg_color)
        if len(color_list) > 10:
            color_list.pop(0)
        for i, c in enumerate(color_list):
            cv2.rectangle(frame, (10+i*30, h-40), (30+i*30, h-10), c.astype(int).tolist(), -1)

    # ROI 사각형 표시
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    frame = put_text_korean(frame, f"MODE: {mode.upper()}", (10, 10), font_size=20, color=(255, 255, 0))

    # 화면에 프레임 출력
    cv2.imshow("Color Recognizer", frame)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:   # 'q' 또는 ESC 키 종료
        break
    elif key == ord('l'):
        mode = 'learn'                  # 학습 모드 전환
    elif key == ord('p'):
        mode = 'predict'               # 예측 모드 전환
    elif key == ord('s'):
        train_model()                  # 모델 재학습 및 저장
    elif key == ord('r'):
        # 데이터셋 초기화
        if os.path.exists(CSV_PATH):
            os.remove(CSV_PATH)
        print("🔄 데이터셋 초기화 완료.")
        X_train, y_train = np.empty((0, 3)), np.empty((0,))
    elif mode == 'learn' and ord('1') <= key <= ord('7'):
        # 학습 모드에서 숫자키 입력 시 샘플 저장
        label = str(key - ord('0'))    # 키 값 → 문자열 라벨 변환
        r, g, b = avg_color.astype(int)
        print(f"➕ 샘플 추가: {r}, {g}, {b}, 라벨: {label}")
        new_data = pd.DataFrame([[r, g, b, label]], columns=['R', 'G', 'B', 'label'])
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            df = pd.concat([df, new_data], ignore_index=True)
        else:
            df = new_data
        df.to_csv(CSV_PATH, index=False)

cap.release()
cv2.destroyAllWindows()
```

**[예측 모드 화면]**

<img width="638" height="507" alt="image" src="https://github.com/user-attachments/assets/f999e8cd-1776-4c82-be5c-3011353a98f4" />

<br><br>

**[초기 실행시 터미널 출력]**

<img width="597" height="39" alt="image" src="https://github.com/user-attachments/assets/5b431d66-4097-4f5a-87bf-5166b44896f9" />

<br><br>

**[학습 모드 화면]**

<img width="637" height="507" alt="image" src="https://github.com/user-attachments/assets/81539830-762e-4290-8a2b-ae5c5aab172f" />

<br><br>

**[학습 모드 데이터 학습 및 저장시 터미널 출력]**

<img width="242" height="38" alt="image" src="https://github.com/user-attachments/assets/3ea5e481-70a2-48c1-afaa-b0d580df942a" />

</div>
</details>
