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
   - 실습 예제

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

## **4-5. 실습 예제**

```python3

```
