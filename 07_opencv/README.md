# 머신러닝 / K-평균 클러스터링 /

## 목차

1. 머신러닝
2. K-평균 클러스터링

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

