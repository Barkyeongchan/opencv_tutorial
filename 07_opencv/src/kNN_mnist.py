import numpy as np, cv2
import mnist
import matplotlib.pyplot as plt

# 훈련 데이타와 테스트 데이타 가져오기
train, train_labels = mnist.getTrain()
test, test_labels = mnist.getTest()

# kNN 객체 생성 및 훈련
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

# 정확도 저장용 리스트
k_values = list(range(1, 11))
accuracies = []

# k값을 1~10까지 변경하면서 예측
for k in range(1, 11):
    # 결과 예측
    ret, result, neighbors, distance = knn.findNearest(test, k=k)

    # 정확도 계산 및 출력
    correct = np.sum(result == test_labels)
    accuracy = correct / result.size * 100.0
    accuracies.append(accuracy)
    print("K:%d, Accuracy :%.2f%%(%d/%d)" % (k, accuracy, correct, result.size))

# --- 시각화 ---
plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')
plt.title('k-NN Accuracy on MNIST (Test Size: %d)' % result.size)
plt.xlabel('k value')
plt.ylabel('Accuracy (%)')
plt.xticks(k_values)
plt.grid(True)
plt.ylim(min(accuracies) - 1, max(accuracies) + 1)

plt.tight_layout()
plt.show()