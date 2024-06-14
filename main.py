# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import svm
# from sklearn.datasets import load_digits
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.metrics import accuracy_score

# digits = load_digits()
# X = digits.data
# y = digits.target

# normal_labels = (y < 5).astype(int)  

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# clf_svm = svm.OneClassSVM(kernel='rbf', gamma=0.01, nu=0.3)

# normal_data = X_scaled[y < 5]
# clf_svm.fit(normal_data)

# predictions_svm = clf_svm.predict(X_scaled)
# predictions_svm = (predictions_svm == 1).astype(int)  # -1을 0으로, 1을 1로 변환

# accuracy_svm = accuracy_score(normal_labels, predictions_svm) * 100

# plt.figure(figsize=(10, 6))
# plt.title(f"One-Class SVM Anomaly Detection\nAccuracy: {accuracy_svm:.2f}%")
# colors_svm = np.array(['blue' if pred == 1 else 'red' for pred in predictions_svm])
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_svm, s=10, edgecolors='k', alpha=0.6)

# from matplotlib.lines import Line2D
# legend_elements = [Line2D([0], [0], marker='o', color='w', label='Normal Data',
#                           markerfacecolor='blue', markersize=10),
#                    Line2D([0], [0], marker='o', color='w', label='Anomaly Data',
#                           markerfacecolor='red', markersize=10)]
# plt.legend(handles=legend_elements, loc='upper right')

# plt.xlabel('First Principal Component')
# plt.ylabel('Second Principal Component')
# plt.show()

# from sklearn.ensemble import IsolationForest

# clf_if = IsolationForest(contamination=0.3, random_state=42)

# clf_if.fit(normal_data)

# predictions_if = clf_if.predict(X_scaled)
# predictions_if = (predictions_if == 1).astype(int)  # -1을 0으로, 1을 1로 변환

# accuracy_if = accuracy_score(normal_labels, predictions_if) * 100

# plt.figure(figsize=(10, 6))
# plt.title(f"Isolation Forest Anomaly Detection\nAccuracy: {accuracy_if:.2f}%")
# colors_if = np.array(['blue' if pred == 1 else 'red' for pred in predictions_if])
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_if, s=10, edgecolors='k', alpha=0.6)

# legend_elements = [Line2D([0], [0], marker='o', color='w', label='Normal Data',
#                           markerfacecolor='blue', markersize=10),
#                    Line2D([0], [0], marker='o', color='w', label='Anomaly Data',
#                           markerfacecolor='red', markersize=10)]
# plt.legend(handles=legend_elements, loc='upper right')

# plt.xlabel('First Principal Component')
# plt.ylabel('Second Principal Component')
# plt.show()

# print(f"One-Class SVM Accuracy: {accuracy_svm:.2f}%")
# print(f"Isolation Forest Accuracy: {accuracy_if:.2f}%")


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from matplotlib.animation import FuncAnimation

digits = load_digits()
X = digits.data
y = digits.target

# 정상 데이터 (0-4)와 이상 데이터 (5-9)로 나누기
normal_labels = (y < 5).astype(int)  # 0-4는 1로, 5-9는 0으로

# 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA를 사용해 2차원으로 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# One-Class SVM 
clf_svm = svm.OneClassSVM(kernel='rbf', gamma=0.01, nu=0.3)
normal_data = X_scaled[y < 5]
clf_svm.fit(normal_data)

# Isolation Forest 
clf_if = IsolationForest(contamination=0.3, random_state=42)
clf_if.fit(normal_data)

# 실시간 데이터 스트리밍 및 예측
def update(frame, X, clf_svm, clf_if, pca, scaler, ax1, ax2):
    
    # 데이터 전처리
    new_data = X[frame]
    new_data_scaled = scaler.transform([new_data])
    new_data_pca = pca.transform(new_data_scaled)
    
    # One-Class SVM 예측
    prediction_svm = clf_svm.predict(new_data_scaled)
    prediction_svm = (prediction_svm == 1).astype(int)  
    color_svm = 'blue' if prediction_svm == 1 else 'red'
    ax1.scatter(new_data_pca[0, 0], new_data_pca[0, 1], c=color_svm, edgecolor='k')

    # Isolation Forest 예측
    prediction_if = clf_if.predict(new_data_scaled)
    prediction_if = (prediction_if == 1).astype(int)  
    color_if = 'blue' if prediction_if == 1 else 'red'
    ax2.scatter(new_data_pca[0, 0], new_data_pca[0, 1], c=color_if, edgecolor='k')

    return ax1, ax2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

ax1.set_title("One-Class SVM Real-Time Anomaly Detection")
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')
colors_svm = np.array(['blue' if label == 1 else 'red' for label in normal_labels])
ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_svm, s=10, edgecolors='k', alpha=0.1)

ax2.set_title("Isolation Forest Real-Time Anomaly Detection")
ax2.set_xlabel('First Principal Component')
ax2.set_ylabel('Second Principal Component')
colors_if = np.array(['blue' if label == 1 else 'red' for label in normal_labels])
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_if, s=10, edgecolors='k', alpha=0.1)

ani = FuncAnimation(fig, update, frames=len(X), fargs=(X, clf_svm, clf_if, pca, scaler, ax1, ax2), interval=50, repeat=False)

plt.show()