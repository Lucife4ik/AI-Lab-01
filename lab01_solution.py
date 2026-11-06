import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Для збереження графіків без відкриття вікон
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'lab01.csv')

# Файл не має заголовка; значення розділено символом ';'.
data = pd.read_csv(CSV_PATH, sep=';', header=None, names=['x', 'y'])
X = data[['x', 'y']].to_numpy(dtype=float)
if len(X) < 3:
    raise ValueError('Потрібно щонайменше три точки.')

# 1. Метод зсуву середнього
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=len(X))
if bandwidth <= 0:
    raise ValueError('Не вдалося оцінити ширину вікна Mean Shift.')
mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms_labels = mean_shift.fit_predict(X)
ms_centers = mean_shift.cluster_centers_
print(f'Mean Shift: bandwidth = {bandwidth:.3f}; кластерів = {len(ms_centers)}')

# 2. Silhouette Score для k від 2 до 15
scores = {}
for k in range(2, min(15, len(X) - 1) + 1):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    if 2 <= len(np.unique(labels)) < len(X):
        scores[k] = silhouette_score(X, labels)

if not scores:
    raise ValueError('Не вдалося обчислити Silhouette Score.')
optimal_k = max(scores, key=scores.get)
print('Silhouette Score:')
for k, score in scores.items():
    print(f'  k = {k:2d}: {score:.4f}')
print(f'Оптимальна кількість кластерів: {optimal_k}')

# 3. K-Means з оптимальним k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X)
km_centers = kmeans.cluster_centers_

# 4. Чотири рисунки
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='steelblue', edgecolors='black', s=45)
plt.title('1. Вихідні точки')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '01_original_points.png'), dpi=160)
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=ms_labels, cmap='tab20', s=45, edgecolors='black')
plt.scatter(ms_centers[:, 0], ms_centers[:, 1], c='red', marker='X', s=240,
            edgecolors='black', label='Центри Mean Shift')
plt.title(f'2. Центри кластерів Mean Shift (k={len(ms_centers)})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '02_mean_shift_centers.png'), dpi=160)
plt.close()

plt.figure(figsize=(9, 5))
bars = plt.bar(list(scores.keys()), list(scores.values()), color='steelblue')
bars[list(scores.keys()).index(optimal_k)].set_color('tomato')
plt.xticks(list(scores.keys()))
plt.xlabel('Кількість кластерів (k)')
plt.ylabel('Silhouette Score')
plt.title(f'3. Silhouette Score (оптимальне k={optimal_k})')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '03_silhouette_scores.png'), dpi=160)
plt.close()

padding = max(np.ptp(X, axis=0)) * 0.08 + 1
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - padding, X[:, 0].max() + padding, 400),
    np.linspace(X[:, 1].min() - padding, X[:, 1].max() + padding, 400)
)
region_labels = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, region_labels, levels=np.arange(optimal_k + 1) - 0.5,
             cmap='tab20', alpha=0.25)
plt.scatter(X[:, 0], X[:, 1], c=km_labels, cmap='tab20', s=48, edgecolors='black')
plt.scatter(km_centers[:, 0], km_centers[:, 1], c='red', marker='X', s=220,
            edgecolors='black', label='Центри K-Means')
plt.title(f'4. K-Means: точки та області кластеризації (k={optimal_k})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '04_kmeans_regions.png'), dpi=160)
plt.close()
print('Чотири рисунки збережено в папці програми.')
