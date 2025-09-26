import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. 读取图像
img = cv2.imread("data/white_bankuai.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. 提取红色区域（用阈值）
# 转 HSV 更好分离红色
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

# 提取红色区域坐标
coords = np.column_stack(np.where(mask > 0))

# 3. 聚类 (KMeans, n_clusters=2)
kmeans = KMeans(n_clusters=2, random_state=0).fit(coords)
labels = kmeans.labels_

# 4. 可视化
plt.figure(figsize=(4, 12))
plt.imshow(img_rgb)
plt.scatter(coords[:,1], coords[:,0], c=labels, cmap="coolwarm", s=5)
plt.axis("off")
plt.show()
