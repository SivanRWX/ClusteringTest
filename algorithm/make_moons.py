import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# 生成两个半月形数据集
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# 创建与 Mall_Customers.csv 相同的结构
data = pd.DataFrame({
    'CustomerID': range(1, 201),
    'Genre': np.random.choice(['Male', 'Female'], size=200),  # 随机生成性别
    'Age': np.random.randint(18, 70, size=200),  # 随机生成年龄
    'Annual Income (k$)': (X[:, 0] * 50 + 50).astype(int),  # 将第一个维度映射为年收入
    'Spending Score (1-100)': (X[:, 1] * 50 + 50).astype(int)  # 将第二个维度映射为消费分数
})

# 保存为 CSV 文件
data.to_csv('Half_Moons_Customers.csv', index=False)

# 可视化生成的二维数据
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
plt.title('Two Half-Moons Dataset')
plt.xlabel('Feature 1 (Mapped to Annual Income)')
plt.ylabel('Feature 2 (Mapped to Spending Score)')
plt.grid(True)
plt.show()