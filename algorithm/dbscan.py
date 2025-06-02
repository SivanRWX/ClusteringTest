import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
import numpy as np
from sklearn.preprocessing import StandardScaler  # 导入StandardScaler用于数据标准化
from sklearn.cluster import DBSCAN  # 导入DBSCAN用于密度聚类
import seaborn as sns  # 导入seaborn用于可视化


def dbscan_clustering(data, features, eps, min_samples):
    """
    执行 DBSCAN 聚类并可视化结果。

    参数:
        data (pd.DataFrame): 输入数据。
        features (list): 用于聚类的特征列名。
        eps (float): DBSCAN 的 ε 参数。
        min_samples (int): DBSCAN 的 min_samples 参数。

    返回:
        dbscan_labels (np.ndarray): 每个数据点的聚类标签。
    """
    # 提取特征并标准化数据
    X = data[features].values  # 从数据中提取指定特征的值
    scaler = StandardScaler()  # 创建标准化器对象
    X_scaled = scaler.fit_transform(X)  # 对特征数据进行标准化

    # 训练 DBSCAN 模型
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # 创建DBSCAN聚类模型，设置参数
    dbscan_labels = dbscan.fit_predict(X_scaled)  # 对标准化后的数据进行聚类，获取聚类标签

    # 计算聚类性能指标
    unique_labels = np.unique(dbscan_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # 排除噪声点
    print(f"聚类簇数量: {n_clusters}")
    print(f"噪声点数量: {np.sum(dbscan_labels == -1)}")

    # 可视化聚类结果
    plt.figure(figsize=(10, 6))  # 设置图像大小
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=dbscan_labels, palette='viridis', s=100)  # 绘制聚类散点图
    plt.title(f'DBSCAN聚类结果 (ε={eps}, 最小样本数={min_samples})')  # 设置标题
    plt.xlabel('年收入 (k$)')
    plt.ylabel('消费分数 (1-100)')
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图像

    return dbscan_labels  # 返回聚类标签


def dbscan_clustering_gender(data, features, eps, min_samples, gender=None):
    """
    执行 DBSCAN 聚类并可视化结果，支持性别筛选。

    参数:
        data (pd.DataFrame): 输入数据。
        features (list): 用于聚类的特征列名。
        eps (float): DBSCAN 的 ε 参数。
        min_samples (int): DBSCAN 的 min_samples 参数。
        gender (str): 筛选的性别 ('Male' 或 'Female')。

    返回:
        dbscan_labels (np.ndarray): 每个数据点的聚类标签。
    """
    # 根据性别筛选数据
    if gender:
        data = data[data['Genre'] == gender]

    # 提取特征并标准化数据
    X = data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练 DBSCAN 模型
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    # 可视化聚类结果
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=dbscan_labels, palette='viridis', s=100)
    plt.title(f'DBSCAN聚类结果 (ε={eps}, 最小样本数={min_samples}),性别={gender}')  # 设置标题
    plt.xlabel('年收入 (k$)')
    plt.ylabel('消费分数 (1-100)')
    plt.grid(True)
    plt.show()

    return dbscan_labels
