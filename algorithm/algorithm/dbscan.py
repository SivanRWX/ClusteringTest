import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
import numpy as np
from sklearn.preprocessing import StandardScaler  # 导入StandardScaler用于数据标准化
from sklearn.cluster import DBSCAN  # 导入DBSCAN用于密度聚类
import seaborn as sns  # 导入seaborn用于可视化


def dbscan_clustering(data, features, eps, min_samples):
    """
    手动实现DBSCAN聚类算法并可视化结果。需要计算距离矩阵

    参数:
        data (pd.DataFrame): 输入数据
        features (list): 用于聚类的特征列名
        eps (float): 邻域半径参数
        min_samples (int): 最小样本数参数

    返回:
        labels (np.ndarray): 聚类标签
    """
    # 提取特征并标准化数据
    X = data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 计算距离矩阵
    n_samples = X_scaled.shape[0]
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            distances[i, j] = np.sqrt(np.sum((X_scaled[i] - X_scaled[j])**2))
    
    # 初始化标签 (-1表示未分类点)
    labels = np.full(n_samples, -1)
    cluster_id = 0
    
    def find_neighbors(point):
        """找出给定点的邻域内的所有点的索引"""
        return np.where(distances[point] <= eps)[0]
    
    # 遍历所有点
    for point_idx in range(n_samples):
        # 跳过已经分类的点
        if labels[point_idx] != -1:
            continue
            
        neighbors = find_neighbors(point_idx)
        
        # 如果不是核心点，继续下一个点
        if len(neighbors) < min_samples:
            continue
            
        # 发现一个新簇
        cluster_id += 1
        labels[point_idx] = cluster_id
        
        # 扩展簇
        seed_queue = list(neighbors)
        while seed_queue:
            current_point = seed_queue.pop(0)
            
            # 如果是噪声点，将其加入当前簇
            if labels[current_point] == -1:
                labels[current_point] = cluster_id
                
                # 如果是核心点，将其未分类邻居加入队列
                current_neighbors = find_neighbors(current_point)
                if len(current_neighbors) >= min_samples:
                    for neighbor in current_neighbors:
                        if labels[neighbor] == -1 and neighbor not in seed_queue:
                            seed_queue.append(neighbor)
    
    # 计算聚类结果统计
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"聚类簇数量: {n_clusters}")
    print(f"噪声点数量: {np.sum(labels == -1)}")

    # 可视化聚类结果
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels, palette='viridis', s=100)
    plt.title(f'DBSCAN聚类结果 (ε={eps}, 最小样本数={min_samples})')  # 设置标题
    plt.xlabel('年收入 (k$)')
    plt.ylabel('消费分数 (1-100)')
    plt.grid(True)
    plt.show()
    
    return labels

# def dbscan_clustering(data, features, eps, min_samples):
#     """
#     执行 DBSCAN 聚类并可视化结果。

#     参数:
#         data (pd.DataFrame): 输入数据。
#         features (list): 用于聚类的特征列名。
#         eps (float): DBSCAN 的 ε 参数。
#         min_samples (int): DBSCAN 的 min_samples 参数。

#     返回:
#         dbscan_labels (np.ndarray): 每个数据点的聚类标签。
#     """
#     # 提取特征并标准化数据
#     X = data[features].values  # 从数据中提取指定特征的值
#     scaler = StandardScaler()  # 创建标准化器对象
#     X_scaled = scaler.fit_transform(X)  # 对特征数据进行标准化

#     # 训练 DBSCAN 模型
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # 创建DBSCAN聚类模型，设置参数
#     dbscan_labels = dbscan.fit_predict(X_scaled)  # 对标准化后的数据进行聚类，获取聚类标签

#     # 计算聚类性能指标
#     unique_labels = np.unique(dbscan_labels)
#     n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # 排除噪声点 (-1 表示噪声点)
#     print(f"聚类簇数量: {n_clusters}")
#     print(f"噪声点数量: {np.sum(dbscan_labels == -1)}")

#     # 可视化聚类结果
#     plt.figure(figsize=(10, 6))  # 设置图像大小
#     palette = sns.color_palette("viridis", n_colors=n_clusters)
#     # 使用 sns.color_palette 动态生成颜色，确保簇编号与颜色一致。
#     sns.scatterplot(
#         x=X_scaled[:, 0], y=X_scaled[:, 1], hue=dbscan_labels, palette=palette + [(0.5, 0.5, 0.5)], s=100
#     )  # 绘制聚类散点图
#     plt.title(f'DBSCAN聚类结果 (ε={eps}, 最小样本数={min_samples})')  # 设置标题
#     plt.xlabel('年收入 (k$)')
#     plt.ylabel('消费分数 (1-100)')
#     plt.grid(True)  # 显示网格
#     plt.show()  # 显示图像

#     return dbscan_labels  # 返回聚类标签


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
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=dbscan_labels, palette='viridis', s=100)
    plt.title(f'DBSCAN聚类结果 (ε={eps}, 最小样本数={min_samples}),性别={gender}')  # 设置标题
    plt.xlabel('年收入 (k$)')
    plt.ylabel('消费分数 (1-100)')
    plt.grid(True)
    plt.show()

    return dbscan_labels
