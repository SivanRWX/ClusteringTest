import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
import numpy as np
from sklearn.preprocessing import StandardScaler  # 导入StandardScaler用于数据标准化
from sklearn.cluster import AgglomerativeClustering  # 导入AgglomerateClustering用于层次聚类
import seaborn as sns  # 导入seaborn用于可视化



# def hierarchical_clustering(data, features, method, n_clusters):
#     """
#     优化后的手动实现层次聚类算法，降低时间复杂度
#
#     参数:
#         data (pd.DataFrame): 输入数据
#         features (list): 用于聚类的特征列名
#         method (str): 聚类方法（未使用）
#         n_clusters (int): 期望的簇数量
#
#     返回:
#         labels (np.ndarray): 聚类标签
#     """
#     # 提取特征并标准化数据
#     X = data[features].values
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # 使用向量化操作计算距离矩阵
#     n_samples = X_scaled.shape[0]
#     distances = np.zeros((n_samples, n_samples))
#     for i in range(n_samples):
#         diff = X_scaled - X_scaled[i]
#         distances[i] = np.sqrt(np.sum(diff * diff, axis=1))
#
#     # 初始化簇和合并历史
#     current_clusters = {i: np.array([i]) for i in range(n_samples)}
#     next_cluster_id = n_samples
#     merge_history = []
#     heights = []
#
#     # 优化的簇间距离计算
#     def cluster_distance(ci, cj):
#         # 使用向量化操作计算簇间最小距离
#         dist_matrix = distances[ci][:, cj]
#         return np.min(dist_matrix)
#
#     # 主循环：合并簇直到达到目标簇数
#     while len(current_clusters) > n_clusters:
#         min_dist = float('inf')
#         merge_i, merge_j = None, None
#
#         # 找到最近的两个簇
#         cluster_ids = list(current_clusters.keys())
#         for i in range(len(cluster_ids)):
#             for j in range(i + 1, len(cluster_ids)):
#                 ci, cj = cluster_ids[i], cluster_ids[j]
#                 dist = cluster_distance(current_clusters[ci], current_clusters[cj])
#                 if dist < min_dist:
#                     min_dist = dist
#                     merge_i, merge_j = ci, cj
#
#         # 记录合并历史
#         merge_history.append((merge_i, merge_j))
#         heights.append(min_dist)
#
#         # 合并簇
#         new_cluster = np.concatenate([current_clusters[merge_i], current_clusters[merge_j]])
#         current_clusters[next_cluster_id] = new_cluster
#         del current_clusters[merge_i]
#         del current_clusters[merge_j]
#         next_cluster_id += 1
#
#     # 生成最终标签
#     labels = np.zeros(n_samples, dtype=int)
#     for i, (_, cluster) in enumerate(current_clusters.items()):
#         labels[cluster] = i
#
#     # 可视化聚类结果
#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
#                          c=labels, cmap='viridis', s=100)
#     plt.colorbar(scatter)
#     plt.title(f'优化后的层次聚类结果 (簇数量={n_clusters})')
#     plt.xlabel('标准化年收入')
#     plt.ylabel('标准化消费分数')
#     plt.grid(True)
#     plt.show()
#
#     return labels

def hierarchical_clustering(data, features, method, n_clusters):
    """
    执行层次聚类并可视化结果。

    参数:
        data (pd.DataFrame): 输入数据。
        features (list): 用于聚类的特征列名。
        method (str): 层次聚类的链接方法。
        n_clusters (int): 最终划分的簇数量。

    返回:
        labels (np.ndarray): 每个数据点的聚类标签。
    """
    # 提取特征并标准化数据
    X = data[features].values  # 从数据中提取指定特征的值
    scaler = StandardScaler()  # 创建标准化器对象
    X_scaled = scaler.fit_transform(X)  # 对特征数据进行标准化

    # # 生成层次聚类的链接矩阵
    # linkage_matrix = linkage(X_scaled, method=method)  # 计算层次聚类的链接矩阵
    #
    # # 绘制树状图
    # plt.figure(figsize=(20, 9))  # 设置图像大小
    # dendrogram(linkage_matrix)  # 绘制树状图
    # plt.title(f'层次聚类树状图 ({method} 方法)')  # 设置标题
    # plt.xlabel('样本索引')  # 设置x轴标签
    # plt.ylabel('距离')  # 设置y轴标签
    # plt.grid(True)  # 显示网格
    # plt.show()  # 显示图像

    # 训练层次聚类模型
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)  # 创建层次聚类模型
    labels = model.fit_predict(X_scaled)  # 获取聚类标签

    # 计算聚类性能指标
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # 排除噪声点
    print(f"聚类簇数量: {len(unique_labels)}")
    print(f"噪声点数量: {np.sum(labels == -1)}")  # 计算噪声点数量

    # 可视化聚类结果
    plt.figure(figsize=(10, 6))  # 设置图像大小
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels, palette='viridis', s=100)  # 绘制聚类散点图
    plt.title(f'层次聚类结果 (簇数量={n_clusters}, 方法={method})')  # 设置标题
    plt.xlabel('年收入 (k$)')
    plt.ylabel('消费分数 (1-100)')
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图像

    return labels  # 返回聚类标签


def hierarchical_clustering_gender(data, features, method, n_clusters, gender=None):
    """
    执行层次聚类并可视化结果，支持性别筛选。

    参数:
        data (pd.DataFrame): 输入数据。
        features (list): 用于聚类的特征列名。
        method (str): 层次聚类的链接方法。
        n_clusters (int): 最终划分的簇数量。
        gender (str): 筛选的性别 ('Male' 或 'Female')。

    返回:
        labels (np.ndarray): 每个数据点的聚类标签。
    """
    # 根据性别筛选数据
    if gender:
        data = data[data['Genre'] == gender]

    # 提取特征并标准化数据
    X = data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # # 生成层次聚类的链接矩阵
    # linkage_matrix = linkage(X_scaled, method=method)
    #
    # # 绘制树状图
    # plt.figure(figsize=(20, 9))
    # dendrogram(linkage_matrix)
    # plt.title(f'层次聚类结果 (簇数量={n_clusters}, 方法={method}),性别={gender}')  # 设置标题
    # plt.xlabel('年收入 (k$)')
    # plt.ylabel('消费分数 (1-100)')
    # plt.grid(True)
    # plt.show()

    # 训练层次聚类模型
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    labels = model.fit_predict(X_scaled)

    # 可视化聚类结果
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels, palette='viridis', s=100)
    plt.title(f'层次聚类结果 (簇数量={n_clusters}, 方法={method}),性别={gender}')  # 设置标题
    plt.xlabel('年收入 (k$)')
    plt.ylabel('消费分数 (1-100)')
    plt.grid(True)
    plt.show()

    return labels
