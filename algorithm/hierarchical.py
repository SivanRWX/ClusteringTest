import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
import numpy as np
from sklearn.preprocessing import StandardScaler  # 导入StandardScaler用于数据标准化
from sklearn.cluster import AgglomerativeClustering  # 导入AgglomerateClustering用于层次聚类
import seaborn as sns  # 导入seaborn用于可视化
from scipy.cluster.hierarchy import dendrogram, linkage  # 导入dendrogram和linkage用于绘制树状图和计算链接矩阵


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
    print(f"聚类簇数量: {len(unique_labels)}")

    # 可视化聚类结果
    plt.figure(figsize=(10, 6))  # 设置图像大小
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', s=100)  # 绘制聚类散点图
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
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', s=100)
    plt.title(f'层次聚类结果 (簇数量={n_clusters}, 方法={method}),性别={gender}')  # 设置标题
    plt.xlabel('年收入 (k$)')
    plt.ylabel('消费分数 (1-100)')
    plt.grid(True)
    plt.show()

    return labels
