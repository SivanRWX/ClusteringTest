import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from algorithm.dbscan import dbscan_clustering
from algorithm.k_means import k_means_clustering
from algorithm.hierarchical import hierarchical_clustering
from algorithm.k_means_plus import k_means_plus_clustering
from algorithm.dbscan import dbscan_clustering_gender
from algorithm.k_means import k_means_clustering_gender
from algorithm.hierarchical import hierarchical_clustering_gender


def rc():
    """基础图像与字体设置"""
    # plt.rcParams是一个字典，它存储了matplotlib的配置参数，用于控制图形的外观和行为
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    # 默认使用Unicode字符U+2212，可导致保存图像时的乱码问题。设置为False则用ASCII字符U+002D，避免混淆并解决保存问题
    plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


def rawdata():
    """可视化原始数据"""
    # 选择特征：年收入和消费分数
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

    # 数据标准化
    scaler = StandardScaler()
    scaler.fit_transform(X)

    # 可视化原始数据
    plt.figure(figsize=(10, 6))
    # plt.subplot(221)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], color='blue')
    plt.title('原始数据分布')
    plt.xlabel('年收入 (k$)')
    plt.ylabel('消费分数 (1-100)')
    plt.grid(True)


if __name__ == "__main__":
    rc()
    # 加载数据集
    data = pd.read_csv('Mall_Customers.csv', index_col='CustomerID')
    print(data.head(n=200))  # 打印数据集
    data.Genre.value_counts()  # 统计性别分布

    # 原始数据
    rawdata()

    # k-means 聚类
    # 定义特征和最佳 K 值
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    best_k = 5
    # 调用 K-means 聚类方法
    k_means_clustering(data, features, best_k)
    k_means_clustering_gender(data, features, best_k, gender='Male')
    k_means_clustering_gender(data, features, best_k, gender='Female')
    print("已完成 K-means 聚类")

    # k-means++ 聚类
    # 定义特征和最佳 K 值
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    best_k = 5
    # 调用 k-means++ 聚类方法
    k_means_plus_clustering(data, features, best_k)
    print("已完成 K-means++ 聚类")

    # DBSCAN 聚类
    # 定义特征和 DBSCAN 参数
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    eps = 0.3
    min_samples = 5
    # 调用 DBSCAN 聚类方法
    dbscan_clustering(data, features, eps, min_samples)
    dbscan_clustering_gender(data, features, eps, min_samples, gender='Male')
    dbscan_clustering_gender(data, features, eps, min_samples, gender='Female')
    print("已完成 DBSCAN 聚类")

    # 层次 聚类
    # 定义特征和层次聚类参数
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    method = 'ward'  # 基于最小化簇内方差
    n_clusters = 5
    # 调用层次聚类方法
    hierarchical_clustering(data, features, method, n_clusters)
    hierarchical_clustering_gender(data, features, method, n_clusters, gender='Male')
    hierarchical_clustering_gender(data, features, method, n_clusters, gender='Female')
    print("已完成层次聚类")

    # 月牙型簇分析
    # 加载数据集
    data = pd.read_csv('Half_Moons_Customers.csv', index_col='CustomerID')
    print(data.head(n=200))  # 打印数据集
    data.Genre.value_counts()  # 统计性别分布

    # 原始数据
    rawdata()

    # k-means 聚类
    # 定义特征和最佳 K 值
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    best_k = 5
    # 调用 K-means 聚类方法
    k_means_clustering(data, features, best_k)
    print("已完成 K-means 聚类")

    # k-means++ 聚类
    # 定义特征和最佳 K 值
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    best_k = 5

    # 调用 k-means++ 聚类方法
    k_means_plus_clustering(data, features, best_k)
    print("已完成 K-means++ 聚类")

    # DBSCAN 聚类
    # 定义特征和 DBSCAN 参数
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    eps = 0.3
    min_samples = 5
    # 调用 DBSCAN 聚类方法
    dbscan_clustering(data, features, eps, min_samples)
    print("已完成 DBSCAN 聚类")

    # 层次 聚类
    # 定义特征和层次聚类参数
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    method = 'ward'  # 基于最小化簇内方差
    n_clusters = 5
    # 调用层次聚类方法
    hierarchical_clustering(data, features, method, n_clusters)
    print("已完成层次聚类")

    # 球形簇型分析
    # 加载数据集
    data = pd.read_csv('Spherical_Clusters_Customers.csv', index_col='CustomerID')
    print(data.head(n=200))  # 打印数据集
    data.Genre.value_counts()  # 统计性别分布

    # 原始数据
    rawdata()

    # k-means 聚类
    # 定义特征和最佳 K 值
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    best_k = 5
    # 调用 K-means 聚类方法
    k_means_clustering(data, features, best_k)
    print("已完成 K-means 聚类")

    # k-means++ 聚类
    # 定义特征和最佳 K 值
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    best_k = 5

    # 调用 k-means++ 聚类方法
    k_means_plus_clustering(data, features, best_k)
    print("已完成 K-means++ 聚类")

    # DBSCAN 聚类
    # 定义特征和 DBSCAN 参数
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    eps = 0.3
    min_samples = 5
    # 调用 DBSCAN 聚类方法
    dbscan_clustering(data, features, eps, min_samples)
    print("已完成 DBSCAN 聚类")

    # 层次 聚类
    # 定义特征和层次聚类参数
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    method = 'ward'  # 基于最小化簇内方差
    n_clusters = 5
    # 调用层次聚类方法
    hierarchical_clustering(data, features, method, n_clusters)
    print("已完成层次聚类")
