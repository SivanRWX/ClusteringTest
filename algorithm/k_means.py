import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
from sklearn.preprocessing import StandardScaler  # 导入StandardScaler用于数据标准化
from sklearn.cluster import KMeans  # 导入KMeans用于K-means聚类
import seaborn as sns  # 导入seaborn用于可视化


def k_means_clustering(data, features, best_k):
    """
    执行 K-means 聚类并可视化结果。

    参数:
        data (pd.DataFrame): 输入数据。
        features (list): 用于聚类的特征列名。
        best_k (int): 最佳聚类数量。

    返回:
        kmeans_labels (np.ndarray): 每个数据点的聚类标签。
    """
    # 提取特征并标准化数据
    X = data[features].values  # 从数据中提取指定特征的值
    scaler = StandardScaler()  # 创建标准化器对象
    X_scaled = scaler.fit_transform(X)  # 对特征数据进行标准化

    # 训练 K-means 模型
    kmeans = KMeans(n_clusters=best_k, init='random', random_state=42)  # 创建K-means聚类模型，设置参数
    kmeans_labels = kmeans.fit_predict(X_scaled)  # 对标准化后的数据进行聚类，获取聚类标签

    # 可视化聚类结果
    plt.figure(figsize=(10, 6))  # 设置图像大小
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=kmeans_labels, palette='viridis', s=100)  # 绘制聚类散点图
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=300, c='red', marker='X', label='聚类中心')  # 绘制聚类中心
    plt.title(f'K-means聚类结果 (K={best_k})')  # 设置标题
    plt.xlabel('年收入 (k$)')
    plt.ylabel('消费分数 (1-100)')
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图像

    return kmeans_labels  # 返回聚类标签


def k_means_clustering_gender(data, features, best_k, gender=None):
    """
    执行 K-means 聚类并可视化结果，支持性别筛选。

    参数:
        data (pd.DataFrame): 输入数据。
        features (list): 用于聚类的特征列名。
        best_k (int): 最佳聚类数量。
        gender (str): 筛选的性别 ('Male' 或 'Female')。

    返回:
        kmeans_labels (np.ndarray): 每个数据点的聚类标签。
    """
    # 根据性别筛选数据
    if gender:
        data = data[data['Genre'] == gender]

    # 提取特征并标准化数据
    X = data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练 K-means 模型
    kmeans = KMeans(n_clusters=best_k, init='random', random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # 可视化聚类结果
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=kmeans_labels, palette='viridis', s=100)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=300, c='red', marker='X', label='聚类中心')
    plt.title(f'K-means聚类结果 (K={best_k}, 性别={gender})')  # 设置标题
    plt.xlabel('年收入 (k$)')
    plt.ylabel('消费分数 (1-100)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return kmeans_labels
