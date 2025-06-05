import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
from sklearn.preprocessing import StandardScaler  # 导入StandardScaler用于数据标准化
from sklearn.cluster import KMeans  # 导入KMeans用于K-means聚类
import seaborn as sns  # 导入seaborn用于可视化
import numpy as np  # 导入numpy用于数值计算


def k_means_clustering(data, features, best_k):
    """
    手动实现 K-means 聚类并可视化结果。

    参数:
        data (pd.DataFrame): 输入数据。
        features (list): 用于聚类的特征列名。
        best_k (int): 最佳聚类数量。

    返回:
        labels (np.ndarray): 每个数据点的聚类标签。
    """
    # 提取特征数据

    global kmeans_labels
    X = data[features].values

    # 初始化聚类中心（随机选择K个点作为初始中心）
    np.random.seed(42)
    centers = X[np.random.choice(X.shape[0], best_k, replace=False)]

    # 迭代更新聚类中心
    for _ in range(100):  # 最大迭代次数
        # 计算每个点到各聚类中心的距离
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)

        # 分配每个点到最近的聚类中心
        kmeans_labels = np.argmin(distances, axis=1)

        # 计算新的聚类中心
        new_centers = np.array([X[kmeans_labels == i].mean(axis=0) for i in range(best_k)])

        # 如果聚类中心不再变化，则停止迭代
        if np.all(centers == new_centers):
            break
        centers = new_centers

    # 可视化聚类结果
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=kmeans_labels, palette='viridis', s=100)
    plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='X', label='聚类中心')
    plt.title(f'K-means聚类结果 (K={best_k})')
    plt.xlabel('年收入 (k$)')
    plt.ylabel('消费分数 (1-100)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return kmeans_labels


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

    # 检查数据是否为空
    if data.empty:
        raise ValueError("输入数据为空，请检查数据加载或筛选条件。")

    # 根据性别筛选数据
    if gender:
        data = data[data['Genre'] == gender]

    # 提取特征并标准化数据
    X = data[features].values
    if np.all(X == 0):
        raise ValueError("特征值全为零，请检查数据预处理步骤。")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 检查标准化后的数据是否异常
    if np.all(X_scaled == 0):
        raise ValueError("标准化后的数据全为零，请检查数据分布或标准化逻辑。")

    # 训练 K-means 模型
    kmeans = KMeans(n_clusters=best_k, init='random', random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # 可视化聚类结果
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=kmeans_labels, palette='viridis', s=100)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=300, c='red', marker='X', label='聚类中心')
    plt.title(f'K-means聚类结果 (K={best_k}, 性别={gender})')  # 设置标题
    plt.xlabel('年收入 (k$)')
    plt.ylabel('消费分数 (1-100)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return kmeans_labels
