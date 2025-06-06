import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate spherical clusters
X, y = make_blobs(n_samples=200, centers=5, cluster_std=1.5, random_state=42)

# Create a dataset with the same structure as Mall_Customers.csv
data = pd.DataFrame({
    'CustomerID': range(1, 201),
    'Genre': np.random.choice(['Male', 'Female'], size=200),  # Randomly generate gender
    'Age': np.random.randint(18, 70, size=200),  # Randomly generate age
    'Annual Income (k$)': (X[:, 0] * 10 + 50).astype(int),  # Map first dimension to annual income
    'Spending Score (1-100)': (X[:, 1] * 10 + 50).astype(int)  # Map second dimension to spending score
})

# Save the dataset to a CSV file
data.to_csv('Spherical_Clusters_Customers.csv', index=False)

# Visualize the generated data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
plt.title('Spherical Clusters Dataset')
plt.xlabel('Feature 1 (Mapped to Annual Income)')
plt.ylabel('Feature 2 (Mapped to Spending Score)')
plt.grid(True)
plt.show()