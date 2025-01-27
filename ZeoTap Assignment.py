import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: EDA and Business Insights
data=pd.read_csv('Customers.csv')
print("Basic Info:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Visualizations
plt.figure(figsize=(10, 6))
plt.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# # Distribution of numerical columns
# data.hist(bins=20, figsize=(15, 10), edgecolor='black')
# plt.suptitle("Numerical Feature Distributions")
# plt.show()
#
# # Pair plot for numerical columns
# sns.pairplot(data.select_dtypes(include=['float64', 'int64']))
# plt.title("Pair Plot")
# plt.show()

#
# # Task 2: Lookalike Model
# def lookalike_model(customers, transactions):
#     # Merge datasets
#     data = customers.merge(transactions, on="CustomerID", how="left")
#
#     # Feature engineering
#     customer_features = data.groupby('CustomerID').agg({
#         'TransactionAmount': ['sum', 'mean', 'count'],
#         'ProductCategory': 'nunique'
#     }).reset_index()
#     customer_features.columns = ['CustomerID', 'TotalSpend', 'AvgSpend', 'TransactionCount', 'UniqueCategories']
#
#     # Normalize data
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(customer_features.iloc[:, 1:])
#
#     # Compute similarity
#     similarity_matrix = cosine_similarity(features_scaled)
#
#     # Generate top 3 lookalikes for each customer
#     lookalikes = {}
#     for idx, customer_id in enumerate(customer_features['CustomerID']):
#         similar_indices = np.argsort(similarity_matrix[idx])[-4:-1][::-1]  # Top 3 similar excluding self
#         lookalikes[customer_id] = [
#             (customer_features['CustomerID'][i], similarity_matrix[idx][i]) for i in similar_indices
#         ]
#
#     # Save to Lookalike.csv
#     lookalike_data = [
#         {"CustomerID": cust, "Lookalikes": lookalike}
#         for cust, lookalike in lookalikes.items()
#     ]
#     lookalike_df = pd.DataFrame(lookalike_data)
#     lookalike_df.to_csv("Lookalike.csv", index=False)
#     print("Lookalike Model Results Saved as Lookalike.csv")
#
#
# # Task 3: Customer Segmentation / Clustering
# def customer_clustering(customers, transactions):
#     # Merge datasets
#     data = customers.merge(transactions, on="CustomerID", how="left")
#
#     # Feature engineering
#     customer_features = data.groupby('CustomerID').agg({
#         'TransactionAmount': ['sum', 'mean', 'count'],
#         'ProductCategory': 'nunique'
#     }).reset_index()
#     customer_features.columns = ['CustomerID', 'TotalSpend', 'AvgSpend', 'TransactionCount', 'UniqueCategories']
#
#     # Normalize data
#     scaler = MinMaxScaler()
#     features_scaled = scaler.fit_transform(customer_features.iloc[:, 1:])
#
#     # KMeans Clustering
#     kmeans = KMeans(n_clusters=4, random_state=42)
#     cluster_labels = kmeans.fit_predict(features_scaled)
#
#     # Append cluster labels
#     customer_features['Cluster'] = cluster_labels
#
#     # Davies-Bouldin Index
#     db_index = davies_bouldin_score(features_scaled, cluster_labels)
#     print(f"Davies-Bouldin Index: {db_index}")
#
#     # Silhouette Score
#     silhouette_avg = silhouette_score(features_scaled, cluster_labels)
#     print(f"Silhouette Score: {silhouette_avg}")
#
#     # Cluster Visualization
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(
#         x=features_scaled[:, 0], y=features_scaled[:, 1],
#         hue=cluster_labels, palette="viridis", legend="full"
#     )
#     plt.title("Customer Clustering Visualization")
#     plt.show()
#
#
# # Main Script Execution
# if __name__ == "__main__":
#     # Load datasets
#     customers = pd.read_csv("Customers.csv")
#     transactions = pd.read_csv("Transactions.csv")
#
#     # Task 1: Perform EDA
#     print("Performing EDA...")
#     perform_eda(customers)
#
#     # Task 2: Lookalike Model
#     print("Building Lookalike Model...")
#     lookalike_model(customers, transactions)
#
#     # Task 3: Customer Segmentation
#     print("Performing Customer Segmentation...")
#     customer_clustering(customers, transactions)
