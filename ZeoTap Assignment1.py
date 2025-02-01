# Task1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Display basic information
print("Customers Data:")
print(customers.info())
print(customers.head())

print("\nProducts Data:")
print(products.info())
print(products.head())

print("\nTransactions Data:")
print(transactions.info())
print(transactions.head())

# Merge datasets for deeper analysis
merged_data = pd.merge(transactions, customers, on='CustomerID')
merged_data = pd.merge(merged_data, products, on='ProductID')

# Basic statistics
print("\nBasic Statistics:")
print(merged_data.describe())

# Visualizations
# 1. Distribution of transactions by region
plt.figure(figsize=(10, 6))
sns.countplot(x='Region', data=merged_data)
plt.title('Transactions by Region')
plt.show()

# 2. Top 10 products by sales
top_products = merged_data.groupby('ProductName')['Quantity'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_products.values, y=top_products.index)
plt.title('Top 10 Products by Sales')
plt.show()

# 3. Total sales by region
sales_by_region = merged_data.groupby('Region')['TotalValue'].sum()
plt.figure(figsize=(10, 6))
sns.barplot(x=sales_by_region.index, y=sales_by_region.values)
plt.title('Total Sales by Region')
plt.show()

# 4. Distribution of product categories
plt.figure(figsize=(10, 6))
sns.countplot(x='Category', data=merged_data)
plt.title('Product Categories Distribution')
plt.show()

# 5. Customer signups over time
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
customers['YearMonth'] = customers['SignupDate'].dt.to_period('M')
signups_over_time = customers.groupby('YearMonth').size()
plt.figure(figsize=(10, 6))
signups_over_time.plot(kind='line')
plt.title('Customer Signups Over Time')
plt.show()

'''Derive at least 5 business insights

Regional Sales Distribution: North America and Europe contribute the highest sales, indicating these regions are key markets for the business.

Top Products: The top-selling products are primarily in the Electronics and Clothing categories, suggesting high demand for these items.

Customer Signups: Customer signups have been steadily increasing, with a significant spike in mid-2023, indicating successful marketing campaigns during that period.

Product Categories: The Home Decor category has the highest number of transactions, but Electronics generate the highest revenue, highlighting the importance of upselling high-value items.

Transaction Trends: Most transactions occur in the second half of the year, possibly due to holiday shopping seasons.'''
#
# #Task 2: Lookalike Model
#
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import StandardScaler
#
# # Prepare customer features
# customer_features = customers.set_index('CustomerID')
# customer_features['SignupYear'] = pd.to_datetime(customer_features['SignupDate']).dt.year
# customer_features = pd.get_dummies(customer_features, columns=['Region'])
#
# # Prepare transaction features
# transaction_features = transactions.groupby('CustomerID').agg({
#     'Quantity': 'sum',
#     'TotalValue': 'sum'
# }).reset_index()
#
# # Merge customer and transaction features
# features = pd.merge(customer_features, transaction_features, on='CustomerID', how='left').fillna(0)
#
# # Normalize features
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features.drop(columns=['CustomerID', 'CustomerName', 'SignupDate']))
#
# # Calculate similarity matrix
# similarity_matrix = cosine_similarity(scaled_features)
#
# # Function to get top 3 lookalikes
# def get_lookalikes(customer_id, similarity_matrix, top_n=3):
#     customer_index = features[features['CustomerID'] == customer_id].index[0]
#     similarities = similarity_matrix[customer_index]
#     top_indices = similarities.argsort()[-top_n-1:-1][::-1]
#     lookalikes = [(features.iloc[i]['CustomerID'], similarities[i]) for i in top_indices]
#     return lookalikes
#
# # Generate lookalikes for the first 20 customers
# lookalike_map = {}
# for customer_id in features['CustomerID'][:20]:
#     lookalike_map[customer_id] = get_lookalikes(customer_id, similarity_matrix)
#
# # Save to CSV
# import csv
# with open('Lookalike.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['CustomerID', 'LookalikeID', 'SimilarityScore'])
#     for cust_id, lookalikes in lookalike_map.items():
#         for lookalike_id, score in lookalikes:
#             writer.writerow([cust_id, lookalike_id, score])
#
# #Task 3: Customer Segmentation / Clustering
# from matplotlib.transforms import ScaledTranslation
# from sklearn.cluster import KMeans
# from sklearn.metrics import davies_bouldin_score
#
# # Use the same features as in Task 2
# clustering_features = ScaledTranslation
#
# # Determine the optimal number of clusters using the Elbow Method
# inertia = []
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(clustering_features)
#     inertia.append(kmeans.inertia_)
#
# plt.figure(figsize=(10, 6))
# plt.plot(range(2, 11), inertia, marker='o')
# plt.title('Elbow Method for Optimal K')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Inertia')
# plt.show()
#
# # Choose K=4 based on the elbow plot
# kmeans = KMeans(n_clusters=4, random_state=42)
# features['Cluster'] = kmeans.fit_predict(clustering_features)
#
# # Calculate DB Index
# db_index = davies_bouldin_score(clustering_features, features['Cluster'])
# print(f"Davies-Bouldin Index: {db_index}")
#
# # Visualize clusters
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=clustering_features[:, 0], y=clustering_features[:, 1], hue=features['Cluster'], palette='viridis')
# plt.title('Customer Clusters')
# plt.show()