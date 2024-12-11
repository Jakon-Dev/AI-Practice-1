import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ELECTRICITY_CONSUMPTION_DATA = "dataset/electricity_consumption.parquet"

file_path = ELECTRICITY_CONSUMPTION_DATA

data = pd.read_parquet(file_path)

data['time'] = pd.to_datetime(data['time'])
data['date'] = data['time'].dt.date
data['hour'] = data['time'].dt.hour

pivot_df = data.pivot_table(index=['postalcode', 'date'], columns='hour', values='consumption', aggfunc='sum').reset_index()

pivot_df = pivot_df.dropna().reset_index(drop=True)
load_curves = pivot_df.drop(columns=['postalcode', 'date'])

scaler = StandardScaler()
normalized_load_curves = scaler.fit_transform(load_curves)

k_values = range(2, 11)
inertia_values = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_load_curves)
    inertia_values.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(normalized_load_curves, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')

plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o', color='orange')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different k Values')

plt.tight_layout()
plt.show()

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(normalized_load_curves)

pivot_df['cluster'] = clusters

plt.figure(figsize=(15, 10))

for cluster in range(optimal_k):
    sample_curves = pivot_df[pivot_df['cluster'] == cluster].sample(5, random_state=42).drop(columns=['postalcode', 'date', 'cluster'])
    for i, curve in sample_curves.iterrows():
        plt.plot(curve.values, alpha=0.6, label=f'Cluster {cluster}' if i == 0 else "")

plt.xlabel('Hour of the Day')
plt.ylabel('Electricity Consumption (kWh)')
plt.title('Example Daily Load Curves for Each Cluster')
plt.legend()
plt.show()

pca = PCA(n_components=2)
pca_components = pca.fit_transform(normalized_load_curves)

plt.figure(figsize=(10, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=clusters, cmap='viridis', s=10)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Clusters of Daily Load Curves Visualized in 2D (PCA)')
plt.colorbar(label='Cluster')
plt.show()

cluster_counts = pivot_df['cluster'].value_counts().sort_index()

print("\nNumber of Days per Cluster:\n")
print(cluster_counts)

pivot_df.to_csv('processed_data/clustered_daily_load_curves.csv', index=False)

print("\nTask 1: Identify Common Daily Load Curves completed successfully!\n")
