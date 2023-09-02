import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

iris = datasets.load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

algorithms = [
    KMeans(n_clusters=3),
    AgglomerativeClustering(n_clusters=3),
    DBSCAN(eps=0.5, min_samples=5),
    SpectralClustering(n_clusters=3),
    GaussianMixture(n_components=3)
]

algorithm_names = [
    "KMeans",
    "Agglomerative",
    "DBSCAN",
    "Spectral",
    "GaussianMixture"
]

# Sonuçları görselleştirme
plt.figure(figsize=(15, 10))

for i, algorithm in enumerate(algorithms):
    algorithm.fit(X_scaled)

    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(int)
    else:
        y_pred = algorithm.predict(X_scaled)

    plt.subplot(2, 3, i + 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis')
    plt.title(algorithm_names[i])

    score = silhouette_score(X_scaled, y_pred)
    plt.xlabel(f"Silhouette Score: {score:.2f}")

plt.tight_layout()
plt.show()
