# clustering/kmeans_clusterer.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import umap  # <-- NEW


class KMeansClusterer:
    """
    Dimension-agnostic K-means clustering utility.
    Works with any BaseDataHandler subclass.
    """

    def __init__(self, n_clusters=3, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    # ---------------------------------------------------------
    # 1. Load data from handler (any dimensionality)
    # ---------------------------------------------------------
    def load_data_from_handler(self, handler):
        X = handler.get_inputs()
        y = handler.get_outputs()

        df = pd.DataFrame(X, columns=handler.input_columns)
        df[handler.output_label] = y
        return df

    # ---------------------------------------------------------
    # 2. Normalize numeric columns
    # ---------------------------------------------------------
    def normalize(self, df):
        numeric = df.select_dtypes(include=[np.number]).copy()

        # Avoid division by zero if a column is constant
        denom = (numeric.max() - numeric.min()).replace(0, 1)
        df_scaled = (numeric - numeric.min()) / denom

        return df_scaled

    # ---------------------------------------------------------
    # 3. Fit K-means
    # ---------------------------------------------------------
    def fit(self, df):
        km = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init
        )
        labels = km.fit_predict(df)
        return labels, km

    # ---------------------------------------------------------
    # 4. Plot clusters (2D or 3D)
    # ---------------------------------------------------------
    def plot_clusters(self, df, labels):
        cols = df.columns.tolist()

        if len(cols) == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(df[cols[0]], df[cols[1]], c=labels, cmap="jet")
            plt.xlabel(cols[0])
            plt.ylabel(cols[1])
            plt.title("2D K-means Clustering")
            plt.show()

        elif len(cols) == 3:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df[cols[0]], df[cols[1]], df[cols[2]], c=labels, cmap="jet")
            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[1])
            ax.set_zlabel(cols[2])
            plt.title("3D K-means Clustering")
            plt.show()

        else:
            print(f"Data has {len(cols)} dimensions — using UMAP for visualisation.")
            self.plot_umap(df, labels)

    # ---------------------------------------------------------
    # 5. UMAP visualisation (NEW)
    # ---------------------------------------------------------
    def plot_umap(self, df, labels):
        reducer = umap.UMAP(
            n_components=2,
            random_state=self.random_state,
            n_neighbors=15,
            min_dist=0.1
        )

        embedding = reducer.fit_transform(df)

        plt.figure(figsize=(8, 6))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="jet", s=10)
        plt.title("UMAP Projection of Clusters")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.show()
