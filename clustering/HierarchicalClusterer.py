# clustering/hierarchical_clusterer.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D
import umap
import matplotlib.pyplot as plt



class HierarchicalClusterer:
    """
    Dimension-agnostic hierarchical clustering utility.
    Works with any BaseDataHandler subclass.
    """

    def __init__(self, n_clusters=2, linkage="ward", metric="euclidean",
                 distance_threshold=0.5):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.distance_threshold = distance_threshold

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
    # 2. Normalize numeric columns (safe version)
    # ---------------------------------------------------------
    def normalize(self, df):
        numeric = df.select_dtypes(include=[np.number]).copy()

        # Avoid division by zero for constant columns
        denom = (numeric.max() - numeric.min()).replace(0, 1)
        df_scaled = (numeric - numeric.min()) / denom

        return df_scaled

    # ---------------------------------------------------------
    # 3. Fit hierarchical clustering
    # ---------------------------------------------------------
    def fit(self, df):
        """
        Fit hierarchical clustering using:
        - n_clusters > 0 → standard flat clustering
        - n_clusters < 0 → distance threshold mode
        - n_clusters = None → full tree, no cut
        """

        # Ward linkage *requires* Euclidean metric
        metric = "euclidean" if self.linkage == "ward" else self.metric

        if self.n_clusters is None:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0,
                linkage=self.linkage,
                metric=metric
            )
        elif self.n_clusters < 0:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                linkage=self.linkage,
                metric=metric
            )
        else:
            model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage=self.linkage,
                metric=metric
            )

        # Always use fit_predict so labels are returned explicitly
        labels = model.fit_predict(df)

        # Always return exactly TWO values
        return labels, model
    # ---------------------------------------------------------
    # 4. Dendrogram
    # ---------------------------------------------------------
    def plot_dendrogram(self, df, threshold=None):
        plt.figure(figsize=(10, 7))
        plt.title("Dendrogram")

        # Use scipy linkage for dendrogram
        Z = shc.linkage(df, method=self.linkage)
        shc.dendrogram(Z)

        if threshold is not None:
            plt.axhline(y=threshold, color='r', linestyle='--')

        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.show()

        # ---------------------------------------------------------
        # 5. Cluster visualization (2D or 3D)
        # ---------------------------------------------------------

    def plot_clusters(self, df, labels):
        cols = df.columns.tolist()
        dim = len(cols)

        # 2D case
        if dim == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(df[cols[0]], df[cols[1]], c=labels, cmap="jet")
            plt.xlabel(cols[0])
            plt.ylabel(cols[1])
            plt.title("2D Hierarchical Clustering")
            plt.show()
            return

        # 3D case
        if dim == 3:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df[cols[0]], df[cols[1]], df[cols[2]], c=labels, cmap="jet")
            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[1])
            ax.set_zlabel(cols[2])
            plt.title("3D Hierarchical Clustering")
            plt.show()
            return

        # High-dimensional case → UMAP fallback
        print(f"Data has {dim} dimensions — using UMAP for visualisation.")

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=42
        )

        reduced = reducer.fit_transform(df.values)

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="jet")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title(f"UMAP Projection of {dim}D Clusters")
        plt.show()