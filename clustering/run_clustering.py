# clustering/run_clustering.py

from clustering.handler_factory import get_handler
from clustering.KMeansClusterer import KMeansClusterer
from clustering.HierarchicalClusterer import HierarchicalClusterer

def run_clustering(function_name: str, method: str = "kmeans", n_clusters: int = 4):
    """
    Generic clustering runner.
    method: 'kmeans' or 'hierarchical'
    """

    # 1. Get the correct DataHandler
    handler = get_handler(function_name)
    handler.add_weekly_updates()

    # 2. Choose clustering method
    method = method.lower().strip()

    if method == "kmeans":
        clusterer = KMeansClusterer(n_clusters=n_clusters)

    elif method == "hierarchical":
        clusterer = HierarchicalClusterer(n_clusters=n_clusters)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # 3. Load + normalize
    df = clusterer.load_data_from_handler(handler)
    print("Data:\n", df)
    df_scaled = clusterer.normalize(df[handler.input_columns])
    print("Scaled data:\n", df_scaled)

    # 4. Fit
    labels, model = clusterer.fit(df_scaled)
    print("Cluster labels:", labels)

    # 5. Plot
    clusterer.plot_clusters(df_scaled, labels)

    return labels, model, df, df_scaled