# clustering/cluster_analysis.py

import pandas as pd
from clustering.cluster_semantics import label_cluster

class ClusterReport:
    """
    Generates human-readable insights about clusters,
    including semantic labels based on function description.
    """

    def __init__(self, df, labels, output_label, function_name):
        self.df = df.copy()
        self.df["cluster"] = labels
        self.output_label = output_label
        self.function_name = function_name

    def generate(self):
        df = self.df
        y = self.output_label

        grouped = df.groupby("cluster")[y]

        cluster_sizes = df.groupby("cluster").size()
        avg_output = grouped.mean()
        max_output = grouped.max()
        min_output = grouped.min()
        var_output = grouped.var()

        # Identify globally best point
        best_idx = df[y].idxmax()
        best_cluster = df.loc[best_idx, "cluster"]

        print("\n📊 **Cluster Analysis Summary**")
        print("==================================")

        results = {}

        for c in sorted(df["cluster"].unique()):
            stats = {
                "mean": avg_output[c],
                "max": max_output[c],
                "min": min_output[c],
                "var": var_output[c]
            }
            global_max = df[self.output_label].max()
            semantic = label_cluster(self.function_name, stats, global_max)

            results[c] = {
                "stats": stats,
                "semantic_label": semantic
            }

            # Pretty printed block for each cluster
            print(f"\n🔹 Cluster {c} — {semantic}")
            print("   -----------------------------")
            print(f"   Size:            {cluster_sizes[c]}")
            print(f"   Mean output:     {stats['mean']:.6f}")
            print(f"   Max output:      {stats['max']:.6f}")
            print(f"   Min output:      {stats['min']:.6f}")
            print(f"   Variability:     {stats['var']:.6f}")

        print("\n🌟 **Best Cluster Identified**")
        print("----------------------------------")
        print(f"Cluster {best_cluster} → {results[best_cluster]['semantic_label']}")

        return results, best_cluster