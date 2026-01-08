# clustering/cluster_semantics.py

def label_cluster(function_name, stats, global_max):
    """
    Generic semantic labeler for all functions.
    Uses relative thresholds based on global max.
    """

    fn = function_name.lower()
    max_val = stats["max"]
    mean_val = stats["mean"]

    # Treat tiny negatives as zero
    if max_val < 0:
        max_val = 0

    # Handle degenerate case: all outputs are zero
    if global_max <= 0:
        return "background region"

    # Compute relative strength
    rel = max_val / global_max

    # Generic interpretation rules
    if rel < 0.05:
        return "background region"
    elif rel < 0.30:
        return "weak region"
    elif rel < 0.70:
        return "moderate region"
    else:
        return "strong region"