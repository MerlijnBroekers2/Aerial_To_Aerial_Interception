import numpy as np
from sklearn.cluster import DBSCAN


def robust_stats(data):
    """Compute robust statistics for a list of data values."""
    if len(data) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "Q1": np.nan,
            "Q3": np.nan,
        }
    return {
        "mean": np.mean(data),
        "std": np.std(data),
        "median": np.median(data),
        "Q1": np.percentile(data, 25),
        "Q3": np.percentile(data, 75),
    }


def compute_simulation_metrics(
    simulation_results, near_miss_threshold=0.30, cluster_radius=0.15
):
    """
    Compute metrics for each simulation in the raw simulation data.

    Compatible with the updated Simulation logging structure.
    """
    analysis_data = []
    for sim in simulation_results:
        history = sim.get("history", [])
        interceptions = sim.get("interceptions", [])

        # Initialize arrays
        if len(history) == 0:
            distances = np.array([])
            closest_distance = float("inf")
            time_in_near_miss = 0.0
            pursuer_pos = evader_pos = np.empty((0, 3))
        else:
            pursuer_pos = np.array([h["p_state"]["true_position"] for h in history])
            evader_pos = np.array([h["e_state"]["true_position"] for h in history])
            distances = np.linalg.norm(pursuer_pos - evader_pos, axis=1)
            closest_distance = np.min(distances)
            time_in_near_miss = (
                100.0
                * np.count_nonzero(distances < near_miss_threshold)
                / len(distances)
            )

        # Interception stats
        total_interceptions = len(interceptions)
        if total_interceptions > 0:
            interception_positions = [i["evader_pos"] for i in interceptions]
            clustering = DBSCAN(eps=cluster_radius, min_samples=1).fit(
                interception_positions
            )
            num_clusters = len(set(clustering.labels_))
            first_interception_time = interceptions[0]["time"]
        else:
            num_clusters = 0
            first_interception_time = float("inf")

        computed_metrics = {
            "name": sim.get("name", ""),
            "first_interception_time": first_interception_time,
            "closest_distance": closest_distance,
            "time_in_near_miss": time_in_near_miss,
            "total_interceptions": total_interceptions,
            "num_clusters": num_clusters,
            "mean_distance": np.mean(distances) if distances.size > 0 else np.nan,
            "std_distance": np.std(distances) if distances.size > 0 else np.nan,
        }

        analysis_data.append(
            {
                "name": sim.get("name", ""),
                "evader_pos": evader_pos,
                "pursuer_pos": pursuer_pos,
                "interceptions": interceptions,
                "metrics": computed_metrics,
                "history": history,
            }
        )

    return analysis_data
