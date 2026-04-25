"""Pseudo-labelling module using KMeans clusters for ExplainInvest."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans


def generate_pseudo_labels(
    X: np.ndarray,
    current_close: np.ndarray,
    next_close: np.ndarray,
    clean_mask: np.ndarray,
    n_clusters: int = 25,
    move_threshold: float = 0.00005,
) -> Tuple[np.ndarray, KMeans, Dict[int, int]]:
    """Create buy/sell/hold pseudo-labels by mapping KMeans clusters to future returns."""
    if X.ndim != 3:
        raise ValueError("X must be 3D with shape (samples, time_steps, features).")

    X_flat = X.reshape(X.shape[0], -1)
    clean_X = X_flat[clean_mask]
    clean_curr = current_close[clean_mask]
    clean_next = next_close[clean_mask]

    if len(clean_X) < n_clusters:
        raise ValueError("Not enough clean samples to fit KMeans with requested clusters.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clean_cluster_ids = kmeans.fit_predict(clean_X)

    # Determine each cluster's expected move using next-period return.
    clean_returns = (clean_next - clean_curr) / (clean_curr + 1e-12)
    cluster_to_label: Dict[int, int] = {}
    for cluster_id in range(n_clusters):
        cluster_mask = clean_cluster_ids == cluster_id
        if np.sum(cluster_mask) == 0:
            cluster_to_label[cluster_id] = 2
            continue

        avg_ret = float(np.mean(clean_returns[cluster_mask]))
        if avg_ret > move_threshold:
            cluster_to_label[cluster_id] = 0  # BUY
        elif avg_ret < -move_threshold:
            cluster_to_label[cluster_id] = 1  # SELL
        else:
            cluster_to_label[cluster_id] = 2  # HOLD

    labels = np.full(shape=(len(X_flat),), fill_value=2, dtype=np.int32)
    predicted_clusters_all = kmeans.predict(X_flat)

    # Label clean samples from their assigned cluster while keeping noisy rows as HOLD.
    labels[clean_mask] = np.array(
        [cluster_to_label[c] for c in predicted_clusters_all[clean_mask]],
        dtype=np.int32,
    )

    # If any class collapses to zero, fall back to a return-quantile split so the
    # model always sees BUY, SELL, and HOLD examples during training.
    clean_labels = labels[clean_mask]
    class_counts = np.bincount(clean_labels, minlength=3)
    class_total = int(np.sum(class_counts))
    class_shares = class_counts / class_total if class_total > 0 else np.zeros_like(class_counts, dtype=np.float32)
    if np.any(class_counts == 0) or float(np.min(class_shares)) < 0.15:
        lower_q, upper_q = np.quantile(clean_returns, [0.33, 0.67])
        quantile_labels = np.full(shape=(len(clean_returns),), fill_value=2, dtype=np.int32)
        quantile_labels[clean_returns <= lower_q] = 1
        quantile_labels[clean_returns >= upper_q] = 0
        labels[clean_mask] = quantile_labels

        # As a second pass, make sure SELL exists even in strongly trending windows.
        clean_labels = labels[clean_mask]
        class_counts = np.bincount(clean_labels, minlength=3)
        if class_counts[1] == 0 and len(clean_returns) > 0:
            worst_idx = int(np.argmin(clean_returns))
            labels[np.where(clean_mask)[0][worst_idx]] = 1

    return labels, kmeans, cluster_to_label
