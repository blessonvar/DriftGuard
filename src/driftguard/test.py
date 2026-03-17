import json
from pathlib import Path

d = {
    "datasets": ["DG5", "PACS", "DomainNet"],
    "models": ["CRST-S", "CRST-M", "CVIT", "CVIT-S"],
    "strategies": [
        "Never",
        "FCL-AveTrig",
        "FCL-perDevice",
        "PFL-AveTrig",
        "PFL-perDevice",
        "Cluster-based ",
        "DriftGuard",
    ],
    "retraining_thresholds": 0.8,
    "global_retraining_thresholds": 0.8, # reletive to retrainging threshold, only for DriftGuard
    "clustering_distance": 0.3, # only for DriftGuard
    "min_cluster_size": 4, # only for DriftGuard
}
with open("config.json", "w") as f:
    json.dump(d, f, indent=4)