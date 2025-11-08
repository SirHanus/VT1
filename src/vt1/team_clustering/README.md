# Team Clustering (migrated to apps/team_clustering)

Place original scripts from `offline_pipeline/team_clustering/` here:
- build_training_set.py
- cluster_umap_kmeans.py
- audit_training_set.py
- eval_clustering.py
- clustering/ (model artifacts and generated embeddings)
- outputs/ (audit/eval results)

Model artifacts (umap.pkl, kmeans.pkl) can remain temporarily in the legacy folder until copied.

Update GUI defaults after moving artifacts:
- Change references from `offline_pipeline/team_clustering/clustering` to `apps/team_clustering/clustering`.

This directory currently contains placeholders—replace them with the full implementations.
"""Placeholder for list_sources.py
Original location: offline_pipeline/list_sources.py
Copy the full implementation here and adjust imports if needed.
"""
from __future__ import annotations
import sys


def main(argv: list[str] | None = None) -> int:
    print("[PLACEHOLDER] list_sources.py moved to apps/pipeline. Copy original code here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

