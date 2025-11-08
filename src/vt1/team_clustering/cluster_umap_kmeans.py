# python
"""
Cluster SigLIP embeddings from the training set using UMAP + KMeans.

Inputs layout expected (created by build_training_set.py):
- <in_root>/<video_stem>/embeddings.npy
- <in_root>/<video_stem>/index.csv

Outputs in <out_dir>:
- combined_embeddings.npy        # [N, D]
- combined_index.csv             # aggregated rows across videos
- umap_D.npy                     # reduced embeddings [N, D]
- labels_kK.npy                  # KMeans labels [N]
- labeled_index.csv              # combined_index + cluster label
- summary.json                   # basic stats, silhouette if available
- (optional) umap_scatter.png    # if --plot

Models (.pkl) are saved to the configured team models directory (see vt1.config),
or override via --models-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

# Optional plotting (lazy)
_HAS_MPL = False
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# New: project config for defaults
try:
    from vt1.config import settings  # type: ignore
except Exception:
    settings = None  # type: ignore


def parse_args() -> argparse.Namespace:
    # Default input/output from config if available; fallback to adjacent-to-script
    ap = argparse.ArgumentParser("UMAP + KMeans clustering for team separation")
    local_base = None
    models_base = None
    if settings is not None:
        try:
            cfg = settings()
            local_base = cfg.team_output_dir
            models_base = cfg.team_models_dir
        except Exception:
            local_base = None
            models_base = None
    if local_base is None:
        local_base = Path(__file__).resolve().parent / "clustering"
    if models_base is None:
        models_base = Path(__file__).resolve().parent / "clustering"

    ap.add_argument("--in-root", type=str, default=str(local_base),
                    help="Root directory with per-video embeddings (default: config team_output_dir)")
    ap.add_argument("--out-dir", type=str, default=str(local_base),
                    help="Output directory for clustering artifacts (default: config team_output_dir)")
    # New: where to save the fitted UMAP/KMeans models
    ap.add_argument("--models-dir", type=str, default=str(models_base),
                    help="Directory to save umap.pkl and kmeans.pkl (default: config team_models_dir)")

    ap.add_argument("--k", type=int, default=2, help="Number of clusters for KMeans (e.g., 2 teams)")
    ap.add_argument("--umap-dim", type=int, default=16, help="UMAP output dimensionality")
    ap.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors")
    ap.add_argument("--umap-metric", type=str, default="cosine", help="UMAP distance metric")
    ap.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist (lower=more compact clusters)")
    ap.add_argument("--reuse-umap", type=str, default="", help="Path to a pre-trained umap.pkl to reuse (skip fitting)")

    ap.add_argument("--limit", type=int, default=0, help="Limit N total rows (0=all)")
    ap.add_argument("--plot", action="store_true", help="Save a 2D scatter plot if umap-dim>=2 and matplotlib available")

    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--save-models", action="store_true", help="Persist fitted UMAP and KMeans models (umap.pkl, kmeans.pkl)")
    return ap.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_training_set(in_root: Path) -> Tuple[np.ndarray, List[List[str]]]:
    """Load and concatenate embeddings + indexes across subfolders.

    Returns:
      E: np.ndarray [N, D]
      rows: list of CSV rows including header row at index 0
    """
    subs = [p for p in in_root.iterdir() if p.is_dir()]
    all_embeddings: List[np.ndarray] = []
    combined_rows: List[List[str]] = []
    header = ["idx", "frame_idx", "time_s", "x1", "y1", "x2", "y2", "score", "label", "crop_relpath", "video_stem"]

    for sub in subs:
        emb_p = sub / "embeddings.npy"
        idx_p = sub / "index.csv"
        if not emb_p.exists() or not idx_p.exists():
            continue
        E = np.load(emb_p)
        with idx_p.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            continue
        # Skip header in per-video index; append video stem
        for r in rows[1:]:
            combined_rows.append([*r, sub.name])
        all_embeddings.append(E)

    if not all_embeddings:
        raise RuntimeError(f"No embeddings/index found in: {in_root}")

    E = np.concatenate(all_embeddings, axis=0)
    rows = [header, *combined_rows]
    if len(rows) - 1 != E.shape[0]:
        raise RuntimeError(f"Row/embedding mismatch: rows={len(rows)-1}, E={E.shape}")
    return E, rows


def run_umap(E: np.ndarray, dim: int, metric: str, neighbors: int, seed: int, min_dist: float):
    try:
        import umap.umap_ as umap
    except Exception as e:
        raise RuntimeError("umap-learn is not installed. pip install umap-learn") from e
    reducer = umap.UMAP(n_components=int(dim), n_neighbors=int(neighbors), metric=metric,
                        min_dist=float(min_dist), random_state=seed)
    Z = reducer.fit_transform(E)
    return Z, reducer


def run_kmeans(Z: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, Dict[str, Any], 'KMeans']:
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception as e:
        raise RuntimeError("scikit-learn is not installed. pip install scikit-learn") from e
    km = KMeans(n_clusters=int(k), n_init=10, random_state=seed)
    labels = km.fit_predict(Z)
    info: Dict[str, Any] = {
        "inertia": float(km.inertia_),
        "centroids": km.cluster_centers_.tolist(),
    }
    try:
        if Z.shape[0] >= 10:
            info["silhouette_cosine"] = float(silhouette_score(Z, labels, metric="cosine"))
    except Exception:
        pass
    return labels.astype(int), info, km


def main() -> int:
    args = parse_args()
    if args.seed:
        np.random.seed(args.seed)

    in_root = Path(args.in_root)
    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    E, rows = load_training_set(in_root)
    N = E.shape[0]

    # Optional limit
    if args.limit and args.limit > 0 and args.limit < N:
        E = E[: args.limit]
        rows = [rows[0], *rows[1: 1 + args.limit]]
        N = E.shape[0]

    # Save combined for reference
    np.save(out_root / "combined_embeddings.npy", E)
    with (out_root / "combined_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # UMAP: reuse if provided, else fit
    reducer = None
    Z = None
    if args.reuse_umap:
        try:
            import joblib
            reducer = joblib.load(args.reuse_umap)
            Z = reducer.transform(E)
        except Exception as e:
            print(f"[WARN] Failed to reuse UMAP from {args.reuse_umap}: {e}. Fitting a new one instead.")
            Z, reducer = run_umap(E, dim=args.umap_dim, metric=args.umap_metric, neighbors=args.umap_neighbors,
                                   seed=args.seed, min_dist=args.umap_min_dist)
    else:
        Z, reducer = run_umap(E, dim=args.umap_dim, metric=args.umap_metric, neighbors=args.umap_neighbors,
                               seed=args.seed, min_dist=args.umap_min_dist)

    np.save(out_root / f"umap_{args.umap_dim}.npy", Z)

    # KMeans
    labels, km_info, km_model = run_kmeans(Z, k=args.k, seed=args.seed)
    np.save(out_root / f"labels_k{args.k}.npy", labels)

    # Optionally persist models for inference (to models-dir)
    saved_umap = None
    saved_kmeans = None
    if args.save_models:
        try:
            import joblib
            models_dir = Path(args.models_dir)
            ensure_dir(models_dir)
            # Save UMAP reducer
            if reducer is not None:
                saved_umap = models_dir / "umap.pkl"
                joblib.dump(reducer, saved_umap)
            # Save KMeans model
            saved_kmeans = models_dir / "kmeans.pkl"
            joblib.dump(km_model, saved_kmeans)
            print(f"[INFO] Saved models to: {models_dir}")
        except Exception as e:
            print(f"[WARN] Failed to save models: {e}")
    else:
        print("[INFO] Models not saved (no --save-models flag). Re-run with --save-models to create umap.pkl and kmeans.pkl.")

    # Write labeled index
    labeled_header = [*rows[0], "cluster"]
    labeled_rows = [labeled_header]
    for i in range(N):
        labeled_rows.append([*rows[1 + i], int(labels[i])])
    with (out_root / "labeled_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(labeled_rows)

    # Summary
    uniq, cnts = np.unique(labels, return_counts=True)
    summary = {
        "N": int(N),
        "emb_dim": int(E.shape[1]),
        "umap_dim": int(Z.shape[1]),
        "k": int(args.k),
        "counts": {int(k): int(v) for k, v in zip(uniq.tolist(), cnts.tolist())},
        "kmeans": km_info,
        "umap_min_dist": float(args.umap_min_dist),
        "reuse_umap": (str(args.reuse_umap) if args.reuse_umap else None),
        "models_dir": (str(Path(args.models_dir).resolve()) if args.save_models else None),
        "umap_pkl": (str(Path(saved_umap).resolve()) if saved_umap else None),
        "kmeans_pkl": (str(Path(saved_kmeans).resolve()) if saved_kmeans else None),
    }
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Optional plot
    if args.plot and _HAS_MPL and Z.shape[1] >= 2:
        try:
            import matplotlib.pyplot as plt  # ensure available
            plt.figure(figsize=(6, 5))
            cs = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
            for lab in np.unique(labels):
                m = labels == lab
                plt.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.7, c=cs[int(lab) % len(cs)], label=f"c{int(lab)}")
            plt.legend(frameon=False)
            plt.title("UMAP + KMeans clusters")
            plt.tight_layout()
            plt.savefig(out_root / "umap_scatter.png", dpi=200)
            plt.close()
        except Exception:
            pass

    print(f"Clustering complete. Output in: {out_root}")
    if args.save_models:
        print(f"Models available at: {Path(args.models_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
