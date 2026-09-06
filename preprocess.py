from pathlib import Path
from copy import deepcopy
import json
import re
import glob

import ijson
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ERROR_EDGE_PATTERNS = [
    r"error",
    r"fail",
    r"failure",
    r"status",
    r"return",
    r"response.*code",
    r"code",
    # r"[45]xx",
    # r"\brc_[45]",
]


def is_error_related_feature(name: str) -> bool:
    name = name.lower()

    return any(
        re.search(pattern, name)
        for pattern in ERROR_EDGE_PATTERNS
    )

def place_series(
    out: np.ndarray,
    feature_idx: int,
    record: dict,
    reference_steps: list,
    step_to_idx: dict,
):
    """
    Записывает одну временную серию в out[:, feature_idx].

    out shape:
        [T, num_features]
    """

    values = np.asarray(
        record["values"],
        dtype=np.float32
    )

    steps = record["steps"]

    if len(values) != len(steps):
        raise ValueError(
            f"values/steps mismatch: {len(values)} != {len(steps)}"
        )

    # Быстрый путь: временная ось полностью совпадает.
    if (
        len(steps) == len(reference_steps)
        and steps[0] == reference_steps[0]
        and steps[-1] == reference_steps[-1]
    ):
        out[:, feature_idx] = values
        return

    # Общий случай: часть временных точек отсутствует.
    for step, value in zip(steps, values):
        pos = step_to_idx.get(step)

        if pos is not None:
            out[pos, feature_idx] = value
            
def preprocess_node_features(node_json_path: Path):
    """
    Returns
    -------
    node_x:
        [T, N, F_node]

    node_ids:
        list[str]

    feature_names:
        list[str]

    time_steps:
        list
    """

    node_ids = []
    node_matrices = []

    with open(node_json_path, "rb") as f:
        iterator = ijson.kvitems(f, "")

        # Первый узел нужен для определения schema.
        first_node_id, first_metrics = next(iterator)

        feature_names = list(first_metrics.keys())

        reference_feature = feature_names[0]
        reference_steps = list(
            first_metrics[reference_feature]["steps"]
        )

        T = len(reference_steps)
        F = len(feature_names)

        step_to_idx = {
            step: i
            for i, step in enumerate(reference_steps)
        }

        print("Node features:")
        for feature in feature_names:
            print("  ", feature)

        print()
        print("T =", T)
        print("F_node =", F)

        def build_node_matrix(metrics):
            matrix = np.full(
                (T, F),
                np.nan,
                dtype=np.float32,
            )

            for feature_idx, feature_name in enumerate(feature_names):
                if feature_name not in metrics:
                    continue

                place_series(
                    matrix,
                    feature_idx,
                    metrics[feature_name],
                    reference_steps,
                    step_to_idx,
                )

            return matrix

        # Первый узел.
        node_ids.append(str(first_node_id))
        node_matrices.append(
            build_node_matrix(first_metrics)
        )

        # Остальные.
        for node_id, metrics in tqdm(
            iterator,
            desc="Reading nodes"
        ):
            node_ids.append(str(node_id))
            node_matrices.append(
                build_node_matrix(metrics)
            )

    # [N, T, F] -> [T, N, F]
    node_x = np.stack(
        node_matrices,
        axis=0
    ).transpose(1, 0, 2)

    return (
        node_x,
        node_ids,
        feature_names,
        reference_steps,
    )
    

def build_edge_index(
    edges_path: Path,
    node_ids: list[str],
):
    edges = pd.read_csv(
        edges_path,
        dtype={
            "source": str,
            "target": str,
        }
    )

    node_to_idx = {
        node_id: idx
        for idx, node_id in enumerate(node_ids)
    }

    unknown_sources = set(edges["source"]) - set(node_to_idx)
    unknown_targets = set(edges["target"]) - set(node_to_idx)

    if unknown_sources:
        raise ValueError(
            f"Unknown source nodes: {list(unknown_sources)[:10]}"
        )

    if unknown_targets:
        raise ValueError(
            f"Unknown target nodes: {list(unknown_targets)[:10]}"
        )

    src = edges["source"].map(node_to_idx).to_numpy()
    dst = edges["target"].map(node_to_idx).to_numpy()

    edge_index = np.stack(
        [src, dst],
        axis=0,
    ).astype(np.int64)

    edge_keys = [
        f"{source}->{target}"
        for source, target in zip(
            edges["source"],
            edges["target"]
        )
    ]

    return edges, edge_index, edge_keys


def inspect_edge_features(edge_json_path: Path):
    with open(edge_json_path, "rb") as f:
        iterator = ijson.kvitems(f, "")
        edge_key, metrics = next(iterator)

    feature_names = list(metrics.keys())

    return edge_key, feature_names

def preprocess_edge_features(
    edge_feature_paths: list[Path],
    edge_keys: list[str],
    kept_features: list[str],
    reference_steps: list,
):
    """
    Returns
    -------
    edge_x:
        [T, E, F_edge]
    """

    T = len(reference_steps)
    E = len(edge_keys)
    F = len(kept_features)

    edge_to_idx = {
        key: idx
        for idx, key in enumerate(edge_keys)
    }

    step_to_idx = {
        step: i
        for i, step in enumerate(reference_steps)
    }

    edge_x = np.full(
        (T, E, F),
        np.nan,
        dtype=np.float32,
    )

    seen_edges = set()

    for path in edge_feature_paths:
        print("Reading:", path)

        with open(path, "rb") as f:
            iterator = ijson.kvitems(f, "")

            for edge_key, metrics in tqdm(
                iterator,
                desc=path.name
            ):
                edge_idx = edge_to_idx.get(edge_key)

                # Игнорируем edges, которых нет в static topology.
                if edge_idx is None:
                    continue

                seen_edges.add(edge_key)

                temp = np.full(
                    (T, F),
                    np.nan,
                    dtype=np.float32,
                )

                for feature_idx, feature_name in enumerate(kept_features):
                    if feature_name not in metrics:
                        continue

                    place_series(
                        temp,
                        feature_idx,
                        metrics[feature_name],
                        reference_steps,
                        step_to_idx,
                    )

                edge_x[:, edge_idx, :] = temp

    print(
        f"Found temporal features for "
        f"{len(seen_edges)} / {E} edges"
    )

    return edge_x


from decimal import Decimal


def make_json_serializable(obj):
    """
    Рекурсивно преобразует типы, которые json.dump
    не умеет сериализовать.
    """
    if isinstance(obj, Decimal):
        # Не теряем целочисленный тип, если значение целое
        if obj == obj.to_integral_value():
            return int(obj)
        return float(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {
            key: make_json_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            make_json_serializable(value)
            for value in obj
        ]

    return obj

def preprocess_chronograph(
    data_dir: Path,
    cache_dir: Path,
    force=False,
):
    node_cache = cache_dir / "node_raw.npy"
    edge_cache = cache_dir / "edge_raw.npy"
    edge_index_cache = cache_dir / "edge_index.npy"
    metadata_cache = cache_dir / "metadata.json"

    if (
        not force
        and node_cache.exists()
        and edge_cache.exists()
        and edge_index_cache.exists()
        and metadata_cache.exists()
    ):
        print("Using cached preprocessing.")

        with open(metadata_cache) as f:
            metadata = json.load(f)

        return metadata

    # -----------------------
    # Nodes
    # -----------------------

    (
        node_x,
        node_ids,
        node_feature_names,
        time_steps,
    ) = preprocess_node_features(
        data_dir / "node_features.json"
    )

    # -----------------------
    # Static graph
    # -----------------------

    (
        edges,
        edge_index,
        edge_keys,
    ) = build_edge_index(
        data_dir / "edges.csv",
        node_ids,
    )

    # -----------------------
    # Edge schema
    # -----------------------

    edge_paths = sorted(
        data_dir.glob("edge_features_part*.json")
    )

    _, all_edge_features = inspect_edge_features(
        edge_paths[0]
    )

    kept_edge_features = [
        f
        for f in all_edge_features
        if not is_error_related_feature(f)
    ]

    excluded_edge_features = [
        f
        for f in all_edge_features
        if is_error_related_feature(f)
    ]

    print("\nAll edge features:")
    print(all_edge_features)

    print("\nKept edge features:")
    print(kept_edge_features)

    print("\nExcluded edge features:")
    print(excluded_edge_features)

    if not kept_edge_features:
        raise ValueError(
            "All edge features were excluded. "
            "Check ERROR_EDGE_PATTERNS."
        )

    # -----------------------
    # Temporal edge features
    # -----------------------

    edge_x = preprocess_edge_features(
        edge_paths,
        edge_keys,
        kept_edge_features,
        time_steps,
    )

    # -----------------------
    # Save
    # -----------------------

    np.save(node_cache, node_x)
    np.save(edge_cache, edge_x)
    np.save(edge_index_cache, edge_index)

    metadata = {
    "node_ids": node_ids,
    "node_features": node_feature_names,
    "all_edge_features": all_edge_features,
    "edge_features": kept_edge_features,
    "excluded_edge_features": excluded_edge_features,
    "time_steps": time_steps,
    "num_nodes": int(node_x.shape[1]),
    "num_edges": int(edge_x.shape[1]),
    "num_timesteps": int(node_x.shape[0]),
    }

    metadata = make_json_serializable(metadata)

    with open(metadata_cache, "w") as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    print("\nSaved:")
    print("node_x:", node_x.shape)
    print("edge_x:", edge_x.shape)
    print("edge_index:", edge_index.shape)

    return metadata

def fit_robust_scaler(
    x: np.ndarray,
    train_end: int,
    eps: float = 1e-6,
):
    """
    x:
        [T, entities, features]
    """

    train = x[:train_end]

    median = np.nanmedian(
        train,
        axis=0
    ).astype(np.float32)

    q25 = np.nanpercentile(
        train,
        25,
        axis=0
    ).astype(np.float32)

    q75 = np.nanpercentile(
        train,
        75,
        axis=0
    ).astype(np.float32)

    scale = q75 - q25

    median = np.nan_to_num(
        median,
        nan=0.0
    )

    scale = np.nan_to_num(
        scale,
        nan=1.0
    )

    scale[scale < eps] = 1.0

    return median, scale

def forward_fill(
    x: np.ndarray,
    initial_fill: np.ndarray,
):
    """
    Forward-fill по temporal axis.

    x:
        [T, N, F]

    initial_fill:
        [N, F]
    """

    out = x.copy()

    T = out.shape[0]
    flat = out.reshape(T, -1)
    initial_flat = initial_fill.reshape(-1)

    for col_idx in tqdm(
        range(flat.shape[1]),
        desc="Forward filling"
    ):
        col = flat[:, col_idx]

        valid = ~np.isnan(col)

        if not valid.any():
            col[:] = initial_flat[col_idx]
            continue

        first_valid = np.argmax(valid)

        # Leading missing values.
        if first_valid > 0:
            col[:first_valid] = initial_flat[col_idx]

        # Standard forward-fill.
        indices = np.where(
            ~np.isnan(col),
            np.arange(T),
            0
        )

        np.maximum.accumulate(
            indices,
            out=indices
        )

        col[:] = col[indices]

    return out