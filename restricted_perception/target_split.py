"""Frozen target manifests and leakage checks for restricted-perception tests."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HASH_QUANTIZATION_DIGITS = 6


def canonical_point(point_mm: Iterable[float]) -> tuple[float, float, float]:
    point = np.asarray(list(point_mm), dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("target point must be a finite 3-vector")
    return tuple(round(float(value), HASH_QUANTIZATION_DIGITS) for value in point)


def target_hash(point_mm: Iterable[float], seed: int | None = None) -> str:
    point = canonical_point(point_mm)
    payload = json.dumps(
        {"point_mm_6dp": point, "seed": seed},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def nearest_neighbor_distances(
    left: Iterable[Iterable[float]],
    right: Iterable[Iterable[float]],
    *,
    exclude_self: bool = False,
) -> list[float]:
    left_array = np.asarray(list(left), dtype=float)
    right_array = np.asarray(list(right), dtype=float)
    if left_array.ndim != 2 or left_array.shape[1] != 3:
        raise ValueError("left must contain 3-vectors")
    if right_array.ndim != 2 or right_array.shape[1] != 3:
        raise ValueError("right must contain 3-vectors")
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ValueError("target arrays must be finite")
    result = []
    for index, point in enumerate(left_array):
        distances = np.linalg.norm(right_array - point, axis=1)
        if exclude_self:
            distances[index] = np.inf
        if not np.isfinite(distances).any():
            raise ValueError("nearest-neighbor reference set is empty")
        result.append(float(np.min(distances)))
    return result


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _role_entries_from_reports(
    paths: Iterable[Path],
    *,
    role: str,
    generator_version: str,
    root: Path,
    target_key: str = "target_robot_mm",
) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(paths):
        payload = _read_json(path)
        point = payload.get(target_key)
        if point is None:
            continue
        seed = payload.get("seed")
        entry = {
            "role": role,
            "point_mm": canonical_point(point),
            "seed": int(seed) if seed is not None else None,
            "generator_version": generator_version,
            "source": str(path.relative_to(root)),
        }
        entry["target_hash"] = target_hash(entry["point_mm"], entry["seed"])
        entries.append(entry)
    return entries


def _development_entries(root: Path) -> list[dict[str, Any]]:
    evaluation_roots = (
        root / "outputs" / "evaluation_v7_guided_dagger1_cached_20261106",
        root / "outputs" / "evaluation_v7_guided_20mm_repeat_20261106",
    )
    entries = []
    for evaluation_root in evaluation_roots:
        for path in sorted(evaluation_root.glob("g*/evaluation_results.json")):
            payload = _read_json(path)
            for episode in payload.get("episodes", []):
                point = episode.get("target_robot_mm")
                if point is None:
                    continue
                seed = episode.get("seed")
                entry = {
                    "role": "development",
                    "point_mm": canonical_point(point),
                    "seed": int(seed) if seed is not None else None,
                    "generator_version": "v7_runtime_random_fundus_targets",
                    "source": str(path.relative_to(root)),
                }
                entry["target_hash"] = target_hash(entry["point_mm"], entry["seed"])
                entries.append(entry)
    return entries


def _baseline_seeds(root: Path) -> dict[str, int]:
    protocol = _read_json(root / "runs" / "paper_stage1" / "train" / "collection_protocol.json")
    base = int(protocol["seed"])
    return {
        f"episode_{index:06d}": base + index for index in range(int(protocol["requested_episodes"]))
    }


def _attach_baseline_seeds(entries: list[dict[str, Any]], seed_by_episode: dict[str, int]) -> None:
    for entry in entries:
        episode_name = Path(entry["source"]).parent.name
        if episode_name in seed_by_episode:
            entry["seed"] = seed_by_episode[episode_name]
            entry["target_hash"] = target_hash(entry["point_mm"], entry["seed"])


def _deduplicate(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[canonical_point(entry["point_mm"])].append(entry)
    result = []
    for point, group in sorted(grouped.items()):
        seeds = sorted({item["seed"] for item in group if item["seed"] is not None})
        primary = group[0]
        result.append(
            {
                "id": f"target_{target_hash(point, seeds[0] if seeds else None)[:12].lower()}",
                "point_mm": point,
                "seed": seeds[0] if seeds else None,
                "seeds": seeds,
                "generator_versions": sorted({item["generator_version"] for item in group}),
                "roles": sorted({item["role"] for item in group}),
                "sources": sorted({item["source"] for item in group}),
                "target_hash": target_hash(point, seeds[0] if seeds else None),
            }
        )
    return result


def _generate_final_targets(
    root: Path,
    existing_points: list[tuple[float, float, float]],
    *,
    count: int,
    seed: int,
    minimum_separation_mm: float = 1.0,
    minimum_normalized_posterior_z: float = 0.8,
    validate_reachable: bool = False,
    validate_scene_reachable: bool = False,
    inter_target_minimum_separation_mm: float | None = None,
) -> tuple[list[dict[str, Any]], int]:
    from rcm_fundus import eye_frame
    from rcm_reference import Reference

    reference = Reference()
    center, rotation, radii = eye_frame(reference)
    from scipy.spatial import cKDTree

    accepted = []
    rng = np.random.default_rng(seed)
    existing_array = np.asarray(existing_points, dtype=float)
    existing_tree = cKDTree(existing_array)
    accepted_array = np.empty((0, 3), dtype=float)
    reachability_scene = None
    if validate_scene_reachable:
        from rcm_scene_v9 import RCMSceneV9

        reachability_scene = RCMSceneV9()
    attempts = 0
    batch_size = 20000
    while len(accepted) < count and attempts < 2_000_000:
        batch_count = min(batch_size, 2_000_000 - attempts)
        samples = rng.uniform(size=(batch_count, 2))
        attempts += batch_count
        z = (
            minimum_normalized_posterior_z
            + (1.0 - minimum_normalized_posterior_z) * samples[:, 0]
        )
        azimuth = 2.0 * np.pi * samples[:, 1]
        radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        unit = np.column_stack(
            [radial * np.cos(azimuth), radial * np.sin(azimuth), z]
        )
        worlds = center + (rotation @ (radii[:, None] * unit.T)).T
        candidates = np.round(
            (worlds - reference.rcm_mm) @ reference.rotation_robot,
            HASH_QUANTIZATION_DIGITS,
        )
        inter_separation = (
            minimum_separation_mm
            if inter_target_minimum_separation_mm is None
            else inter_target_minimum_separation_mm
        )
        separation_sq = (inter_separation - 1e-12) ** 2
        existing_distances, _ = existing_tree.query(candidates)
        for candidate in candidates[existing_distances >= minimum_separation_mm - 1e-12]:
            if validate_reachable:
                try:
                    state = reference.nominal.inverse(
                        candidate,
                        enforce_limits=False,
                        reference_state=reference.state_from_task(
                            reference.current_task()
                        ),
                    )
                    reference.validate_task(reference.task_from_state(state))
                except ValueError:
                    continue
            if reachability_scene is not None:
                try:
                    reachability_scene.set_target_world(
                        reachability_scene.robot_to_world(candidate)
                    )
                    target_task = reachability_scene.target_task(validate=True)
                    start_task = target_task.copy()
                    start_task[2] -= 0.0002
                    reachability_scene.set_task(start_task)
                except ValueError:
                    continue
            if accepted_array.size and np.min(
                np.sum((accepted_array - candidate) ** 2, axis=1)
            ) < separation_sq:
                continue
            accepted.append(canonical_point(candidate))
            accepted_array = np.asarray(accepted, dtype=float)
            if len(accepted) == count:
                break
    if len(accepted) < count:
        raise RuntimeError(
            f"unable to generate {count} separated final targets after {attempts} attempts"
        )
    if reachability_scene is not None:
        reachability_scene.close()
    accepted.sort(key=lambda point: target_hash(point, seed))
    entries = []
    for index, point in enumerate(accepted):
        entries.append(
            {
                "id": f"final_{index:03d}_{target_hash(point, seed)[:10].lower()}",
                "point_mm": point,
                "seed": seed,
                "generator_version": "restricted_target_split_v1",
                "role": "final_test",
                "source": "rcm_fundus.make_random_fundus_targets",
                "target_hash": target_hash(point, seed),
            }
        )
    return entries, seed


def generate_training_target_set(
    root: Path,
    *,
    seed: int,
    count: int = 60,
    minimum_normalized_posterior_z: float = 0.9,
) -> dict[str, Any]:
    manifest = json.loads(
        (Path(root) / "configs" / "target_split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    existing = [
        item["point_mm"]
        for key in ("development_targets", "final_test_targets")
        for item in manifest[key]
    ]
    entries, actual_seed = _generate_final_targets(
        Path(root),
        existing,
        count=count,
        seed=seed,
        minimum_separation_mm=1.0,
        minimum_normalized_posterior_z=minimum_normalized_posterior_z,
        validate_reachable=True,
        validate_scene_reachable=True,
        inter_target_minimum_separation_mm=0.1,
    )
    start_edges_um = (20.0, 50.0, 80.0, 110.0, 140.0, 170.0, 200.0)
    for index, entry in enumerate(entries):
        lower = start_edges_um[index % (len(start_edges_um) - 1)]
        upper = start_edges_um[index % (len(start_edges_um) - 1) + 1]
        entry["role"] = "stage2_training"
        entry["starts"] = [
            {
                "start_id": f"{entry['id']}_s0",
                "initial_error_um": float(lower),
                "seed": int(seed + index * 2),
            },
            {
                "start_id": f"{entry['id']}_s1",
                "initial_error_um": float(upper),
                "seed": int(seed + index * 2 + 1),
            },
        ]
    return {
        "schema_version": 1,
        "generator_version": "restricted_stage2_targets_v1",
        "seed": actual_seed,
        "count": count,
        "minimum_separation_mm": 1.0,
        "minimum_normalized_posterior_z": minimum_normalized_posterior_z,
        "selection_used_model_results": False,
        "targets": entries,
    }


def build_manifest(root: Path, *, final_seed: int = 20261215) -> dict[str, Any]:
    root = Path(root).resolve()
    training_paths = (
        list((root / "runs/paper_stage1/train").glob("episode_*/visual_servo_report.json"))
        + list((root / "runs/recovery_pilot/train").glob("episode_*/visual_servo_report.json"))
        + list((root / "runs/recovery_pilot_wave2/train").glob("episode_*/visual_servo_report.json"))
        + list(
            (root / "runs/recovery_pilot_wave2_parallel/train_a").glob(
                "episode_*/visual_servo_report.json"
            )
        )
        + list(
            (root / "runs/recovery_pilot_wave2_parallel/train_b").glob(
                "episode_*/visual_servo_report.json"
            )
        )
    )
    training = _role_entries_from_reports(
        training_paths,
        role="training",
        generator_version="v9_random_fundus_training",
        root=root,
    )
    _attach_baseline_seeds(training, _baseline_seeds(root))
    dagger_paths = list(
        (root / "runs/dagger_round1_guided_v6_20261105").glob(
            "*/episode_*/control_report.json"
        )
    )
    dagger = _role_entries_from_reports(
        dagger_paths,
        role="dagger",
        generator_version="v6_guided_dagger_target_sampler",
        root=root,
    )
    development = _development_entries(root)
    training_unique = _deduplicate(training)
    dagger_unique = _deduplicate(dagger)
    development_unique = _deduplicate(development)
    existing = (
        [item["point_mm"] for item in training_unique]
        + [item["point_mm"] for item in dagger_unique]
        + [item["point_mm"] for item in development_unique]
    )
    final, actual_seed = _generate_final_targets(
        root,
        existing,
        count=60,
        seed=final_seed,
        minimum_separation_mm=1.0,
    )
    final = _deduplicate(final)
    final_points = [item["point_mm"] for item in final]
    training_points = [item["point_mm"] for item in training_unique]
    dagger_points = [item["point_mm"] for item in dagger_unique]
    development_points = [item["point_mm"] for item in development_unique]
    min_final_final = min(nearest_neighbor_distances(final_points, final_points, exclude_self=True))
    min_final_training = min(nearest_neighbor_distances(final_points, training_points))
    min_final_dagger = min(nearest_neighbor_distances(final_points, dagger_points))
    min_final_development = min(nearest_neighbor_distances(final_points, development_points))
    min_development_training = min(
        nearest_neighbor_distances(development_points, training_points)
    )
    min_development_dagger = min(
        nearest_neighbor_distances(development_points, dagger_points)
    )
    min_dagger_training = min(
        nearest_neighbor_distances(dagger_points, training_points)
    )
    median_subset = sorted(
        final,
        key=lambda item: target_hash(item["point_mm"], 0),
    )[:20]
    return {
        "schema_version": 1,
        "generator_version": "restricted_target_split_v1",
        "source_root": str(root),
        "hash_quantization_digits": HASH_QUANTIZATION_DIGITS,
        "final_generation": {
            "requested_seed": final_seed,
            "actual_generation_seed": actual_seed,
            "required_count": 60,
            "minimum_separation_mm": 1.0,
            "minimum_normalized_posterior_z": 0.8,
            "region_change_reason": (
                "z>=0.9 admitted zero additional 1 mm-separated targets; "
                "extended region frozen before any model outcome"
            ),
            "selection_used_model_results": False,
        },
        "counts": {
            "training_observations": len(training),
            "training_unique": len(training_unique),
            "dagger_observations": len(dagger),
            "dagger_unique": len(dagger_unique),
            "development_observations": len(development),
            "development_unique": len(development_unique),
            "final_test": len(final),
        },
        "training_targets": training_unique,
        "dagger_targets": dagger_unique,
        "development_targets": development_unique,
        "final_test_targets": final,
        "median_subset_20": [
            {"id": item["id"], "point_mm": item["point_mm"], "target_hash": item["target_hash"]}
            for item in median_subset
        ],
        "nearest_neighbor_mm": {
            "final_to_final_min": min_final_final,
            "final_to_training_min": min_final_training,
            "final_to_dagger_min": min_final_dagger,
            "final_to_development_min": min_final_development,
            "development_to_training_min": min_development_training,
            "development_to_dagger_min": min_development_dagger,
            "dagger_to_training_min": min_dagger_training,
        },
        "acceptance": {
            "final_to_final_min_ge_1mm": min_final_final >= 1.0 - 1e-12,
            "final_to_training_min_ge_1mm": min_final_training >= 1.0 - 1e-12,
            "final_to_dagger_min_ge_1mm": min_final_dagger >= 1.0 - 1e-12,
            "final_to_development_min_ge_1mm": min_final_development >= 1.0 - 1e-12,
            "final_count_is_60": len(final) == 60,
            "median_subset_count_is_20": len(median_subset) == 20,
        },
    }


def verify_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    training = manifest["training_targets"]
    dagger = manifest["dagger_targets"]
    development = manifest["development_targets"]
    final = manifest["final_test_targets"]
    training_points = [canonical_point(item["point_mm"]) for item in training]
    dagger_points = [canonical_point(item["point_mm"]) for item in dagger]
    development_points = [canonical_point(item["point_mm"]) for item in development]
    final_points = [canonical_point(item["point_mm"]) for item in final]

    def entry_seed(item: dict[str, Any]) -> int | None:
        seeds = item.get("seeds")
        if seeds:
            return int(seeds[0])
        seed = item.get("seed")
        return None if seed is None else int(seed)

    checks = {
        "training_hashes": all(
            item["target_hash"] == target_hash(item["point_mm"], entry_seed(item))
            for item in training
        ),
        "dagger_hashes": all(
            item["target_hash"] == target_hash(item["point_mm"], entry_seed(item))
            for item in dagger
        ),
        "development_hashes": all(
            item["target_hash"] == target_hash(item["point_mm"], entry_seed(item))
            for item in development
        ),
        "final_hashes": all(item["target_hash"] == target_hash(item["point_mm"], item["seed"]) for item in final),
        "training_unique": len(training_points) == len(set(training_points)),
        "dagger_unique": len(dagger_points) == len(set(dagger_points)),
        "development_unique": len(development_points) == len(set(development_points)),
        "final_unique": len(final_points) == len(set(final_points)),
        "final_count": len(final_points) == 60,
        "median_subset_count": len(manifest["median_subset_20"]) == 20,
        "final_final_separation": min(
            nearest_neighbor_distances(final_points, final_points, exclude_self=True)
        ) >= 1.0 - 1e-12,
        "final_training_separation": min(
            nearest_neighbor_distances(final_points, training_points)
        ) >= 1.0 - 1e-12,
        "final_dagger_separation": min(
            nearest_neighbor_distances(final_points, dagger_points)
        ) >= 1.0 - 1e-12,
        "final_development_separation": min(
            nearest_neighbor_distances(final_points, development_points)
        ) >= 1.0 - 1e-12,
    }
    return {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
