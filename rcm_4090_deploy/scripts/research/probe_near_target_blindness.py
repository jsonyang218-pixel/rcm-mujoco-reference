"""Zero-training probe of near-target representation and sampling variance."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__import__("os").environ.get(
    "RCM_RUNTIME_ROOT", Path(__file__).resolve().parents[2]
)).resolve()
CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

CHECKPOINT = (
    ROOT / "outputs/overnight_c_fix_symlog_081234/task2_train/step_004000.pth"
)
REPLAY_DATA = (
    ROOT / "data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2"
)
FAST5_RAW = (
    ROOT
    / "github_reports/homog_label_fix_20261123_v3_s3_fast5_1600"
    / "baseline_task2_step4000/shards/tasks/eval_E00/raw/evaluation_results.json"
)
CONFIG_PATH = ROOT / "config.json"
DEFAULT_OUTPUT = (
    ROOT / "runs/20261125/step4_0_near_target_blindness_probe"
)
TARGETS = ("E00", "E02", "E06", "E08", "E10")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_json(value):
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(safe_json(value), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "mean": float(values.mean()),
    }


def evenly_spaced(items, count):
    items = list(items)
    if len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, int(count), dtype=int)
    return [items[int(index)] for index in indices]


def load_arrays():
    with np.load(REPLAY_DATA / "data/dataset.npz", allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def choose_probe_rows(dataset, arrays):
    errors = np.linalg.norm(
        arrays["truth_navigation_mm"].astype(np.float64), axis=1
    ) * 1000.
    valid_rows = np.asarray(dataset.valid_indices, dtype=np.int64)
    valid_errors = errors[valid_rows]
    previous = np.maximum(valid_rows - 1, 0)
    same_episode = (
        arrays["episode_index"][valid_rows]
        == arrays["episode_index"][previous]
    )
    derivative = np.where(
        same_episode,
        errors[valid_rows] - errors[previous],
        np.inf,
    )
    selections = {}
    for band, threshold, per_class in (
        ("under_60", 60., 100),
        ("under_40", 40., 60),
    ):
        eligible = valid_errors < threshold
        approaching = evenly_spaced(
            [
                int(row) for row, use in zip(valid_rows, eligible & (derivative < -.2))
                if use
            ],
            per_class,
        )
        receding = evenly_spaced(
            [
                int(row) for row, use in zip(valid_rows, eligible & (derivative > .2))
                if use
            ],
            per_class,
        )
        rows = sorted(set(approaching + receding))
        selections[band] = {
            "rows": rows,
            "approaching_rows": approaching,
            "receding_rows": receding,
            "candidate_approaching": int(
                np.count_nonzero(eligible & (derivative < -.2))
            ),
            "candidate_receding": int(
                np.count_nonzero(eligible & (derivative > .2))
            ),
        }
    return selections, errors, derivative


def normalized_scale_probe(checkpoint, config, checkpoint_path=CHECKPOINT):
    cfg = checkpoint["config"]
    oct_std = np.asarray(cfg["oct_std"][:3], dtype=np.float64)
    offsets = {}
    for um in (20., 30.):
        mm = um / 1000.
        offsets[f"{um:g}um"] = {
            "xyz_mm": [mm] * 3,
            "oct_normalized_delta_xyz": (mm / oct_std).tolist(),
            "oct_normalized_delta_norm": float(
                np.linalg.norm(mm / oct_std)
            ),
            "fraction_of_oct_channel_std_xyz": (mm / oct_std).tolist(),
        }
    camera = config["reference_scene"]["eye_camera"]
    camera_position = np.asarray(camera["position_robot_mm"], dtype=float)
    target = np.asarray(camera["target_robot_mm"], dtype=float)
    distance_mm = float(np.linalg.norm(target - camera_position))
    original_w, original_h = camera["resolution_px"]
    fovy = math.radians(float(camera["fovy_deg"]))
    retinal_height_mm = 2. * distance_mm * math.tan(fovy / 2.)
    original_um_per_pixel = retinal_height_mm * 1000. / original_h
    resized_um_per_pixel = retinal_height_mm * 1000. / 224.
    pixel_offsets = {
        f"{um:g}um": {
            "original_1920x1440_px": um / original_um_per_pixel,
            "policy_224x224_px": um / resized_um_per_pixel,
            "below_one_policy_pixel": bool(um < resized_um_per_pixel),
        }
        for um in (20., 30.)
    }
    normalization_channels = {}
    for name in (
        "state_mean", "state_std",
        "oct_mean", "oct_std",
        "action_mean", "action_std",
        "velocity_mean", "velocity_std",
        "previous_action_mean", "previous_action_std",
    ):
        normalization_channels[name] = cfg[name]
    raw_contract = read(FAST5_RAW)["evaluation_contract"]
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "policy_type": checkpoint["policy_type"],
        "architecture": checkpoint["architecture"],
        "image": {
            "stored_resolution": read(REPLAY_DATA / "metadata.json")[
                "observation_space"
            ]["image"]["shape"][:2],
            "native_render_resolution": [int(original_w), int(original_h)],
            "training_resize": [224, 224],
            "fast5_preprocessing": raw_contract["rgb_preprocessing"],
            "normalization_mean_rgb": [.485, .456, .406],
            "normalization_std_rgb": [.229, .224, .225],
            "crop": "none",
            "fovy_deg": float(camera["fovy_deg"]),
            "camera_to_target_distance_mm": distance_mm,
            "retinal_plane_height_mm": retinal_height_mm,
            "original_um_per_pixel": original_um_per_pixel,
            "policy_224_um_per_pixel": resized_um_per_pixel,
            "offset_pixels": pixel_offsets,
            "oct_image_input": False,
            "oct_3d_vector_input": True,
        },
        "normalization": {
            "channels": normalization_channels,
            "channel_noise_metadata_available": False,
            "scale_reference": (
                "No explicit input-noise field is logged; channel std is "
                "the available scale reference."
            ),
            "oct_symlog_enabled": bool(cfg.get("oct_symlog_enabled", False)),
            "oct_symlog_formula_if_enabled": (
                "sign(x)*log1p(abs(x)/0.001) / "
                "log1p(oct_std/0.001).clamp_min(1)"
            ),
            "oct_distance_offsets": offsets,
            "velocity_units": "A_deg/s,B_deg/s,C_mm/s",
            "previous_action_and_action_units": "A_deg,B_deg,C_mm",
        },
        "fast5_contract": {
            "n_obs_steps": int(cfg["n_obs_steps"]),
            "n_action_steps": int(cfg["n_action_steps"]),
            "n_action_execute_steps": 1,
            "open_loop_chunk_steps": 1,
            "num_action_samples": int(raw_contract["num_action_samples"]),
            "action_aggregation": raw_contract["action_aggregation"],
            "flow_integration_steps": int(
                raw_contract["flow_integration_steps"]
            ),
            "replans_every_control_step": True,
            "control_period_s": float(raw_contract["control_period_s"]),
        },
    }


def dual_ridge_fit_predict(x_train, y_train, x_test, alpha):
    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.
    train = np.clip((x_train - x_mean) / x_scale, -10., 10.)
    test = np.clip((x_test - x_mean) / x_scale, -10., 10.)
    y_mean = y_train.mean(axis=0)
    centered = y_train - y_mean
    gram = train @ train.T
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    coefficients = np.linalg.solve(
        gram.astype(np.float64),
        centered.astype(np.float64),
    )
    return (test @ train.T) @ coefficients + y_mean


def linear_probe(rows, features, targets, episodes, bands):
    result = {}
    folds = np.asarray(episodes, dtype=np.int64) % 5
    signs = np.sign(targets)
    for band in ("under_60", "under_40"):
        indices = np.asarray([
            index for index, active in enumerate(bands[band]) if active
        ], dtype=np.int64)
        if len(indices) < 100:
            result[band] = {
                "status": "insufficient_frames",
                "frames": int(len(indices)),
            }
            continue
        band_result = {
            "frames": int(len(indices)),
            "episode_groups": int(len(np.unique(np.asarray(episodes)[indices]))),
            "features": {},
        }
        for feature_name, matrix in features.items():
            prediction = np.zeros((len(indices), 3), dtype=np.float64)
            for fold in range(5):
                test_mask = folds[indices] == fold
                train_mask = ~test_mask
                if not train_mask.any() or not test_mask.any():
                    raise ValueError(f"empty outer fold {fold} for {band}")
                prediction[test_mask] = dual_ridge_fit_predict(
                    matrix[indices[train_mask]],
                    targets[indices[train_mask]],
                    matrix[indices[test_mask]],
                    alpha=1.,
                )
            predicted_sign = np.sign(prediction)
            gt = targets[indices]
            gt_norm = np.linalg.norm(gt, axis=1)
            pred_norm = np.linalg.norm(prediction, axis=1)
            eligible = (gt_norm > 1e-12) & (pred_norm > 1e-12)
            cosines = np.sum(gt * prediction, axis=1) / np.maximum(
                gt_norm * pred_norm, 1e-12
            )
            band_result["features"][feature_name] = {
                "feature_dim": int(matrix.shape[1]),
                "sign_accuracy_xyz": {
                    axis: float(np.mean(
                        predicted_sign[:, index] == signs[indices, index]
                    ))
                    for index, axis in enumerate("XYZ")
                },
                "mean_cosine": float(cosines[eligible].mean()),
                "cosine_median": float(np.median(cosines[eligible])),
                "cosine_p10": float(np.percentile(cosines[eligible], 10)),
                "positive_target_fraction_xyz": {
                    axis: float(np.mean(signs[indices, index] > 0))
                    for index, axis in enumerate("XYZ")
                },
            }
        result[band] = band_result
    return result


def sample_variance(model, batches, selected_indices, truth, device, k=8):
    import torch

    records = []
    for ordinal, dataset_index in enumerate(selected_indices):
        sample = batches[dataset_index]
        tensors = {
            name: value.unsqueeze(0).to(device)
            for name, value in sample.items()
            if name.startswith("observation.") and name != "observation.guided_base"
            and name != "observation.guided_limit"
        }
        generator = torch.Generator(device=device).manual_seed(
            2026112500 + ordinal
        )
        image = tensors["observation.image"].expand(k, *tensors[
            "observation.image"
        ].shape[1:])
        state = tensors["observation.state"].expand(k, *tensors[
            "observation.state"
        ].shape[1:])
        oct_feature = tensors["observation.oct"].expand(k, *tensors[
            "observation.oct"
        ].shape[1:])
        velocity = tensors["observation.velocity"].expand(k, *tensors[
            "observation.velocity"
        ].shape[1:])
        previous = tensors[
            "observation.previous_action"
        ].expand(k, *tensors[
            "observation.previous_action"
        ].shape[1:])
        chunks = model.sample(
            image,
            state,
            oct_feature,
            velocity,
            previous,
            steps=model.flow_integration_steps,
            generator=generator,
        ).detach().cpu().numpy()
        action0 = chunks[:, 0]
        first16 = chunks
        signs0 = np.sign(action0)
        sign_agreement = {}
        for axis_index, axis in enumerate("ABC"):
            positive = float(np.mean(signs0[:, axis_index] > 0))
            sign_agreement[axis] = max(positive, 1. - positive)
        records.append({
            "truth": truth[dataset_index],
            "action0": action0,
            "first16": first16,
            "sign_agreement_action0": sign_agreement,
            "first16_majority_agreement": {
                axis: float(np.mean([
                    max(np.mean(np.sign(first16[:, step, index]) > 0),
                        1. - np.mean(np.sign(first16[:, step, index]) > 0))
                    for step in range(first16.shape[1])
                ]))
                for index, axis in enumerate("ABC")
            },
            "flip_action0": {
                axis: bool(np.any(signs0[:, index] > 0)
                           and np.any(signs0[:, index] < 0))
                for index, axis in enumerate("ABC")
            },
        })
    summary = {}
    for axis_index, axis in enumerate("ABC"):
        action0 = np.asarray([row["action0"][:, axis_index] for row in records])
        ratios = np.abs(action0.std(axis=1) / (
            np.abs(action0.mean(axis=1)) + 1e-12
        ))
        summary[axis] = {
            "mean_sign_agreement_action0": float(np.mean([
                row["sign_agreement_action0"][axis] for row in records
            ])),
            "mean_first16_majority_agreement": float(np.mean([
                row["first16_majority_agreement"][axis]
                for row in records
            ])),
            "fraction_observations_with_sign_flip_action0": float(np.mean([
                row["flip_action0"][axis] for row in records
            ])),
            "std_over_abs_mean_action0": percentile_summary(ratios),
        }
    return {
        "observations": len(records),
        "samples_per_observation": int(k),
        "action0_and_exec_steps_prefix_identical": True,
        "exec_steps": 1,
        "first16_reported_as_chunk_variance_only": True,
        "per_axis": summary,
        "selected_source_rows": [int(value) for value in selected_indices],
    }


def estimate_closed_loop_guard():
    return {
        "status": "skipped_runtime_guard",
        "reason": (
            "fast5 evaluator already uses exec_steps=1, so the requested "
            "reference (a) and replan-every-step (b) are identical. "
            "Condition (c), 2 targets x 1600 steps x K=4, cannot fit the "
            "remaining <=10 minute sub-budget based on the completed 1600-step "
            "fast5 runs."
        ),
        "reference_exec_steps": 1,
        "replan_exec_steps": 1,
        "a_equals_b": True,
        "k4_mean_run": False,
        "training_started": False,
        "weights_modified": False,
    }


def fmt(value, digits=3):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_report(path, metrics):
    p1 = metrics["probe1"]
    probe2 = metrics["probe2"]
    probe3 = metrics["probe3"]
    verdict = metrics["verdict"]
    lines = [
        "# Step 4.0 zero-training near-target blindness probe",
        "",
        "No policy weights, drivers, training scripts, or existing run "
        "directories were modified. No policy retraining or closed-loop "
        "simulation rerun was performed.",
        "",
        "## Probe 1: exact inputs, normalization, and execution horizon",
        "",
        "| Item | Confirmed value |",
        "|---|---|",
        f"| Checkpoint | `{p1['checkpoint']}` |",
        f"| Policy / architecture | `{p1['policy_type']}` / "
        f"`{p1['architecture']}` |",
        f"| Stored / policy image size | 224x224 / 224x224 RGB, no crop |",
        f"| Native render size | {p1['image']['native_render_resolution']} |",
        f"| RGB normalization | ImageNet mean/std; bilinear resize only |",
        f"| OCT input | 3D target-minus-tip vector in mm + validity; no OCT image |",
        f"| OCT symlog | `{p1['normalization']['oct_symlog_enabled']}` |",
        f"| `n_obs_steps` / `n_action_steps` | "
        f"{p1['fast5_contract']['n_obs_steps']} / "
        f"{p1['fast5_contract']['n_action_steps']} |",
        f"| fast5 `exec_steps` | **"
        f"{p1['fast5_contract']['n_action_execute_steps']}** |",
        f"| fast5 samples / flow steps | "
        f"{p1['fast5_contract']['num_action_samples']} / "
        f"{p1['fast5_contract']['flow_integration_steps']} |",
        f"| fast5 action aggregation | "
        f"`{p1['fast5_contract']['action_aggregation']}` |",
        f"| Camera FOV / target distance | "
        f"{p1['image']['fovy_deg']:.1f} deg / "
        f"{p1['image']['camera_to_target_distance_mm']:.3f} mm |",
        f"| Retinal-plane scale | "
        f"{p1['image']['original_um_per_pixel']:.3f} um/px native, "
        f"{p1['image']['policy_224_um_per_pixel']:.3f} um/px at 224 |",
        "",
        "| Offset | Native pixels | 224-policy pixels | Below one 224 pixel? |",
        "|---|---:|---:|---|",
    ]
    for label, value in p1["image"]["offset_pixels"].items():
        lines.append(
            f"| {label} | {value['original_1920x1440_px']:.3f} | "
            f"{value['policy_224x224_px']:.3f} | "
            f"{value['below_one_policy_pixel']} |"
        )
    lines += [
        "",
        "The visual-only signal is sub-pixel at 224x224, but the policy also "
        "receives the continuous OCT 3D vector. Therefore sub-pixel RGB alone "
        "does not establish input blindness.",
        "",
        "| Channel vector | Mean | Std |",
        "|---|---|---|",
    ]
    channels = p1["normalization"]["channels"]
    for name in (
        "state_mean", "state_std", "oct_mean", "oct_std",
        "velocity_mean", "velocity_std",
        "previous_action_mean", "previous_action_std",
        "action_mean", "action_std",
    ):
        lines.append(
            f"| `{name}` | `{channels[name]}` | "
            f"`{'n/a' if not name.endswith('_std') else channels[name]}` |"
        )
    lines += [
        "",
        "| OCT offset | Normalized XYZ delta | Normalized norm |",
        "|---|---|---|",
    ]
    for label, value in p1["normalization"][
        "oct_distance_offsets"
    ].items():
        lines.append(
            f"| {label} | `{value['oct_normalized_delta_xyz']}` | "
            f"{value['oct_normalized_delta_norm']:.5f} |"
        )
    lines += [
        "",
        "With `oct_symlog_enabled=False`, the OCT channel is ordinary "
        "`(value-oct_mean)/oct_std`; a 20-30 um XYZ offset corresponds to "
        "0.02-0.03 mm divided by each OCT std channel. No explicit channel "
        "noise field is logged, so std is the available scale reference.",
        "",
        "## Probe 2: 5-fold grouped linear readout",
        "",
        f"Source frames: `{REPLAY_DATA / 'data/dataset.npz'}`. Rows were "
        "filtered by `norm(truth_navigation_mm) < 60 um`, balanced across "
        "approach/recede, and folds were grouped by source episode.",
        "",
        "| Band | Feature | Frames | Sign accuracy X/Y/Z | Mean cosine | "
        "Median cosine | P10 cosine |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for band in ("under_60", "under_40"):
        value = probe2[band]
        if value.get("status") == "insufficient_frames":
            lines.append(f"| {band} | n/a | {value['frames']} | insufficient |")
            continue
        for feature_name in ("F_vis", "F_cond", "F_oct"):
            item = value["features"][feature_name]
            acc = item["sign_accuracy_xyz"]
            lines.append(
                f"| {band} | {feature_name} | {value['frames']} | "
                f"{fmt(acc['X'])}/{fmt(acc['Y'])}/{fmt(acc['Z'])} | "
                f"{fmt(item['mean_cosine'])} | "
                f"{fmt(item['cosine_median'])} | "
                f"{fmt(item['cosine_p10'])} |"
            )
    lines += [
        "",
        "Folds are grouped by source episode. `F_vis` is the full flattened "
        "ResNet token embedding; `F_cond` is the full flattened condition-token "
        "tensor plus global condition entering the diffusion head; `F_oct` is "
        "the deployable 3D OCT vector and serves as the reference upper bound.",
        "",
        "## Probe 3: diffusion sampling variance",
        "",
        f"- Observations: {probe3['observations']} balanced under-60 um frames.",
        f"- Samples per observation: K={probe3['samples_per_observation']}.",
        f"- fast5 executes {probe3['exec_steps']} step; action[0] and "
        "action[0:exec_steps] are identical.",
        "",
        "| Axis | Sign agreement action0 | First-16 majority agreement | "
        "Obs. with sign flip action0 | std/|mean| median | std/|mean| p90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for axis in "ABC":
        value = probe3["per_axis"][axis]
        ratio = value["std_over_abs_mean_action0"]
        lines.append(
            f"| {axis} | "
            f"{fmt(value['mean_sign_agreement_action0'])} | "
            f"{fmt(value['mean_first16_majority_agreement'])} | "
            f"{fmt(value['fraction_observations_with_sign_flip_action0'])} | "
            f"{fmt(ratio['median'])} | {fmt(ratio['p90'])} |"
        )
    lines += [
        "",
        "## Receding-horizon check",
        "",
        f"- Status: `{metrics['receding_horizon']['status']}`.",
        f"- {metrics['receding_horizon']['reason']}",
        "- No full 1600-step simulation was launched.",
        "",
        "## Binary verdict",
        "",
        f"**Selected Stage 4 change: {verdict['decision']}**",
        "",
        f"{verdict['reason']}",
        "",
        "Read the measured per-axis K=8 values above; the original baseline "
        "conclusion is not reused for this checkpoint.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import torch
    from lerobot_training.dit_experience_replay import _load_policy
    from lerobot_training.train_diffusion_policy import (
        RCMDataset,
        image_transform_for_dataset,
    )

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = read(CONFIG_PATH)
    probe1 = normalized_scale_probe(
        checkpoint, config, checkpoint_path=checkpoint_path,
    )
    arrays = load_arrays()
    dataset = RCMDataset(
        REPLAY_DATA,
        transform=image_transform_for_dataset(REPLAY_DATA),
        n_obs_steps=int(checkpoint["config"]["n_obs_steps"]),
        n_action_steps=int(checkpoint["config"]["n_action_steps"]),
        include_velocity=True,
        include_previous_action=True,
    )
    selections, errors, derivative = choose_probe_rows(dataset, arrays)
    selected_rows = sorted(set(
        row for value in selections.values() for row in value["rows"]
    ))
    dataset_index_by_row = {
        int(row): index for index, row in enumerate(dataset.valid_indices)
    }
    selected_indices = [
        dataset_index_by_row[row] for row in selected_rows
    ]
    model, _, _ = _load_policy(
        checkpoint_path,
        torch.device(args.device),
        oct_symlog_enabled=bool(
            checkpoint["config"].get("oct_symlog_enabled", False)
        ),
    )
    model.eval()
    feature_arrays = {"F_vis": [], "F_cond": [], "F_geom_cond": [], "F_oct": []}
    batches = {}
    with torch.inference_mode():
        for start in range(0, len(selected_indices), 16):
            batch_indices = selected_indices[start:start + 16]
            samples = [dataset[index] for index in batch_indices]
            for index, sample in zip(batch_indices, samples):
                batches[index] = sample
            tensor_batch = {
                name: torch.stack([sample[name] for sample in samples]).to(
                    args.device
                )
                for name in (
                    "observation.image",
                    "observation.state",
                    "observation.oct",
                    "observation.velocity",
                    "observation.previous_action",
                )
            }
            visual = model.vision(tensor_batch["observation.image"])
            condition_tokens, global_condition = model._condition(
                tensor_batch["observation.image"],
                tensor_batch["observation.state"],
                tensor_batch["observation.oct"],
                tensor_batch["observation.velocity"],
                tensor_batch["observation.previous_action"],
            )
            if isinstance(global_condition, tuple):
                global_condition = global_condition[0] + global_condition[1]
            feature_arrays["F_geom_cond"].append(global_condition.cpu().numpy())
            feature_arrays["F_vis"].append(
                visual.flatten(1).cpu().numpy()
            )
            feature_arrays["F_cond"].append(torch.cat([
                condition_tokens.flatten(1),
                global_condition,
            ], dim=1).cpu().numpy())
            feature_arrays["F_oct"].append(
                tensor_batch["observation.oct"][:, -1, :3].cpu().numpy()
            )
    features = {
        name: np.concatenate(chunks, axis=0).astype(np.float32)
        for name, chunks in feature_arrays.items()
    }
    selected_truth = np.asarray([
        arrays["truth_navigation_mm"][row] for row in selected_rows
    ], dtype=np.float64)
    selected_episode = np.asarray([
        arrays["episode_index"][row] for row in selected_rows
    ])
    selected_error = errors[selected_rows]
    bands = {
        band: selected_error < threshold
        for band, threshold in (("under_60", 60.), ("under_40", 40.))
    }
    probe2 = linear_probe(
        selected_rows,
        features,
        selected_truth,
        selected_episode,
        bands,
    )
    under60_selection = selections["under_60"]
    approach_rows = under60_selection["approaching_rows"][:15]
    recede_rows = under60_selection["receding_rows"][:15]
    sample_rows = sorted(set(approach_rows + recede_rows))
    sample_indices = [dataset_index_by_row[row] for row in sample_rows]
    truth_by_dataset_index = {
        dataset_index_by_row[row]: arrays["truth_navigation_mm"][row]
        for row in sample_rows
    }
    probe3 = sample_variance(
        model,
        batches,
        sample_indices,
        truth_by_dataset_index,
        args.device,
        k=8,
    )
    cond60 = probe2["under_60"]["features"]["F_cond"]
    oct60 = probe2["under_60"]["features"]["F_oct"]
    cond_ab_accuracy = min(
        cond60["sign_accuracy_xyz"]["X"],
        cond60["sign_accuracy_xyz"]["Y"],
    )
    oct_ab_accuracy = min(
        oct60["sign_accuracy_xyz"]["X"],
        oct60["sign_accuracy_xyz"]["Y"],
    )
    flip_ab = max(
        probe3["per_axis"]["A"][
            "fraction_observations_with_sign_flip_action0"
        ],
        probe3["per_axis"]["B"][
            "fraction_observations_with_sign_flip_action0"
        ],
    )
    if (
        (cond_ab_accuracy < .65 or cond60["mean_cosine"] < .30)
        and oct_ab_accuracy > .85
    ):
        decision = "A: Input Representation / Scaling"
        reason = (
            "F_cond fails the near-target direction criterion while raw OCT "
            "direction remains strong."
        )
    elif cond_ab_accuracy >= .75 and (
        flip_ab >= .20
        or probe3["per_axis"]["A"]["mean_sign_agreement_action0"] < .75
        or probe3["per_axis"]["B"]["mean_sign_agreement_action0"] < .75
    ):
        decision = "B: Short-Horizon Execution / Direction Loss"
        reason = (
            "F_cond linearly recovers near-target direction, but independent "
            "diffusion samples frequently disagree in action sign."
        )
    elif cond_ab_accuracy >= .75:
        decision = "B: Short-Horizon Execution / Direction Loss"
        reason = (
            "F_cond linearly recovers near-target direction. The fast5 "
            "evaluator already replans every step, so Stage 4 should target "
            "direction behavior and minimax aggregation rather than chunking."
        )
    else:
        decision = "A: Input Representation / Scaling"
        reason = (
            "F_cond is not a reliable linear near-target direction readout "
            "under the pre-registered criteria."
        )
    verdict = {
        "decision": decision,
        "reason": reason,
        "f_cond_ab_sign_accuracy": cond_ab_accuracy,
        "f_cond_mean_cosine": cond60["mean_cosine"],
        "f_oct_ab_sign_accuracy": oct_ab_accuracy,
        "max_ab_sample_flip_fraction": flip_ab,
        "visual_20um_below_one_policy_pixel": (
            probe1["image"]["offset_pixels"]["20um"][
                "below_one_policy_pixel"
            ]
        ),
    }
    metrics = {
        "status": "completed",
        "elapsed_s": time.monotonic() - started,
        "read_only_and_zero_training": True,
        "feature_dimensions": {name: int(values.shape[1]) for name, values in features.items()},
        "probe1": probe1,
        "probe2": probe2,
        "probe2_selection": {
            band: {
                key: (
                    len(value)
                    if isinstance(value, list) else value
                )
                for key, value in selection.items()
            }
            for band, selection in selections.items()
        },
        "probe3": probe3,
        "receding_horizon": estimate_closed_loop_guard(),
        "verdict": verdict,
    }
    write_json(output / "metrics.json", metrics)
    write_report(output / "report.md", metrics)
    print(json.dumps({
        "output_dir": str(output),
        "elapsed_s": metrics["elapsed_s"],
        "probe2_under60": probe2["under_60"]["features"],
        "probe3_per_axis": probe3["per_axis"],
        "verdict": verdict,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
