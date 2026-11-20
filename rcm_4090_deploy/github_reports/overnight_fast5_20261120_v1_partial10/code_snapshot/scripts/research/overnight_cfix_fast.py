"""Authorized fast5 Task 3 -> gated symlog Task 4 -> one conditional dev20."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.research.overnight_cfix import (
    DATA, EVAL_MODULE, MODULE, TARGETS, Interrupted, Runner, continuation_identity,
    continuation_inputs, digest, read, write,
)

FAST5 = (0, 2, 6, 8, 10)
HISTORICAL = Path("runs/20261119/overnight_c_fix_symlog_081234/task2_step4000")
MODULE_NAME = "scripts.research.overnight_cfix_fast"


def identity():
    return {**continuation_identity(), "fast_runner_sha256": digest(Path(__file__))}


def episode_rows(folder):
    with (folder / "episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {row["target_id"]: row for row in rows}
    if len(result) != len(rows) or not result:
        raise ValueError("duplicate or absent target_id")
    return result


def paired_rows(baseline, candidate):
    """Pair a candidate subset by ID, rejecting mismatched seeds or nonfinite data."""
    if not candidate or not set(candidate).issubset(baseline):
        raise ValueError("candidate target_id missing from baseline")
    pairs = []
    for target_id, trial in candidate.items():
        base = baseline[target_id]
        if int(base["seed"]) != int(trial["seed"]) or base["episode"] != trial["episode"]:
            raise ValueError(f"seed/episode mismatch for {target_id}")
        b, c = [float(row["final_error_um"]) for row in (base, trial)]
        bm, cm = [float(row["minimum_error_um"]) for row in (base, trial)]
        if not all(math.isfinite(value) and value >= 0 for value in (b, c, bm, cm)):
            raise ValueError("nonfinite or negative paired errors")
        pairs.append({"target_id": target_id, "episode": trial["episode"], "seed": int(trial["seed"]),
                      "baseline_final_um": b, "candidate_final_um": c,
                      "baseline_minimum_um": bm, "candidate_minimum_um": cm,
                      "final_delta_um": c-b, "minimum_delta_um": cm-bm})
    return pairs


def comparison(baseline_folder, candidate_folder, output=None):
    base, trial = [read(folder / "summary.json") for folder in (baseline_folder, candidate_folder)]
    for key in ("status", "arm", "max_steps", "tier", "error_threshold_um", "flow_steps", "samples"):
        if base[key] != trial[key] or base["status"] != "completed":
            raise ValueError(f"paired protocol mismatch: {key}")
    pairs = paired_rows(episode_rows(baseline_folder), episode_rows(candidate_folder))
    if len(pairs) != trial["target_count"]:
        raise ValueError("partial comparison")
    means = {key: sum(row[key] for row in pairs) / len(pairs) for key in (
        "baseline_final_um", "candidate_final_um", "baseline_minimum_um", "candidate_minimum_um")}
    final_gain = 1 - means["candidate_final_um"] / means["baseline_final_um"] if means["baseline_final_um"] else 0
    min_gain = 1 - means["candidate_minimum_um"] / means["baseline_minimum_um"] if means["baseline_minimum_um"] else 0
    result = {**means, "paired_targets": len(pairs), "arm": trial["arm"],
              "baseline": str(baseline_folder), "candidate": str(candidate_folder),
              "final_improvement_fraction": final_gain, "minimum_improvement_fraction": min_gain,
              "improved_final_count": sum(row["final_delta_um"] < 0 for row in pairs),
              "gate_passed": final_gain >= 0.10 - 1e-12 or min_gain >= 0.10 - 1e-12,
              "rule": "same-arm paired fast5 final mean OR minimum mean improves at least 10%"}
    if output is not None:
        write(output, {**result, "pairs": pairs})
    return result


def near_target_plot(folder):
    """Reconstruct recorded FK only; no policy inference or simulation rollout."""
    import numpy as np
    from lerobot_training.plot_closed_loop_errors import plt
    from lerobot_training.plot_trajectory_3d import state_to_task
    from rcm.scene_v9 import RCMSceneV9

    fig, axes = plt.subplots(4, 5, figsize=(16, 12), layout="constrained")
    scene = RCMSceneV9()
    evidence = []
    try:
        for index, ax in enumerate(axes.flat):
            episode = read(folder / f"E{index:02d}/evaluation_results.json")["episodes"][0]
            target = np.asarray(episode["target_world_mm"])
            states = np.vstack((episode["trajectory"]["states"], episode["final_state_A_deg_B_deg_C_mm"]))
            relative = []
            for state in states:
                scene.set_task(state_to_task(state))
                relative.append(scene.site_mm("needle_tip") - target)
            relative = np.asarray(relative)
            errors = np.asarray([*episode["trajectory"]["errors"], episode["final_error_mm"]])
            delta = float(np.max(np.abs(np.linalg.norm(relative, axis=1) - errors)))
            if delta >= 1e-3:
                raise ValueError(f"E{index:02d}: FK reconstruction mismatch {delta} mm")
            xy = relative * 1000
            ax.plot(xy[:, 0], xy[:, 1], linewidth=0.9)
            ax.scatter(0, 0, marker="*", c="black", s=45)
            ax.scatter(*xy[-1, :2], marker="s", s=25, label="Final")
            ax.set(xlim=(-150, 150), ylim=(-150, 150), aspect="equal", xlabel="X (um)", ylabel="Y (um)",
                   title=f"E{index:02d}: {errors[-1]*1000:.2f} um (3D)")
            ax.grid(alpha=0.2)
            evidence.append({"episode": index, "max_reconstruction_delta_mm": delta})
        fig.suptitle("Winning candidate: target-relative needle-tip XY paths, clipped to +/-150 um")
        fig.savefig(folder / "needle_tip_near_target.png", dpi=150)
    finally:
        scene.close()
        plt.close(fig)
    write(folder / "needle_tip_reconstruction.json", {"status": "passed", "episodes": evidence})


class FastRunner(Runner):
    def __init__(self, output, workers):
        super().__init__(output, continuation=True)
        self.workers = workers
        self.hours = 12
        self.deadline = time.monotonic() + self.hours * 3600

    def evaluate_tier(self, name, checkpoint, arm, tier="fast5"):
        out = self.output / name
        count, steps = (5, 1000) if tier == "fast5" else (20, 1600)
        with self.stage(name, output=str(out), checkpoint=str(checkpoint), arm=arm, tier=tier,
                        protocol=f"{tier}; original indices; seeds 42+index; {steps} steps; 50 ms; 0.35 um") as detail:
            current = identity()
            if current != self.source_identity:
                raise ValueError("code/scene changed after preflight")
            weight_hash = digest(checkpoint)
            self.require_ok(self.cmd(name, EVAL_MODULE, "--checkpoint", checkpoint,
                                    "--target-manifest", TARGETS, "--output-dir", out, "--arm", arm,
                                    "--tier", tier, "--workers", self.workers,
                                    timeout_s=5400 if tier == "fast5" else 14400))
            result = read(out / "summary.json")
            if (result["status"] != "completed" or result["target_count"] != count
                    or result["max_steps"] != steps or result["arm"] != arm
                    or Path(result["checkpoint"]) != checkpoint
                    or result["episode_indices"] != (list(FAST5) if tier == "fast5" else list(range(20)))):
                raise ValueError("partial or mismatched evaluation")
            if identity() != current or digest(checkpoint) != weight_hash:
                raise ValueError("inputs changed during evaluation")
            write(out / "provenance.json", {**current, "checkpoint_sha256": weight_hash,
                  "summary_sha256": digest(out / "summary.json"), "tier": tier,
                  "episode_indices": result["episode_indices"], "seeds": [42+i for i in result["episode_indices"]],
                  "max_steps": steps, "arm": arm, "workers": self.workers})
            detail.update(result)
        return result

    def run_fast(self, receipt):
        with self.stage("continuation_preflight", asset_receipt=str(receipt)) as detail:
            import torch
            checkpoint, overlay = continuation_inputs(receipt)
            if self.checkpoints.exists():
                raise FileExistsError(self.checkpoints)
            if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
                raise RuntimeError("RTX 4090 CUDA required")
            self.source_identity = identity()
            detail.update(checkpoint=str(checkpoint), dataset=str(DATA), overlay=str(overlay),
                          source_identity=self.source_identity, workers=self.workers,
                          gpu=torch.cuda.get_device_name(0), total_budget_hours=self.hours)
        arms = {arm: self.evaluate_tier(f"task3_{arm}", checkpoint, arm) for arm in ("I0", "I1", "I3")}
        baseline = arms["I0"]
        eligible = [row for arm, row in arms.items() if arm != "I0"
                    and row["inference_median_ms"] < 50 and row["inference_p90_ms"] < 50
                    and row["final_mean_um"] <= baseline["final_mean_um"]
                    and (row["final_mean_um"] <= 0.9 * baseline["final_mean_um"]
                         or row["minimum_mean_um"] <= 0.9 * baseline["minimum_mean_um"])]
        best_arm = min(eligible, key=lambda row: row["final_mean_um"])["arm"] if eligible else "I0"
        task3_pairs = {}
        for arm in ("I1", "I3"):
            task3_pairs[arm] = paired_rows(episode_rows(self.output / "task3_I0"),
                                           episode_rows(self.output / f"task3_{arm}"))
        write(self.output / "task3_paired.json", task3_pairs)
        self.record("task3_selection", "completed", {"arm": best_arm, "checkpoint": str(checkpoint),
                    "rule": ">=10% final/minimum mean improvement; final mean cannot regress; median AND P90 <50ms; otherwise I0",
                    "latency_context": "five concurrent GPU workers; conservative observed realtime gate",
                    "paired_evidence": str(self.output / "task3_paired.json"),
                    "summaries": arms})
        with self.stage("task4_step_zero", checkpoint=str(checkpoint), dataset=str(DATA),
                        output=str(self.output / "step_zero_selected.json")) as detail:
            if continuation_inputs(receipt) != (checkpoint, overlay) or identity() != self.source_identity:
                raise ValueError("assets/code changed before step-zero")
            self.require_ok(self.cmd("task4_step_zero", MODULE, "step-zero", "--checkpoint", checkpoint,
                                    "--output-dir", self.output, timeout_s=600))
            gate = read(self.output / "step_zero_selected.json")
            detail.update(gate)
            if gate["status"] != "PASS" or not 0 <= gate["max_abs_diff"] < 1e-6:
                raise ValueError("step-zero gate failed; training stopped")
        trained = self.train("task4_train", checkpoint, overlay, symlog=True)
        candidates = []
        for step in (2000, 4000):
            for arm in dict.fromkeys(("I0", best_arm)):
                name = f"task4_step{step}_{arm}"
                score = self.evaluate_tier(name, trained / f"step_{step:06d}.pth", arm)
                paired = comparison(self.output / f"task3_{arm}", self.output / name,
                                    self.output / name / "paired_comparison.json")
                candidates.append((name, score, paired))
        name, winner, paired = min(candidates, key=lambda item: item[1]["final_mean_um"])
        self.record("task4_selection", "completed", {"winner": name, "checkpoint": winner["checkpoint"],
                    "arm": winner["arm"], "paired": paired,
                    "rule": "minimum fast5 final mean across both checkpoints and I0/BEST_ARM"})
        if not paired["gate_passed"]:
            self.record("task4_dev20", "skipped", {"reason": "fast5 final/minimum mean did not improve >=10% under same arm",
                        "paired": paired, "dependent_stages": ["dev20_plots", "dev20_historical_comparison"]})
            return
        result = self.evaluate_tier("task4_dev20", Path(winner["checkpoint"]), winner["arm"], "dev20")
        with self.stage("task4_dev20_report", output=str(self.output / "task4_dev20")) as detail:
            out = self.output / "task4_dev20"
            near_target_plot(out)
            historical_pairs = paired_rows(episode_rows(HISTORICAL), episode_rows(out))
            reference = {"reference": str(HISTORICAL), "reference_final_mean_um": 53.9499234,
                         "reference_arm": "I0", "candidate_arm": winner["arm"],
                         "provenance_matched": False,
                         "limitation": "Historical reference lacks code/scene fingerprints; cross-arm if candidate is not I0. Not a fresh matched baseline.",
                         "paired_targets": len(historical_pairs),
                         "improved_final_count": sum(row["final_delta_um"] < 0 for row in historical_pairs),
                         "candidate_final_mean_um": result["final_mean_um"], "pairs": historical_pairs}
            write(out / "historical_reference.json", reference)
            required = ("summary.json", "episodes.csv", "error_curves_all.png", "best_and_final_error.png",
                        "c_min_aligned.png", "needle_tip_near_target.png")
            if any(not (out / file).is_file() for file in required):
                raise ValueError("final artifacts missing")
            detail.update(artifacts=[str(out / file) for file in required],
                          historical_provenance_matched=False, strict_success_count=result["canonical_success_count"],
                          target_count=20, historical_reference=str(out / "historical_reference.json"))

    def finish(self, status):
        lines = ["# Task 3 / Task 4 fast5 experiment", "", f"Workflow: **{status}**", "",
                 "MuJoCo simulation only. Fast5 screening is distinct from dev20 confirmation.",
                 "Metric improvement and strict success are separate; partial shards are never final results.", "",
                 "| Stage | State | Targets / steps | Final mean um | Min mean um | Strict success | Median/P90 ms |",
                 "|---|---|---|---|---|---|---|"]
        for name, row in self.results.items():
            if "final_mean_um" in row:
                lines.append(f"| {name} | {row['status']} | {row['target_count']}/{row['max_steps']} | "
                             f"{row['final_mean_um']:.3f} | {row['minimum_mean_um']:.3f} | "
                             f"{row['canonical_success_count']}/{row['target_count']} | "
                             f"{row['inference_median_ms']:.2f}/{row['inference_p90_ms']:.2f} |")
            else:
                lines.append(f"| {name} | {row['status']} | | | | | |")
            if "reason" in row:
                lines += ["", f"{name}: {row['reason']}", ""]
        for stage in ("task3_selection", "task4_selection", "task4_dev20"):
            if stage in self.results:
                import json
                lines += ["", f"## {stage}", "", "```json", json.dumps(self.results[stage], indent=2), "```"]
        lines += ["", "Historical 53.950 um is an I0 reference without source/scene provenance, not a fresh matched baseline.",
                  f"Evidence: `{self.output}`; per-stage records: `{self.reports}`."]
        (self.reports / "SUMMARY.md").write_text("\n".join(lines) + "\n")
        self.status(status)
        print(f"Experiment {status}: {self.reports / 'SUMMARY.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5, choices=range(2, 6))
    args = parser.parse_args()

    def stop(signum, _frame):
        raise Interrupted(f"received {signal.Signals(signum).name}")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    runner = FastRunner(args.output_dir, args.workers)
    try:
        runner.run_fast(args.asset_receipt)
    except (Interrupted, KeyboardInterrupt, subprocess.TimeoutExpired):
        runner.finish("interrupted")
        return 130
    except BaseException:
        runner.finish("failed")
        raise
    runner.finish("completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
