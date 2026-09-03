from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def bootstrap_mean_ci(
    values: list[float],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    rng = random.Random(seed)
    size = len(values)
    means = [
        statistics.fmean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    ]
    means.sort()
    lower = means[int(0.025 * (samples - 1))]
    upper = means[int(0.975 * (samples - 1))]
    return lower, upper


def quality_relation(row: dict[str, str]) -> int:
    selected = (int(row["selected_swap"]), int(row["selected_depth"]))
    baseline = (
        int(row["lightsabre_swap"]),
        int(row["lightsabre_depth"]),
    )
    return -1 if selected < baseline else 1 if selected > baseline else 0


def group_rows(
    rows: list[dict[str, str]],
    field: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)

    output: list[dict[str, Any]] = []
    for value in sorted(grouped, key=str):
        items = grouped[value]
        ls_total = sum(int(row["lightsabre_swap"]) for row in items)
        selected_total = sum(int(row["selected_swap"]) for row in items)
        normalized = [
            (
                int(row["selected_swap"])
                - int(row["lightsabre_swap"])
            )
            / int(row["cx_count"])
            for row in items
        ]
        output.append(
            {
                "group_field": field,
                "group_value": value,
                "cases": len(items),
                "lightsabre_swap": ls_total,
                "selected_swap": selected_total,
                "gain": ls_total - selected_total,
                "relative_gain_pct": (
                    100.0 * (ls_total - selected_total) / ls_total
                    if ls_total
                    else 0.0
                ),
                "swap_wins": sum(
                    int(row["selected_swap"])
                    < int(row["lightsabre_swap"])
                    for row in items
                ),
                "swap_ties": sum(
                    int(row["selected_swap"])
                    == int(row["lightsabre_swap"])
                    for row in items
                ),
                "swap_losses": sum(
                    int(row["selected_swap"])
                    > int(row["lightsabre_swap"])
                    for row in items
                ),
                "mean_normalized_gap": statistics.fmean(normalized),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 56: preregistered analysis of blind V6 results"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/benchmarks_v6/blind_test.json"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("results/stage55_v6_blind/cases.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage56_v6_analysis"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    rows = read_csv(args.cases)
    expected_ids = [str(case["name"]) for case in dataset["cases"]]
    actual_ids = [str(row["instance"]) for row in rows]

    if len(rows) != 120:
        raise ValueError(f"expected 120 result rows, got {len(rows)}")
    if actual_ids != expected_ids:
        raise ValueError("result case order/identity does not match V6 dataset")

    normalized_gaps = [
        (int(row["selected_swap"]) - int(row["lightsabre_swap"]))
        / int(row["cx_count"])
        for row in rows
    ]
    ci_low, ci_high = bootstrap_mean_ci(
        normalized_gaps,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )

    ls_total = sum(int(row["lightsabre_swap"]) for row in rows)
    oaabr_total = sum(int(row["oaabr_swap"]) for row in rows)
    dual_total = sum(int(row["safe_dual_swap"]) for row in rows)
    selected_total = sum(int(row["selected_swap"]) for row in rows)
    swap_wins = sum(
        int(row["selected_swap"]) < int(row["lightsabre_swap"])
        for row in rows
    )
    swap_losses = sum(
        int(row["selected_swap"]) > int(row["lightsabre_swap"])
        for row in rows
    )
    quality_relations = [quality_relation(row) for row in rows]
    semantic_passed = sum(is_true(row["semantic_equivalent"]) for row in rows)
    topology_passed = sum(is_true(row["topology_valid"]) for row in rows)
    metric_passed = sum(is_true(row["metric_valid"]) for row in rows)

    checks = {
        "case_count_120": len(rows) == 120,
        "semantic_120": semantic_passed == 120,
        "topology_120": topology_passed == 120,
        "metric_120": metric_passed == 120,
        "no_per_case_swap_regressions": swap_losses == 0,
        "selected_total_not_above_lightsabre": selected_total <= ls_total,
        "bootstrap_ci_upper_below_zero": ci_high < 0.0,
    }
    passed = all(checks.values())

    group_output = [
        item
        for field in (
            "topology",
            "circuit_mode",
            "num_qubits",
        )
        for item in group_rows(rows, field)
    ]
    # length_factor is not currently emitted by Stage 53 cases.csv; CX and n
    # retain enough information to reconstruct it exactly.
    for row in rows:
        row["length_factor"] = str(
            int(row["cx_count"]) // int(row["num_qubits"])
        )
    group_output.extend(group_rows(rows, "length_factor"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    group_path = args.output_dir / "groups.csv"
    summary_path = args.output_dir / "summary.json"
    write_csv(group_path, group_output)

    summary = {
        "stage": 56,
        "purpose": "preregistered blind V6 evaluation",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "cases": str(args.cases.resolve()),
        "cases_sha256": sha256(args.cases),
        "case_count": len(rows),
        "totals": {
            "lightsabre_swap": ls_total,
            "oaabr_swap": oaabr_total,
            "safe_dual_swap": dual_total,
            "selected_swap": selected_total,
            "gain_vs_lightsabre": ls_total - selected_total,
            "gain_vs_safe_dual": dual_total - selected_total,
            "relative_gain_vs_lightsabre_pct": (
                100.0 * (ls_total - selected_total) / ls_total
            ),
        },
        "swap_win_tie_loss": {
            "win": swap_wins,
            "tie": len(rows) - swap_wins - swap_losses,
            "loss": swap_losses,
        },
        "quality_win_tie_loss": {
            "win": quality_relations.count(-1),
            "tie": quality_relations.count(0),
            "loss": quality_relations.count(1),
        },
        "selected_backends": dict(
            Counter(row["selected_backend"] for row in rows)
        ),
        "semantic_passed": semantic_passed,
        "topology_passed": topology_passed,
        "metric_passed": metric_passed,
        "mean_normalized_gap": statistics.fmean(normalized_gaps),
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "ci95": [ci_low, ci_high],
        },
        "checks": checks,
        "passed": passed,
        "groups": str(group_path.resolve()),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("===== Stage 56 blind V6 analysis =====")
    print("cases:", len(rows))
    print("LightSABRE total SWAP:", ls_total)
    print("OAABR total SWAP:", oaabr_total)
    print("safe dual total SWAP:", dual_total)
    print("selected total SWAP:", selected_total)
    print("gain vs LightSABRE:", ls_total - selected_total)
    print("gain vs safe dual:", dual_total - selected_total)
    print(
        "SWAP W/T/L:",
        f"{swap_wins}/{len(rows)-swap_wins-swap_losses}/{swap_losses}",
    )
    print(
        "quality W/T/L:",
        f"{quality_relations.count(-1)}/"
        f"{quality_relations.count(0)}/"
        f"{quality_relations.count(1)}",
    )
    print(
        "mean normalized gap:",
        f"{statistics.fmean(normalized_gaps):+.6f}",
    )
    print("bootstrap 95% CI:", f"[{ci_low:+.6f}, {ci_high:+.6f}]")
    print("audits:", semantic_passed, topology_passed, metric_passed)
    print("checks:", checks)
    print("PASSED:", passed)
    print("summary:", summary_path.resolve())
    print("groups:", group_path.resolve())


if __name__ == "__main__":
    main()
