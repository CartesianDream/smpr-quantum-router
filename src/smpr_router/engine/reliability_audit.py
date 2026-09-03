from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_TOPOLOGIES = {"grid", "line", "random", "ring"}
EXPECTED_MODES = {
    "alternating",
    "far",
    "parallel_layers",
    "phase_shift",
    "uniform",
}
EXPECTED_QUBITS = {6, 8, 10}
EXPECTED_LENGTH_FACTORS = {3, 6, 12}
REQUIRED_COLUMNS = {
    "case_index",
    "instance",
    "topology",
    "circuit_mode",
    "num_qubits",
    "cx_count",
    "lightsabre_swap",
    "lightsabre_depth",
    "independent_rollout_swap",
    "independent_rollout_depth",
    "baseline_rollout_safe_swap",
    "baseline_rollout_safe_depth",
    "selected_backend",
    "selected_swap",
    "selected_depth",
    "gain_vs_lightsabre",
    "gain_vs_baseline_rollout",
    "initial_mapping",
    "final_mapping",
    "semantic_equivalent",
    "topology_valid",
    "metric_valid",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def mapping_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def length_factor(row: dict[str, str]) -> int:
    num_qubits = int(row["num_qubits"])
    cx_count = int(row["cx_count"])
    if cx_count % num_qubits:
        raise ValueError(
            f"{row['instance']}: cx_count={cx_count} is not divisible by "
            f"num_qubits={num_qubits}"
        )
    return cx_count // num_qubits


def factor_cell(row: dict[str, str]) -> tuple[str, str, int]:
    return (
        row["topology"],
        row["circuit_mode"],
        length_factor(row),
    )


def normalized_gap(row: dict[str, str]) -> float:
    return (
        int(row["selected_swap"]) - int(row["lightsabre_swap"])
    ) / int(row["cx_count"])


def validate_rows(
    rows: list[dict[str, str]],
    *,
    name: str,
    expected_count: int,
    expected_cluster_size: int,
) -> dict[str, Any]:
    if len(rows) != expected_count:
        raise ValueError(f"{name}: expected {expected_count} rows, got {len(rows)}")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"{name}: missing columns: {sorted(missing)}")

    checks: dict[str, bool] = {}
    checks["case_indices_contiguous"] = [
        int(row["case_index"]) for row in rows
    ] == list(range(expected_count))
    checks["instances_unique"] = len({row["instance"] for row in rows}) == len(rows)
    checks["topology_levels_exact"] = {
        row["topology"] for row in rows
    } == EXPECTED_TOPOLOGIES
    checks["mode_levels_exact"] = {
        row["circuit_mode"] for row in rows
    } == EXPECTED_MODES
    checks["qubit_levels_exact"] = {
        int(row["num_qubits"]) for row in rows
    } == EXPECTED_QUBITS
    checks["length_levels_exact"] = {
        length_factor(row) for row in rows
    } == EXPECTED_LENGTH_FACTORS
    checks["all_semantic_audits_pass"] = all(
        is_true(row["semantic_equivalent"]) for row in rows
    )
    checks["all_topology_audits_pass"] = all(
        is_true(row["topology_valid"]) for row in rows
    )
    checks["all_metric_audits_pass"] = all(
        is_true(row["metric_valid"]) for row in rows
    )
    checks["selected_swap_floor"] = all(
        int(row["selected_swap"]) <= int(row["lightsabre_swap"])
        for row in rows
    )
    checks["selected_quality_floor"] = all(
        (int(row["selected_swap"]), int(row["selected_depth"]))
        <= (int(row["lightsabre_swap"]), int(row["lightsabre_depth"]))
        for row in rows
    )
    checks["baseline_rollout_safe_swap_floor"] = all(
        int(row["baseline_rollout_safe_swap"]) <= int(row["lightsabre_swap"])
        for row in rows
    )
    checks["gain_vs_lightsabre_consistent"] = all(
        int(row["gain_vs_lightsabre"])
        == int(row["lightsabre_swap"]) - int(row["selected_swap"])
        for row in rows
    )
    checks["gain_vs_baseline_rollout_consistent"] = all(
        int(row["gain_vs_baseline_rollout"])
        == int(row["baseline_rollout_safe_swap"]) - int(row["selected_swap"])
        for row in rows
    )

    mapping_checks: list[bool] = []
    for row in rows:
        expected_mapping = tuple(range(int(row["num_qubits"])))
        for column in ("initial_mapping", "final_mapping"):
            mapping = mapping_tuple(row[column])
            mapping_checks.append(
                len(mapping) == len(expected_mapping)
                and tuple(sorted(mapping)) == expected_mapping
            )
    checks["mappings_are_permutations"] = all(mapping_checks)

    clusters: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        clusters[factor_cell(row)].append(row)
    checks["full_factorial_60_cells"] = len(clusters) == 60
    checks["factor_cell_sizes_exact"] = all(
        len(items) == expected_cluster_size for items in clusters.values()
    )

    failed = [check for check, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{name}: validation failed: {failed}")

    return {
        "name": name,
        "row_count": len(rows),
        "factor_cells": len(clusters),
        "factor_cell_size": expected_cluster_size,
        "checks": checks,
        "passed": True,
    }


def percentile_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    )
    return (
        means[int(0.025 * (samples - 1))],
        means[int(0.975 * (samples - 1))],
    )


def cluster_percentile_ci(
    rows: list[dict[str, str]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    clusters: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        clusters[factor_cell(row)].append(normalized_gap(row))
    if len(clusters) != 60 or any(len(values) != 2 for values in clusters.values()):
        raise ValueError("V6 cluster bootstrap requires 60 cells of size 2")
    cluster_means = [
        statistics.fmean(values)
        for _key, values in sorted(clusters.items())
    ]
    return percentile_ci(cluster_means, samples=samples, seed=seed)


def metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    ls_total = sum(int(row["lightsabre_swap"]) for row in rows)
    independent_rollout_total = sum(int(row["independent_rollout_swap"]) for row in rows)
    dual_total = sum(int(row["baseline_rollout_safe_swap"]) for row in rows)
    selected_total = sum(int(row["selected_swap"]) for row in rows)
    absolute_gains = [
        int(row["lightsabre_swap"]) - int(row["selected_swap"])
        for row in rows
    ]
    gaps = [normalized_gap(row) for row in rows]
    selected_quality = [
        (int(row["selected_swap"]), int(row["selected_depth"]))
        for row in rows
    ]
    baseline_quality = [
        (int(row["lightsabre_swap"]), int(row["lightsabre_depth"]))
        for row in rows
    ]
    return {
        "cases": len(rows),
        "lightsabre_swap": ls_total,
        "native_independent_rollout_swap": independent_rollout_total,
        "baseline_rollout_safe_swap": dual_total,
        "selected_swap": selected_total,
        "gain_vs_lightsabre": ls_total - selected_total,
        "relative_gain_pct": 100.0 * (ls_total - selected_total) / ls_total,
        "swap_wins": sum(gain > 0 for gain in absolute_gains),
        "swap_ties": sum(gain == 0 for gain in absolute_gains),
        "swap_losses": sum(gain < 0 for gain in absolute_gains),
        "quality_wins": sum(
            selected < baseline
            for selected, baseline in zip(selected_quality, baseline_quality)
        ),
        "quality_ties": sum(
            selected == baseline
            for selected, baseline in zip(selected_quality, baseline_quality)
        ),
        "quality_losses": sum(
            selected > baseline
            for selected, baseline in zip(selected_quality, baseline_quality)
        ),
        "mean_normalized_gap": statistics.fmean(gaps),
        "median_normalized_gap": statistics.median(gaps),
        "mean_absolute_gain": statistics.fmean(absolute_gains),
        "median_absolute_gain": statistics.median(absolute_gains),
        "maximum_absolute_gain": max(absolute_gains),
        "selected_backends": dict(
            sorted(Counter(row["selected_backend"] for row in rows).items())
        ),
        "semantic_passed": sum(
            is_true(row["semantic_equivalent"]) for row in rows
        ),
        "topology_passed": sum(
            is_true(row["topology_valid"]) for row in rows
        ),
        "metric_passed": sum(is_true(row["metric_valid"]) for row in rows),
    }


def group_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    fields = ("topology", "circuit_mode", "num_qubits", "length_factor")
    for field in fields:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            value = (
                str(length_factor(row))
                if field == "length_factor"
                else str(row[field])
            )
            grouped[value].append(row)
        for value, items in sorted(grouped.items()):
            result = metrics(items)
            output.append(
                {
                    "group_field": field,
                    "group_value": value,
                    "cases": result["cases"],
                    "lightsabre_swap": result["lightsabre_swap"],
                    "selected_swap": result["selected_swap"],
                    "gain": result["gain_vs_lightsabre"],
                    "relative_gain_pct": result["relative_gain_pct"],
                    "swap_wins": result["swap_wins"],
                    "swap_ties": result["swap_ties"],
                    "swap_losses": result["swap_losses"],
                    "mean_normalized_gap": result["mean_normalized_gap"],
                }
            )
    return output


def leave_one_level_out(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    fields = ("topology", "circuit_mode", "length_factor")
    for field in fields:
        values = sorted(
            {
                str(length_factor(row))
                if field == "length_factor"
                else str(row[field])
                for row in rows
            }
        )
        for excluded in values:
            included = [
                row
                for row in rows
                if (
                    str(length_factor(row))
                    if field == "length_factor"
                    else str(row[field])
                )
                != excluded
            ]
            result = metrics(included)
            output.append(
                {
                    "excluded_field": field,
                    "excluded_value": excluded,
                    "remaining_cases": result["cases"],
                    "gain": result["gain_vs_lightsabre"],
                    "relative_gain_pct": result["relative_gain_pct"],
                    "mean_normalized_gap": result["mean_normalized_gap"],
                    "swap_wins": result["swap_wins"],
                    "swap_ties": result["swap_ties"],
                    "swap_losses": result["swap_losses"],
                }
            )
    return output


def format_ci(interval: Iterable[float]) -> str:
    low, high = interval
    return f"[{low:.6f}, {high:.6f}]"


def build_report(summary: dict[str, Any]) -> str:
    v5 = summary["v5"]["metrics"]
    v6 = summary["v6"]["metrics"]
    iid_ci = summary["v6"]["iid_bootstrap_95_ci"]
    cluster_ci = summary["v6"]["cluster_bootstrap_95_ci"]
    loo = summary["v6"]["leave_one_level_out"]
    min_loo = min(item["relative_gain_pct"] for item in loo)
    max_loo = max(item["relative_gain_pct"] for item in loo)
    return f"""# reliability audit：V5/V6 结果可靠性审计

## 结论

两份逐实例结果文件通过结构、因子设计、映射、质量下界和三项审计字段检查。
V5 与 V6 的汇总值均与 historical development 一致。V6 的 60 个
“拓扑 × 线路模式 × 长度因子”单元均恰含两个实现。

V6 最终方法将总 SWAP 从 {v6['lightsabre_swap']} 降至
{v6['selected_swap']}，减少 {v6['gain_vs_lightsabre']} 个
（{v6['relative_gain_pct']:.4f}%）；逐实例胜/平/负为
{v6['swap_wins']}/{v6['swap_ties']}/{v6['swap_losses']}。

## 复算结果

| 数据集 | 实例 | LightSABRE | 原生 independent rollout | 安全双路 | 最终选择 | 改善 |
|---|---:|---:|---:|---:|---:|---:|
| V5 | {v5['cases']} | {v5['lightsabre_swap']} | {v5['native_independent_rollout_swap']} | {v5['baseline_rollout_safe_swap']} | {v5['selected_swap']} | {v5['relative_gain_pct']:.4f}% |
| V6 | {v6['cases']} | {v6['lightsabre_swap']} | {v6['native_independent_rollout_swap']} | {v6['baseline_rollout_safe_swap']} | {v6['selected_swap']} | {v6['relative_gain_pct']:.4f}% |

V6 的语义、拓扑和指标审计均为 {v6['cases']}/{v6['cases']}。
平均归一化配对差值为 {v6['mean_normalized_gap']:.9f}。

## Bootstrap 敏感性

- 原预注册逐实例 percentile bootstrap：
  {format_ci(iid_ci)}。
- 新增因子单元 cluster bootstrap：
  {format_ci(cluster_ci)}。

cluster bootstrap 以 60 个因子单元为抽样单位，每次抽中一个单元时同时
保留其中两个实现。区间仍完全低于零，并与原区间接近，因此主要统计结论
对单元内相关性的处理不敏感。该 cluster 分析属于 reliability audit 的事后稳健性
检查，不能追溯标记为 historical development 的预注册分析。

## 留一层级敏感性

分别删除一种拓扑、一种线路模式或一个长度因子后，剩余样本的总 SWAP
相对改善仍为正，改善范围为 {min_loo:.4f}%–{max_loo:.4f}%。
这说明总体收益没有完全依赖某一个单独的拓扑、模式或长度层级。

## 解释边界

1. V6 有 {v6['swap_ties']} 个 SWAP 平局，中位逐实例改善为
   {v6['median_absolute_gain']:.0f}；总改善来自
   {v6['swap_wins']} 个获胜实例，应同时报告总量、胜平负与分布。
2. 逐实例无退化主要来自候选集中显式保留 LightSABRE 的结构性保证，
   不能单独解释为 independent rollout 普遍强于 LightSABRE。
3. 当前 CSV 中的 QASM/QPY/manifest 路径是原 Windows 机器上的绝对路径。
   本次审计只验证 CSV 字段，未重新打开这些线路文件。
4. 数据规模仍限于 6、8、10 比特的合成线路；外部 benchmark 和新 V7
   盲测仍是投稿前的重要工作。
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="reliability audit: audit V5/V6 case-level reliability"
    )
    parser.add_argument("--v5-cases", type=Path, required=True)
    parser.add_argument("--v6-cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_723)
    args = parser.parse_args()

    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    for path in (args.v5_cases, args.v6_cases):
        if not path.is_file():
            raise FileNotFoundError(path)

    v5_rows = read_rows(args.v5_cases)
    v6_rows = read_rows(args.v6_cases)
    v5_validation = validate_rows(
        v5_rows,
        name="V5 engineering check",
        expected_count=60,
        expected_cluster_size=1,
    )
    v6_validation = validate_rows(
        v6_rows,
        name="V6 preregistered blind",
        expected_count=120,
        expected_cluster_size=2,
    )

    v5_metrics = metrics(v5_rows)
    v6_metrics = metrics(v6_rows)
    gaps = [normalized_gap(row) for row in v6_rows]
    iid_ci = percentile_ci(
        gaps,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    cluster_ci = cluster_percentile_ci(
        v6_rows,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    groups = group_summary(v6_rows)
    leave_one_out = leave_one_level_out(v6_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups_path = args.output_dir / "v6_group_summary.csv"
    leave_one_out_path = args.output_dir / "v6_leave_one_level_out.csv"
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "REPORT.md"
    write_csv(groups_path, groups)
    write_csv(leave_one_out_path, leave_one_out)

    summary = {
        "schema_version": 1,
        "component": "reliability_audit",
        "purpose": "case-level reliability and clustered-bootstrap audit",
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "method": "percentile",
        },
        "inputs": {
            "v5_cases": str(args.v5_cases.resolve()),
            "v5_sha256": sha256(args.v5_cases),
            "v6_cases": str(args.v6_cases.resolve()),
            "v6_sha256": sha256(args.v6_cases),
        },
        "v5": {
            "validation": v5_validation,
            "metrics": v5_metrics,
        },
        "v6": {
            "validation": v6_validation,
            "metrics": v6_metrics,
            "iid_bootstrap_95_ci": list(iid_ci),
            "cluster_bootstrap_95_ci": list(cluster_ci),
            "cluster_definition": [
                "topology",
                "circuit_mode",
                "length_factor",
            ],
            "group_summary": groups,
            "leave_one_level_out": leave_one_out,
        },
        "checks": {
            "final_evidence_v5_totals_reproduced": (
                v5_metrics["lightsabre_swap"] == 1691
                and v5_metrics["native_independent_rollout_swap"] == 1843
                and v5_metrics["baseline_rollout_safe_swap"] == 1670
                and v5_metrics["selected_swap"] == 1641
            ),
            "final_evidence_v6_totals_reproduced": (
                v6_metrics["lightsabre_swap"] == 3176
                and v6_metrics["native_independent_rollout_swap"] == 3386
                and v6_metrics["baseline_rollout_safe_swap"] == 3119
                and v6_metrics["selected_swap"] == 3051
            ),
            "original_iid_ci_reproduced": (
                abs(iid_ci[0] - (-0.024346064814814817)) < 1e-15
                and abs(iid_ci[1] - (-0.01363425925925926)) < 1e-15
            ),
            "cluster_ci_upper_below_zero": cluster_ci[1] < 0.0,
            "all_leave_one_level_out_gains_positive": all(
                item["gain"] > 0 for item in leave_one_out
            ),
        },
    }
    summary["passed"] = all(summary["checks"].values())
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(build_report(summary), encoding="utf-8")

    hash_paths = [
        Path(__file__).resolve(),
        args.v5_cases.resolve(),
        args.v6_cases.resolve(),
        groups_path.resolve(),
        leave_one_out_path.resolve(),
        summary_path.resolve(),
        report_path.resolve(),
    ]
    hashes_path = args.output_dir / "SHA256SUMS.txt"
    hashes_path.write_text(
        "\n".join(
            f"{sha256(path)}  {path.name}"
            for path in sorted(hash_paths, key=lambda item: item.name.lower())
        )
        + "\n",
        encoding="utf-8",
    )

    print("===== reliability audit V5/V6 reliability audit =====")
    print("V5 selected / LightSABRE:", v5_metrics["selected_swap"], "/", v5_metrics["lightsabre_swap"])
    print("V6 selected / LightSABRE:", v6_metrics["selected_swap"], "/", v6_metrics["lightsabre_swap"])
    print("V6 relative gain:", f"{v6_metrics['relative_gain_pct']:.4f}%")
    print("V6 iid bootstrap 95% CI:", iid_ci)
    print("V6 cluster bootstrap 95% CI:", cluster_ci)
    print("factor cells:", v6_validation["factor_cells"], "x", v6_validation["factor_cell_size"])
    print("PASSED:", summary["passed"])
    print("report:", report_path.resolve())
    print("summary:", summary_path.resolve())
    print("hashes:", hashes_path.resolve())
    if not summary["passed"]:
        raise RuntimeError("reliability audit reliability audit failed")


if __name__ == "__main__":
    main()
