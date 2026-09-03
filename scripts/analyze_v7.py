from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


MULTIPLIERS = (0.25, 0.5, 1.0, 2.0)


def load_rows(results_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(results_root.glob("v7_repeat_*/paired_frontier.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def load_environments(results_root: Path) -> list[dict[str, Any]]:
    environments: list[dict[str, Any]] = []
    for path in sorted(results_root.glob("v7_repeat_*/environment.json")):
        environments.append(json.loads(path.read_text(encoding="utf-8")))
    return environments


def verify_repeat_hashes(results_root: Path) -> tuple[int, list[str]]:
    verified = 0
    failures: list[str] = []
    for directory in sorted(results_root.glob("v7_repeat_*")):
        if not directory.is_dir():
            continue
        manifest = directory / "SHA256SUMS.txt"
        if not manifest.is_file():
            failures.append(f"{directory.name}: missing SHA256SUMS.txt")
            continue
        directory_failed = False
        for line in manifest.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split(maxsplit=1)
            path = directory / relative.strip()
            if not path.is_file() or sha256(path) != digest:
                failures.append(f"{directory.name}: {relative.strip()}")
                directory_failed = True
        if not directory_failed:
            verified += 1
    return verified, failures


def truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_ci(
    rows: list[dict[str, str]],
    *,
    samples: int = 10_000,
    seed: int = 20_260_725,
) -> tuple[float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["instance"]].append(
            (
                int(row["smpr_swap"]) - int(row["lightsabre_swap"])
            )
            / max(1, int(row["routing_cx_count"]))
        )
    cluster_values = [
        statistics.mean(values) for _, values in sorted(grouped.items())
    ]
    rng = random.Random(seed)
    bootstrap = [
        statistics.mean(
            cluster_values[rng.randrange(len(cluster_values))]
            for _ in cluster_values
        )
        for _ in range(samples)
    ]
    return percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)


def summarize_multiplier(rows: list[dict[str, str]]) -> dict[str, Any]:
    smpr = sum(int(row["smpr_swap"]) for row in rows)
    lightsabre = sum(int(row["lightsabre_swap"]) for row in rows)
    gaps = [
        int(row["smpr_swap"]) - int(row["lightsabre_swap"]) for row in rows
    ]
    low, high = cluster_ci(rows)
    return {
        "observations": len(rows),
        "cases": len({row["instance"] for row in rows}),
        "repeats": len({row["repeat"] for row in rows}),
        "smpr_total_swap": smpr,
        "lightsabre_total_swap": lightsabre,
        "swap_gap": smpr - lightsabre,
        "relative_smpr_reduction": (
            (lightsabre - smpr) / lightsabre if lightsabre else 0.0
        ),
        "wins": sum(gap < 0 for gap in gaps),
        "ties": sum(gap == 0 for gap in gaps),
        "losses": sum(gap > 0 for gap in gaps),
        "cluster_bootstrap_ci95_normalized_gap": [low, high],
        "median_smpr_wall_ms": statistics.median(
            float(row["smpr_wall_ms"]) for row in rows
        ),
        "median_lightsabre_elapsed_ms": statistics.median(
            float(row["lightsabre_elapsed_ms"]) for row in rows
        ),
        "median_lightsabre_trials": statistics.median(
            int(row["lightsabre_trials"]) for row in rows
        ),
        "max_smpr_peak_rss_bytes": max(
            int(row["smpr_peak_rss_bytes"]) for row in rows
        ),
        "max_lightsabre_peak_rss_bytes": max(
            int(row["lightsabre_peak_rss_bytes"]) for row in rows
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def report(summary: dict[str, Any]) -> str:
    lines = [
        "# SMPR V7 外部验证报告",
        "",
        "## 审计结论",
        "",
        f"- 预期观测数：120；实际：{summary['row_count']}；",
        f"- 公共线路：{summary['case_count']}；同机重复：{summary['repeat_count']}；",
        f"- 冻结清单一致：{summary['frozen_manifest_consistent']}；",
        f"- 环境记录：{summary['environment_count']}/3；",
        f"- 重复结果哈希：{summary['verified_repeat_hashes']}/3；",
        f"- 路由轨迹审计：{summary['routing_trace_passed']}/{summary['row_count']}；",
        f"- 拓扑审计：{summary['topology_passed']}/{summary['row_count']}；",
        f"- 内部 LightSABRE 质量下界：{summary['safe_floor_passed']}/{summary['row_count']}；",
        f"- 总体通过：`{summary['passed']}`。",
        "",
        "## 同机质量—时间前沿",
        "",
        "| LightSABRE 时间预算 | SMPR SWAP | LightSABRE SWAP | 差值 | 胜/平/负 | cluster 95% CI |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for multiplier in MULTIPLIERS:
        item = summary["frontier"][str(multiplier)]
        low, high = item["cluster_bootstrap_ci95_normalized_gap"]
        lines.append(
            f"| {multiplier:.2f}× | {item['smpr_total_swap']} | "
            f"{item['lightsabre_total_swap']} | {item['swap_gap']:+d} | "
            f"{item['wins']}/{item['ties']}/{item['losses']} | "
            f"[{low:.6f}, {high:.6f}] |"
        )
    lines.extend(
        [
            "",
            "差值定义为 `SMPR - time-matched LightSABRE`；负值表示 SMPR 使用更少",
            "SWAP。bootstrap 的抽样单位为公共线路，同一线路的三次重复保留为簇。",
            "",
            "## 解释边界",
            "",
            "V7 比较的是路由交互骨架：规范化保留全部 CX 与门依赖，一比特门被替换为",
            "不影响路由的占位门。轨迹审计不能表述为原始非 Clifford 线路的完整幺正",
            "等价证明。运行时间只覆盖候选生成和选择，不含文件导出及统计分析。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="analyze V7 paired results")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.results_root)
    environments = load_environments(args.results_root)
    verified_repeat_hashes, repeat_hash_failures = verify_repeat_hashes(
        args.results_root
    )
    expected = 10 * 3 * len(MULTIPLIERS)
    if not rows:
        raise RuntimeError("no V7 paired results found")

    frontier: dict[str, Any] = {}
    for multiplier in MULTIPLIERS:
        selected = [
            row
            for row in rows
            if abs(float(row["budget_multiplier"]) - multiplier) < 1e-12
        ]
        frontier[str(multiplier)] = summarize_multiplier(selected)

    row_manifest_hashes = {
        row.get("frozen_manifest_sha256", "") for row in rows
    }
    environment_manifest_hashes = {
        str(item.get("frozen_manifest_sha256", "")) for item in environments
    }
    frozen_manifest_consistent = (
        len(row_manifest_hashes) == 1
        and len(environment_manifest_hashes) == 1
        and row_manifest_hashes == environment_manifest_hashes
        and len(next(iter(row_manifest_hashes), "")) == 64
    )

    summary = {
        "schema_version": 1,
        "purpose": "prospectively frozen V7 external validation",
        "row_count": len(rows),
        "expected_row_count": expected,
        "case_count": len({row["instance"] for row in rows}),
        "repeat_count": len({row["repeat"] for row in rows}),
        "environment_count": len(environments),
        "frozen_manifest_consistent": frozen_manifest_consistent,
        "frozen_manifest_sha256": next(iter(row_manifest_hashes), ""),
        "verified_repeat_hashes": verified_repeat_hashes,
        "repeat_hash_failures": repeat_hash_failures,
        "all_qubits_over_10": all(
            int(row["num_logical_qubits"]) > 10 for row in rows
        ),
        "all_heavy_hex": all(
            row["topology"].startswith("heavy_hex_") for row in rows
        ),
        "routing_trace_passed": sum(
            truth(row["routing_trace_valid"]) for row in rows
        ),
        "topology_passed": sum(truth(row["topology_valid"]) for row in rows),
        "safe_floor_passed": sum(
            truth(row["safe_quality_floor"]) for row in rows
        ),
        "frontier": frontier,
    }
    summary["passed"] = (
        summary["row_count"] == expected
        and summary["case_count"] == 10
        and summary["repeat_count"] == 3
        and summary["environment_count"] == 3
        and summary["frozen_manifest_consistent"]
        and summary["verified_repeat_hashes"] == 3
        and not summary["repeat_hash_failures"]
        and summary["all_qubits_over_10"]
        and summary["all_heavy_hex"]
        and summary["routing_trace_passed"] == expected
        and summary["topology_passed"] == expected
        and summary["safe_floor_passed"] == expected
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "REPORT.md"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report_path.write_text(report(summary), encoding="utf-8", newline="\n")
    hashes = args.output_dir / "SHA256SUMS.txt"
    hashes.write_text(
        f"{sha256(report_path)}  REPORT.md\n"
        f"{sha256(summary_path)}  summary.json\n",
        encoding="utf-8",
        newline="\n",
    )
    print(report(summary))
    if not summary["passed"]:
        raise RuntimeError("V7 audit did not meet the preregistered integrity checks")


if __name__ == "__main__":
    main()
