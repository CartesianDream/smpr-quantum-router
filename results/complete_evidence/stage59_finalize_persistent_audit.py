from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PRIMARY_COLUMNS = (
    "lightsabre_swap",
    "oaabr_swap",
    "safe_dual_backend",
    "safe_dual_swap",
    "hybrid_swap",
    "selected_backend",
    "selected_swap",
    "selected_depth",
    "gain_vs_lightsabre",
    "gain_vs_safe_dual",
    "semantic_equivalent",
    "topology_valid",
    "metric_valid",
)

AUXILIARY_COLUMNS = (
    "lightsabre_depth",
    "oaabr_depth",
    "safe_dual_depth",
    "hybrid_depth",
    "hybrid_attempts",
)

ISOLATED_SELECTION_COLUMNS = (
    "selected_backend",
    "selected_swap",
    "selected_depth",
    "semantic_equivalent",
    "topology_valid",
    "metric_valid",
)


@dataclass(frozen=True)
class Mismatch:
    instance: str
    column: str
    severity: str
    reference: str
    persistent: str
    isolated: str
    isolated_confirmed: bool
    direction: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def index_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = read_rows(path)
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        instance = str(row.get("instance", ""))
        if not instance:
            raise ValueError(f"row without instance in: {path}")
        if instance in indexed:
            raise ValueError(f"duplicate instance {instance!r} in: {path}")
        indexed[instance] = row
    return indexed


def is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def numeric_direction(column: str, reference: str, persistent: str) -> str:
    if not column.endswith("_depth"):
        return "changed"
    try:
        old = float(reference)
        new = float(persistent)
    except ValueError:
        return "changed"
    if new < old:
        return "improved"
    if new > old:
        return "regressed"
    return "unchanged"


def isolated_confirms(
    column: str,
    persistent_row: dict[str, str],
    isolated_row: dict[str, str] | None,
) -> bool:
    if isolated_row is None:
        return False
    if str(isolated_row.get(column, "")) != str(persistent_row.get(column, "")):
        return False
    return all(
        str(isolated_row.get(name, "")) == str(persistent_row.get(name, ""))
        for name in ISOLATED_SELECTION_COLUMNS
    )


def collect_mismatches(
    reference: dict[str, dict[str, str]],
    persistent: dict[str, dict[str, str]],
    isolated: dict[str, dict[str, str]],
) -> list[Mismatch]:
    mismatches: list[Mismatch] = []
    shared = sorted(set(reference) & set(persistent))
    for instance in shared:
        old = reference[instance]
        new = persistent[instance]
        isolated_row = isolated.get(instance)
        for column in PRIMARY_COLUMNS:
            old_value = str(old.get(column, ""))
            new_value = str(new.get(column, ""))
            if old_value == new_value:
                continue
            mismatches.append(
                Mismatch(
                    instance=instance,
                    column=column,
                    severity="primary",
                    reference=old_value,
                    persistent=new_value,
                    isolated=(
                        ""
                        if isolated_row is None
                        else str(isolated_row.get(column, ""))
                    ),
                    isolated_confirmed=False,
                    direction=numeric_direction(column, old_value, new_value),
                )
            )
        for column in AUXILIARY_COLUMNS:
            old_value = str(old.get(column, ""))
            new_value = str(new.get(column, ""))
            if old_value == new_value:
                continue
            mismatches.append(
                Mismatch(
                    instance=instance,
                    column=column,
                    severity="auxiliary",
                    reference=old_value,
                    persistent=new_value,
                    isolated=(
                        ""
                        if isolated_row is None
                        else str(isolated_row.get(column, ""))
                    ),
                    isolated_confirmed=isolated_confirms(
                        column,
                        new,
                        isolated_row,
                    ),
                    direction=numeric_direction(column, old_value, new_value),
                )
            )
    return mismatches


def write_mismatches(path: Path, mismatches: list[Mismatch]) -> None:
    fieldnames = [
        "instance",
        "column",
        "severity",
        "reference",
        "persistent",
        "isolated",
        "isolated_confirmed",
        "direction",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mismatch in mismatches:
            writer.writerow(
                {
                    "instance": mismatch.instance,
                    "column": mismatch.column,
                    "severity": mismatch.severity,
                    "reference": mismatch.reference,
                    "persistent": mismatch.persistent,
                    "isolated": mismatch.isolated,
                    "isolated_confirmed": mismatch.isolated_confirmed,
                    "direction": mismatch.direction,
                }
            )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    runtime = summary["runtime"]
    mismatch = summary["mismatches"]
    lines = [
        "# Stage 59：持久化 V6 最终审计",
        "",
        f"- 最终结论：**{'PASSED' if summary['passed'] else 'FAILED'}**",
        f"- 实例集合一致：{summary['instance_sets_equal']}",
        f"- 主字段差异：{mismatch['primary']}",
        f"- 辅助字段差异：{mismatch['auxiliary']}",
        f"- 已由独立单例确认的辅助差异：{mismatch['isolated_confirmed']}",
        f"- 未确认的辅助差异：{mismatch['unconfirmed_auxiliary']}",
        f"- 持久执行审计：{summary['audits_passed']}/{summary['case_count']}",
        f"- safe quality floor：{summary['safe_quality_floor']}",
        "",
        "## 运行时间",
        "",
        f"- 原顺序执行：{runtime['sequential_wall_seconds']:.3f} 秒",
        f"- Stage 58 持久执行：{runtime['persistent_wall_seconds']:.3f} 秒",
        f"- 加速比：{runtime['speedup']:.3f}×",
        f"- 墙钟时间减少：{runtime['wall_reduction_fraction']:.2%}",
        "",
        "## 解释口径",
        "",
        "Stage 55 是预注册盲测的质量主结果，保持原样，不以后续执行覆盖。",
        "Stage 58 只用于验证持久分片实现的效率与最终决策等价性。",
        "辅助候选深度的差异只有在最终选择、SWAP 主指标和有效性审计均不变，",
        "且独立单实例复现与持久结果一致时，才归类为已确认的执行顺序差异。",
        "",
    ]
    if summary["mismatch_records"]:
        lines.extend(
            [
                "## 差异记录",
                "",
                "| 实例 | 字段 | 参考 | 持久 | 单例 | 方向 | 已确认 |",
                "|---|---|---:|---:|---:|---|---|",
            ]
        )
        for row in summary["mismatch_records"]:
            lines.append(
                f"| {row['instance']} | {row['column']} | "
                f"{row['reference']} | {row['persistent']} | "
                f"{row['isolated']} | {row['direction']} | "
                f"{row['isolated_confirmed']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 59: finalize the persistent V6 equivalence audit"
    )
    parser.add_argument("--reference-cases", type=Path, required=True)
    parser.add_argument("--persistent-cases", type=Path, required=True)
    parser.add_argument("--persistent-summary", type=Path, required=True)
    parser.add_argument(
        "--isolated-cases",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--sequential-wall-seconds",
        type=float,
        default=583.84,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage59_v6_final_audit"),
    )
    args = parser.parse_args()

    input_paths = [
        args.reference_cases,
        args.persistent_cases,
        args.persistent_summary,
        *args.isolated_cases,
    ]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.sequential_wall_seconds <= 0:
        raise ValueError("sequential-wall-seconds must be positive")
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; use a fresh path: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)

    reference = index_rows(args.reference_cases)
    persistent = index_rows(args.persistent_cases)
    isolated: dict[str, dict[str, str]] = {}
    for path in args.isolated_cases:
        for instance, row in index_rows(path).items():
            if instance in isolated:
                raise ValueError(
                    f"duplicate isolated instance {instance!r}: {path}"
                )
            isolated[instance] = row

    persistent_summary = json.loads(
        args.persistent_summary.read_text(encoding="utf-8")
    )
    mismatches = collect_mismatches(reference, persistent, isolated)
    primary = [item for item in mismatches if item.severity == "primary"]
    auxiliary = [item for item in mismatches if item.severity == "auxiliary"]
    confirmed = [item for item in auxiliary if item.isolated_confirmed]
    unconfirmed = [item for item in auxiliary if not item.isolated_confirmed]

    instance_sets_equal = set(reference) == set(persistent)
    case_count = len(persistent)
    audits_passed = min(
        int(persistent_summary.get("semantic_passed", -1)),
        int(persistent_summary.get("topology_passed", -1)),
        int(persistent_summary.get("metric_passed", -1)),
    )
    safe_quality_floor = bool(
        persistent_summary.get("safe_quality_floor", False)
    )
    persistent_wall = float(
        persistent_summary.get("parallel_wall_seconds", 0.0)
    )
    speedup = (
        args.sequential_wall_seconds / persistent_wall
        if persistent_wall > 0
        else 0.0
    )
    reduction = (
        1.0 - persistent_wall / args.sequential_wall_seconds
        if persistent_wall > 0
        else 0.0
    )
    passed = (
        instance_sets_equal
        and not primary
        and not unconfirmed
        and audits_passed == case_count
        and safe_quality_floor
        and persistent_wall > 0
    )

    mismatch_records = [
        {
            "instance": item.instance,
            "column": item.column,
            "severity": item.severity,
            "reference": item.reference,
            "persistent": item.persistent,
            "isolated": item.isolated,
            "isolated_confirmed": item.isolated_confirmed,
            "direction": item.direction,
        }
        for item in mismatches
    ]
    summary = {
        "stage": 59,
        "purpose": "finalize persistent V6 decision-equivalence audit",
        "passed": passed,
        "case_count": case_count,
        "reference_case_count": len(reference),
        "instance_sets_equal": instance_sets_equal,
        "audits_passed": audits_passed,
        "safe_quality_floor": safe_quality_floor,
        "mismatches": {
            "total": len(mismatches),
            "primary": len(primary),
            "auxiliary": len(auxiliary),
            "isolated_confirmed": len(confirmed),
            "unconfirmed_auxiliary": len(unconfirmed),
        },
        "runtime": {
            "sequential_wall_seconds": args.sequential_wall_seconds,
            "persistent_wall_seconds": persistent_wall,
            "speedup": speedup,
            "wall_reduction_fraction": reduction,
            "persistent_processes": persistent_summary.get(
                "persistent_processes"
            ),
            "avoided_process_restarts": persistent_summary.get(
                "avoided_process_restarts"
            ),
            "effective_parallelism": persistent_summary.get(
                "effective_parallelism"
            ),
        },
        "input_hashes": {
            "reference_cases": sha256(args.reference_cases),
            "persistent_cases": sha256(args.persistent_cases),
            "persistent_summary": sha256(args.persistent_summary),
            "isolated_cases": {
                str(path.resolve()): sha256(path)
                for path in args.isolated_cases
            },
        },
        "mismatch_records": mismatch_records,
    }

    mismatch_path = args.output_dir / "mismatches.csv"
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "REPORT.md"
    write_mismatches(mismatch_path, mismatches)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(report_path, summary)
    hashes_path = args.output_dir / "SHA256SUMS.txt"
    hashes_path.write_text(
        "".join(
            f"{sha256(path)}  {path}\n"
            for path in (mismatch_path, summary_path, report_path)
        ),
        encoding="utf-8",
    )

    print("===== Stage 59 persistent V6 final audit =====")
    print("cases:", case_count)
    print("instance sets equal:", instance_sets_equal)
    print("primary mismatches:", len(primary))
    print("auxiliary mismatches:", len(auxiliary))
    print("isolated-confirmed auxiliary:", len(confirmed))
    print("unconfirmed auxiliary:", len(unconfirmed))
    print("audits:", audits_passed, "/", case_count)
    print("safe quality floor:", safe_quality_floor)
    print("sequential wall seconds:", round(args.sequential_wall_seconds, 3))
    print("persistent wall seconds:", round(persistent_wall, 3))
    print("speedup:", round(speedup, 3))
    print("wall reduction:", f"{reduction:.2%}")
    print("PASSED:", passed)
    print("report:", report_path.resolve())
    print("summary:", summary_path.resolve())
    print("hashes:", hashes_path.resolve())

    if not passed:
        raise RuntimeError("Stage 59 final audit failed")


if __name__ == "__main__":
    main()
