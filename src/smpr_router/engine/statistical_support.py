from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import score_support as score_support

from layout_pool import (
    MappingCandidate,
    generate_base_mapping_pool,
)
from case_io import DevCase, load_cases
from validation_support import (
    compare_quality,
    read_case_metadata,
    run_qiskit_multiseed,
)


# ============================================================
# 1. 验证集已经冻结的唯一最终配置
# ============================================================

FINAL_CONFIG = score_support.ScoreConfig(
    name="d0_b025",
    theta=5,
    future_weight=0.5,
    gamma=0.5,
    alpha=0.5,
    beta_unlock=0.25,
    lambda_depth=0.0,
)


# ============================================================
# 2. 结果结构
# ============================================================

@dataclass
class FinalCaseResult:
    instance: str
    split: str
    topology: str
    circuit_mode: str
    num_qubits: int
    length_factor: int
    cx_count: int

    best_mapping: str
    mapping_candidate_count: int
    independent_rollout_swap: int
    independent_rollout_added_two_qubit_count: int
    independent_rollout_depth: int
    mapping_generation_ms: float
    independent_rollout_routing_budget_ms: float
    independent_rollout_total_budget_ms: float
    independent_rollout_swap_per_cx: float

    qiskit_best_seed: int
    qiskit_swap: int
    qiskit_added_two_qubit_count: int
    qiskit_depth: int
    qiskit_total_budget_ms: float
    qiskit_swap_per_cx: float

    swap_gap: int
    depth_gap: int
    normalized_swap_gap: float
    outcome: str


@dataclass
class GroupSummary:
    group_type: str
    group_value: str
    case_count: int
    total_cx: int

    independent_rollout_total_swap: int
    qiskit_total_swap: int
    swap_gap: int

    independent_rollout_total_depth: int
    qiskit_total_depth: int
    depth_gap: int

    independent_rollout_swap_per_cx: float
    qiskit_swap_per_cx: float
    mean_normalized_swap_gap: float
    median_normalized_swap_gap: float

    independent_rollout_wins: int
    ties: int
    qiskit_wins: int

    independent_rollout_total_budget_ms: float
    qiskit_total_budget_ms: float
    budget_ratio_independent_rollout_over_qiskit: float


@dataclass
class OverallSummary:
    dataset_path: str
    dataset_sha256: str
    final_config: str
    theta: int
    future_weight: float
    gamma: float
    alpha: float
    beta_unlock: float
    lambda_depth: float

    case_count: int
    total_cx: int

    independent_rollout_total_swap: int
    qiskit_total_swap: int
    swap_gap: int
    relative_swap_gap: float

    independent_rollout_total_added_two_qubit_count: int
    qiskit_total_added_two_qubit_count: int

    independent_rollout_total_depth: int
    qiskit_total_depth: int
    depth_gap: int

    independent_rollout_wins: int
    ties: int
    qiskit_wins: int

    mean_normalized_swap_gap: float
    median_normalized_swap_gap: float
    bootstrap_ci95_low: float
    bootstrap_ci95_high: float

    independent_rollout_total_budget_ms: float
    qiskit_total_budget_ms: float
    budget_ratio_independent_rollout_over_qiskit: float


# ============================================================
# 3. 基础工具
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def bootstrap_mean_ci(
    values: list[float],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap 输入不能为空")

    if len(values) == 1:
        return values[0], values[0]

    rng = random.Random(seed)
    n = len(values)

    boot_means = []

    for _ in range(samples):
        sample = [
            values[rng.randrange(n)]
            for _ in range(n)
        ]

        boot_means.append(
            statistics.mean(sample)
        )

    boot_means.sort()

    low_index = int(0.025 * (samples - 1))
    high_index = int(0.975 * (samples - 1))

    return (
        boot_means[low_index],
        boot_means[high_index],
    )


def run_independent_rollout_pool(
    case: DevCase,
    candidates: list[MappingCandidate],
) -> tuple[str, int, int, float]:
    return score_support.run_pool(
        case=case.test_case,
        candidates=candidates,
    )


# ============================================================
# 4. 汇总
# ============================================================

def summarize_group(
    group_type: str,
    group_value: str,
    rows: list[FinalCaseResult],
) -> GroupSummary:
    outcomes = Counter(
        row.outcome
        for row in rows
    )

    total_cx = sum(
        row.cx_count
        for row in rows
    )

    independent_rollout_total_swap = sum(
        row.independent_rollout_swap
        for row in rows
    )

    qiskit_total_swap = sum(
        row.qiskit_swap
        for row in rows
    )

    independent_rollout_total_depth = sum(
        row.independent_rollout_depth
        for row in rows
    )

    qiskit_total_depth = sum(
        row.qiskit_depth
        for row in rows
    )

    independent_rollout_budget = sum(
        row.independent_rollout_total_budget_ms
        for row in rows
    )

    qiskit_budget = sum(
        row.qiskit_total_budget_ms
        for row in rows
    )

    normalized_gaps = [
        row.normalized_swap_gap
        for row in rows
    ]

    return GroupSummary(
        group_type=group_type,
        group_value=group_value,
        case_count=len(rows),
        total_cx=total_cx,
        independent_rollout_total_swap=independent_rollout_total_swap,
        qiskit_total_swap=qiskit_total_swap,
        swap_gap=(
            independent_rollout_total_swap
            - qiskit_total_swap
        ),
        independent_rollout_total_depth=(
            independent_rollout_total_depth
        ),
        qiskit_total_depth=(
            qiskit_total_depth
        ),
        depth_gap=(
            independent_rollout_total_depth
            - qiskit_total_depth
        ),
        independent_rollout_swap_per_cx=(
            independent_rollout_total_swap / total_cx
        ),
        qiskit_swap_per_cx=(
            qiskit_total_swap / total_cx
        ),
        mean_normalized_swap_gap=(
            statistics.mean(
                normalized_gaps
            )
        ),
        median_normalized_swap_gap=(
            statistics.median(
                normalized_gaps
            )
        ),
        independent_rollout_wins=outcomes[
            "independent rollout胜"
        ],
        ties=outcomes["平局"],
        qiskit_wins=outcomes[
            "Qiskit胜"
        ],
        independent_rollout_total_budget_ms=(
            independent_rollout_budget
        ),
        qiskit_total_budget_ms=(
            qiskit_budget
        ),
        budget_ratio_independent_rollout_over_qiskit=(
            independent_rollout_budget / qiskit_budget
            if qiskit_budget > 0
            else float("inf")
        ),
    )


def build_group_summaries(
    rows: list[FinalCaseResult],
) -> list[GroupSummary]:
    summaries: list[GroupSummary] = []

    summaries.append(
        summarize_group(
            group_type="overall",
            group_value="all",
            rows=rows,
        )
    )

    attributes = (
        ("topology", "topology"),
        ("circuit_mode", "circuit_mode"),
        ("length_factor", "length_factor"),
        ("num_qubits", "num_qubits"),
    )

    for group_type, attribute in attributes:
        grouped: dict[
            str,
            list[FinalCaseResult],
        ] = defaultdict(list)

        for row in rows:
            grouped[
                str(getattr(row, attribute))
            ].append(row)

        for group_value in sorted(
            grouped,
            key=str,
        ):
            summaries.append(
                summarize_group(
                    group_type=group_type,
                    group_value=group_value,
                    rows=grouped[group_value],
                )
            )

    return summaries


def build_overall_summary(
    dataset_path: Path,
    rows: list[FinalCaseResult],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> OverallSummary:
    overall = summarize_group(
        group_type="overall",
        group_value="all",
        rows=rows,
    )

    normalized_gaps = [
        row.normalized_swap_gap
        for row in rows
    ]

    ci_low, ci_high = bootstrap_mean_ci(
        normalized_gaps,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )

    relative_swap_gap = (
        overall.swap_gap
        / overall.qiskit_total_swap
        if overall.qiskit_total_swap > 0
        else float("inf")
    )

    return OverallSummary(
        dataset_path=str(
            dataset_path.resolve()
        ),
        dataset_sha256=sha256_file(
            dataset_path
        ),
        final_config=FINAL_CONFIG.name,
        theta=FINAL_CONFIG.theta,
        future_weight=(
            FINAL_CONFIG.future_weight
        ),
        gamma=FINAL_CONFIG.gamma,
        alpha=FINAL_CONFIG.alpha,
        beta_unlock=(
            FINAL_CONFIG.beta_unlock
        ),
        lambda_depth=(
            FINAL_CONFIG.lambda_depth
        ),
        case_count=overall.case_count,
        total_cx=overall.total_cx,
        independent_rollout_total_swap=(
            overall.independent_rollout_total_swap
        ),
        qiskit_total_swap=(
            overall.qiskit_total_swap
        ),
        swap_gap=overall.swap_gap,
        relative_swap_gap=(
            relative_swap_gap
        ),
        independent_rollout_total_added_two_qubit_count=(
            3 * overall.independent_rollout_total_swap
        ),
        qiskit_total_added_two_qubit_count=(
            3 * overall.qiskit_total_swap
        ),
        independent_rollout_total_depth=(
            overall.independent_rollout_total_depth
        ),
        qiskit_total_depth=(
            overall.qiskit_total_depth
        ),
        depth_gap=overall.depth_gap,
        independent_rollout_wins=overall.independent_rollout_wins,
        ties=overall.ties,
        qiskit_wins=(
            overall.qiskit_wins
        ),
        mean_normalized_swap_gap=(
            overall.mean_normalized_swap_gap
        ),
        median_normalized_swap_gap=(
            overall.median_normalized_swap_gap
        ),
        bootstrap_ci95_low=ci_low,
        bootstrap_ci95_high=ci_high,
        independent_rollout_total_budget_ms=(
            overall.independent_rollout_total_budget_ms
        ),
        qiskit_total_budget_ms=(
            overall.qiskit_total_budget_ms
        ),
        budget_ratio_independent_rollout_over_qiskit=(
            overall.budget_ratio_independent_rollout_over_qiskit
        ),
    )


# ============================================================
# 5. CSV
# ============================================================

def save_dataclass_rows(
    path: Path,
    rows: list[object],
    fields: tuple[str, ...],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: getattr(row, field)
                    for field in fields
                }
            )


def save_overall_json(
    path: Path,
    summary: OverallSummary,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        field: getattr(summary, field)
        for field
        in OverallSummary.__annotations__
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# 6. 输出
# ============================================================

def print_group_table(
    summaries: list[GroupSummary],
) -> None:
    print()
    print("=" * 118)
    print("分组结果")
    print("=" * 118)

    print(
        f"{'分组':<18}"
        f"{'值':<18}"
        f"{'n':>4}"
        f"{'independent rollout':>9}"
        f"{'Qiskit':>9}"
        f"{'gap':>8}"
        f"{'胜':>5}"
        f"{'平':>5}"
        f"{'负':>5}"
        f"{'mean gap/CX':>14}"
    )

    print("-" * 118)

    for row in summaries:
        print(
            f"{row.group_type:<18}"
            f"{row.group_value:<18}"
            f"{row.case_count:>4}"
            f"{row.independent_rollout_total_swap:>9}"
            f"{row.qiskit_total_swap:>9}"
            f"{row.swap_gap:>+8}"
            f"{row.independent_rollout_wins:>5}"
            f"{row.ties:>5}"
            f"{row.qiskit_wins:>5}"
            f"{row.mean_normalized_swap_gap:>14.4f}"
        )


def print_final_summary(
    summary: OverallSummary,
) -> None:
    print()
    print("=" * 118)
    print("historical development：Benchmark V3 最终测试结果")
    print("=" * 118)

    print(
        "最终配置："
        f"{summary.final_config}, "
        f"lambda_depth={summary.lambda_depth}, "
        f"beta_unlock={summary.beta_unlock}, "
        f"alpha={summary.alpha}, "
        f"theta={summary.theta}, "
        f"W={summary.future_weight}, "
        f"gamma={summary.gamma}"
    )

    print(
        "测试集 SHA-256：",
        summary.dataset_sha256,
    )

    print()
    print(
        "总 SWAP："
        f"independent rollout={summary.independent_rollout_total_swap}, "
        f"Qiskit={summary.qiskit_total_swap}, "
        f"gap={summary.swap_gap:+d}"
    )

    print(
        "相对 SWAP 差："
        f"{summary.relative_swap_gap:+.2%}"
    )

    print(
        "总附加双比特门："
        f"independent rollout="
        f"{summary.independent_rollout_total_added_two_qubit_count}, "
        f"Qiskit="
        f"{summary.qiskit_total_added_two_qubit_count}"
    )

    print(
        "总加权深度："
        f"independent rollout={summary.independent_rollout_total_depth}, "
        f"Qiskit={summary.qiskit_total_depth}, "
        f"gap={summary.depth_gap:+d}"
    )

    print(
        "逐实例胜负："
        f"independent rollout胜={summary.independent_rollout_wins}, "
        f"平局={summary.ties}, "
        f"Qiskit胜={summary.qiskit_wins}"
    )

    print(
        "平均归一化 SWAP 差："
        f"{summary.mean_normalized_swap_gap:+.4f}"
    )

    print(
        "平均归一化差的 bootstrap 95% 区间："
        f"[{summary.bootstrap_ci95_low:+.4f}, "
        f"{summary.bootstrap_ci95_high:+.4f}]"
    )

    print(
        "总计算预算："
        f"independent rollout={summary.independent_rollout_total_budget_ms:.2f} ms, "
        f"Qiskit={summary.qiskit_total_budget_ms:.2f} ms, "
        f"比值="
        f"{summary.budget_ratio_independent_rollout_over_qiskit:.3f}"
    )


# ============================================================
# 7. 主程序
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "historical development：使用验证集冻结的唯一配置，"
            "在 Benchmark V3 测试集上进行最终一次评估。"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/benchmarks_v3/test.json"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "仅用于在 validation.json 上做程序冒烟测试；"
            "正式 test.json 运行时不要设置"
        ),
    )

    parser.add_argument(
        "--qiskit-seeds",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--qiskit-repeats",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260716,
    )

    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path(
            "results/statistical_support_raw.csv"
        ),
    )

    parser.add_argument(
        "--group-output",
        type=Path,
        default=Path(
            "results/statistical_support_groups.csv"
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "results/statistical_support_summary.json"
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"找不到输入数据："
            f"{args.input.resolve()}"
        )

    if args.limit is not None and args.limit <= 0:
        raise ValueError(
            "--limit 必须为正整数"
        )

    if min(
        args.qiskit_seeds,
        args.qiskit_repeats,
        args.bootstrap_samples,
    ) <= 0:
        raise ValueError(
            "种子数、重复次数和 bootstrap 次数必须为正整数"
        )

    cases = load_cases(
        path=args.input,
        limit=args.limit,
    )

    metadata = read_case_metadata(
        args.input
    )

    print("===== historical development：最终测试 =====")
    print("输入数据：", args.input.resolve())
    print("实例数：", len(cases))
    print(
        "最终配置：",
        FINAL_CONFIG.name,
    )
    print(
        "Qiskit："
        f"{args.qiskit_seeds} seeds × "
        f"{args.qiskit_repeats} repeats"
    )

    rows: list[FinalCaseResult] = []

    score_support.apply_config(
        FINAL_CONFIG
    )

    for index, case in enumerate(
        cases,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(cases)}] "
            f"{case.name}"
        )

        mapping_start = time.perf_counter()

        candidates = generate_base_mapping_pool(
            case.test_case
        )

        mapping_generation_ms = (
            time.perf_counter()
            - mapping_start
        ) * 1000.0

        (
            best_mapping,
            independent_rollout_swap,
            independent_rollout_depth,
            independent_rollout_routing_budget_ms,
        ) = run_independent_rollout_pool(
            case=case,
            candidates=candidates,
        )

        (
            qiskit_best,
            qiskit_budget_ms,
        ) = run_qiskit_multiseed(
            case=case,
            seeds=args.qiskit_seeds,
            repeats=args.qiskit_repeats,
        )

        case_meta = metadata[
            case.name
        ]

        cx_count = int(
            case_meta[
                "two_qubit_gate_count"
            ]
        )

        qiskit_swap = int(
            qiskit_best.swap_count
        )

        qiskit_depth = int(
            qiskit_best.weighted_depth
        )

        outcome = compare_quality(
            swap_count=independent_rollout_swap,
            depth=independent_rollout_depth,
            qiskit_swap=qiskit_swap,
            qiskit_depth=qiskit_depth,
        )

        independent_rollout_total_budget_ms = (
            mapping_generation_ms
            + independent_rollout_routing_budget_ms
        )

        row = FinalCaseResult(
            instance=case.name,
            split=str(
                case_meta["split"]
            ),
            topology=case.topology,
            circuit_mode=(
                case.circuit_mode
            ),
            num_qubits=(
                case.num_qubits
            ),
            length_factor=int(
                case_meta[
                    "length_factor"
                ]
            ),
            cx_count=cx_count,
            best_mapping=best_mapping,
            mapping_candidate_count=(
                len(candidates)
            ),
            independent_rollout_swap=independent_rollout_swap,
            independent_rollout_added_two_qubit_count=(
                3 * independent_rollout_swap
            ),
            independent_rollout_depth=independent_rollout_depth,
            mapping_generation_ms=(
                mapping_generation_ms
            ),
            independent_rollout_routing_budget_ms=(
                independent_rollout_routing_budget_ms
            ),
            independent_rollout_total_budget_ms=(
                independent_rollout_total_budget_ms
            ),
            independent_rollout_swap_per_cx=(
                independent_rollout_swap / cx_count
            ),
            qiskit_best_seed=int(
                qiskit_best.seed
            ),
            qiskit_swap=qiskit_swap,
            qiskit_added_two_qubit_count=(
                3 * qiskit_swap
            ),
            qiskit_depth=qiskit_depth,
            qiskit_total_budget_ms=(
                qiskit_budget_ms
            ),
            qiskit_swap_per_cx=(
                qiskit_swap / cx_count
            ),
            swap_gap=(
                independent_rollout_swap
                - qiskit_swap
            ),
            depth_gap=(
                independent_rollout_depth
                - qiskit_depth
            ),
            normalized_swap_gap=(
                (
                    independent_rollout_swap
                    - qiskit_swap
                )
                / cx_count
            ),
            outcome=outcome,
        )

        rows.append(row)

        group_summaries = (
            build_group_summaries(rows)
        )

        overall_summary = (
            build_overall_summary(
                dataset_path=args.input,
                rows=rows,
                bootstrap_samples=(
                    args.bootstrap_samples
                ),
                bootstrap_seed=(
                    args.bootstrap_seed
                ),
            )
        )

        save_dataclass_rows(
            args.raw_output,
            rows,
            tuple(
                FinalCaseResult
                .__annotations__
                .keys()
            ),
        )

        save_dataclass_rows(
            args.group_output,
            group_summaries,
            tuple(
                GroupSummary
                .__annotations__
                .keys()
            ),
        )

        save_overall_json(
            args.summary_output,
            overall_summary,
        )

        print(
            "  independent rollout："
            f"mapping={best_mapping}, "
            f"SWAP={independent_rollout_swap}, "
            f"depth={independent_rollout_depth}, "
            f"budget={independent_rollout_total_budget_ms:.2f} ms"
        )

        print(
            "  Qiskit："
            f"seed={qiskit_best.seed}, "
            f"SWAP={qiskit_swap}, "
            f"depth={qiskit_depth}, "
            f"budget={qiskit_budget_ms:.2f} ms"
        )

        print(
            "  结论："
            f"gap={row.swap_gap:+d} SWAP, "
            f"gap/CX="
            f"{row.normalized_swap_gap:+.4f}, "
            f"{outcome}"
        )

    group_summaries = (
        build_group_summaries(rows)
    )

    overall_summary = (
        build_overall_summary(
            dataset_path=args.input,
            rows=rows,
            bootstrap_samples=(
                args.bootstrap_samples
            ),
            bootstrap_seed=(
                args.bootstrap_seed
            ),
        )
    )

    print_final_summary(
        overall_summary
    )

    print_group_table(
        group_summaries
    )

    print()
    print(
        "逐实例结果：",
        args.raw_output.resolve(),
    )
    print(
        "分组汇总：",
        args.group_output.resolve(),
    )
    print(
        "总体摘要：",
        args.summary_output.resolve(),
    )


if __name__ == "__main__":
    main()
