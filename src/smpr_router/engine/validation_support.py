from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import score_support as score_support

from layout_pool import (
    MappingCandidate,
    MappingRunResult,
    generate_base_mapping_pool,
    run_qiskit_auto_layout,
)
from case_io import DevCase, load_cases


# ============================================================
# 1. 冻结的三组候选配置
# ============================================================

CONFIGS: tuple[score_support.ScoreConfig, ...] = (
    score_support.ScoreConfig(
        name="baseline",
        theta=5,
        future_weight=0.5,
        gamma=0.5,
        alpha=0.5,
        beta_unlock=2.0,
        lambda_depth=0.2,
    ),
    score_support.ScoreConfig(
        name="anchor_d0_b2",
        theta=5,
        future_weight=0.5,
        gamma=0.5,
        alpha=0.5,
        beta_unlock=2.0,
        lambda_depth=0.0,
    ),
    score_support.ScoreConfig(
        name="d0_b025",
        theta=5,
        future_weight=0.5,
        gamma=0.5,
        alpha=0.5,
        beta_unlock=0.25,
        lambda_depth=0.0,
    ),
)


# ============================================================
# 2. 结果结构
# ============================================================

@dataclass
class ValidationCaseResult:
    instance: str
    topology: str
    circuit_mode: str
    num_qubits: int
    length_factor: int
    cx_count: int

    config: str
    best_mapping: str
    swap_count: int
    added_two_qubit_count: int
    weighted_depth: int
    independent_rollout_budget_ms: float

    qiskit_best_seed: int
    qiskit_swap: int
    qiskit_added_two_qubit_count: int
    qiskit_depth: int
    qiskit_budget_ms: float

    outcome: str
    swap_gap: int
    normalized_swap_gap: float


@dataclass
class ConfigSummary:
    rank: int
    config: str

    theta: int
    future_weight: float
    gamma: float
    alpha: float
    beta_unlock: float
    lambda_depth: float

    case_count: int
    total_swap: int
    total_added_two_qubit_count: int
    total_weighted_depth: int
    total_independent_rollout_budget_ms: float

    total_qiskit_swap: int
    total_qiskit_depth: int
    total_qiskit_budget_ms: float

    swap_gap: int
    mean_normalized_swap_gap: float
    median_normalized_swap_gap: float

    independent_rollout_wins: int
    ties: int
    qiskit_wins: int


# ============================================================
# 3. V3 元数据读取
# ============================================================

def read_case_metadata(
    path: Path,
) -> dict[str, dict]:
    import json

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    return {
        str(case["name"]): case
        for case in payload["cases"]
    }


# ============================================================
# 4. independent rollout 基础映射池
# ============================================================

def run_independent_rollout_pool(
    case: DevCase,
    candidates: list[MappingCandidate],
) -> tuple[str, int, int, float]:
    return score_support.run_pool(
        case=case.test_case,
        candidates=candidates,
    )


# ============================================================
# 5. Qiskit SABRE：预热、多种子、重复计时
# ============================================================

def result_quality(
    row: MappingRunResult,
) -> tuple[int, int]:
    if (
        row.added_two_qubit_count is None
        or row.weighted_depth is None
    ):
        return 10**9, 10**9

    return (
        row.added_two_qubit_count,
        row.weighted_depth,
    )


def best_valid(
    rows: list[MappingRunResult],
) -> MappingRunResult:
    valid = [
        row
        for row in rows
        if row.valid
    ]

    if not valid:
        raise RuntimeError(
            "Qiskit 没有合法结果："
            + "; ".join(
                row.error
                for row in rows
                if row.error
            )
        )

    return min(
        valid,
        key=lambda row: (
            result_quality(row),
            row.total_ms,
            row.seed if row.seed is not None else -1,
        ),
    )


def run_qiskit_multiseed(
    case: DevCase,
    seeds: int,
    repeats: int,
) -> tuple[MappingRunResult, float]:
    # 预热结果丢弃。
    run_qiskit_auto_layout(
        case=case.test_case,
        seed=0,
    )

    seed_rows: list[MappingRunResult] = []

    for seed in range(seeds):
        repeated = [
            run_qiskit_auto_layout(
                case=case.test_case,
                seed=seed,
            )
            for _ in range(repeats)
        ]

        valid = [
            row
            for row in repeated
            if row.valid
        ]

        if not valid:
            seed_rows.append(repeated[0])
            continue

        reference = valid[0]
        reference_quality = result_quality(
            reference
        )

        if any(
            result_quality(row)
            != reference_quality
            for row in valid[1:]
        ):
            raise RuntimeError(
                f"{case.name} 的 Qiskit seed={seed} "
                "重复运行中产生不同质量结果"
            )

        median_ms = statistics.median(
            row.total_ms
            for row in valid
        )

        seed_rows.append(
            MappingRunResult(
                instance=reference.instance,
                method=reference.method,
                seed=seed,
                mapping=reference.mapping,
                swap_count=reference.swap_count,
                added_two_qubit_count=(
                    reference.added_two_qubit_count
                ),
                weighted_depth=(
                    reference.weighted_depth
                ),
                generation_ms=0.0,
                routing_ms=median_ms,
                total_ms=median_ms,
                valid=True,
                error="",
            )
        )

    best = best_valid(seed_rows)

    budget = sum(
        row.total_ms
        for row in seed_rows
        if row.valid
    )

    return best, budget


# ============================================================
# 6. 比较与汇总
# ============================================================

def compare_quality(
    swap_count: int,
    depth: int,
    qiskit_swap: int,
    qiskit_depth: int,
) -> str:
    left = (swap_count, depth)
    right = (qiskit_swap, qiskit_depth)

    if left < right:
        return "independent rollout胜"

    if left > right:
        return "Qiskit胜"

    return "平局"


def build_summaries(
    rows: list[ValidationCaseResult],
) -> list[ConfigSummary]:
    grouped: dict[
        str,
        list[ValidationCaseResult],
    ] = defaultdict(list)

    for row in rows:
        grouped[row.config].append(row)

    config_lookup = {
        config.name: config
        for config in CONFIGS
    }

    summaries: list[ConfigSummary] = []

    for config_name, config_rows in grouped.items():
        config = config_lookup[config_name]
        outcomes = Counter(
            row.outcome
            for row in config_rows
        )

        total_swap = sum(
            row.swap_count
            for row in config_rows
        )

        total_qiskit_swap = sum(
            row.qiskit_swap
            for row in config_rows
        )

        normalized_gaps = [
            row.normalized_swap_gap
            for row in config_rows
        ]

        summaries.append(
            ConfigSummary(
                rank=0,
                config=config.name,
                theta=config.theta,
                future_weight=(
                    config.future_weight
                ),
                gamma=config.gamma,
                alpha=config.alpha,
                beta_unlock=(
                    config.beta_unlock
                ),
                lambda_depth=(
                    config.lambda_depth
                ),
                case_count=len(config_rows),
                total_swap=total_swap,
                total_added_two_qubit_count=(
                    3 * total_swap
                ),
                total_weighted_depth=sum(
                    row.weighted_depth
                    for row in config_rows
                ),
                total_independent_rollout_budget_ms=sum(
                    row.independent_rollout_budget_ms
                    for row in config_rows
                ),
                total_qiskit_swap=(
                    total_qiskit_swap
                ),
                total_qiskit_depth=sum(
                    row.qiskit_depth
                    for row in config_rows
                ),
                total_qiskit_budget_ms=sum(
                    row.qiskit_budget_ms
                    for row in config_rows
                ),
                swap_gap=(
                    total_swap
                    - total_qiskit_swap
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
            )
        )

    summaries.sort(
        key=lambda row: (
            row.total_swap,
            row.total_weighted_depth,
            row.total_independent_rollout_budget_ms,
        )
    )

    ranked: list[ConfigSummary] = []

    for rank, row in enumerate(
        summaries,
        start=1,
    ):
        ranked.append(
            ConfigSummary(
                **{
                    **{
                        field: getattr(row, field)
                        for field
                        in ConfigSummary.__annotations__
                    },
                    "rank": rank,
                }
            )
        )

    return ranked


# ============================================================
# 7. CSV
# ============================================================

CASE_FIELDS = tuple(
    ValidationCaseResult.__annotations__.keys()
)


def save_case_rows(
    path: Path,
    rows: list[ValidationCaseResult],
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
            fieldnames=CASE_FIELDS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: getattr(row, field)
                    for field in CASE_FIELDS
                }
            )


SUMMARY_FIELDS = tuple(
    ConfigSummary.__annotations__.keys()
)


def save_summaries(
    path: Path,
    rows: list[ConfigSummary],
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
            fieldnames=SUMMARY_FIELDS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: getattr(row, field)
                    for field in SUMMARY_FIELDS
                }
            )


# ============================================================
# 8. 打印
# ============================================================

def print_summaries(
    summaries: list[ConfigSummary],
) -> None:
    print()
    print("=" * 118)
    print("historical development：V3 验证集配置选择")
    print("配置排序：总 SWAP 数优先；平局时总加权深度优先")
    print("=" * 118)

    print(
        f"{'rank':>4} "
        f"{'config':<16}"
        f"{'SWAP':>8}"
        f"{'gap':>8}"
        f"{'depth':>10}"
        f"{'independent rollout/ms':>12}"
        f"{'胜':>6}"
        f"{'平':>6}"
        f"{'负':>6}"
        f"{'mean gap/CX':>14}"
    )

    print("-" * 118)

    for row in summaries:
        print(
            f"{row.rank:>4} "
            f"{row.config:<16}"
            f"{row.total_swap:>8}"
            f"{row.swap_gap:>+8}"
            f"{row.total_weighted_depth:>10}"
            f"{row.total_independent_rollout_budget_ms:>12.2f}"
            f"{row.independent_rollout_wins:>6}"
            f"{row.ties:>6}"
            f"{row.qiskit_wins:>6}"
            f"{row.mean_normalized_swap_gap:>14.4f}"
        )

    winner = summaries[0]

    print()
    print("验证集选出的最终配置：")
    print(
        f"  {winner.config}: "
        f"lambda_depth={winner.lambda_depth}, "
        f"beta_unlock={winner.beta_unlock}, "
        f"alpha={winner.alpha}, "
        f"theta={winner.theta}, "
        f"W={winner.future_weight}, "
        f"gamma={winner.gamma}"
    )


# ============================================================
# 9. 主程序
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "historical development：在 Benchmark V3 验证集上，"
            "只比较三组冻结配置，并选定最终测试配置。"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/benchmarks_v3/validation.json"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只运行前若干个验证实例，用于快速检查",
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
        "--raw-output",
        type=Path,
        default=Path(
            "results/validation_support_validation_raw.csv"
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "results/validation_support_validation_summary.csv"
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"找不到验证集：{args.input.resolve()}"
        )

    if args.limit is not None and args.limit <= 0:
        raise ValueError(
            "--limit 必须为正整数"
        )

    if min(
        args.qiskit_seeds,
        args.qiskit_repeats,
    ) <= 0:
        raise ValueError(
            "Qiskit 种子数和重复次数必须为正整数"
        )

    cases = load_cases(
        path=args.input,
        limit=args.limit,
    )

    metadata = read_case_metadata(
        args.input
    )

    # 初始映射池与评分参数无关，只生成一次。
    mapping_pools: dict[
        str,
        list[MappingCandidate],
    ] = {
        case.name: generate_base_mapping_pool(
            case.test_case
        )
        for case in cases
    }

    # Qiskit 基线也与 independent rollout 参数无关，每个实例只计算一次。
    qiskit_results: dict[
        str,
        tuple[MappingRunResult, float],
    ] = {}

    print("===== historical development：V3 验证集配置选择 =====")
    print("验证实例数：", len(cases))
    print(
        "Qiskit："
        f"{args.qiskit_seeds} seeds × "
        f"{args.qiskit_repeats} repeats"
    )
    print(
        "冻结配置：",
        ", ".join(
            config.name
            for config in CONFIGS
        ),
    )

    print()
    print("先计算 Qiskit 验证基线：")

    for index, case in enumerate(
        cases,
        start=1,
    ):
        best, budget = run_qiskit_multiseed(
            case=case,
            seeds=args.qiskit_seeds,
            repeats=args.qiskit_repeats,
        )

        qiskit_results[case.name] = (
            best,
            budget,
        )

        print(
            f"  [{index}/{len(cases)}] "
            f"{case.name}: "
            f"seed={best.seed}, "
            f"SWAP={best.swap_count}, "
            f"depth={best.weighted_depth}"
        )

    case_rows: list[
        ValidationCaseResult
    ] = []

    baseline_config = CONFIGS[0]

    try:
        for config_index, config in enumerate(
            CONFIGS,
            start=1,
        ):
            score_support.apply_config(config)

            print()
            print(
                f"[配置 {config_index}/"
                f"{len(CONFIGS)}] "
                f"{config.name}"
            )

            for case_index, case in enumerate(
                cases,
                start=1,
            ):
                (
                    best_mapping,
                    swap_count,
                    weighted_depth,
                    independent_rollout_budget_ms,
                ) = run_independent_rollout_pool(
                    case=case,
                    candidates=(
                        mapping_pools[
                            case.name
                        ]
                    ),
                )

                (
                    qiskit_best,
                    qiskit_budget,
                ) = qiskit_results[
                    case.name
                ]

                qiskit_swap = int(
                    qiskit_best.swap_count
                )

                qiskit_depth = int(
                    qiskit_best.weighted_depth
                )

                case_meta = metadata[
                    case.name
                ]

                cx_count = int(
                    case_meta[
                        "two_qubit_gate_count"
                    ]
                )

                outcome = compare_quality(
                    swap_count=swap_count,
                    depth=weighted_depth,
                    qiskit_swap=qiskit_swap,
                    qiskit_depth=qiskit_depth,
                )

                row = ValidationCaseResult(
                    instance=case.name,
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
                    config=config.name,
                    best_mapping=(
                        best_mapping
                    ),
                    swap_count=swap_count,
                    added_two_qubit_count=(
                        3 * swap_count
                    ),
                    weighted_depth=(
                        weighted_depth
                    ),
                    independent_rollout_budget_ms=(
                        independent_rollout_budget_ms
                    ),
                    qiskit_best_seed=int(
                        qiskit_best.seed
                    ),
                    qiskit_swap=(
                        qiskit_swap
                    ),
                    qiskit_added_two_qubit_count=(
                        3 * qiskit_swap
                    ),
                    qiskit_depth=(
                        qiskit_depth
                    ),
                    qiskit_budget_ms=(
                        qiskit_budget
                    ),
                    outcome=outcome,
                    swap_gap=(
                        swap_count
                        - qiskit_swap
                    ),
                    normalized_swap_gap=(
                        (
                            swap_count
                            - qiskit_swap
                        )
                        / cx_count
                    ),
                )

                case_rows.append(row)

                print(
                    f"  [{case_index}/"
                    f"{len(cases)}] "
                    f"{case.name}: "
                    f"mapping={best_mapping}, "
                    f"SWAP={swap_count}, "
                    f"depth={weighted_depth}, "
                    f"{outcome}"
                )

            summaries = build_summaries(
                case_rows
            )

            save_case_rows(
                args.raw_output,
                case_rows,
            )

            save_summaries(
                args.summary_output,
                summaries,
            )

    finally:
        score_support.apply_config(
            baseline_config
        )

    summaries = build_summaries(
        case_rows
    )

    print_summaries(summaries)

    print()
    print(
        "原始结果：",
        args.raw_output.resolve(),
    )
    print(
        "汇总结果：",
        args.summary_output.resolve(),
    )


if __name__ == "__main__":
    main()
