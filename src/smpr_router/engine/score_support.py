from __future__ import annotations

import argparse
import csv
import io
import time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import one_step_policy as one_step_policy

from layout_pool import (
    MappingCandidate,
    generate_base_mapping_pool,
)
from scaling_support import (
    FAMILIES,
    build_scale_cases,
    parse_factors,
)


# ============================================================
# 1. 参数配置
# ============================================================

@dataclass(frozen=True)
class ScoreConfig:
    name: str
    theta: int = 5
    future_weight: float = 0.5
    gamma: float = 0.5
    alpha: float = 0.5
    beta_unlock: float = 2.0
    lambda_depth: float = 0.2


CONFIGS: tuple[ScoreConfig, ...] = (
    ScoreConfig(name="baseline"),
    ScoreConfig(
        name="depth_0",
        lambda_depth=0.0,
    ),
    ScoreConfig(
        name="depth_005",
        lambda_depth=0.05,
    ),
    ScoreConfig(
        name="unlock_0",
        beta_unlock=0.0,
    ),
    ScoreConfig(
        name="unlock_05",
        beta_unlock=0.5,
    ),
    ScoreConfig(
        name="unlock_1",
        beta_unlock=1.0,
    ),
    ScoreConfig(
        name="alpha_0",
        alpha=0.0,
    ),
    ScoreConfig(
        name="alpha_1",
        alpha=1.0,
    ),
    ScoreConfig(
        name="future_025",
        future_weight=0.25,
    ),
    ScoreConfig(
        name="future_1",
        future_weight=1.0,
    ),
    ScoreConfig(
        name="theta_8",
        theta=8,
    ),
    ScoreConfig(
        name="long_future",
        theta=10,
        future_weight=1.0,
        gamma=0.75,
    ),
)


def apply_config(config: ScoreConfig) -> None:
    """
    one_step_policy 的路由函数从模块全局变量读取参数，
    因此在每组实验前显式写入。
    """

    one_step_policy.THETA = config.theta
    one_step_policy.W_FUTURE = config.future_weight
    one_step_policy.GAMMA = config.gamma
    one_step_policy.ALPHA = config.alpha
    one_step_policy.BETA_UNLOCK = config.beta_unlock
    one_step_policy.LAMBDA_DEPTH = config.lambda_depth


# ============================================================
# 2. 结果结构
# ============================================================

@dataclass
class CaseResult:
    config: str
    family: str
    factor: int
    cx_count: int
    topology: str
    circuit_mode: str
    best_mapping: str
    swap_count: int
    added_two_qubit_count: int
    weighted_depth: int
    runtime_ms: float
    qiskit_swap: int
    qiskit_depth: int
    outcome: str


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
    total_runtime_ms: float
    total_qiskit_swap: int
    swap_gap: int
    independent_rollout_wins: int
    ties: int
    qiskit_wins: int


# ============================================================
# 3. Qiskit 基线
# ============================================================

def load_qiskit_baseline(
    path: Path,
) -> dict[tuple[str, int], tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 historical development CSV：{path.resolve()}"
        )

    baseline: dict[
        tuple[str, int],
        tuple[int, int],
    ] = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            key = (
                row["family"],
                int(row["length_factor"]),
            )

            baseline[key] = (
                int(row["qiskit_swap"]),
                int(row["qiskit_depth"]),
            )

    return baseline


# ============================================================
# 4. 运行 OneStep + 基础映射池
# ============================================================

def run_candidate(
    case,
    candidate: MappingCandidate,
) -> tuple[int, int, float]:
    dag, hardware, _ = case.build()

    buffer = io.StringIO()
    start = time.perf_counter()

    with redirect_stdout(buffer):
        state = one_step_policy.run_one_step_router(
            dag=dag,
            hardware=hardware,
            initial_mapping=list(candidate.mapping),
        )

    runtime_ms = (
        time.perf_counter() - start
    ) * 1000.0

    state.assert_valid(
        dag=dag,
        hardware=hardware,
    )

    if len(state.executed_gates) != len(dag.gates):
        raise AssertionError(
            "路由结束后仍有逻辑门未执行"
        )

    swap_count = sum(
        operation[0] == "swap"
        for operation in state.physical_operations
    )

    return (
        swap_count,
        state.current_depth(),
        runtime_ms,
    )


def run_pool(
    case,
    candidates: list[MappingCandidate],
) -> tuple[str, int, int, float]:
    rows: list[
        tuple[str, int, int, float]
    ] = []

    for candidate in candidates:
        swap_count, depth, runtime_ms = (
            run_candidate(
                case=case,
                candidate=candidate,
            )
        )

        rows.append(
            (
                candidate.name,
                swap_count,
                depth,
                runtime_ms,
            )
        )

    best = min(
        rows,
        key=lambda row: (
            row[1],
            row[2],
            row[3],
            row[0],
        ),
    )

    total_runtime = sum(
        row[3]
        for row in rows
    )

    return (
        best[0],
        best[1],
        best[2],
        total_runtime,
    )


def compare(
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


# ============================================================
# 5. CSV
# ============================================================

CASE_FIELDS = tuple(
    CaseResult.__annotations__.keys()
)


def save_case_results(
    path: Path,
    rows: list[CaseResult],
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
# 6. 汇总
# ============================================================

def build_summaries(
    case_rows: list[CaseResult],
) -> list[ConfigSummary]:
    grouped: dict[
        str,
        list[CaseResult],
    ] = defaultdict(list)

    for row in case_rows:
        grouped[row.config].append(row)

    config_lookup = {
        config.name: config
        for config in CONFIGS
    }

    unranked: list[ConfigSummary] = []

    for config_name, rows in grouped.items():
        config = config_lookup[config_name]
        outcomes = Counter(
            row.outcome
            for row in rows
        )

        total_swap = sum(
            row.swap_count
            for row in rows
        )

        total_qiskit_swap = sum(
            row.qiskit_swap
            for row in rows
        )

        unranked.append(
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
                case_count=len(rows),
                total_swap=total_swap,
                total_added_two_qubit_count=(
                    3 * total_swap
                ),
                total_weighted_depth=sum(
                    row.weighted_depth
                    for row in rows
                ),
                total_runtime_ms=sum(
                    row.runtime_ms
                    for row in rows
                ),
                total_qiskit_swap=(
                    total_qiskit_swap
                ),
                swap_gap=(
                    total_swap
                    - total_qiskit_swap
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

    unranked.sort(
        key=lambda row: (
            row.total_swap,
            row.total_weighted_depth,
            row.total_runtime_ms,
        )
    )

    return [
        ConfigSummary(
            **{
                **{
                    field: getattr(row, field)
                    for field in SUMMARY_FIELDS
                },
                "rank": rank,
            }
        )
        for rank, row in enumerate(
            unranked,
            start=1,
        )
    ]


def print_summaries(
    rows: list[ConfigSummary],
) -> None:
    print()
    print("=" * 118)
    print("historical development：评分项消融汇总")
    print("排序：总 SWAP 数优先，其次总加权深度，再其次运行时间")
    print("=" * 118)

    print(
        f"{'rank':>4} "
        f"{'config':<16}"
        f"{'SWAP':>8}"
        f"{'gap':>8}"
        f"{'depth':>10}"
        f"{'time/ms':>12}"
        f"{'胜':>6}"
        f"{'平':>6}"
        f"{'负':>6}"
    )

    print("-" * 118)

    for row in rows:
        print(
            f"{row.rank:>4} "
            f"{row.config:<16}"
            f"{row.total_swap:>8}"
            f"{row.swap_gap:>+8}"
            f"{row.total_weighted_depth:>10}"
            f"{row.total_runtime_ms:>12.2f}"
            f"{row.independent_rollout_wins:>6}"
            f"{row.ties:>6}"
            f"{row.qiskit_wins:>6}"
        )


# ============================================================
# 7. 主程序
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "historical development：在中长线路上对 OneStepPotential "
            "进行单因素评分消融。"
        )
    )

    parser.add_argument(
        "--factors",
        type=str,
        default="6,12",
        help="默认使用 6n 和 12n 个 CX",
    )

    parser.add_argument(
        "--limit-families",
        type=int,
        default=None,
        help="只运行前若干个线路族",
    )

    parser.add_argument(
        "--qiskit-csv",
        type=Path,
        default=Path(
            "results/scaling_support.csv"
        ),
    )

    parser.add_argument(
        "--case-output",
        type=Path,
        default=Path(
            "results/score_support_raw.csv"
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "results/score_support_summary.csv"
        ),
    )

    args = parser.parse_args()

    factors = parse_factors(args.factors)

    selected_families = FAMILIES

    if args.limit_families is not None:
        if args.limit_families <= 0:
            raise ValueError(
                "--limit-families 必须为正整数"
            )

        selected_families = FAMILIES[
            :args.limit_families
        ]

    # build_scale_cases 读取 scaling_support 模块中的全局 FAMILIES。
    # 这里临时替换，再立即恢复。
    import scaling_support as scaling_support

    original_families = scaling_support.FAMILIES
    scaling_support.FAMILIES = selected_families

    try:
        scale_cases = build_scale_cases(
            factors
        )
    finally:
        scaling_support.FAMILIES = original_families

    qiskit_baseline = load_qiskit_baseline(
        args.qiskit_csv
    )

    # 映射池与评分参数无关，只生成一次。
    mapping_pools: dict[
        str,
        list[MappingCandidate],
    ] = {}

    for family, factor, case in scale_cases:
        mapping_pools[case.name] = (
            generate_base_mapping_pool(case)
        )

    case_rows: list[CaseResult] = []

    print("===== historical development：OneStep 评分项消融 =====")
    print("线路族数：", len(selected_families))
    print("长度因子：", factors)
    print("配置数：", len(CONFIGS))
    print("总配置—实例组合：", (
        len(CONFIGS) * len(scale_cases)
    ))

    baseline_config = CONFIGS[0]

    try:
        for config_index, config in enumerate(
            CONFIGS,
            start=1,
        ):
            apply_config(config)

            print()
            print(
                f"[配置 {config_index}/{len(CONFIGS)}] "
                f"{config.name}"
            )

            for case_index, (
                family,
                factor,
                case,
            ) in enumerate(
                scale_cases,
                start=1,
            ):
                key = (
                    family.name,
                    factor,
                )

                if key not in qiskit_baseline:
                    raise KeyError(
                        f"historical development CSV 缺少基线："
                        f"{key}"
                    )

                (
                    qiskit_swap,
                    qiskit_depth,
                ) = qiskit_baseline[key]

                (
                    best_mapping,
                    swap_count,
                    weighted_depth,
                    runtime_ms,
                ) = run_pool(
                    case=case,
                    candidates=(
                        mapping_pools[case.name]
                    ),
                )

                outcome = compare(
                    swap_count=swap_count,
                    depth=weighted_depth,
                    qiskit_swap=qiskit_swap,
                    qiskit_depth=qiskit_depth,
                )

                row = CaseResult(
                    config=config.name,
                    family=family.name,
                    factor=factor,
                    cx_count=(
                        factor
                        * family.num_qubits
                    ),
                    topology=family.topology,
                    circuit_mode=(
                        family.circuit_mode
                    ),
                    best_mapping=best_mapping,
                    swap_count=swap_count,
                    added_two_qubit_count=(
                        3 * swap_count
                    ),
                    weighted_depth=(
                        weighted_depth
                    ),
                    runtime_ms=runtime_ms,
                    qiskit_swap=qiskit_swap,
                    qiskit_depth=qiskit_depth,
                    outcome=outcome,
                )

                case_rows.append(row)

                print(
                    f"  [{case_index}/{len(scale_cases)}] "
                    f"{family.name}, f={factor}: "
                    f"SWAP={swap_count}, "
                    f"depth={weighted_depth}, "
                    f"{outcome}"
                )

            summaries = build_summaries(
                case_rows
            )

            save_case_results(
                args.case_output,
                case_rows,
            )

            save_summaries(
                args.summary_output,
                summaries,
            )

    finally:
        apply_config(baseline_config)

    summaries = build_summaries(case_rows)
    print_summaries(summaries)

    print()
    print(
        "原始结果：",
        args.case_output.resolve(),
    )
    print(
        "汇总结果：",
        args.summary_output.resolve(),
    )


if __name__ == "__main__":
    main()
