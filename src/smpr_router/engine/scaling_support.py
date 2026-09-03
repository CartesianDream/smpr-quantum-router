from __future__ import annotations

import argparse
import csv
import io
import random
import statistics
import time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from one_step_policy import run_one_step_router
from qiskit_adapter import TestCase
from layout_pool import (
    MappingCandidate,
    MappingRunResult,
    generate_base_mapping_pool,
    run_qiskit_auto_layout,
)
from synthetic_cases import (
    generate_gate_specs,
    make_topology,
)


# ============================================================
# 1. 受控规模实验族
# ============================================================

@dataclass(frozen=True)
class FamilySpec:
    name: str
    num_qubits: int
    topology: str
    circuit_mode: str
    seed: int


FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        name="line8_uniform",
        num_qubits=8,
        topology="line",
        circuit_mode="uniform",
        seed=12001,
    ),
    FamilySpec(
        name="ring8_far",
        num_qubits=8,
        topology="ring",
        circuit_mode="far",
        seed=12002,
    ),
    FamilySpec(
        name="grid10_phase_shift",
        num_qubits=10,
        topology="grid",
        circuit_mode="phase_shift",
        seed=12003,
    ),
    FamilySpec(
        name="random10_alternating",
        num_qubits=10,
        topology="random",
        circuit_mode="alternating",
        seed=12004,
    ),
    FamilySpec(
        name="random10_parallel",
        num_qubits=10,
        topology="random",
        circuit_mode="parallel_layers",
        seed=12005,
    ),
)


# ============================================================
# 2. 结果数据结构
# ============================================================

@dataclass
class ProbeResult:
    family: str
    num_qubits: int
    topology: str
    circuit_mode: str
    length_factor: int
    cx_count: int

    independent_rollout_best_mapping: str
    independent_rollout_swap: int
    independent_rollout_added_2q: int
    independent_rollout_depth: int
    independent_rollout_pool_budget_ms: float
    independent_rollout_swap_per_cx: float

    qiskit_best_seed: int
    qiskit_swap: int
    qiskit_added_2q: int
    qiskit_depth: int
    qiskit_budget_ms: float
    qiskit_swap_per_cx: float

    swap_gap: int
    normalized_swap_gap: float
    outcome: str


# ============================================================
# 3. 生成前缀线路
# ============================================================

def parse_factors(text: str) -> tuple[int, ...]:
    values = tuple(
        sorted(
            {
                int(item.strip())
                for item in text.split(",")
                if item.strip()
            }
        )
    )

    if not values or min(values) <= 0:
        raise ValueError(
            "--factors 必须是逗号分隔的正整数，例如 3,6,12"
        )

    return values


def take_cx_prefix(
    gate_specs: tuple[tuple, ...],
    cx_count: int,
) -> tuple[tuple, ...]:
    """
    保留最前面的所有初始单比特门，
    再保留前 cx_count 个 CX。

    不重新随机生成短线路，因此同一家族的短线路
    是长线路的真实前缀。
    """

    prefix: list[tuple] = []
    seen_cx = 0

    for gate in gate_specs:
        if gate[0] == "cx":
            if seen_cx >= cx_count:
                break

            seen_cx += 1
            prefix.append(tuple(gate))
        else:
            prefix.append(tuple(gate))

    if seen_cx != cx_count:
        raise ValueError(
            f"母线路只有 {seen_cx} 个 CX，"
            f"无法截取 {cx_count} 个"
        )

    return tuple(prefix)


def build_scale_cases(
    factors: tuple[int, ...],
) -> list[tuple[FamilySpec, int, TestCase]]:
    cases: list[tuple[FamilySpec, int, TestCase]] = []
    max_factor = max(factors)

    for family in FAMILIES:
        rng = random.Random(family.seed)

        edges = make_topology(
            topology=family.topology,
            n=family.num_qubits,
            rng=rng,
        )

        maximum_cx = max_factor * family.num_qubits

        mother_gates = generate_gate_specs(
            n=family.num_qubits,
            num_two_qubit_gates=maximum_cx,
            circuit_mode=family.circuit_mode,
            edges=edges,
            rng=rng,
        )

        for factor in factors:
            cx_count = factor * family.num_qubits
            gates = take_cx_prefix(
                mother_gates,
                cx_count,
            )

            case = TestCase(
                name=(
                    f"scale_{family.name}_"
                    f"f{factor}_cx{cx_count}"
                ),
                num_qubits=family.num_qubits,
                edges=edges,
                gate_specs=gates,
                initial_mapping=tuple(
                    range(family.num_qubits)
                ),
            )

            cases.append(
                (family, factor, case)
            )

    return cases


# ============================================================
# 4. independent rollout：当前最佳结构 OneStep + 基础映射池
# ============================================================

def run_independent_rollout_candidate(
    case: TestCase,
    candidate: MappingCandidate,
) -> MappingRunResult:
    dag, hardware, _ = case.build()

    output_buffer = io.StringIO()
    start = time.perf_counter()

    try:
        with redirect_stdout(output_buffer):
            state = run_one_step_router(
                dag=dag,
                hardware=hardware,
                initial_mapping=list(
                    candidate.mapping
                ),
            )

        routing_ms = (
            time.perf_counter() - start
        ) * 1000.0

        state.assert_valid(
            dag=dag,
            hardware=hardware,
        )

        if len(state.executed_gates) != len(dag.gates):
            raise AssertionError(
                "independent rollout 结束后仍有逻辑门未执行"
            )

        swap_count = sum(
            operation[0] == "swap"
            for operation in state.physical_operations
        )

        return MappingRunResult(
            instance=case.name,
            method=f"independent rollout:{candidate.name}",
            seed=None,
            mapping=str(list(candidate.mapping)),
            swap_count=swap_count,
            added_two_qubit_count=3 * swap_count,
            weighted_depth=state.current_depth(),
            generation_ms=candidate.generation_ms,
            routing_ms=routing_ms,
            total_ms=(
                candidate.generation_ms
                + routing_ms
            ),
            valid=True,
            error="",
        )

    except Exception as exc:
        routing_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return MappingRunResult(
            instance=case.name,
            method=f"independent rollout:{candidate.name}",
            seed=None,
            mapping=str(list(candidate.mapping)),
            swap_count=None,
            added_two_qubit_count=None,
            weighted_depth=None,
            generation_ms=candidate.generation_ms,
            routing_ms=routing_ms,
            total_ms=(
                candidate.generation_ms
                + routing_ms
            ),
            valid=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def quality_key(
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
    valid = [row for row in rows if row.valid]

    if not valid:
        raise RuntimeError(
            "当前实例没有合法结果："
            + "; ".join(
                row.error
                for row in rows
                if row.error
            )
        )

    return min(
        valid,
        key=lambda row: (
            quality_key(row),
            row.total_ms,
            row.seed if row.seed is not None else -1,
        ),
    )


def run_independent_rollout_pool(
    case: TestCase,
) -> tuple[MappingRunResult, float]:
    candidates = generate_base_mapping_pool(case)

    rows = [
        run_independent_rollout_candidate(
            case=case,
            candidate=candidate,
        )
        for candidate in candidates
    ]

    best = best_valid(rows)

    budget = (
        sum(
            candidate.generation_ms
            for candidate in candidates
        )
        + sum(
            row.routing_ms
            for row in rows
            if row.valid
        )
    )

    return best, budget


# ============================================================
# 5. Qiskit：预热、多 seed、重复计时
# ============================================================

def run_qiskit_pool(
    case: TestCase,
    seeds: int,
    repeats: int,
) -> tuple[MappingRunResult, float]:
    # 预热，不计入预算。
    run_qiskit_auto_layout(
        case=case,
        seed=0,
    )

    seed_rows: list[MappingRunResult] = []

    for seed in range(seeds):
        repeated = [
            run_qiskit_auto_layout(
                case=case,
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
        reference_quality = quality_key(reference)

        if any(
            quality_key(row) != reference_quality
            for row in valid[1:]
        ):
            raise RuntimeError(
                f"{case.name} 的 Qiskit seed={seed} "
                "重复运行产生不同质量结果"
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
# 6. 比较与 CSV
# ============================================================

def compare_quality(
    independent_rollout: MappingRunResult,
    qiskit: MappingRunResult,
) -> str:
    left = quality_key(independent_rollout)
    right = quality_key(qiskit)

    if left < right:
        return "independent rollout胜"

    if left > right:
        return "Qiskit胜"

    return "平局"


CSV_FIELDS = tuple(
    ProbeResult.__annotations__.keys()
)


def save_csv(
    path: Path,
    rows: list[ProbeResult],
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
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: getattr(row, field)
                    for field in CSV_FIELDS
                }
            )


# ============================================================
# 7. 汇总
# ============================================================

def print_scale_summary(
    rows: list[ProbeResult],
) -> None:
    grouped: dict[
        int,
        list[ProbeResult],
    ] = defaultdict(list)

    for row in rows:
        grouped[row.length_factor].append(row)

    print()
    print("=" * 104)
    print("规模效应汇总")
    print("=" * 104)

    for factor in sorted(grouped):
        factor_rows = grouped[factor]
        outcomes = Counter(
            row.outcome
            for row in factor_rows
        )

        mean_gap = statistics.mean(
            row.normalized_swap_gap
            for row in factor_rows
        )

        median_gap = statistics.median(
            row.normalized_swap_gap
            for row in factor_rows
        )

        mean_independent_rollout_rate = statistics.mean(
            row.independent_rollout_swap_per_cx
            for row in factor_rows
        )

        mean_qiskit_rate = statistics.mean(
            row.qiskit_swap_per_cx
            for row in factor_rows
        )

        print()
        print(
            f"[长度因子 {factor}：CX={factor}n]"
        )
        print("  胜负：", dict(outcomes))
        print(
            "  平均归一化 SWAP 差 "
            "(independent rollout-Qiskit)/CX："
            f"{mean_gap:.4f}"
        )
        print(
            "  中位归一化 SWAP 差："
            f"{median_gap:.4f}"
        )
        print(
            "  independent rollout 平均 SWAP/CX："
            f"{mean_independent_rollout_rate:.4f}"
        )
        print(
            "  Qiskit 平均 SWAP/CX："
            f"{mean_qiskit_rate:.4f}"
        )

    print()
    print("判断尺度效应时主要看：")
    print(
        "  若长度增加时归一化 SWAP 差持续下降，"
        "说明 independent rollout 在长线路上相对改善。"
    )
    print(
        "  若归一化差不降反升，"
        "说明问题不是小规模造成的，"
        "当前评分会随线路增长累积误差。"
    )


# ============================================================
# 8. 主程序
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "historical development：受控规模探针。"
            "同一母线路取 3n、6n、12n 个 CX 前缀，"
            "检验 independent rollout 与 SABRE 的差距是否随线路增长缩小。"
        )
    )

    parser.add_argument(
        "--factors",
        type=str,
        default="3,6,12",
        help="CX 数量因子，例如 3,6,12 表示 3n、6n、12n",
    )

    parser.add_argument(
        "--qiskit-seeds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--qiskit-repeats",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--limit-families",
        type=int,
        default=None,
        help="仅运行前若干个线路族，用于快速测试",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/scaling_support.csv"
        ),
    )

    args = parser.parse_args()

    factors = parse_factors(args.factors)

    if min(
        args.qiskit_seeds,
        args.qiskit_repeats,
    ) <= 0:
        raise ValueError(
            "Qiskit 种子数和重复次数必须为正整数"
        )

    global FAMILIES

    if args.limit_families is not None:
        if args.limit_families <= 0:
            raise ValueError(
                "--limit-families 必须为正整数"
            )

        selected_families = FAMILIES[
            :args.limit_families
        ]
    else:
        selected_families = FAMILIES

    original_families = FAMILIES
    FAMILIES = selected_families

    cases = build_scale_cases(factors)

    # 恢复常量，避免交互式导入时留下副作用。
    FAMILIES = original_families

    results: list[ProbeResult] = []

    print("===== historical development：受控规模探针 =====")
    print("线路族数：", len(selected_families))
    print("长度因子：", factors)
    print(
        "Qiskit："
        f"{args.qiskit_seeds} seeds × "
        f"{args.qiskit_repeats} repeats"
    )

    for index, (
        family,
        factor,
        case,
    ) in enumerate(cases, start=1):
        cx_count = (
            factor * family.num_qubits
        )

        print()
        print(
            f"[{index}/{len(cases)}] "
            f"{family.name}, "
            f"CX={cx_count}"
        )

        independent_rollout_best, independent_rollout_budget = (
            run_independent_rollout_pool(case)
        )

        qiskit_best, qiskit_budget = (
            run_qiskit_pool(
                case=case,
                seeds=args.qiskit_seeds,
                repeats=args.qiskit_repeats,
            )
        )

        independent_rollout_swap = int(
            independent_rollout_best.swap_count
        )
        qiskit_swap = int(
            qiskit_best.swap_count
        )

        result = ProbeResult(
            family=family.name,
            num_qubits=family.num_qubits,
            topology=family.topology,
            circuit_mode=family.circuit_mode,
            length_factor=factor,
            cx_count=cx_count,
            independent_rollout_best_mapping=(
                independent_rollout_best.method
            ),
            independent_rollout_swap=independent_rollout_swap,
            independent_rollout_added_2q=int(
                independent_rollout_best.added_two_qubit_count
            ),
            independent_rollout_depth=int(
                independent_rollout_best.weighted_depth
            ),
            independent_rollout_pool_budget_ms=(
                independent_rollout_budget
            ),
            independent_rollout_swap_per_cx=(
                independent_rollout_swap / cx_count
            ),
            qiskit_best_seed=int(
                qiskit_best.seed
            ),
            qiskit_swap=qiskit_swap,
            qiskit_added_2q=int(
                qiskit_best.added_two_qubit_count
            ),
            qiskit_depth=int(
                qiskit_best.weighted_depth
            ),
            qiskit_budget_ms=(
                qiskit_budget
            ),
            qiskit_swap_per_cx=(
                qiskit_swap / cx_count
            ),
            swap_gap=(
                independent_rollout_swap - qiskit_swap
            ),
            normalized_swap_gap=(
                (independent_rollout_swap - qiskit_swap)
                / cx_count
            ),
            outcome=compare_quality(
                independent_rollout_best,
                qiskit_best,
            ),
        )

        results.append(result)
        save_csv(args.output, results)

        print(
            "  independent rollout："
            f"mapping={independent_rollout_best.method}, "
            f"SWAP={independent_rollout_swap}, "
            f"SWAP/CX={result.independent_rollout_swap_per_cx:.4f}, "
            f"budget={independent_rollout_budget:.2f} ms"
        )

        print(
            "  Qiskit："
            f"seed={qiskit_best.seed}, "
            f"SWAP={qiskit_swap}, "
            f"SWAP/CX={result.qiskit_swap_per_cx:.4f}, "
            f"budget={qiskit_budget:.2f} ms"
        )

        print(
            "  差值："
            f"{result.swap_gap:+d} SWAP, "
            f"归一化差={result.normalized_swap_gap:+.4f}, "
            f"{result.outcome}"
        )

    print_scale_summary(results)

    print()
    print(
        "CSV 已保存到：",
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
