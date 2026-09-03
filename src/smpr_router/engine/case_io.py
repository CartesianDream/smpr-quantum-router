from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from qiskit_adapter import TestCase
from layout_pool import (
    MappingCandidate,
    MappingRunResult,
    deduplicate_candidates,
    forward_backward_polish,
    generate_base_mapping_pool,
    run_mapping_candidate,
    run_qiskit_auto_layout,
)


# ============================================================
# 1. 开发集元数据
# ============================================================

@dataclass(frozen=True)
class DevCase:
    test_case: TestCase
    split: str
    topology: str
    circuit_mode: str
    source_seed: int

    @property
    def name(self) -> str:
        return self.test_case.name

    @property
    def num_qubits(self) -> int:
        return self.test_case.num_qubits

    @property
    def two_qubit_gate_count(self) -> int:
        return sum(
            gate[0] == "cx"
            for gate in self.test_case.gate_specs
        )


@dataclass
class SummaryRow:
    instance: str
    split: str
    topology: str
    circuit_mode: str
    num_qubits: int
    two_qubit_gate_count: int

    base_best_method: str
    base_swap: int
    base_added_2q: int
    base_depth: int
    base_budget_ms: float

    selective_best_method: str
    selective_swap: int
    selective_added_2q: int
    selective_depth: int
    selective_budget_ms: float
    fb_improved: bool

    qiskit_best_seed: int
    qiskit_swap: int
    qiskit_added_2q: int
    qiskit_depth: int
    qiskit_budget_ms: float

    base_vs_qiskit: str
    selective_vs_qiskit: str


# ============================================================
# 2. 读取 historical development 数据
# ============================================================

def load_cases(
    path: Path,
    limit: int | None,
) -> list[DevCase]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    raw_cases = payload["cases"]

    if limit is not None:
        raw_cases = raw_cases[:limit]

    cases: list[DevCase] = []

    for raw in raw_cases:
        test_case = TestCase(
            name=str(raw["name"]),
            num_qubits=int(raw["num_qubits"]),
            edges=tuple(
                (int(edge[0]), int(edge[1]))
                for edge in raw["edges"]
            ),
            gate_specs=tuple(
                tuple(gate)
                for gate in raw["gate_specs"]
            ),
            initial_mapping=tuple(
                int(value)
                for value in raw["initial_mapping"]
            ),
            physical_qubits=int(
                raw.get("physical_qubits", raw["num_qubits"])
            ),
        )

        cases.append(
            DevCase(
                test_case=test_case,
                split=str(raw["split"]),
                topology=str(raw["topology"]),
                circuit_mode=str(
                    raw["circuit_mode"]
                ),
                source_seed=int(raw["seed"]),
            )
        )

    return cases


# ============================================================
# 3. 通用比较
# ============================================================

def valid_mapping_results(
    rows: list[MappingRunResult],
) -> list[MappingRunResult]:
    return [
        row
        for row in rows
        if row.valid
    ]


def result_score(
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


def best_result(
    rows: list[MappingRunResult],
) -> MappingRunResult:
    valid = valid_mapping_results(rows)

    if not valid:
        error_text = "; ".join(
            row.error
            for row in rows
            if row.error
        )

        raise RuntimeError(
            "没有合法结果。"
            f"错误信息：{error_text}"
        )

    return min(
        valid,
        key=lambda row: (
            result_score(row),
            row.total_ms,
            row.seed if row.seed is not None else -1,
        ),
    )


def compare_scores(
    left: tuple[int, int],
    right: tuple[int, int],
) -> str:
    if left < right:
        return "independent rollout胜"
    if left > right:
        return "Qiskit胜"
    return "平局"


# ============================================================
# 4. independent rollout 基础映射池
# ============================================================

def run_base_pool(
    case: DevCase,
    repeats: int,
    verbose: bool,
) -> tuple[
    list[MappingCandidate],
    list[MappingRunResult],
]:
    candidates = generate_base_mapping_pool(
        case.test_case
    )

    results: list[MappingRunResult] = []

    for candidate in candidates:
        repeated: list[MappingRunResult] = []

        for _ in range(repeats):
            repeated.append(
                run_mapping_candidate(
                    case=case.test_case,
                    candidate=candidate,
                    verbose=verbose,
                )
            )

        best_quality = best_result(repeated)
        valid = valid_mapping_results(repeated)

        median_routing = statistics.median(
            row.routing_ms
            for row in valid
        )

        results.append(
            MappingRunResult(
                instance=best_quality.instance,
                method=best_quality.method,
                seed=None,
                mapping=best_quality.mapping,
                swap_count=best_quality.swap_count,
                added_two_qubit_count=(
                    best_quality.added_two_qubit_count
                ),
                weighted_depth=(
                    best_quality.weighted_depth
                ),
                generation_ms=(
                    candidate.generation_ms
                ),
                routing_ms=median_routing,
                total_ms=(
                    candidate.generation_ms
                    + median_routing
                ),
                valid=True,
                error="",
            )
        )

    return candidates, results


def base_pool_budget(
    candidates: list[MappingCandidate],
    results: list[MappingRunResult],
) -> float:
    return (
        sum(
            candidate.generation_ms
            for candidate in candidates
        )
        + sum(
            row.routing_ms
            for row in results
            if row.valid
        )
    )


# ============================================================
# 5. 选择性正反遍历
# ============================================================

def find_candidate_by_method(
    candidates: list[MappingCandidate],
    method: str,
) -> MappingCandidate:
    mapping_name = method.split(":", 1)[1]

    for candidate in candidates:
        if candidate.name == mapping_name:
            return candidate

    raise KeyError(
        f"找不到映射候选：{mapping_name}"
    )


def select_fb_sources(
    candidates: list[MappingCandidate],
    base_results: list[MappingRunResult],
) -> list[MappingCandidate]:
    """
    选择性规则：

    1. identity；
    2. 基础映射池中完整路由质量最好的候选。

    两者相同则只保留一个。
    """

    identity = next(
        candidate
        for candidate in candidates
        if candidate.name == "identity"
    )

    base_best = best_result(base_results)

    best_candidate = find_candidate_by_method(
        candidates,
        base_best.method,
    )

    return deduplicate_candidates(
        [identity, best_candidate]
    )


def run_selective_fb(
    case: DevCase,
    sources: list[MappingCandidate],
    repeats: int,
    verbose: bool,
) -> tuple[
    list[MappingCandidate],
    list[MappingRunResult],
    float,
]:
    polished_candidates: list[MappingCandidate] = []

    for source in sources:
        polished_candidates.append(
            forward_backward_polish(
                case=case.test_case,
                candidate=source,
                verbose=verbose,
            )
        )

    polished_candidates = deduplicate_candidates(
        polished_candidates
    )

    results: list[MappingRunResult] = []
    extra_generation_budget = 0.0

    source_generation = {
        source.name: source.generation_ms
        for source in sources
    }

    for candidate in polished_candidates:
        base_name = candidate.name.removesuffix(
            "_fb"
        )

        extra_generation_budget += max(
            0.0,
            candidate.generation_ms
            - source_generation.get(base_name, 0.0),
        )

        repeated: list[MappingRunResult] = []

        for _ in range(repeats):
            repeated.append(
                run_mapping_candidate(
                    case=case.test_case,
                    candidate=candidate,
                    verbose=verbose,
                )
            )

        best_quality = best_result(repeated)
        valid = valid_mapping_results(repeated)

        median_routing = statistics.median(
            row.routing_ms
            for row in valid
        )

        results.append(
            MappingRunResult(
                instance=best_quality.instance,
                method=best_quality.method,
                seed=None,
                mapping=best_quality.mapping,
                swap_count=best_quality.swap_count,
                added_two_qubit_count=(
                    best_quality.added_two_qubit_count
                ),
                weighted_depth=(
                    best_quality.weighted_depth
                ),
                generation_ms=(
                    candidate.generation_ms
                ),
                routing_ms=median_routing,
                total_ms=(
                    candidate.generation_ms
                    + median_routing
                ),
                valid=True,
                error="",
            )
        )

    selective_extra_budget = (
        extra_generation_budget
        + sum(
            row.routing_ms
            for row in results
            if row.valid
        )
    )

    return (
        polished_candidates,
        results,
        selective_extra_budget,
    )


# ============================================================
# 6. Qiskit SABRE：预热与重复计时
# ============================================================

def warmup_qiskit(
    case: DevCase,
) -> None:
    """
    预热结果丢弃，减少首次导入、缓存和系统抖动影响。
    """

    run_qiskit_auto_layout(
        case=case.test_case,
        seed=0,
    )


def run_qiskit_multiseed(
    case: DevCase,
    seeds: int,
    repeats: int,
) -> list[MappingRunResult]:
    warmup_qiskit(case)

    results: list[MappingRunResult] = []

    for seed in range(seeds):
        repeated = [
            run_qiskit_auto_layout(
                case=case.test_case,
                seed=seed,
            )
            for _ in range(repeats)
        ]

        valid = valid_mapping_results(repeated)

        if not valid:
            results.append(repeated[0])
            continue

        reference = valid[0]
        reference_score = result_score(reference)

        unstable = any(
            result_score(row) != reference_score
            for row in valid[1:]
        )

        if unstable:
            raise RuntimeError(
                f"{case.name} 的 Qiskit seed={seed} "
                "在重复运行中产生了不同质量结果"
            )

        median_total = statistics.median(
            row.total_ms
            for row in valid
        )

        results.append(
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
                routing_ms=median_total,
                total_ms=median_total,
                valid=True,
                error="",
            )
        )

    return results


# ============================================================
# 7. CSV
# ============================================================

RAW_FIELDS = (
    "instance",
    "split",
    "topology",
    "circuit_mode",
    "num_qubits",
    "two_qubit_gate_count",
    "category",
    "method",
    "seed",
    "mapping",
    "swap_count",
    "added_two_qubit_count",
    "weighted_depth",
    "generation_ms",
    "routing_ms",
    "total_ms",
    "valid",
    "error",
)


def save_raw_csv(
    path: Path,
    rows: list[dict[str, object]],
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
            fieldnames=RAW_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)


SUMMARY_FIELDS = tuple(
    SummaryRow.__annotations__.keys()
)


def save_summary_csv(
    path: Path,
    rows: list[SummaryRow],
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


def raw_record(
    case: DevCase,
    category: str,
    row: MappingRunResult,
) -> dict[str, object]:
    return {
        "instance": case.name,
        "split": case.split,
        "topology": case.topology,
        "circuit_mode": case.circuit_mode,
        "num_qubits": case.num_qubits,
        "two_qubit_gate_count": (
            case.two_qubit_gate_count
        ),
        "category": category,
        "method": row.method,
        "seed": (
            "" if row.seed is None else row.seed
        ),
        "mapping": row.mapping,
        "swap_count": row.swap_count,
        "added_two_qubit_count": (
            row.added_two_qubit_count
        ),
        "weighted_depth": row.weighted_depth,
        "generation_ms": (
            f"{row.generation_ms:.3f}"
        ),
        "routing_ms": (
            f"{row.routing_ms:.3f}"
        ),
        "total_ms": f"{row.total_ms:.3f}",
        "valid": row.valid,
        "error": row.error,
    }


# ============================================================
# 8. 输出汇总
# ============================================================

def print_case_summary(
    summary: SummaryRow,
) -> None:
    print(
        "  Base pool："
        f"{summary.base_best_method}, "
        f"SWAP={summary.base_swap}, "
        f"depth={summary.base_depth}, "
        f"budget={summary.base_budget_ms:.2f} ms"
    )

    print(
        "  Selective FB："
        f"{summary.selective_best_method}, "
        f"SWAP={summary.selective_swap}, "
        f"depth={summary.selective_depth}, "
        f"budget={summary.selective_budget_ms:.2f} ms, "
        f"improved={summary.fb_improved}"
    )

    print(
        "  Qiskit："
        f"seed={summary.qiskit_best_seed}, "
        f"SWAP={summary.qiskit_swap}, "
        f"depth={summary.qiskit_depth}, "
        f"budget={summary.qiskit_budget_ms:.2f} ms"
    )

    print(
        "  质量："
        f"base={summary.base_vs_qiskit}, "
        f"selective={summary.selective_vs_qiskit}"
    )


def print_global_summary(
    rows: list[SummaryRow],
) -> None:
    base_outcomes = Counter(
        row.base_vs_qiskit
        for row in rows
    )

    selective_outcomes = Counter(
        row.selective_vs_qiskit
        for row in rows
    )

    print()
    print("=" * 96)
    print("开发集总体汇总")
    print("=" * 96)

    print(
        "Base pool vs Qiskit：",
        dict(base_outcomes),
    )

    print(
        "Selective FB vs Qiskit：",
        dict(selective_outcomes),
    )

    print(
        "Selective FB 改善实例数：",
        sum(row.fb_improved for row in rows),
        "/",
        len(rows),
    )

    total_base_budget = sum(
        row.base_budget_ms
        for row in rows
    )

    total_selective_budget = sum(
        row.selective_budget_ms
        for row in rows
    )

    total_qiskit_budget = sum(
        row.qiskit_budget_ms
        for row in rows
    )

    print(
        "Base pool 总预算："
        f"{total_base_budget:.3f} ms"
    )

    print(
        "Selective FB 总预算："
        f"{total_selective_budget:.3f} ms"
    )

    print(
        "Qiskit 总预算："
        f"{total_qiskit_budget:.3f} ms"
    )

    if total_qiskit_budget > 0:
        print(
            "Base/Qiskit 预算比："
            f"{total_base_budget / total_qiskit_budget:.3f}"
        )

        print(
            "Selective/Qiskit 预算比："
            f"{total_selective_budget / total_qiskit_budget:.3f}"
        )

    print()
    print("按线路模式统计 selective 质量：")

    grouped: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        grouped[row.circuit_mode][
            row.selective_vs_qiskit
        ] += 1

    for mode in sorted(grouped):
        print(
            f"  {mode}: {dict(grouped[mode])}"
        )


# ============================================================
# 9. 主程序
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "historical development：开发集评测。"
            "比较基础映射池、选择性正反遍历和 "
            "预热后的多种子 Qiskit SABRE。"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/benchmarks/dev.json"
        ),
        help="historical development 生成的数据文件",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只运行前若干个实例，用于快速检查",
    )

    parser.add_argument(
        "--qiskit-seeds",
        type=int,
        default=10,
        help="Qiskit SABRE 随机种子数",
    )

    parser.add_argument(
        "--qiskit-repeats",
        type=int,
        default=2,
        help="每个 Qiskit seed 重复计时次数",
    )

    parser.add_argument(
        "--independent_rollout-repeats",
        type=int,
        default=1,
        help="每个 independent rollout 映射重复计时次数",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示 independent rollout 的逐步路由日志",
    )

    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path(
            "results/case_io_dev_raw.csv"
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "results/case_io_dev_summary.csv"
        ),
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须为正整数")

    if min(
        args.qiskit_seeds,
        args.qiskit_repeats,
        args.independent_rollout_repeats,
    ) <= 0:
        raise ValueError(
            "种子数和重复次数必须为正整数"
        )

    if not args.input.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{args.input.resolve()}"
        )

    cases = load_cases(
        args.input,
        args.limit,
    )

    raw_rows: list[dict[str, object]] = []
    summaries: list[SummaryRow] = []

    print("===== historical development：开发集评测 =====")
    print("实例数：", len(cases))
    print(
        "Qiskit："
        f"{args.qiskit_seeds} seeds × "
        f"{args.qiskit_repeats} repeats"
    )
    print(
        "independent rollout 每映射重复次数：",
        args.independent_rollout_repeats,
    )

    total_start = time.perf_counter()

    for index, case in enumerate(
        cases,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(cases)}] "
            f"{case.name}"
        )

        base_candidates, base_results = (
            run_base_pool(
                case=case,
                repeats=args.independent_rollout_repeats,
                verbose=args.verbose,
            )
        )

        base_best = best_result(base_results)
        base_budget = base_pool_budget(
            base_candidates,
            base_results,
        )

        for row in base_results:
            raw_rows.append(
                raw_record(
                    case,
                    "independent_rollout_base",
                    row,
                )
            )

        fb_sources = select_fb_sources(
            base_candidates,
            base_results,
        )

        _, fb_results, fb_extra_budget = (
            run_selective_fb(
                case=case,
                sources=fb_sources,
                repeats=args.independent_rollout_repeats,
                verbose=args.verbose,
            )
        )

        for row in fb_results:
            raw_rows.append(
                raw_record(
                    case,
                    "independent_rollout_selective_fb",
                    row,
                )
            )

        selective_results = (
            base_results + fb_results
        )

        selective_best = best_result(
            selective_results
        )

        selective_budget = (
            base_budget + fb_extra_budget
        )

        qiskit_results = run_qiskit_multiseed(
            case=case,
            seeds=args.qiskit_seeds,
            repeats=args.qiskit_repeats,
        )

        for row in qiskit_results:
            raw_rows.append(
                raw_record(
                    case,
                    "qiskit",
                    row,
                )
            )

        qiskit_best = best_result(
            qiskit_results
        )

        qiskit_budget = sum(
            row.total_ms
            for row in qiskit_results
            if row.valid
        )

        base_score = result_score(base_best)
        selective_score = result_score(
            selective_best
        )
        qiskit_score = result_score(
            qiskit_best
        )

        summary = SummaryRow(
            instance=case.name,
            split=case.split,
            topology=case.topology,
            circuit_mode=case.circuit_mode,
            num_qubits=case.num_qubits,
            two_qubit_gate_count=(
                case.two_qubit_gate_count
            ),
            base_best_method=base_best.method,
            base_swap=int(base_best.swap_count),
            base_added_2q=int(
                base_best.added_two_qubit_count
            ),
            base_depth=int(
                base_best.weighted_depth
            ),
            base_budget_ms=base_budget,
            selective_best_method=(
                selective_best.method
            ),
            selective_swap=int(
                selective_best.swap_count
            ),
            selective_added_2q=int(
                selective_best.added_two_qubit_count
            ),
            selective_depth=int(
                selective_best.weighted_depth
            ),
            selective_budget_ms=(
                selective_budget
            ),
            fb_improved=(
                selective_score < base_score
            ),
            qiskit_best_seed=int(
                qiskit_best.seed
            ),
            qiskit_swap=int(
                qiskit_best.swap_count
            ),
            qiskit_added_2q=int(
                qiskit_best.added_two_qubit_count
            ),
            qiskit_depth=int(
                qiskit_best.weighted_depth
            ),
            qiskit_budget_ms=qiskit_budget,
            base_vs_qiskit=compare_scores(
                base_score,
                qiskit_score,
            ),
            selective_vs_qiskit=compare_scores(
                selective_score,
                qiskit_score,
            ),
        )

        summaries.append(summary)
        print_case_summary(summary)

        save_raw_csv(
            args.raw_output,
            raw_rows,
        )

        save_summary_csv(
            args.summary_output,
            summaries,
        )

    elapsed = (
        time.perf_counter() - total_start
    )

    print_global_summary(summaries)

    print()
    print(
        "总运行时间："
        f"{elapsed:.2f} s"
    )

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
