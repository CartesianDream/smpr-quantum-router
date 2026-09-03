from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import structural_policy as structural_policy
import mapping_portfolio as mapping_portfolio
import rollout_policy as rollout_policy

from layout_pool import (
    MappingCandidate,
    generate_complete_mapping_pool,
)
from case_io import DevCase, load_cases


@dataclass
class MappingScore:
    instance: str
    mapping_name: str
    source: str
    search_round: int
    mapping: str
    base_swap: int | None
    base_depth: int | None
    base_runtime_ms: float
    selected_for_rollout: bool
    rollout_swap: int | None
    rollout_depth: int | None
    rollout_runtime_ms: float
    valid: bool
    error: str


@dataclass
class CaseSummary:
    instance: str
    circuit_mode: str
    num_qubits: int
    mapping_candidates_scored: int
    rollout_mappings: int
    old_mapping: str
    old_base_swap: int
    old_rollout_swap: int
    selected_mapping: str
    selected_base_swap: int
    selected_rollout_swap: int
    improvement_vs_old_rollout: int
    total_runtime_ms: float


def count_swaps(state) -> int:
    return sum(op[0] == "swap" for op in state.physical_operations)


def mapping_key(mapping: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in mapping)


def swapped_neighbors(mapping: tuple[int, ...]) -> list[tuple[int, ...]]:
    output: list[tuple[int, ...]] = []
    for left in range(len(mapping)):
        for right in range(left + 1, len(mapping)):
            candidate = list(mapping)
            candidate[left], candidate[right] = (
                candidate[right],
                candidate[left],
            )
            output.append(tuple(candidate))
    return output


def score_base_mapping(
    case: DevCase,
    mapping: MappingCandidate,
    config: structural_policy.StructuralConfig,
) -> tuple[int, int, float]:
    dag, hardware, _ = case.test_case.build()
    start = time.perf_counter()
    state = structural_policy.run_router(
        dag=dag,
        hardware=hardware,
        initial_mapping=list(mapping.mapping),
        config=config,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    state.assert_valid(dag, hardware)
    if len(state.executed_gates) != len(dag.gates):
        raise AssertionError("基线映射评分后仍有未执行逻辑门")
    return count_swaps(state), state.current_depth(), elapsed_ms


def search_mapping_candidates(
    case: DevCase,
    config: structural_policy.StructuralConfig,
    beam_width: int,
    search_rounds: int,
    use_forward_backward: bool,
) -> tuple[
    list[MappingScore],
    dict[tuple[int, ...], MappingCandidate],
]:
    initial = generate_complete_mapping_pool(
        case.test_case,
        use_forward_backward=use_forward_backward,
        verbose=False,
    )
    candidates: dict[tuple[int, ...], MappingCandidate] = {
        item.mapping: item for item in initial
    }
    source_by_mapping = {
        item.mapping: ("forward_backward" if item.name.endswith("_fb") else "base")
        for item in initial
    }
    round_by_mapping = {item.mapping: 0 for item in initial}
    cache: dict[tuple[int, ...], tuple[int, int, float, str]] = {}

    def evaluate(mapping: tuple[int, ...]) -> tuple[int, int, float, str]:
        if mapping in cache:
            return cache[mapping]
        item = candidates[mapping]
        try:
            swaps, depth, elapsed = score_base_mapping(case, item, config)
            cache[mapping] = (swaps, depth, elapsed, "")
        except Exception as error:
            cache[mapping] = (
                10**9,
                10**9,
                0.0,
                f"{type(error).__name__}: {error}",
            )
        return cache[mapping]

    for mapping in list(candidates):
        evaluate(mapping)

    for search_round in range(1, search_rounds + 1):
        valid = [
            mapping
            for mapping in candidates
            if evaluate(mapping)[0] < 10**9
        ]
        beam = sorted(
            valid,
            key=lambda mapping: (
                evaluate(mapping)[0],
                evaluate(mapping)[1],
                mapping,
            ),
        )[:beam_width]

        new_mappings: list[tuple[int, ...]] = []
        for parent_index, parent in enumerate(beam):
            for neighbor_index, neighbor in enumerate(swapped_neighbors(parent)):
                if neighbor in candidates:
                    continue
                item = MappingCandidate(
                    name=(
                        f"local_r{search_round}_p{parent_index}_"
                        f"n{neighbor_index}"
                    ),
                    mapping=neighbor,
                    generation_ms=0.0,
                )
                candidates[neighbor] = item
                source_by_mapping[neighbor] = "route_local_search"
                round_by_mapping[neighbor] = search_round
                new_mappings.append(neighbor)

        for mapping in new_mappings:
            evaluate(mapping)

    rows: list[MappingScore] = []
    for mapping, item in candidates.items():
        swaps, depth, elapsed, error = evaluate(mapping)
        rows.append(
            MappingScore(
                instance=case.name,
                mapping_name=item.name,
                source=source_by_mapping[mapping],
                search_round=round_by_mapping[mapping],
                mapping=mapping_key(mapping),
                base_swap=None if swaps >= 10**9 else swaps,
                base_depth=None if depth >= 10**9 else depth,
                base_runtime_ms=elapsed,
                selected_for_rollout=False,
                rollout_swap=None,
                rollout_depth=None,
                rollout_runtime_ms=0.0,
                valid=not error,
                error=error,
            )
        )

    return rows, candidates


def run_case(
    case: DevCase,
    config: structural_policy.StructuralConfig,
    beam_width: int,
    search_rounds: int,
    rollout_top: int,
    top_k: int,
    use_forward_backward: bool,
) -> tuple[CaseSummary, list[MappingScore]]:
    start = time.perf_counter()
    rows, candidates = search_mapping_candidates(
        case,
        config,
        beam_width,
        search_rounds,
        use_forward_backward,
    )
    valid = [row for row in rows if row.valid]
    if not valid:
        raise RuntimeError(f"{case.name} 没有合法初始映射")

    by_name = {item.name: item for item in candidates.values()}
    old_mappings = mapping_portfolio.make_mappings(case, "v1_best")
    old_attempt = rollout_policy.run_attempt(case, old_mappings[0], config, top_k)
    if not old_attempt.valid:
        raise RuntimeError(f"旧 historical development 映射失败：{old_attempt.error}")

    ranked = sorted(
        valid,
        key=lambda row: (
            int(row.base_swap),
            int(row.base_depth),
            row.mapping_name,
        ),
    )

    selected_names: list[str] = []
    old_mapping_tuple = old_mappings[0].mapping
    old_pool_name = next(
        (
            item.name
            for item in candidates.values()
            if item.mapping == old_mapping_tuple
        ),
        None,
    )
    if old_pool_name is not None:
        selected_names.append(old_pool_name)

    for row in ranked:
        if len(selected_names) >= rollout_top:
            break
        if row.mapping_name not in selected_names:
            selected_names.append(row.mapping_name)

    rollout_attempts: list[rollout_policy.AttemptRow] = []
    for name in selected_names:
        attempt = rollout_policy.run_attempt(case, by_name[name], config, top_k)
        rollout_attempts.append(attempt)
        target = next(row for row in rows if row.mapping_name == name)
        target.selected_for_rollout = True
        target.rollout_runtime_ms = attempt.rollout_runtime_ms
        if attempt.valid:
            target.rollout_swap = attempt.rollout_swap
            target.rollout_depth = attempt.rollout_depth
        else:
            target.valid = False
            target.error = attempt.error

    valid_rollouts = [item for item in rollout_attempts if item.valid]
    if not valid_rollouts:
        raise RuntimeError(f"{case.name} 所有候选 rollout 均失败")
    best_attempt = min(
        valid_rollouts,
        key=lambda item: (
            int(item.rollout_swap),
            int(item.rollout_depth),
            item.mapping,
        ),
    )

    return (
        CaseSummary(
            instance=case.name,
            circuit_mode=case.circuit_mode,
            num_qubits=case.num_qubits,
            mapping_candidates_scored=len(rows),
            rollout_mappings=len(valid_rollouts),
            old_mapping=old_attempt.mapping,
            old_base_swap=int(old_attempt.base_swap),
            old_rollout_swap=int(old_attempt.rollout_swap),
            selected_mapping=best_attempt.mapping,
            selected_base_swap=int(best_attempt.base_swap),
            selected_rollout_swap=int(best_attempt.rollout_swap),
            improvement_vs_old_rollout=(
                int(old_attempt.rollout_swap)
                - int(best_attempt.rollout_swap)
            ),
            total_runtime_ms=(time.perf_counter() - start) * 1000.0,
        ),
        rows,
    )


def write_csv(path: Path, rows: list, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="historical development：ring-n10 的路由代价驱动初始映射局部搜索"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmarks_v4/dev.json"),
    )
    parser.add_argument("--case-indices", type=str, default="7,8,10")
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--search-rounds", type=int, default=1)
    parser.add_argument("--rollout-top", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--no-forward-backward", action="store_true"
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=Path("results/local_layout_search_ring_mapping_candidates.csv"),
    )
    parser.add_argument(
        "--case-output",
        type=Path,
        default=Path("results/local_layout_search_ring_mapping_cases.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/local_layout_search_ring_mapping_summary.json"),
    )
    args = parser.parse_args()

    if args.beam_width <= 0 or args.search_rounds < 0:
        raise ValueError("beam-width 必须为正，search-rounds 必须非负")
    if args.rollout_top <= 0 or args.top_k < 0:
        raise ValueError("rollout-top 必须为正，top-k 必须非负")

    structural_policy.one_step_policy.THETA = 5
    structural_policy.one_step_policy.W_FUTURE = 0.5
    structural_policy.one_step_policy.GAMMA = 0.5
    structural_policy.one_step_policy.ALPHA = 0.5
    structural_policy.one_step_policy.LAMBDA_DEPTH = 0.0
    config = structural_policy.CONFIG_BY_NAME["ordered_potential"]

    all_cases = load_cases(args.input, limit=None)
    cases = mapping_portfolio.choose_cases(
        all_cases,
        mapping_portfolio.parse_indices(args.case_indices),
        None,
    )

    for case in cases:
        if case.topology != "ring" or case.num_qubits < 10:
            raise ValueError(
                f"historical development 当前只允许 ring-n10：{case.name}"
            )

    print("===== historical development：ring-n10 初始映射局部搜索 =====")
    print(f"输入：{args.input.resolve()}")
    print(f"实例数：{len(cases)}")
    print(f"beam width：{args.beam_width}")
    print(f"search rounds：{args.search_rounds}")
    print(f"rollout mappings：{args.rollout_top}")
    print(f"rollout top-k：{args.top_k}")
    print(f"forward/backward：{not args.no_forward_backward}")

    all_mapping_rows: list[MappingScore] = []
    case_rows: list[CaseSummary] = []
    for index, case in enumerate(cases, start=1):
        summary, mapping_rows = run_case(
            case,
            config,
            args.beam_width,
            args.search_rounds,
            args.rollout_top,
            args.top_k,
            not args.no_forward_backward,
        )
        all_mapping_rows.extend(mapping_rows)
        case_rows.append(summary)
        print(
            f"[{index}/{len(cases)}] {case.name}: "
            f"old={summary.old_base_swap}->{summary.old_rollout_swap} "
            f"({summary.old_mapping}), "
            f"search={summary.selected_base_swap}->"
            f"{summary.selected_rollout_swap} "
            f"({summary.selected_mapping}), "
            f"DeltaSWAP=-{summary.improvement_vs_old_rollout}, "
            f"mappings={summary.mapping_candidates_scored}/"
            f"{summary.rollout_mappings}, "
            f"{summary.total_runtime_ms:.1f} ms"
        )
        write_csv(
            args.mapping_output,
            all_mapping_rows,
            MappingScore.__annotations__.keys(),
        )
        write_csv(
            args.case_output,
            case_rows,
            CaseSummary.__annotations__.keys(),
        )

    summary = {
        "case_count": len(case_rows),
        "old_rollout_swap": sum(row.old_rollout_swap for row in case_rows),
        "searched_rollout_swap": sum(
            row.selected_rollout_swap for row in case_rows
        ),
        "swap_improvement": sum(
            row.improvement_vs_old_rollout for row in case_rows
        ),
        "improved_cases": sum(
            row.improvement_vs_old_rollout > 0 for row in case_rows
        ),
        "total_runtime_ms": sum(row.total_runtime_ms for row in case_rows),
        "beam_width": args.beam_width,
        "search_rounds": args.search_rounds,
        "rollout_top": args.rollout_top,
        "top_k": args.top_k,
        "forward_backward": not args.no_forward_backward,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print("===== historical development 汇总 =====")
    print(f"旧 rollout total SWAP：{summary['old_rollout_swap']}")
    print(f"搜索后 total SWAP：{summary['searched_rollout_swap']}")
    print(f"减少 SWAP：{summary['swap_improvement']}")
    print(f"严格改善实例：{summary['improved_cases']}/{summary['case_count']}")
    print(f"总运行时间：{summary['total_runtime_ms']:.1f} ms")
    print(f"映射明细：{args.mapping_output.resolve()}")
    print(f"逐实例结果：{args.case_output.resolve()}")
    print(f"汇总 JSON：{args.summary_output.resolve()}")


if __name__ == "__main__":
    main()
