from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import structural_policy as structural_policy
import mapping_portfolio as mapping_portfolio

from routing_model import CircuitDAG, HardwareGraph, RoutingState, drain
from layout_pool import MappingCandidate
from case_io import DevCase, load_cases


# ============================================================
# 1. 结果结构
# ============================================================


@dataclass
class RolloutStats:
    route_decisions: int = 0
    changed_decisions: int = 0
    rollout_candidates: int = 0
    cache_hits: int = 0


@dataclass
class AttemptRow:
    instance: str
    topology: str
    circuit_mode: str
    num_qubits: int
    cx_count: int
    mapping: str
    base_policy: str
    top_k: int

    base_swap: int | None
    base_depth: int | None
    rollout_swap: int | None
    rollout_depth: int | None
    swap_improvement: int | None

    route_decisions: int
    changed_decisions: int
    rollout_candidates: int
    cache_hits: int
    base_runtime_ms: float
    rollout_runtime_ms: float
    valid: bool
    error: str


@dataclass
class CaseRow:
    instance: str
    topology: str
    circuit_mode: str
    num_qubits: int
    cx_count: int
    mapping_count: int
    base_policy: str
    top_k: int

    base_mapping: str
    base_swap: int
    base_depth: int
    rollout_mapping: str
    rollout_swap: int
    rollout_depth: int
    swap_improvement: int
    changed_decisions: int
    rollout_candidates: int
    total_runtime_ms: float


# ============================================================
# 2. 完整基线 rollout
# ============================================================


def count_swaps(state: RoutingState) -> int:
    return sum(
        operation[0] == "swap" for operation in state.physical_operations
    )


def rollout_fingerprint(
    state: RoutingState,
    last_swap: tuple[int, int] | None,
) -> tuple:
    """
    这些量完整决定后续 one-step 策略的行为。

    physical_depth 必须保留，因为它会影响候选的深度增量平局规则。
    physical_operations 不参与后续决策，因此不放进缓存键。
    """
    return (
        tuple(state.logical_to_physical),
        tuple(state.remaining_predecessors),
        tuple(sorted(state.executed_gates)),
        tuple(state.physical_depth),
        None if last_swap is None else structural_policy.normalize_edge(last_swap),
    )


def complete_base_rollout(
    state_after_action: RoutingState,
    last_swap: tuple[int, int],
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
    base_config: structural_policy.StructuralConfig,
    cache: dict[tuple, tuple[int, int]],
) -> tuple[int, int, bool]:
    """返回候选动作之后还需的 SWAP 数、最终深度增量、是否命中缓存。"""
    key = rollout_fingerprint(state_after_action, last_swap)
    if key in cache:
        swaps, depth = cache[key]
        return swaps, depth, True

    state = copy.deepcopy(state_after_action)
    start_depth = state.current_depth()
    previous = last_swap
    swaps = 0
    used_cached_tail = False
    # (fingerprint, swaps already taken from the requested state, depth here)
    path: list[tuple[tuple, int, int]] = []

    while len(state.executed_gates) < len(dag.gates):
        state_key = rollout_fingerprint(state, previous)
        cached_tail = cache.get(state_key)
        if cached_tail is not None:
            tail_swaps, tail_depth = cached_tail
            total_swaps = swaps + tail_swaps
            final_depth = state.current_depth() + tail_depth
            used_cached_tail = True
            break

        path.append((state_key, swaps, state.current_depth()))
        best = structural_policy.choose_one_step(
            state=state,
            dag=dag,
            hardware=hardware,
            critical_weights=critical_weights,
            last_swap=previous,
            config=base_config,
        )
        state = best.state_after
        previous = best.edge
        swaps += 1
        state.assert_valid(dag, hardware)

        if swaps > structural_policy.MAX_ROUTE_STEPS:
            raise RuntimeError("完整 rollout 超过 MAX_ROUTE_STEPS")
    else:
        total_swaps = swaps
        final_depth = state.current_depth()

    # 不只缓存 rollout 起点，还反向缓存整条基线轨迹上的所有状态。
    # 后续候选只要汇入同一状态，就能直接复用完整尾部成本。
    for state_key, swaps_before, depth_here in path:
        cache[state_key] = (
            total_swaps - swaps_before,
            final_depth - depth_here,
        )

    result = (total_swaps, final_depth - start_depth)
    cache[key] = result
    return result[0], result[1], used_cached_tail


def candidate_rollout_key(
    evaluation: structural_policy.EdgeEvaluation,
    remaining_swaps: int,
    remaining_depth: int,
) -> tuple:
    return (
        1 + remaining_swaps,
        evaluation.depth_increment + remaining_depth,
        structural_policy.evaluation_key(evaluation),
    )


def choose_rollout_improved_action(
    state: RoutingState,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
    last_swap: tuple[int, int] | None,
    base_config: structural_policy.StructuralConfig,
    top_k: int,
    cache: dict[tuple, tuple[int, int]],
    stats: RolloutStats,
) -> structural_policy.EdgeEvaluation:
    """
    对启发式 shortlist 做完整基线 rollout。

    设基线策略为 pi，候选动作代价为

        Q_pi(s, a) = 1 + V_pi(T(s, a)).

    只有候选的 Q_pi 严格少于基线动作时才偏离基线。因此 shortlist
    无论多小都始终包含基线动作，并且不会因为代理分数而接受更多 SWAP。
    """
    preview = structural_policy.build_preview(state, dag, hardware, base_config)
    evaluations = structural_policy.evaluate_all_edges(
        state=state,
        preview=preview,
        dag=dag,
        hardware=hardware,
        critical_weights=critical_weights,
        last_swap=last_swap,
        config=base_config,
    )
    baseline = evaluations[0]
    shortlist = evaluations if top_k == 0 else evaluations[:top_k]

    evaluated: list[tuple[structural_policy.EdgeEvaluation, int, int]] = []
    for item in shortlist:
        remaining_swaps, remaining_depth, cache_hit = complete_base_rollout(
            state_after_action=item.state_after,
            last_swap=item.edge,
            dag=dag,
            hardware=hardware,
            critical_weights=critical_weights,
            base_config=base_config,
            cache=cache,
        )
        evaluated.append((item, remaining_swaps, remaining_depth))
        stats.rollout_candidates += 1
        stats.cache_hits += int(cache_hit)

    by_edge = {item.edge: (item, swaps, depth) for item, swaps, depth in evaluated}
    baseline_item, baseline_swaps, baseline_depth = by_edge[baseline.edge]
    best_item, best_swaps, best_depth = min(
        evaluated,
        key=lambda row: candidate_rollout_key(row[0], row[1], row[2]),
    )

    baseline_q = 1 + baseline_swaps
    best_q = 1 + best_swaps

    # SWAP 是原题主目标。相同 SWAP 时保持基线动作，避免等价 Q 值动作
    # 在重规划时形成无意义循环。
    if best_q < baseline_q:
        stats.changed_decisions += 1
        return best_item

    _ = baseline_depth, best_depth  # 深度仅用于 rollout 内部稳定排序。
    return baseline_item


def run_rollout_router(
    dag: CircuitDAG,
    hardware: HardwareGraph,
    initial_mapping: list[int],
    base_config: structural_policy.StructuralConfig,
    top_k: int,
) -> tuple[RoutingState, RolloutStats]:
    state = structural_policy.one_step_policy.make_initial_state(dag, hardware, initial_mapping)
    critical_weights = structural_policy.one_step_policy.compute_critical_weights(
        dag, structural_policy.one_step_policy.ALPHA
    )
    drain(state, dag, hardware)

    stats = RolloutStats()
    cache: dict[tuple, tuple[int, int]] = {}
    last_swap: tuple[int, int] | None = None

    while len(state.executed_gates) < len(dag.gates):
        best = choose_rollout_improved_action(
            state=state,
            dag=dag,
            hardware=hardware,
            critical_weights=critical_weights,
            last_swap=last_swap,
            base_config=base_config,
            top_k=top_k,
            cache=cache,
            stats=stats,
        )
        state = best.state_after
        last_swap = best.edge
        stats.route_decisions += 1
        state.assert_valid(dag, hardware)

        if stats.route_decisions > structural_policy.MAX_ROUTE_STEPS:
            raise RuntimeError("rollout 改进策略超过 MAX_ROUTE_STEPS")

    return state, stats


# ============================================================
# 3. 实例、映射池和汇总
# ============================================================


def run_attempt(
    case: DevCase,
    mapping: MappingCandidate,
    base_config: structural_policy.StructuralConfig,
    top_k: int,
) -> AttemptRow:
    dag, hardware, _ = case.test_case.build()

    try:
        start = time.perf_counter()
        base_state = structural_policy.run_router(
            dag=dag,
            hardware=hardware,
            initial_mapping=list(mapping.mapping),
            config=base_config,
        )
        base_runtime_ms = (time.perf_counter() - start) * 1000.0
        base_state.assert_valid(dag, hardware)

        start = time.perf_counter()
        rollout_state, stats = run_rollout_router(
            dag=dag,
            hardware=hardware,
            initial_mapping=list(mapping.mapping),
            base_config=base_config,
            top_k=top_k,
        )
        rollout_runtime_ms = (time.perf_counter() - start) * 1000.0
        rollout_state.assert_valid(dag, hardware)

        if len(rollout_state.executed_gates) != len(dag.gates):
            raise AssertionError("rollout 路由结束后仍有未执行逻辑门")

        base_swap = count_swaps(base_state)
        rollout_swap = count_swaps(rollout_state)
        if rollout_swap > base_swap:
            raise AssertionError(
                f"策略改进违反不退化检查：{rollout_swap} > {base_swap}"
            )

        return AttemptRow(
            instance=case.name,
            topology=case.topology,
            circuit_mode=case.circuit_mode,
            num_qubits=case.num_qubits,
            cx_count=case.two_qubit_gate_count,
            mapping=mapping.name,
            base_policy=base_config.name,
            top_k=top_k,
            base_swap=base_swap,
            base_depth=base_state.current_depth(),
            rollout_swap=rollout_swap,
            rollout_depth=rollout_state.current_depth(),
            swap_improvement=base_swap - rollout_swap,
            route_decisions=stats.route_decisions,
            changed_decisions=stats.changed_decisions,
            rollout_candidates=stats.rollout_candidates,
            cache_hits=stats.cache_hits,
            base_runtime_ms=base_runtime_ms,
            rollout_runtime_ms=rollout_runtime_ms,
            valid=True,
            error="",
        )
    except Exception as error:
        return AttemptRow(
            instance=case.name,
            topology=case.topology,
            circuit_mode=case.circuit_mode,
            num_qubits=case.num_qubits,
            cx_count=case.two_qubit_gate_count,
            mapping=mapping.name,
            base_policy=base_config.name,
            top_k=top_k,
            base_swap=None,
            base_depth=None,
            rollout_swap=None,
            rollout_depth=None,
            swap_improvement=None,
            route_decisions=0,
            changed_decisions=0,
            rollout_candidates=0,
            cache_hits=0,
            base_runtime_ms=0.0,
            rollout_runtime_ms=0.0,
            valid=False,
            error=f"{type(error).__name__}: {error}",
        )


def base_key(row: AttemptRow) -> tuple:
    if not row.valid:
        return (10**9, 10**9, row.mapping)
    return (int(row.base_swap), int(row.base_depth), row.mapping)


def rollout_key(row: AttemptRow) -> tuple:
    if not row.valid:
        return (10**9, 10**9, row.mapping)
    return (int(row.rollout_swap), int(row.rollout_depth), row.mapping)


def build_case_row(
    case: DevCase,
    mappings: list[MappingCandidate],
    attempts: list[AttemptRow],
    base_config: structural_policy.StructuralConfig,
    top_k: int,
) -> CaseRow:
    valid = [row for row in attempts if row.valid]
    if not valid:
        raise RuntimeError(
            f"{case.name} 所有映射均失败："
            + " | ".join(row.error for row in attempts)
        )

    base = min(valid, key=base_key)
    rollout = min(valid, key=rollout_key)
    return CaseRow(
        instance=case.name,
        topology=case.topology,
        circuit_mode=case.circuit_mode,
        num_qubits=case.num_qubits,
        cx_count=case.two_qubit_gate_count,
        mapping_count=len(mappings),
        base_policy=base_config.name,
        top_k=top_k,
        base_mapping=base.mapping,
        base_swap=int(base.base_swap),
        base_depth=int(base.base_depth),
        rollout_mapping=rollout.mapping,
        rollout_swap=int(rollout.rollout_swap),
        rollout_depth=int(rollout.rollout_depth),
        swap_improvement=int(base.base_swap) - int(rollout.rollout_swap),
        changed_decisions=rollout.changed_decisions,
        rollout_candidates=rollout.rollout_candidates,
        total_runtime_ms=sum(
            row.base_runtime_ms + row.rollout_runtime_ms for row in attempts
        ),
    )


def write_csv(path: Path, rows: list, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def summarize(rows: list[CaseRow]) -> dict:
    if not rows:
        return {}

    by_topology: dict[str, list[CaseRow]] = defaultdict(list)
    by_mode: dict[str, list[CaseRow]] = defaultdict(list)
    for row in rows:
        by_topology[row.topology].append(row)
        by_mode[row.circuit_mode].append(row)

    def group_summary(group: list[CaseRow]) -> dict:
        base_swap = sum(row.base_swap for row in group)
        rollout_swap = sum(row.rollout_swap for row in group)
        return {
            "case_count": len(group),
            "base_swap": base_swap,
            "rollout_swap": rollout_swap,
            "swap_improvement": base_swap - rollout_swap,
            "relative_improvement": (
                (base_swap - rollout_swap) / base_swap if base_swap else 0.0
            ),
            "improved_cases": sum(
                row.rollout_swap < row.base_swap for row in group
            ),
        }

    result = group_summary(rows)
    result.update({
        "base_policy": rows[0].base_policy,
        "top_k": rows[0].top_k,
        "changed_decisions": sum(row.changed_decisions for row in rows),
        "rollout_candidates": sum(row.rollout_candidates for row in rows),
        "total_runtime_ms": sum(row.total_runtime_ms for row in rows),
        "mean_runtime_ms": statistics.mean(
            row.total_runtime_ms for row in rows
        ),
        "by_topology": {
            key: group_summary(group)
            for key, group in sorted(by_topology.items())
        },
        "by_mode": {
            key: group_summary(group)
            for key, group in sorted(by_mode.items())
        },
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="historical development：完整 rollout 驱动的策略改进"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmarks_v3/dev.json"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-indices", type=str, default=None)
    parser.add_argument(
        "--mapping",
        choices=("identity", "v1_best", "pool"),
        default="pool",
    )
    parser.add_argument(
        "--base-policy",
        choices=tuple(sorted(structural_policy.CONFIG_BY_NAME)),
        default="ordered_potential",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="每步完整 rollout 的启发式前 K 个候选；0 表示全部候选",
    )
    parser.add_argument(
        "--attempt-output",
        type=Path,
        default=Path("results/rollout_policy_rollout_attempts.csv"),
    )
    parser.add_argument(
        "--case-output",
        type=Path,
        default=Path("results/rollout_policy_rollout_cases.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/rollout_policy_rollout_summary.json"),
    )
    args = parser.parse_args()

    if args.top_k < 0:
        raise ValueError("--top-k 必须是非负整数")

    # 与 historical development--24 冻结相同的公共参数。
    structural_policy.one_step_policy.THETA = 5
    structural_policy.one_step_policy.W_FUTURE = 0.5
    structural_policy.one_step_policy.GAMMA = 0.5
    structural_policy.one_step_policy.ALPHA = 0.5
    structural_policy.one_step_policy.LAMBDA_DEPTH = 0.0

    cases = mapping_portfolio.choose_cases(
        load_cases(args.input, limit=None),
        mapping_portfolio.parse_indices(args.case_indices),
        args.limit,
    )
    base_config = structural_policy.CONFIG_BY_NAME[args.base_policy]

    print("===== historical development：完整 rollout 策略改进 =====")
    print(f"输入：{args.input.resolve()}")
    print(f"实例数：{len(cases)}")
    print(f"基线策略：{base_config.name}")
    print(f"映射：{args.mapping}")
    print(f"每步 rollout 候选：{'all' if args.top_k == 0 else args.top_k}")

    all_attempts: list[AttemptRow] = []
    case_rows: list[CaseRow] = []

    for index, case in enumerate(cases, start=1):
        mappings = mapping_portfolio.make_mappings(case, args.mapping)
        attempts = [
            run_attempt(case, mapping, base_config, args.top_k)
            for mapping in mappings
        ]
        row = build_case_row(
            case, mappings, attempts, base_config, args.top_k
        )
        all_attempts.extend(attempts)
        case_rows.append(row)

        print(
            f"[{index}/{len(cases)}] {case.name}: "
            f"base={row.base_swap}/{row.base_depth} ({row.base_mapping}), "
            f"rollout={row.rollout_swap}/{row.rollout_depth} "
            f"({row.rollout_mapping}), "
            f"DeltaSWAP=-{row.swap_improvement}, "
            f"changed={row.changed_decisions}, "
            f"candidate-rollouts={row.rollout_candidates}, "
            f"{row.total_runtime_ms:.1f} ms"
        )

        write_csv(
            args.attempt_output,
            all_attempts,
            AttemptRow.__annotations__.keys(),
        )
        write_csv(
            args.case_output,
            case_rows,
            CaseRow.__annotations__.keys(),
        )
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summarize(case_rows), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = summarize(case_rows)
    # 再做一次最终原子式完整写出，确保中途被查看的增量文件不会成为
    # 批处理结束后的最终文件。
    write_csv(
        args.attempt_output,
        all_attempts,
        AttemptRow.__annotations__.keys(),
    )
    write_csv(
        args.case_output,
        case_rows,
        CaseRow.__annotations__.keys(),
    )
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print("===== historical development 汇总 =====")
    print(f"基线 total SWAP：{summary['base_swap']}")
    print(f"rollout total SWAP：{summary['rollout_swap']}")
    print(f"减少 SWAP：{summary['swap_improvement']}")
    print(f"相对改善：{summary['relative_improvement']:.2%}")
    print(
        f"严格改善实例：{summary['improved_cases']}/"
        f"{summary['case_count']}"
    )
    print(f"实际改选决策：{summary['changed_decisions']}")
    print(f"完整候选 rollout 次数：{summary['rollout_candidates']}")
    print(f"总运行时间：{summary['total_runtime_ms']:.1f} ms")
    print()
    print(f"尝试明细：{args.attempt_output.resolve()}")
    print(f"逐实例结果：{args.case_output.resolve()}")
    print(f"汇总 JSON：{args.summary_output.resolve()}")


if __name__ == "__main__":
    main()
