from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import one_step_policy as one_step_policy
import score_support as score_support

from routing_model import CircuitDAG, HardwareGraph, RoutingState, drain
from layout_pool import MappingCandidate, generate_base_mapping_pool
from case_io import DevCase, load_cases


# ============================================================
# 1. 冻结 independent rollout-v1
# ============================================================

FROZEN_CONFIG = score_support.ScoreConfig(
    name="d0_b025",
    theta=5,
    future_weight=0.5,
    gamma=0.5,
    alpha=0.5,
    beta_unlock=0.25,
    lambda_depth=0.0,
)

INF_COST = 10**9


@dataclass
class TraceState:
    decision: int
    state: RoutingState
    last_swap: tuple[int, int] | None
    chosen_edge: tuple[int, int]


@dataclass
class CandidateRow:
    instance: str
    topology: str
    circuit_mode: str
    num_qubits: int
    length_factor: int
    cx_count: int
    mapping_method: str
    decision: int
    front_size: int
    remaining_two_qubit: int
    edge: str
    in_current: bool
    in_ordered_frontier: bool
    in_multigeodesic: bool
    selected_by_h: bool
    v1_score: float
    delta_phi: float
    unlock_gain: float
    depth_increment: int
    undo_penalty: int
    rollout_swap: int
    rollout_depth_increment: int
    rollout_failed: bool
    rollout_error: str


@dataclass
class StateRow:
    instance: str
    topology: str
    circuit_mode: str
    num_qubits: int
    length_factor: int
    cx_count: int
    mapping_method: str
    decision: int
    total_trace_decisions: int
    front_size: int
    remaining_two_qubit: int
    current_size: int
    ordered_frontier_size: int
    multigeodesic_size: int
    all_edges_size: int
    h_edge: str
    best_current_edge: str
    best_ordered_frontier_edge: str
    best_multigeodesic_edge: str
    best_all_edge: str
    q_h_swap: int
    q_current_swap: int
    q_ordered_frontier_swap: int
    q_multigeodesic_swap: int
    q_all_swap: int
    ranking_regret: int
    coverage_regret: int
    total_regret: int
    ordered_frontier_coverage_regret: int
    multigeodesic_coverage_regret: int
    current_contains_global_oracle: bool
    ordered_frontier_contains_global_oracle: bool
    multigeodesic_contains_global_oracle: bool
    diagnostic_ms: float


# ============================================================
# 2. 基础工具
# ============================================================

def normalize_edge(edge: tuple[int, int]) -> tuple[int, int]:
    return tuple(sorted(edge))


def edge_text(edge: tuple[int, int]) -> str:
    return f"{edge[0]}-{edge[1]}"


def all_hardware_edges(hardware: HardwareGraph) -> list[tuple[int, int]]:
    return sorted({normalize_edge(edge) for edge in hardware.edges})


def count_swaps(state: RoutingState) -> int:
    return sum(operation[0] == "swap" for operation in state.physical_operations)


def remaining_two_qubit_count(state: RoutingState, dag: CircuitDAG) -> int:
    return sum(
        gate.is_two_qubit and gate.gate_id not in state.executed_gates
        for gate in dag.gates
    )


def evaluation_key(item: one_step_policy.CandidateEvaluation) -> tuple:
    return (
        item.score,
        -item.unlock_gain,
        item.delta_phi,
        item.depth_increment,
        item.edge,
    )


def rollout_key(row: CandidateRow) -> tuple[int, int, str]:
    return (
        row.rollout_swap,
        row.rollout_depth_increment,
        row.edge,
    )


def evenly_spaced_indices(total: int, requested: int | None) -> list[int]:
    if total <= 0:
        return []
    if requested is None or requested >= total:
        return list(range(total))
    if requested <= 1:
        return [0]

    raw = [round(i * (total - 1) / (requested - 1)) for i in range(requested)]
    return sorted(set(raw))


def choose_cases(
    cases: list[DevCase],
    case_indices: list[int] | None,
    limit: int | None,
) -> list[DevCase]:
    if case_indices:
        selected = []
        for index in case_indices:
            if not 0 <= index < len(cases):
                raise IndexError(f"case index 越界：{index}")
            selected.append(cases[index])
        return selected

    if limit is not None:
        return cases[:limit]
    return cases


def parse_case_indices(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(value.strip()) for value in text.split(",") if value.strip()]


# ============================================================
# 3. 有序长前瞻与候选集
# ============================================================

def dynamic_lookahead_horizon(
    state: RoutingState,
    dag: CircuitDAG,
    hardware: HardwareGraph,
) -> int:
    n = hardware.num_qubits
    edge_count = len(all_hardware_edges(hardware))
    average_degree = 2.0 * edge_count / n if n else 0.0

    if average_degree >= 3.5:
        base = 250
    elif average_degree >= 2.5:
        base = 300
    else:
        base = 400

    remaining = remaining_two_qubit_count(state, dag)
    return min(base + remaining // 20, 1000)


def build_extended_gate_order(
    state: RoutingState,
    dag: CircuitDAG,
    front: list[int],
    max_size: int,
) -> list[int]:
    """
    使用确定性的 DAG BFS 构造前瞻顺序。

    当前前沿门作为 BFS 根，但不加入 extended set；
    后续新暴露的双比特门依次加入结果。
    """
    if max_size <= 0:
        return []

    remaining = list(state.remaining_predecessors)
    queue: deque[int] = deque(sorted(front))
    output: list[int] = []

    while queue and len(output) < max_size:
        gate_id = queue.popleft()

        for successor in sorted(dag.successors[gate_id]):
            remaining[successor] -= 1

            if remaining[successor] != 0:
                continue

            successor_gate = dag.gates[successor]

            if successor_gate.is_two_qubit:
                output.append(successor)
                if len(output) >= max_size:
                    break

            queue.append(successor)

    return output


def relevant_physical_pairs(
    state: RoutingState,
    dag: CircuitDAG,
    gate_ids: list[int],
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []

    for gate_id in gate_ids:
        logical_a, logical_b = dag.gates[gate_id].qubits
        pairs.append(
            (
                state.logical_to_physical[logical_a],
                state.logical_to_physical[logical_b],
            )
        )

    return pairs


def ordered_frontier_candidate_set(
    state: RoutingState,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    front: list[int],
    extended: list[int],
) -> list[tuple[int, int]]:
    pairs = relevant_physical_pairs(state, dag, front + extended)
    relevant_nodes = {node for pair in pairs for node in pair}
    candidates: set[tuple[int, int]] = set()

    for edge in all_hardware_edges(hardware):
        if edge[0] in relevant_nodes or edge[1] in relevant_nodes:
            candidates.add(edge)

    for physical_a, physical_b in pairs:
        path = hardware.shortest_path(physical_a, physical_b)
        for u, v in zip(path, path[1:]):
            candidates.add(normalize_edge((u, v)))

    return sorted(candidates)


def distance_matrix(hardware: HardwareGraph) -> list[list[int]]:
    return [
        [
            len(hardware.shortest_path(u, v)) - 1
            for v in range(hardware.num_qubits)
        ]
        for u in range(hardware.num_qubits)
    ]


def multigeodesic_candidate_set(
    state: RoutingState,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    front: list[int],
    extended: list[int],
    distances: list[list[int]],
) -> list[tuple[int, int]]:
    """
    有序前沿候选再加：对每个相关门，加入属于任意最短路的硬件边。
    """
    candidates = set(
        ordered_frontier_candidate_set(
            state=state,
            dag=dag,
            hardware=hardware,
            front=front,
            extended=extended,
        )
    )

    pairs = relevant_physical_pairs(state, dag, front + extended)

    for source, target in pairs:
        shortest = distances[source][target]

        for u, v in all_hardware_edges(hardware):
            forward = distances[source][u] + 1 + distances[v][target]
            backward = distances[source][v] + 1 + distances[u][target]

            if forward == shortest or backward == shortest:
                candidates.add((u, v))

    return sorted(candidates)


# ============================================================
# 4. 冻结策略：追踪与 rollout
# ============================================================

def choose_v1_evaluation(
    state: RoutingState,
    last_swap: tuple[int, int] | None,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
) -> one_step_policy.CandidateEvaluation:
    layers = one_step_policy.build_future_layers(state, dag, one_step_policy.THETA)
    if not layers:
        raise RuntimeError("线路未完成，但未来层为空")

    candidates = one_step_policy.generate_swap_candidates(
        state,
        dag,
        hardware,
        layers,
        critical_weights,
    )
    if not candidates:
        raise RuntimeError("当前状态没有候选 SWAP")

    evaluations = [
        one_step_policy.evaluate_candidate(
            state,
            edge,
            layers,
            dag,
            hardware,
            critical_weights,
            last_swap,
        )
        for edge in candidates
    ]

    return min(evaluations, key=evaluation_key)


def trace_v1_route(
    dag: CircuitDAG,
    hardware: HardwareGraph,
    initial_mapping: list[int],
    critical_weights: list[float],
) -> tuple[list[TraceState], RoutingState]:
    state = one_step_policy.make_initial_state(dag, hardware, initial_mapping)
    drain(state, dag, hardware)

    trace: list[TraceState] = []
    last_swap: tuple[int, int] | None = None

    while len(state.executed_gates) < len(dag.gates):
        best = choose_v1_evaluation(
            state,
            last_swap,
            dag,
            hardware,
            critical_weights,
        )

        trace.append(
            TraceState(
                decision=len(trace) + 1,
                state=copy.deepcopy(state),
                last_swap=last_swap,
                chosen_edge=best.edge,
            )
        )

        state = best.state_after
        last_swap = best.edge

        if len(trace) > one_step_policy.MAX_ROUTE_STEPS:
            raise RuntimeError("冻结策略追踪超过 MAX_ROUTE_STEPS")

    return trace, state


def rollout_fingerprint(
    state: RoutingState,
    last_swap: tuple[int, int] | None,
) -> tuple:
    return (
        tuple(state.logical_to_physical),
        tuple(state.remaining_predecessors),
        tuple(sorted(state.executed_gates)),
        tuple(state.physical_depth),
        last_swap,
    )


def complete_v1_rollout(
    state_after_action: RoutingState,
    last_swap: tuple[int, int],
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
    cache: dict[tuple, tuple[int, int]],
) -> tuple[int, int]:
    """返回候选动作之后仍需的 SWAP 数与最终深度增量。"""
    start_depth = state_after_action.current_depth()
    key = rollout_fingerprint(state_after_action, last_swap)

    if key in cache:
        return cache[key]

    state = copy.deepcopy(state_after_action)
    swaps = 0
    previous = last_swap

    while len(state.executed_gates) < len(dag.gates):
        best = choose_v1_evaluation(
            state,
            previous,
            dag,
            hardware,
            critical_weights,
        )
        state = best.state_after
        previous = best.edge
        swaps += 1

        if swaps > one_step_policy.MAX_ROUTE_STEPS:
            raise RuntimeError("rollout 超过 MAX_ROUTE_STEPS")

    result = (swaps, state.current_depth() - start_depth)
    cache[key] = result
    return result


# ============================================================
# 5. 初始映射选择
# ============================================================

def select_mapping(
    case: DevCase,
    requested_method: str,
) -> tuple[MappingCandidate, list[tuple[str, int, int, float]]]:
    candidates = generate_base_mapping_pool(case.test_case)

    if requested_method != "best_pool":
        for candidate in candidates:
            if candidate.name == requested_method:
                return candidate, []
        raise KeyError(f"找不到映射方法：{requested_method}")

    rows: list[tuple[str, int, int, float]] = []

    for candidate in candidates:
        swap_count, depth, runtime_ms = score_support.run_candidate(
            case=case.test_case,
            candidate=candidate,
        )
        rows.append((candidate.name, swap_count, depth, runtime_ms))

    best_name = min(rows, key=lambda row: (row[1], row[2], row[3], row[0]))[0]
    best = next(candidate for candidate in candidates if candidate.name == best_name)
    return best, rows


# ============================================================
# 6. 单状态诊断
# ============================================================

def diagnose_state(
    case: DevCase,
    metadata: dict,
    mapping_method: str,
    trace_state: TraceState,
    total_trace_decisions: int,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
    distances: list[list[int]],
    rollout_cache: dict[tuple, tuple[int, int]],
) -> tuple[StateRow, list[CandidateRow]]:
    start = time.perf_counter()
    state = trace_state.state
    layers = one_step_policy.build_future_layers(state, dag, one_step_policy.THETA)
    front = layers[0]

    current = set(
        one_step_policy.generate_swap_candidates(
            state,
            dag,
            hardware,
            layers,
            critical_weights,
        )
    )

    horizon = dynamic_lookahead_horizon(state, dag, hardware)
    extended = build_extended_gate_order(state, dag, front, horizon)
    ordered_frontier = set(
        ordered_frontier_candidate_set(state, dag, hardware, front, extended)
    )
    multigeodesic = set(
        multigeodesic_candidate_set(
            state,
            dag,
            hardware,
            front,
            extended,
            distances,
        )
    )
    all_edges = set(all_hardware_edges(hardware))

    if not current <= all_edges:
        raise AssertionError("当前候选集中出现非法硬件边")
    if not ordered_frontier <= all_edges:
        raise AssertionError("有序前沿候选集中出现非法硬件边")
    if not multigeodesic <= all_edges:
        raise AssertionError("多最短路候选集中出现非法硬件边")

    evaluations = {
        edge: one_step_policy.evaluate_candidate(
            state,
            edge,
            layers,
            dag,
            hardware,
            critical_weights,
            trace_state.last_swap,
        )
        for edge in sorted(all_edges)
    }
    h_edge = min((evaluations[edge] for edge in current), key=evaluation_key).edge

    rows: list[CandidateRow] = []

    for edge in sorted(all_edges):
        evaluation = evaluations[edge]
        failed = False
        error = ""

        try:
            remaining_swaps, remaining_depth = complete_v1_rollout(
                evaluation.state_after,
                edge,
                dag,
                hardware,
                critical_weights,
                rollout_cache,
            )
            q_swap = 1 + remaining_swaps
            q_depth = evaluation.depth_increment + remaining_depth
        except Exception as exc:
            q_swap = INF_COST
            q_depth = INF_COST
            failed = True
            error = f"{type(exc).__name__}: {exc}"

        rows.append(
            CandidateRow(
                instance=case.name,
                topology=case.topology,
                circuit_mode=case.circuit_mode,
                num_qubits=case.num_qubits,
                length_factor=int(metadata["length_factor"]),
                cx_count=case.two_qubit_gate_count,
                mapping_method=mapping_method,
                decision=trace_state.decision,
                front_size=len(front),
                remaining_two_qubit=remaining_two_qubit_count(state, dag),
                edge=edge_text(edge),
                in_current=edge in current,
                in_ordered_frontier=edge in ordered_frontier,
                in_multigeodesic=edge in multigeodesic,
                selected_by_h=edge == h_edge,
                v1_score=evaluation.score,
                delta_phi=evaluation.delta_phi,
                unlock_gain=evaluation.unlock_gain,
                depth_increment=evaluation.depth_increment,
                undo_penalty=evaluation.undo_penalty,
                rollout_swap=q_swap,
                rollout_depth_increment=q_depth,
                rollout_failed=failed,
                rollout_error=error,
            )
        )

    def best_in(candidate_set: set[tuple[int, int]]) -> CandidateRow:
        edge_names = {edge_text(edge) for edge in candidate_set}
        eligible = [row for row in rows if row.edge in edge_names]
        if not eligible:
            raise RuntimeError("候选集为空")
        return min(eligible, key=rollout_key)

    h_row = next(row for row in rows if row.edge == edge_text(h_edge))
    best_current = best_in(current)
    best_ordered_frontier = best_in(ordered_frontier)
    best_multigeodesic = best_in(multigeodesic)
    best_all = best_in(all_edges)

    ranking_regret = h_row.rollout_swap - best_current.rollout_swap
    coverage_regret = best_current.rollout_swap - best_all.rollout_swap
    total_regret = h_row.rollout_swap - best_all.rollout_swap

    if ranking_regret + coverage_regret != total_regret:
        raise AssertionError("regret 分解不成立")

    global_oracle_edges = {
        row.edge
        for row in rows
        if (
            row.rollout_swap,
            row.rollout_depth_increment,
        )
        == (
            best_all.rollout_swap,
            best_all.rollout_depth_increment,
        )
    }

    state_row = StateRow(
        instance=case.name,
        topology=case.topology,
        circuit_mode=case.circuit_mode,
        num_qubits=case.num_qubits,
        length_factor=int(metadata["length_factor"]),
        cx_count=case.two_qubit_gate_count,
        mapping_method=mapping_method,
        decision=trace_state.decision,
        total_trace_decisions=total_trace_decisions,
        front_size=len(front),
        remaining_two_qubit=remaining_two_qubit_count(state, dag),
        current_size=len(current),
        ordered_frontier_size=len(ordered_frontier),
        multigeodesic_size=len(multigeodesic),
        all_edges_size=len(all_edges),
        h_edge=h_row.edge,
        best_current_edge=best_current.edge,
        best_ordered_frontier_edge=best_ordered_frontier.edge,
        best_multigeodesic_edge=best_multigeodesic.edge,
        best_all_edge=best_all.edge,
        q_h_swap=h_row.rollout_swap,
        q_current_swap=best_current.rollout_swap,
        q_ordered_frontier_swap=best_ordered_frontier.rollout_swap,
        q_multigeodesic_swap=best_multigeodesic.rollout_swap,
        q_all_swap=best_all.rollout_swap,
        ranking_regret=ranking_regret,
        coverage_regret=coverage_regret,
        total_regret=total_regret,
        ordered_frontier_coverage_regret=(
            best_ordered_frontier.rollout_swap - best_all.rollout_swap
        ),
        multigeodesic_coverage_regret=(
            best_multigeodesic.rollout_swap - best_all.rollout_swap
        ),
        current_contains_global_oracle=bool(
            global_oracle_edges
            & {edge_text(edge) for edge in current}
        ),
        ordered_frontier_contains_global_oracle=bool(
            global_oracle_edges
            & {edge_text(edge) for edge in ordered_frontier}
        ),
        multigeodesic_contains_global_oracle=bool(
            global_oracle_edges
            & {edge_text(edge) for edge in multigeodesic}
        ),
        diagnostic_ms=(time.perf_counter() - start) * 1000.0,
    )

    return state_row, rows


# ============================================================
# 7. 汇总与输出
# ============================================================

def write_dataclass_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"没有可写入的数据：{path}")

    fieldnames = list(asdict(rows[0]).keys())

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def summarize_rows(rows: list[StateRow]) -> dict:
    def summarize_group(group: list[StateRow]) -> dict:
        return {
            "states": len(group),
            "mean_ranking_regret": mean([row.ranking_regret for row in group]),
            "mean_coverage_regret": mean([row.coverage_regret for row in group]),
            "mean_total_regret": mean([row.total_regret for row in group]),
            "positive_ranking_rate": mean([
                float(row.ranking_regret > 0) for row in group
            ]),
            "positive_coverage_rate": mean([
                float(row.coverage_regret > 0) for row in group
            ]),
            "current_global_oracle_coverage": mean([
                float(row.current_contains_global_oracle) for row in group
            ]),
            "ordered_frontier_global_oracle_coverage": mean([
                float(row.ordered_frontier_contains_global_oracle) for row in group
            ]),
            "multigeodesic_global_oracle_coverage": mean([
                float(row.multigeodesic_contains_global_oracle) for row in group
            ]),
            "mean_current_candidate_ratio": mean([
                row.current_size / row.all_edges_size for row in group
            ]),
            "mean_ordered_frontier_candidate_ratio": mean([
                row.ordered_frontier_size / row.all_edges_size for row in group
            ]),
            "mean_multigeodesic_candidate_ratio": mean([
                row.multigeodesic_size / row.all_edges_size for row in group
            ]),
            "total_diagnostic_ms": sum(row.diagnostic_ms for row in group),
        }

    summary = {"overall": summarize_group(rows), "groups": {}}

    group_fields = ("topology", "circuit_mode", "num_qubits")
    for field in group_fields:
        grouped: dict[str, list[StateRow]] = defaultdict(list)
        for row in rows:
            grouped[str(getattr(row, field))].append(row)
        summary["groups"][field] = {
            key: summarize_group(group)
            for key, group in sorted(grouped.items())
        }

    return summary


def print_case_result(case: DevCase, mapping: MappingCandidate, trace_len: int) -> None:
    print(
        f"{case.name}: topology={case.topology}, mode={case.circuit_mode}, "
        f"n={case.num_qubits}, CX={case.two_qubit_gate_count}, "
        f"mapping={mapping.name}, trace SWAP={trace_len}"
    )


def print_summary(summary: dict) -> None:
    overall = summary["overall"]
    print()
    print("===== historical development 汇总 =====")
    print(f"诊断状态数：{overall['states']}")
    print(f"mean ranking regret：{overall['mean_ranking_regret']:.4f}")
    print(f"mean coverage regret：{overall['mean_coverage_regret']:.4f}")
    print(f"mean total regret：{overall['mean_total_regret']:.4f}")
    print(
        "当前 / 有序前沿 / 多最短路 oracle 覆盖率："
        f"{overall['current_global_oracle_coverage']:.3f} / "
        f"{overall['ordered_frontier_global_oracle_coverage']:.3f} / "
        f"{overall['multigeodesic_global_oracle_coverage']:.3f}"
    )


# ============================================================
# 8. 主程序
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="historical development：候选覆盖与 rollout regret 诊断"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmarks_v3/dev.json"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--case-indices",
        type=str,
        default=None,
        help="逗号分隔的 0-based case index，例如 5,7,14,16,27,28",
    )
    parser.add_argument(
        "--max-states-per-case",
        type=int,
        default=8,
        help="沿完整 v1 轨迹均匀抽取的状态数；0 表示全部状态",
    )
    parser.add_argument(
        "--mapping",
        choices=("best_pool", "identity", "all", "early", "late"),
        default="best_pool",
    )
    parser.add_argument(
        "--state-output",
        type=Path,
        default=Path("results/ordered_candidates_state_regret.csv"),
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=Path("results/ordered_candidates_candidate_regret.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/ordered_candidates_summary.json"),
    )
    args = parser.parse_args()

    score_support.apply_config(FROZEN_CONFIG)

    all_cases = load_cases(args.input, limit=None)
    cases = choose_cases(
        all_cases,
        parse_case_indices(args.case_indices),
        args.limit,
    )
    raw_payload = json.loads(args.input.read_text(encoding="utf-8"))
    metadata_by_name = {
        str(raw["name"]): raw for raw in raw_payload["cases"]
    }

    state_rows: list[StateRow] = []
    candidate_rows: list[CandidateRow] = []

    print("===== historical development：候选覆盖与 rollout regret 诊断 =====")
    print(f"输入：{args.input.resolve()}")
    print(f"实例数：{len(cases)}")
    print(f"冻结配置：{FROZEN_CONFIG.name}")
    print()

    for case_index, case in enumerate(cases, start=1):
        print(f"[{case_index}/{len(cases)}] {case.name}")
        dag, hardware, _ = case.test_case.build()
        mapping, mapping_rows = select_mapping(case, args.mapping)
        critical_weights = one_step_policy.compute_critical_weights(dag, one_step_policy.ALPHA)
        distances = distance_matrix(hardware)

        trace, final_state = trace_v1_route(
            dag,
            hardware,
            list(mapping.mapping),
            critical_weights,
        )
        final_state.assert_valid(dag, hardware)

        if count_swaps(final_state) != len(trace):
            raise AssertionError("追踪决策数与最终 SWAP 数不一致")

        print_case_result(case, mapping, len(trace))
        if mapping_rows:
            print(
                "  mapping pool："
                + ", ".join(
                    f"{name}={swap} SWAP"
                    for name, swap, _, _ in mapping_rows
                )
            )

        requested = args.max_states_per_case
        sample_count = None if requested == 0 else requested
        selected_indices = evenly_spaced_indices(len(trace), sample_count)
        rollout_cache: dict[tuple, tuple[int, int]] = {}

        for sample_number, trace_index in enumerate(selected_indices, start=1):
            snapshot = trace[trace_index]
            print(
                f"  state {sample_number}/{len(selected_indices)}: "
                f"decision={snapshot.decision}/{len(trace)}",
                end="",
                flush=True,
            )

            state_row, rows = diagnose_state(
                case=case,
                metadata=metadata_by_name[case.name],
                mapping_method=mapping.name,
                trace_state=snapshot,
                total_trace_decisions=len(trace),
                dag=dag,
                hardware=hardware,
                critical_weights=critical_weights,
                distances=distances,
                rollout_cache=rollout_cache,
            )

            state_rows.append(state_row)
            candidate_rows.extend(rows)

            print(
                f", ranking={state_row.ranking_regret}, "
                f"coverage={state_row.coverage_regret}, "
                f"ordered-coverage={state_row.ordered_frontier_coverage_regret}, "
                f"{state_row.diagnostic_ms:.1f} ms"
            )

    write_dataclass_csv(args.state_output, state_rows)
    write_dataclass_csv(args.candidate_output, candidate_rows)
    summary = summarize_rows(state_rows)
    summary["config"] = asdict(FROZEN_CONFIG)
    summary["input"] = str(args.input)
    summary["case_count"] = len(cases)
    summary["state_output"] = str(args.state_output)
    summary["candidate_output"] = str(args.candidate_output)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_summary(summary)
    print()
    print(f"状态 CSV：{args.state_output.resolve()}")
    print(f"候选 CSV：{args.candidate_output.resolve()}")
    print(f"汇总 JSON：{args.summary_output.resolve()}")


if __name__ == "__main__":
    main()
