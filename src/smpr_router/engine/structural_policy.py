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

import one_step_policy as one_step_policy
import ordered_candidates as ordered_candidates

from routing_model import CircuitDAG, HardwareGraph, RoutingState, drain
from layout_pool import MappingCandidate, generate_base_mapping_pool
from case_io import DevCase, load_cases


# ============================================================
# 1. 实验配置
# ============================================================

@dataclass(frozen=True)
class StructuralConfig:
    name: str
    beta_unlock: float
    undo_weight: float
    potential_mode: str = "v1"
    candidate_mode: str = "current"
    search_mode: str = "one_step"
    beam_width_1: int = 8
    beam_width_2: int = 8


CONFIGS: tuple[StructuralConfig, ...] = (
    # historical development 冻结基线：用于确认本脚本没有改变旧结果。
    StructuralConfig(
        name="v1",
        beta_unlock=0.25,
        undo_weight=2.5,
    ),
    # 新文档中的更强真实进展奖励。
    StructuralConfig(
        name="beta2",
        beta_unlock=2.0,
        undo_weight=2.5,
    ),
    # 检验固定 undo=2.5 是否阻止必要的“送回”动作。
    StructuralConfig(
        name="beta2_no_undo",
        beta_unlock=2.0,
        undo_weight=0.0,
    ),
    # 前沿层改用关键性加权总和，避免宽前沿被平均值稀释。
    StructuralConfig(
        name="front_sum",
        beta_unlock=2.0,
        undo_weight=0.0,
        potential_mode="front_sum",
    ),
    # 只扩候选集，隔离 historical development 中 2.3% 的覆盖缺口。
    StructuralConfig(
        name="front_sum_st_candidates",
        beta_unlock=2.0,
        undo_weight=0.0,
        potential_mode="front_sum",
        candidate_mode="ordered_frontier",
    ),
    # 有序长前瞻：前沿距离总和 + 按 DAG 顺序 0.5^i 衰减的扩展项。
    StructuralConfig(
        name="ordered_potential",
        beta_unlock=2.0,
        undo_weight=0.0,
        potential_mode="ordered_potential",
        candidate_mode="ordered_frontier",
    ),
    # 两步只比较终态，不累计 h(a1)+eta*h(a2)，避免重复计算中间代理量。
    StructuralConfig(
        name="terminal2",
        beta_unlock=2.0,
        undo_weight=0.0,
        potential_mode="front_sum",
        candidate_mode="ordered_frontier",
        search_mode="terminal_two_step",
    ),
)

CONFIG_BY_NAME = {config.name: config for config in CONFIGS}
DEFAULT_CONFIG_NAMES = (
    "v1",
    "beta2",
    "beta2_no_undo",
    "front_sum",
    "front_sum_st_candidates",
    "ordered_potential",
)
MAX_ROUTE_STEPS = 4000


# ============================================================
# 2. 预览、候选和势函数
# ============================================================

@dataclass(frozen=True)
class Preview:
    layers: tuple[tuple[int, ...], ...]
    extended: tuple[int, ...]


@dataclass
class EdgeEvaluation:
    edge: tuple[int, int]
    score: float
    delta_phi: float
    unlock_gain: float
    depth_increment: int
    undo_penalty: int
    executed_now: tuple[int, ...]
    state_after: RoutingState


def normalize_edge(edge: tuple[int, int]) -> tuple[int, int]:
    return tuple(sorted(edge))


def build_preview(
    state: RoutingState,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    config: StructuralConfig,
) -> Preview:
    layers = one_step_policy.build_future_layers(state, dag, one_step_policy.THETA)
    if not layers:
        raise RuntimeError("线路未完成，但未来层为空")

    extended: list[int] = []
    if (
        config.candidate_mode == "ordered_frontier"
        or config.potential_mode == "ordered_potential"
    ):
        horizon = ordered_candidates.dynamic_lookahead_horizon(state, dag, hardware)
        extended = ordered_candidates.build_extended_gate_order(
            state=state,
            dag=dag,
            front=layers[0],
            max_size=horizon,
        )

    return Preview(
        layers=tuple(tuple(layer) for layer in layers),
        extended=tuple(extended),
    )


def generate_candidates(
    state: RoutingState,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
    preview: Preview,
    config: StructuralConfig,
) -> list[tuple[int, int]]:
    layers = [list(layer) for layer in preview.layers]

    if config.candidate_mode == "current":
        return one_step_policy.generate_swap_candidates(
            state=state,
            dag=dag,
            hardware=hardware,
            layers=layers,
            critical_weights=critical_weights,
        )

    if config.candidate_mode == "ordered_frontier":
        return ordered_candidates.ordered_frontier_candidate_set(
            state=state,
            dag=dag,
            hardware=hardware,
            front=layers[0],
            extended=list(preview.extended),
        )

    raise ValueError(f"未知 candidate_mode：{config.candidate_mode}")


def future_layer_weight(layer_index: int) -> float:
    if one_step_policy.THETA <= 1:
        return 0.0

    denominator = 1.0 - one_step_policy.GAMMA ** (one_step_policy.THETA - 1)
    return (
        one_step_policy.W_FUTURE
        * (1.0 - one_step_policy.GAMMA)
        * one_step_policy.GAMMA ** (layer_index - 1)
        / denominator
    )


def front_sum_potential(
    preview: Preview,
    mapping: list[int],
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
) -> float:
    if not preview.layers:
        return 0.0

    front = preview.layers[0]
    value = sum(
        critical_weights[gate_id]
        * one_step_policy.routing_distance(gate_id, mapping, dag, hardware)
        for gate_id in front
    )

    for layer_index, layer in enumerate(preview.layers[1:], start=1):
        value += future_layer_weight(layer_index) * one_step_policy.weighted_average_distance(
            list(layer),
            mapping,
            dag,
            hardware,
            critical_weights,
        )

    return value


def ordered_lookahead_potential(
    preview: Preview,
    mapping: list[int],
    dag: CircuitDAG,
    hardware: HardwareGraph,
) -> float:
    if not preview.layers:
        return 0.0

    value = sum(
        one_step_policy.routing_distance(gate_id, mapping, dag, hardware)
        for gate_id in preview.layers[0]
    )

    for index, gate_id in enumerate(preview.extended):
        value += (
            0.5 ** index
            * one_step_policy.routing_distance(gate_id, mapping, dag, hardware)
        )

    return value


def potential_value(
    preview: Preview,
    mapping: list[int],
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
    config: StructuralConfig,
) -> float:
    if config.potential_mode == "v1":
        return one_step_policy.potential(
            [list(layer) for layer in preview.layers],
            mapping,
            dag,
            hardware,
            critical_weights,
        )

    if config.potential_mode == "front_sum":
        return front_sum_potential(
            preview,
            mapping,
            dag,
            hardware,
            critical_weights,
        )

    if config.potential_mode == "ordered_potential":
        return ordered_lookahead_potential(
            preview,
            mapping,
            dag,
            hardware,
        )

    raise ValueError(f"未知 potential_mode：{config.potential_mode}")


# ============================================================
# 3. 候选完整状态转移评价
# ============================================================

def evaluate_edge(
    state: RoutingState,
    edge: tuple[int, int],
    preview: Preview,
    before_phi: float,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
    last_swap: tuple[int, int] | None,
    config: StructuralConfig,
) -> EdgeEvaluation:
    before_depth = state.current_depth()
    state_after = copy.deepcopy(state)

    normalized = normalize_edge(edge)
    state_after.apply_swap(normalized[0], normalized[1], hardware)
    executed_now = tuple(
        drain(
            state_after,
            dag,
            hardware,
            validate=False,
        )
    )

    after_phi = potential_value(
        preview,
        state_after.logical_to_physical,
        dag,
        hardware,
        critical_weights,
        config,
    )
    delta_phi = after_phi - before_phi
    unlock_gain = sum(
        critical_weights[gate_id]
        for gate_id in executed_now
        if dag.gates[gate_id].is_two_qubit
    )
    depth_increment = state_after.current_depth() - before_depth
    undo_penalty = int(
        last_swap is not None
        and normalized == normalize_edge(last_swap)
    )
    score = (
        delta_phi
        - config.beta_unlock * unlock_gain
        + config.undo_weight * undo_penalty
    )

    return EdgeEvaluation(
        edge=normalized,
        score=score,
        delta_phi=delta_phi,
        unlock_gain=unlock_gain,
        depth_increment=depth_increment,
        undo_penalty=undo_penalty,
        executed_now=executed_now,
        state_after=state_after,
    )


def evaluation_key(item: EdgeEvaluation) -> tuple:
    return (
        item.score,
        -item.unlock_gain,
        item.delta_phi,
        item.depth_increment,
        item.edge,
    )


def evaluate_all_edges(
    state: RoutingState,
    preview: Preview,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
    last_swap: tuple[int, int] | None,
    config: StructuralConfig,
) -> list[EdgeEvaluation]:
    candidates = generate_candidates(
        state,
        dag,
        hardware,
        critical_weights,
        preview,
        config,
    )
    if not candidates:
        raise RuntimeError("当前状态没有候选 SWAP")

    before_phi = potential_value(
        preview,
        state.logical_to_physical,
        dag,
        hardware,
        critical_weights,
        config,
    )
    rows = [
        evaluate_edge(
            state=state,
            edge=edge,
            preview=preview,
            before_phi=before_phi,
            dag=dag,
            hardware=hardware,
            critical_weights=critical_weights,
            last_swap=last_swap,
            config=config,
        )
        for edge in candidates
    ]
    rows.sort(key=evaluation_key)
    return rows


def choose_one_step(
    state: RoutingState,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
    last_swap: tuple[int, int] | None,
    config: StructuralConfig,
) -> EdgeEvaluation:
    preview = build_preview(state, dag, hardware, config)
    return evaluate_all_edges(
        state,
        preview,
        dag,
        hardware,
        critical_weights,
        last_swap,
        config,
    )[0]


def choose_terminal_two_step(
    state: RoutingState,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
    last_swap: tuple[int, int] | None,
    config: StructuralConfig,
) -> EdgeEvaluation:
    """
    两步搜索只执行第一步。

    与旧 Beam 的关键区别：同一条分支只在第二步终态上计算一次势函数，
    不再把 h(a1) 和 h(a2|a1) 相加，从而避免重复奖励/惩罚中间距离。
    """
    root_preview = build_preview(state, dag, hardware, config)
    root_phi = potential_value(
        root_preview,
        state.logical_to_physical,
        dag,
        hardware,
        critical_weights,
        config,
    )
    root_depth = state.current_depth()

    first_rows = evaluate_all_edges(
        state,
        root_preview,
        dag,
        hardware,
        critical_weights,
        last_swap,
        config,
    )[: config.beam_width_1]

    best_progress_key: tuple | None = None
    best_progress_first: EdgeEvaluation | None = None

    for first in first_rows:
        first_executed = set(first.executed_now)

        if len(first.state_after.executed_gates) == len(dag.gates):
            terminal_delta = potential_value(
                root_preview,
                first.state_after.logical_to_physical,
                dag,
                hardware,
                critical_weights,
                config,
            ) - root_phi
            total_unlock = sum(
                critical_weights[gate_id]
                for gate_id in first_executed
                if dag.gates[gate_id].is_two_qubit
            )
            key = (
                terminal_delta - config.beta_unlock * total_unlock,
                -total_unlock,
                first.state_after.current_depth() - root_depth,
                first.edge,
                (-1, -1),
            )
            if total_unlock > 0 and (
                best_progress_key is None or key < best_progress_key
            ):
                best_progress_key = key
                best_progress_first = first
            continue

        second_preview = build_preview(first.state_after, dag, hardware, config)
        second_rows = evaluate_all_edges(
            first.state_after,
            second_preview,
            dag,
            hardware,
            critical_weights,
            first.edge,
            config,
        )[: config.beam_width_2]

        for second in second_rows:
            executed = first_executed | set(second.executed_now)
            total_unlock = sum(
                critical_weights[gate_id]
                for gate_id in executed
                if dag.gates[gate_id].is_two_qubit
            )
            terminal_delta = potential_value(
                root_preview,
                second.state_after.logical_to_physical,
                dag,
                hardware,
                critical_weights,
                config,
            ) - root_phi
            key = (
                terminal_delta - config.beta_unlock * total_unlock,
                -total_unlock,
                second.state_after.current_depth() - root_depth,
                first.edge,
                second.edge,
            )
            if total_unlock > 0 and (
                best_progress_key is None or key < best_progress_key
            ):
                best_progress_key = key
                best_progress_first = first

    # 两步内存在真实 DAG 进展时，只在这些分支中比较终态；否则退回
    # 最佳单步。这样不会把“第一步走出、第二步原路返回”的零进展环
    # 误当成优秀终态。
    if best_progress_first is not None:
        return best_progress_first
    return first_rows[0]


# ============================================================
# 4. 完整路由与映射池
# ============================================================

@dataclass
class RouteResult:
    mapping_name: str
    swap_count: int | None
    depth: int | None
    runtime_ms: float
    valid: bool
    error: str


def run_router(
    dag: CircuitDAG,
    hardware: HardwareGraph,
    initial_mapping: list[int],
    config: StructuralConfig,
) -> RoutingState:
    state = one_step_policy.make_initial_state(dag, hardware, initial_mapping)
    critical_weights = one_step_policy.compute_critical_weights(dag, one_step_policy.ALPHA)
    drain(state, dag, hardware)

    last_swap: tuple[int, int] | None = None
    route_steps = 0

    while len(state.executed_gates) < len(dag.gates):
        if config.search_mode == "one_step":
            best = choose_one_step(
                state,
                dag,
                hardware,
                critical_weights,
                last_swap,
                config,
            )
        elif config.search_mode == "terminal_two_step":
            best = choose_terminal_two_step(
                state,
                dag,
                hardware,
                critical_weights,
                last_swap,
                config,
            )
        else:
            raise ValueError(f"未知 search_mode：{config.search_mode}")

        state = best.state_after
        last_swap = best.edge
        route_steps += 1
        state.assert_valid(dag, hardware)

        if route_steps > MAX_ROUTE_STEPS:
            raise RuntimeError(f"超过最大路由步数 {MAX_ROUTE_STEPS}")

    return state


def run_mapping(
    case: DevCase,
    mapping: MappingCandidate,
    config: StructuralConfig,
) -> RouteResult:
    dag, hardware, _ = case.test_case.build()
    start = time.perf_counter()

    try:
        state = run_router(
            dag,
            hardware,
            list(mapping.mapping),
            config,
        )
        state.assert_valid(dag, hardware)
        if len(state.executed_gates) != len(dag.gates):
            raise AssertionError("路由结束后仍有未执行门")

        swap_count = sum(
            operation[0] == "swap"
            for operation in state.physical_operations
        )
        return RouteResult(
            mapping_name=mapping.name,
            swap_count=swap_count,
            depth=state.current_depth(),
            runtime_ms=(time.perf_counter() - start) * 1000.0,
            valid=True,
            error="",
        )
    except Exception as error:  # 实验脚本必须保留失败案例，而不是中断整批。
        return RouteResult(
            mapping_name=mapping.name,
            swap_count=None,
            depth=None,
            runtime_ms=(time.perf_counter() - start) * 1000.0,
            valid=False,
            error=f"{type(error).__name__}: {error}",
        )


@dataclass
class CaseRow:
    config: str
    instance: str
    topology: str
    circuit_mode: str
    num_qubits: int
    cx_count: int
    mapping_count: int
    best_mapping: str
    swap_count: int | None
    depth: int | None
    total_runtime_ms: float
    valid: bool
    error: str


def run_case(
    case: DevCase,
    mappings: list[MappingCandidate],
    config: StructuralConfig,
) -> CaseRow:
    results = [run_mapping(case, mapping, config) for mapping in mappings]
    valid = [row for row in results if row.valid]
    total_runtime = sum(row.runtime_ms for row in results)

    if not valid:
        return CaseRow(
            config=config.name,
            instance=case.name,
            topology=case.topology,
            circuit_mode=case.circuit_mode,
            num_qubits=case.num_qubits,
            cx_count=case.two_qubit_gate_count,
            mapping_count=len(mappings),
            best_mapping="",
            swap_count=None,
            depth=None,
            total_runtime_ms=total_runtime,
            valid=False,
            error=" | ".join(row.error for row in results),
        )

    best = min(
        valid,
        key=lambda row: (
            row.swap_count,
            row.depth,
            row.runtime_ms,
            row.mapping_name,
        ),
    )
    return CaseRow(
        config=config.name,
        instance=case.name,
        topology=case.topology,
        circuit_mode=case.circuit_mode,
        num_qubits=case.num_qubits,
        cx_count=case.two_qubit_gate_count,
        mapping_count=len(mappings),
        best_mapping=best.mapping_name,
        swap_count=best.swap_count,
        depth=best.depth,
        total_runtime_ms=total_runtime,
        valid=True,
        error="",
    )


# ============================================================
# 5. 汇总和输出
# ============================================================

def parse_names(text: str | None) -> list[StructuralConfig]:
    if not text:
        return [CONFIG_BY_NAME[name] for name in DEFAULT_CONFIG_NAMES]

    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in CONFIG_BY_NAME]
    if unknown:
        raise ValueError(
            f"未知配置：{unknown}；可选值：{sorted(CONFIG_BY_NAME)}"
        )
    return [CONFIG_BY_NAME[name] for name in names]


def parse_indices(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def choose_cases(
    cases: list[DevCase],
    indices: list[int] | None,
    limit: int | None,
) -> list[DevCase]:
    if indices is not None:
        selected: list[DevCase] = []
        for index in indices:
            if not 0 <= index < len(cases):
                raise IndexError(f"case index 越界：{index}")
            selected.append(cases[index])
        return selected
    return cases if limit is None else cases[:limit]


def write_csv(path: Path, rows: list[CaseRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CaseRow.__annotations__.keys())
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def summarize(rows: list[CaseRow]) -> list[dict]:
    grouped: dict[str, list[CaseRow]] = defaultdict(list)
    for row in rows:
        grouped[row.config].append(row)

    baseline = {
        row.instance: row
        for row in grouped.get("v1", [])
        if row.valid
    }
    summaries: list[dict] = []

    for config in CONFIGS:
        config_rows = grouped.get(config.name, [])
        if not config_rows:
            continue
        valid = [row for row in config_rows if row.valid]
        comparable = [row for row in valid if row.instance in baseline]
        wins = ties = losses = 0
        paired_swap_delta = 0

        for row in comparable:
            base = baseline[row.instance]
            left = (row.swap_count, row.depth)
            right = (base.swap_count, base.depth)
            if left < right:
                wins += 1
            elif left > right:
                losses += 1
            else:
                ties += 1
            paired_swap_delta += int(row.swap_count) - int(base.swap_count)

        summaries.append(
            {
                "config": config.name,
                "valid_cases": len(valid),
                "failed_cases": len(config_rows) - len(valid),
                "total_swap": sum(int(row.swap_count) for row in valid),
                "mean_swap": statistics.mean(
                    int(row.swap_count) for row in valid
                ) if valid else None,
                "total_depth": sum(int(row.depth) for row in valid),
                "total_runtime_ms": sum(row.total_runtime_ms for row in valid),
                "wins_vs_v1": wins,
                "ties_vs_v1": ties,
                "losses_vs_v1": losses,
                "paired_swap_delta_vs_v1": paired_swap_delta,
                "config_detail": asdict(config),
            }
        )

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="historical development：路线评分结构消融"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmarks_v3/dev.json"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-indices", type=str, default=None)
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="逗号分隔；默认运行全部配置",
    )
    parser.add_argument(
        "--mapping",
        choices=("v1_best", "pool", "identity"),
        default="v1_best",
        help=(
            "v1_best=先用 v1 从基础池选映射并冻结（结构消融推荐）；"
            "pool=每个配置都从完整池选最好结果（最终性能复核）；"
            "identity=只跑恒等映射（快速冒烟）"
        ),
    )
    parser.add_argument(
        "--case-output",
        type=Path,
        default=Path("results/structural_policy_score_structure_cases.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/structural_policy_score_structure_summary.json"),
    )
    args = parser.parse_args()

    # 固定所有未消融项，确保 v1 真的是 historical development 的 d0_b025。
    one_step_policy.THETA = 5
    one_step_policy.W_FUTURE = 0.5
    one_step_policy.GAMMA = 0.5
    one_step_policy.ALPHA = 0.5
    one_step_policy.LAMBDA_DEPTH = 0.0

    all_cases = load_cases(args.input, limit=None)
    cases = choose_cases(
        all_cases,
        parse_indices(args.case_indices),
        args.limit,
    )
    configs = parse_names(args.configs)

    mapping_pools: dict[str, list[MappingCandidate]] = {}
    for case in cases:
        if args.mapping == "identity":
            mapping_pools[case.name] = [
                MappingCandidate(
                    name="identity",
                    mapping=tuple(range(case.num_qubits)),
                    generation_ms=0.0,
                )
            ]
        elif args.mapping == "pool":
            mapping_pools[case.name] = generate_base_mapping_pool(case.test_case)
        else:
            pool = generate_base_mapping_pool(case.test_case)
            selection_rows = [
                run_mapping(case, mapping, CONFIG_BY_NAME["v1"])
                for mapping in pool
            ]
            valid_selection = [row for row in selection_rows if row.valid]
            if not valid_selection:
                raise RuntimeError(f"{case.name} 的 v1 映射池全部失败")
            selected = min(
                valid_selection,
                key=lambda row: (
                    row.swap_count,
                    row.depth,
                    row.runtime_ms,
                    row.mapping_name,
                ),
            )
            mapping_pools[case.name] = [
                next(mapping for mapping in pool if mapping.name == selected.mapping_name)
            ]
            print(
                f"冻结映射 {case.name}: {selected.mapping_name}, "
                f"v1 SWAP={selected.swap_count}"
            )

    print("===== historical development：评分结构消融 =====")
    print(f"输入：{args.input.resolve()}")
    print(f"实例数：{len(cases)}")
    print(f"配置：{', '.join(config.name for config in configs)}")
    print(f"映射：{args.mapping}")

    rows: list[CaseRow] = []
    for config_index, config in enumerate(configs, start=1):
        print()
        print(f"[{config_index}/{len(configs)}] {config.name}")

        for case_index, case in enumerate(cases, start=1):
            row = run_case(case, mapping_pools[case.name], config)
            rows.append(row)
            if row.valid:
                print(
                    f"  [{case_index}/{len(cases)}] {case.name}: "
                    f"SWAP={row.swap_count}, depth={row.depth}, "
                    f"mapping={row.best_mapping}, "
                    f"{row.total_runtime_ms:.1f} ms"
                )
            else:
                print(
                    f"  [{case_index}/{len(cases)}] {case.name}: FAILED "
                    f"{row.error}"
                )

        write_csv(args.case_output, rows)
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summarize(rows), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summaries = summarize(rows)
    print()
    print("===== historical development 汇总 =====")
    for item in summaries:
        print(
            f"{item['config']}: total SWAP={item['total_swap']}, "
            f"W/T/L vs v1={item['wins_vs_v1']}/"
            f"{item['ties_vs_v1']}/{item['losses_vs_v1']}, "
            f"paired DeltaSWAP={item['paired_swap_delta_vs_v1']}, "
            f"failed={item['failed_cases']}, "
            f"runtime={item['total_runtime_ms']:.1f} ms"
        )

    print()
    print(f"逐实例 CSV：{args.case_output.resolve()}")
    print(f"汇总 JSON：{args.summary_output.resolve()}")


if __name__ == "__main__":
    main()
