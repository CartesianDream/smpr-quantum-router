from __future__ import annotations

import copy
from dataclasses import dataclass

from routing_model import (
    CircuitDAG,
    Gate,
    HardwareGraph,
    RoutingState,
    drain,
    print_result,
)

# ============================================================
# 1. 参数
# ============================================================

THETA = 5
W_FUTURE = 0.5
GAMMA = 0.5
ALPHA = 0.5
K_E = 20

LAMBDA_PHI = 1.0
BETA_UNLOCK = 2.0
LAMBDA_DEPTH = 0.2
LAMBDA_UNDO = 2.5

MAX_ROUTE_STEPS = 1000


# ============================================================
# 2. 关键路径权重
# ============================================================

def compute_reverse_longest_paths(dag: CircuitDAG) -> list[int]:
    """r(g)：从门 g 到线路末端的反向最长路径长度。"""
    r = [1] * len(dag.gates)

    for gate_id in reversed(range(len(dag.gates))):
        successors = dag.successors[gate_id]
        if successors:
            r[gate_id] = 1 + max(r[s] for s in successors)

    return r


def compute_critical_weights(
    dag: CircuitDAG,
    alpha: float,
) -> list[float]:
    """c(g)=1+alpha*(r(g)-1)/(r_max-1)。"""
    r = compute_reverse_longest_paths(dag)
    if not r:
        return []

    r_max = max(r)
    if r_max == 1:
        return [1.0] * len(r)

    return [
        1.0 + alpha * (value - 1) / (r_max - 1)
        for value in r
    ]


# ============================================================
# 3. 未来层
# ============================================================

def build_future_layers(
    state: RoutingState,
    dag: CircuitDAG,
    theta: int,
) -> list[list[int]]:
    """
    构造 L^(0),...,L^(theta-1)。

    未来展开只考虑 DAG 依赖，不考虑硬件可执行性。
    新暴露的单比特门会被虚拟执行。
    """
    if theta <= 0:
        raise ValueError("theta 必须为正整数")

    remaining = list(state.remaining_predecessors)
    executed = set(state.executed_gates)
    layers: list[list[int]] = []

    current = [
        gate.gate_id
        for gate in dag.gates
        if (
            gate.gate_id not in executed
            and remaining[gate.gate_id] == 0
            and gate.is_two_qubit
        )
    ]

    for _ in range(theta):
        current = sorted(current)
        if not current:
            break

        layers.append(current)

        # 虚拟删除当前双比特门层。
        for gate_id in current:
            executed.add(gate_id)
            for successor in dag.successors[gate_id]:
                remaining[successor] -= 1

        # 反复虚拟执行新暴露的单比特门。
        made_progress = True
        while made_progress:
            made_progress = False

            for gate in dag.gates:
                gate_id = gate.gate_id

                if gate_id in executed:
                    continue

                if remaining[gate_id] == 0 and gate.is_single_qubit:
                    executed.add(gate_id)

                    for successor in dag.successors[gate_id]:
                        remaining[successor] -= 1

                    made_progress = True

        current = [
            gate.gate_id
            for gate in dag.gates
            if (
                gate.gate_id not in executed
                and remaining[gate.gate_id] == 0
                and gate.is_two_qubit
            )
        ]

    return layers


# ============================================================
# 4. 势函数
# ============================================================

def routing_distance(
    gate_id: int,
    mapping: list[int],
    dag: CircuitDAG,
    hardware: HardwareGraph,
) -> int:
    """delta(g,pi)=d_H(pi(q_i),pi(q_j))-1。"""
    gate = dag.gates[gate_id]
    logical_a, logical_b = gate.qubits
    physical_a = mapping[logical_a]
    physical_b = mapping[logical_b]

    path = hardware.shortest_path(physical_a, physical_b)
    return len(path) - 2


def weighted_average_distance(
    layer: list[int],
    mapping: list[int],
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
) -> float:
    """A(L,pi)：关键性加权平均路由距离。"""
    if not layer:
        return 0.0

    numerator = sum(
        critical_weights[gate_id]
        * routing_distance(gate_id, mapping, dag, hardware)
        for gate_id in layer
    )

    denominator = sum(
        critical_weights[gate_id]
        for gate_id in layer
    )

    return numerator / denominator


def potential(
    layers: list[list[int]],
    mapping: list[int],
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
) -> float:
    """Phi(pi)=A(L0,pi)+sum_j w_j A(Lj,pi)。"""
    if not layers:
        return 0.0

    value = weighted_average_distance(
        layers[0],
        mapping,
        dag,
        hardware,
        critical_weights,
    )

    if THETA <= 1 or len(layers) <= 1:
        return value

    denominator = 1.0 - GAMMA ** (THETA - 1)

    for layer_index, layer in enumerate(layers[1:], start=1):
        weight = (
            W_FUTURE
            * (1.0 - GAMMA)
            * GAMMA ** (layer_index - 1)
            / denominator
        )

        value += weight * weighted_average_distance(
            layer,
            mapping,
            dag,
            hardware,
            critical_weights,
        )

    return value


# ============================================================
# 5. 候选 SWAP：S_inc ∪ S_path
# ============================================================

def choose_important_future_gates(
    layers: list[list[int]],
    critical_weights: list[float],
) -> list[int]:
    """按 c(g)*gamma^j 选择至多 K_E 个未来门。"""
    ranked: list[tuple[float, int]] = []

    for layer_index, layer in enumerate(layers[1:], start=1):
        for gate_id in layer:
            priority = (
                critical_weights[gate_id]
                * GAMMA ** layer_index
            )
            ranked.append((priority, gate_id))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [gate_id for _, gate_id in ranked[:K_E]]


def generate_swap_candidates(
    state: RoutingState,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    layers: list[list[int]],
    critical_weights: list[float],
) -> list[tuple[int, int]]:
    """生成 S_swap=S_inc∪S_path。"""
    if not layers:
        return []

    front_layer = layers[0]
    important_future = choose_important_future_gates(
        layers,
        critical_weights,
    )

    relevant_gate_ids = list(
        dict.fromkeys(front_layer + important_future)
    )

    relevant_nodes: set[int] = set()

    for gate_id in relevant_gate_ids:
        gate = dag.gates[gate_id]

        for logical_qubit in gate.qubits:
            relevant_nodes.add(
                state.logical_to_physical[logical_qubit]
            )

    candidates: set[tuple[int, int]] = set()

    # S_inc：至少一个端点属于相关物理节点。
    for u, v in hardware.edges:
        if u in relevant_nodes or v in relevant_nodes:
            candidates.add(tuple(sorted((u, v))))

    # S_path：相关门两端最短路径上的边。
    for gate_id in relevant_gate_ids:
        gate = dag.gates[gate_id]
        logical_a, logical_b = gate.qubits

        physical_a = state.logical_to_physical[logical_a]
        physical_b = state.logical_to_physical[logical_b]
        path = hardware.shortest_path(physical_a, physical_b)

        for u, v in zip(path, path[1:]):
            candidates.add(tuple(sorted((u, v))))

    return sorted(candidates)


# ============================================================
# 6. 完整候选模拟与评分
# ============================================================

@dataclass
class CandidateEvaluation:
    edge: tuple[int, int]
    score: float
    delta_phi: float
    unlock_gain: float
    depth_increment: int
    undo_penalty: int
    executed_now: list[int]
    state_after: RoutingState


def evaluate_candidate(
    state: RoutingState,
    edge: tuple[int, int],
    layers: list[list[int]],
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
    last_swap: tuple[int, int] | None,
) -> CandidateEvaluation:
    """
    对候选完整执行 SWAP -> Drain，再计算：
    DeltaPhi、U(a)、DeltaD、P_undo。
    """
    before_phi = potential(
        layers,
        state.logical_to_physical,
        dag,
        hardware,
        critical_weights,
    )

    before_depth = state.current_depth()
    simulated_state = copy.deepcopy(state)

    simulated_state.apply_swap(edge[0], edge[1], hardware)
    executed_now = drain(simulated_state, dag, hardware)

    after_phi = potential(
        layers,
        simulated_state.logical_to_physical,
        dag,
        hardware,
        critical_weights,
    )

    delta_phi = after_phi - before_phi

    unlock_gain = sum(
        critical_weights[gate_id]
        for gate_id in executed_now
        if dag.gates[gate_id].is_two_qubit
    )

    depth_increment = (
        simulated_state.current_depth()
        - before_depth
    )

    normalized_edge = tuple(sorted(edge))

    undo_penalty = int(
        last_swap is not None
        and normalized_edge == tuple(sorted(last_swap))
    )

    score = (
        LAMBDA_PHI * delta_phi
        - BETA_UNLOCK * unlock_gain
        + LAMBDA_DEPTH * depth_increment
        + LAMBDA_UNDO * undo_penalty
    )

    return CandidateEvaluation(
        edge=normalized_edge,
        score=score,
        delta_phi=delta_phi,
        unlock_gain=unlock_gain,
        depth_increment=depth_increment,
        undo_penalty=undo_penalty,
        executed_now=executed_now,
        state_after=simulated_state,
    )


# ============================================================
# 7. 初始状态
# ============================================================

def make_initial_state(
    dag: CircuitDAG,
    hardware: HardwareGraph,
    initial_mapping: list[int],
) -> RoutingState:
    if len(initial_mapping) != dag.num_logical_qubits:
        raise ValueError("初始映射长度与逻辑比特数不一致")

    if len(set(initial_mapping)) != len(initial_mapping):
        raise ValueError("初始映射不是单射")

    physical_to_logical: list[int | None] = [
        None for _ in range(hardware.num_qubits)
    ]

    for logical, physical in enumerate(initial_mapping):
        if not 0 <= physical < hardware.num_qubits:
            raise ValueError(f"非法物理节点：{physical}")

        physical_to_logical[physical] = logical

    state = RoutingState(
        logical_to_physical=list(initial_mapping),
        physical_to_logical=physical_to_logical,
        remaining_predecessors=[
            len(dag.predecessors[gate.gate_id])
            for gate in dag.gates
        ],
        physical_depth=[
            0 for _ in range(hardware.num_qubits)
        ],
    )

    state.assert_valid(dag, hardware)
    return state


# ============================================================
# 8. 单步优化感知路由
# ============================================================

def run_one_step_router(
    dag: CircuitDAG,
    hardware: HardwareGraph,
    initial_mapping: list[int],
) -> RoutingState:
    state = make_initial_state(
        dag,
        hardware,
        initial_mapping,
    )

    critical_weights = compute_critical_weights(dag, ALPHA)
    reverse_length = compute_reverse_longest_paths(dag)

    print("===== 关键路径信息 =====")

    for gate in dag.gates:
        gate_id = gate.gate_id

        print(
            f"门 {gate_id}: "
            f"r={reverse_length[gate_id]}, "
            f"c={critical_weights[gate_id]:.3f}"
        )

    print()
    print("===== 开始单步优化感知路由 =====")

    last_swap: tuple[int, int] | None = None
    no_progress_count = 0
    route_step = 0

    initially_executed = drain(state, dag, hardware)

    if initially_executed:
        print("初始 Drain 执行逻辑门：", initially_executed)

    while len(state.executed_gates) < len(dag.gates):
        layers = build_future_layers(state, dag, THETA)

        if not layers:
            raise RuntimeError("线路未完成，但未来层为空")

        candidates = generate_swap_candidates(
            state,
            dag,
            hardware,
            layers,
            critical_weights,
        )

        if not candidates:
            raise RuntimeError("当前状态没有候选 SWAP")

        evaluations = [
            evaluate_candidate(
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

        evaluations.sort(
            key=lambda item: (
                item.score,
                -item.unlock_gain,
                item.delta_phi,
                item.depth_increment,
                item.edge,
            )
        )

        best = evaluations[0]
        route_step += 1

        print()
        print(f"----- 第 {route_step} 次决策 -----")
        print("未来层：", layers)
        print("候选边：", candidates)
        print("评分最优的前三个候选：")

        for candidate in evaluations[:3]:
            print(
                f"  edge={candidate.edge}, "
                f"score={candidate.score:.4f}, "
                f"DeltaPhi={candidate.delta_phi:.4f}, "
                f"unlock={candidate.unlock_gain:.3f}, "
                f"DeltaD={candidate.depth_increment}, "
                f"undo={candidate.undo_penalty}, "
                f"Drain={candidate.executed_now}"
            )

        print("实际选择：", best.edge)

        state = best.state_after
        last_swap = best.edge

        if best.unlock_gain > 0:
            no_progress_count = 0
        else:
            no_progress_count += 1

        print(
            "当前逻辑到物理映射：",
            state.logical_to_physical,
        )
        print("无进展计数：", no_progress_count)

        state.assert_valid(dag, hardware)

        if route_step > MAX_ROUTE_STEPS:
            raise RuntimeError(
                "超过最大路由步数，可能出现循环；"
                "下一阶段将加入 Beam Search 和 repair"
            )

    return state


# ============================================================
# 9. 主程序
# ============================================================

def main() -> None:
    gates = [
        Gate(0, "h", (0,)),
        Gate(1, "cx", (0, 4)),
        Gate(2, "cx", (1, 3)),
        Gate(3, "cx", (4, 2)),
        Gate(4, "cx", (0, 3)),
    ]

    dag = CircuitDAG(
        num_logical_qubits=5,
        gates=gates,
    )

    hardware = HardwareGraph(
        num_qubits=5,
        edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
        ],
    )

    initial_mapping = [0, 1, 2, 3, 4]

    state = run_one_step_router(
        dag,
        hardware,
        initial_mapping,
    )

    print_result(state, dag)


if __name__ == "__main__":
    main()
