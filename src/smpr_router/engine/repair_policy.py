from __future__ import annotations

import argparse
import copy
from collections import deque
from dataclasses import dataclass

from routing_model import (
    CircuitDAG,
    Gate,
    HardwareGraph,
    RoutingState,
    drain,
    print_result,
)

from one_step_policy import (
    ALPHA,
    THETA,
    CandidateEvaluation,
    build_future_layers,
    compute_critical_weights,
    evaluate_candidate,
    generate_swap_candidates,
    make_initial_state,
)


# ============================================================
# 1. 参数
# ============================================================

G_MIN = 0.10

S_TWO_STEP = 2
S_THREE_STEP = 4
S_REPAIR = 8

TWO_STEP_WIDTHS = (8, 8)
THREE_STEP_WIDTHS = (5, 3, 2)

ETA = 0.8

LAMBDA_RECENT = 1.0
RECENT_INCREMENT = 0.03

HISTORY_LENGTH = 32
MAX_ROUTE_STEPS = 1000


# ============================================================
# 2. 扩展状态和候选
# ============================================================

@dataclass
class RouterNode:
    """
    在第一阶段 RoutingState 外，再维护：

    recent_penalty[p]：
        物理节点 p 的近期重复使用值 rho(p)。

    last_swap：
        上一次真实或模拟执行的 SWAP 边。
    """

    state: RoutingState
    recent_penalty: list[float]
    last_swap: tuple[int, int] | None


@dataclass
class ExtendedEvaluation:
    """
    一个加入近期惩罚后的完整候选评价。
    """

    base: CandidateEvaluation
    score: float
    recent_cost: float
    next_node: RouterNode

    @property
    def edge(self) -> tuple[int, int]:
        return self.base.edge

    @property
    def unlock_gain(self) -> float:
        return self.base.unlock_gain

    @property
    def executed_now(self) -> list[int]:
        return self.base.executed_now


@dataclass
class SearchPath:
    evaluations: list[ExtendedEvaluation]
    cumulative_score: float
    total_unlock: float
    final_depth: int

    @property
    def first(self) -> ExtendedEvaluation:
        if not self.evaluations:
            raise RuntimeError("空搜索路径没有第一步")
        return self.evaluations[0]

    @property
    def last_node(self) -> RouterNode:
        if not self.evaluations:
            raise RuntimeError("空搜索路径没有末状态")
        return self.evaluations[-1].next_node

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(item.edge for item in self.evaluations)


# ============================================================
# 3. 带 recent penalty 的候选评价
# ============================================================

def update_recent_penalty(
    current: list[float],
    edge: tuple[int, int],
    unlock_gain: float,
) -> list[float]:
    """
    若候选产生真实双比特门进展，则全部重置为 1；
    否则候选端点的 rho 增加 epsilon。
    """

    if unlock_gain > 0.0:
        return [1.0] * len(current)

    updated = list(current)
    u, v = edge

    updated[u] += RECENT_INCREMENT
    updated[v] += RECENT_INCREMENT

    return updated


def evaluate_extended_candidate(
    node: RouterNode,
    edge: tuple[int, int],
    layers: list[list[int]],
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
) -> ExtendedEvaluation:
    """
    先调用第二阶段候选评价，再加入：

        lambda_r * (max(rho(u),rho(v)) - 1)
    """

    base = evaluate_candidate(
        state=node.state,
        edge=edge,
        layers=layers,
        dag=dag,
        hardware=hardware,
        critical_weights=critical_weights,
        last_swap=node.last_swap,
    )

    u, v = edge

    recent_cost = (
        max(
            node.recent_penalty[u],
            node.recent_penalty[v],
        )
        - 1.0
    )

    score = (
        base.score
        + LAMBDA_RECENT * recent_cost
    )

    next_recent = update_recent_penalty(
        current=node.recent_penalty,
        edge=base.edge,
        unlock_gain=base.unlock_gain,
    )

    next_node = RouterNode(
        state=base.state_after,
        recent_penalty=next_recent,
        last_swap=base.edge,
    )

    return ExtendedEvaluation(
        base=base,
        score=score,
        recent_cost=recent_cost,
        next_node=next_node,
    )


def evaluate_all_candidates(
    node: RouterNode,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
) -> list[ExtendedEvaluation]:
    """
    对当前节点重新构造未来层、候选集并完整模拟。
    """

    layers = build_future_layers(
        state=node.state,
        dag=dag,
        theta=THETA,
    )

    if not layers:
        return []

    candidates = generate_swap_candidates(
        state=node.state,
        dag=dag,
        hardware=hardware,
        layers=layers,
        critical_weights=critical_weights,
    )

    evaluations = [
        evaluate_extended_candidate(
            node=node,
            edge=edge,
            layers=layers,
            dag=dag,
            hardware=hardware,
            critical_weights=critical_weights,
        )
        for edge in candidates
    ]

    evaluations.sort(
        key=lambda item: (
            item.score,
            -item.unlock_gain,
            item.base.delta_phi,
            item.base.depth_increment,
            item.edge,
        )
    )

    return evaluations


# ============================================================
# 4. 自适应 Beam Search
# ============================================================

def normalized_gap(
    evaluations: list[ExtendedEvaluation],
) -> float:
    if not evaluations:
        raise ValueError("候选集合为空")

    if len(evaluations) == 1:
        return float("inf")

    best = evaluations[0].score
    second = evaluations[1].score

    return (second - best) / (1.0 + abs(best))


def beam_search(
    root: RouterNode,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
    widths: tuple[int, ...],
) -> list[SearchPath]:
    """
    通用滚动 Beam Search。

    widths=(8,8)：
        两步搜索。

    widths=(5,3,2)：
        三步窄搜索。
    """

    frontier: list[SearchPath] = [
        SearchPath(
            evaluations=[],
            cumulative_score=0.0,
            total_unlock=0.0,
            final_depth=root.state.current_depth(),
        )
    ]

    completed: list[SearchPath] = []

    for level, width in enumerate(widths):
        expanded: list[SearchPath] = []

        for path in frontier:
            current_node = (
                root
                if not path.evaluations
                else path.last_node
            )

            if (
                len(current_node.state.executed_gates)
                == len(dag.gates)
            ):
                completed.append(path)
                continue

            candidates = evaluate_all_candidates(
                node=current_node,
                dag=dag,
                hardware=hardware,
                critical_weights=critical_weights,
            )

            for candidate in candidates:
                new_path = SearchPath(
                    evaluations=(
                        path.evaluations + [candidate]
                    ),
                    cumulative_score=(
                        path.cumulative_score
                        + ETA ** level * candidate.score
                    ),
                    total_unlock=(
                        path.total_unlock
                        + candidate.unlock_gain
                    ),
                    final_depth=(
                        candidate.next_node.state.current_depth()
                    ),
                )

                expanded.append(new_path)

        if not expanded:
            break

        expanded.sort(
            key=lambda path: (
                path.cumulative_score,
                -path.total_unlock,
                path.final_depth,
                path.edges,
            )
        )

        frontier = expanded[:width]

    all_paths = completed + frontier

    if not all_paths:
        raise RuntimeError("Beam Search 没有生成有效路径")

    all_paths.sort(
        key=lambda path: (
            path.cumulative_score,
            -path.total_unlock,
            path.final_depth,
            path.edges,
        )
    )

    return all_paths


def choose_search_depth(
    one_step: list[ExtendedEvaluation],
    gap: float,
    no_progress_count: int,
    repeated_state: bool,
) -> int:
    best = one_step[0]

    all_fail_to_unlock = all(
        item.unlock_gain <= 0.0
        for item in one_step
    )

    if (
        best.unlock_gain > 0.0
        and gap >= G_MIN
        and no_progress_count < S_TWO_STEP
        and not repeated_state
    ):
        return 1

    if (
        no_progress_count >= S_THREE_STEP
        or repeated_state
    ):
        return 3

    if (
        all_fail_to_unlock
        or gap < G_MIN
        or no_progress_count >= S_TWO_STEP
    ):
        return 2

    return 1


# ============================================================
# 5. 状态签名
# ============================================================

def state_signature(
    state: RoutingState,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    映射和线路进度共同构成状态签名。
    """

    return (
        tuple(state.logical_to_physical),
        tuple(sorted(state.executed_gates)),
    )


def signature_occurrences(
    history: deque[
        tuple[tuple[int, ...], tuple[int, ...]]
    ],
    signature: tuple[tuple[int, ...], tuple[int, ...]],
) -> int:
    """
    计算把当前状态也计入后，该签名出现了几次。
    """

    return (
        sum(item == signature for item in history)
        + 1
    )


# ============================================================
# 6. 回滚式最短路径 repair
# ============================================================

def gate_routing_distance(
    node: RouterNode,
    gate_id: int,
    dag: CircuitDAG,
    hardware: HardwareGraph,
) -> int:
    gate = dag.gates[gate_id]
    logical_a, logical_b = gate.qubits

    physical_a = (
        node.state.logical_to_physical[logical_a]
    )

    physical_b = (
        node.state.logical_to_physical[logical_b]
    )

    path = hardware.shortest_path(
        physical_a,
        physical_b,
    )

    return len(path) - 2


def choose_repair_gate(
    node: RouterNode,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
) -> int:
    """
    R_t(g)=c(g)/delta(g,pi)。

    Drain 后前沿门均不可执行，因此 delta>=1。
    """

    layers = build_future_layers(
        state=node.state,
        dag=dag,
        theta=1,
    )

    if not layers or not layers[0]:
        raise RuntimeError("repair 时前沿层为空")

    front = layers[0]

    return max(
        front,
        key=lambda gate_id: (
            critical_weights[gate_id]
            / gate_routing_distance(
                node=node,
                gate_id=gate_id,
                dag=dag,
                hardware=hardware,
            ),
            critical_weights[gate_id],
            -gate_id,
        ),
    )


def repair_until_progress(
    stable_node: RouterNode,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
) -> tuple[RouterNode, int, int]:
    """
    1. 从上一次真实进展后的稳定状态重新开始；
    2. 选择 repair 优先级最高的前沿门；
    3. 在最短路径两端候选 SWAP 中选评分较小者；
    4. 重新计算最短路径；
    5. 直到目标门被执行。

    返回：
        repair 后节点；
        repair 使用的 SWAP 数；
        目标逻辑门编号。
    """

    node = copy.deepcopy(stable_node)

    target_gate_id = choose_repair_gate(
        node=node,
        dag=dag,
        hardware=hardware,
        critical_weights=critical_weights,
    )

    target_gate = dag.gates[target_gate_id]
    repair_swaps = 0

    print()
    print("===== 进入回滚式 repair =====")
    print("已回滚到上一次真实进展后的状态")
    print("repair 目标逻辑门：", target_gate_id)

    while target_gate_id not in node.state.executed_gates:
        logical_a, logical_b = target_gate.qubits

        physical_a = (
            node.state.logical_to_physical[logical_a]
        )

        physical_b = (
            node.state.logical_to_physical[logical_b]
        )

        path = hardware.shortest_path(
            physical_a,
            physical_b,
        )

        # 已相邻但尚未执行时，主动 Drain。
        if len(path) == 2:
            executed_now = drain(
                state=node.state,
                dag=dag,
                hardware=hardware,
            )

            if target_gate_id in executed_now:
                node.recent_penalty = (
                    [1.0] * hardware.num_qubits
                )
                break

            raise RuntimeError(
                "repair 目标门已相邻，但 Drain 未执行它"
            )

        left_edge = tuple(
            sorted((path[0], path[1]))
        )

        right_edge = tuple(
            sorted((path[-1], path[-2]))
        )

        repair_edges = list(
            dict.fromkeys([left_edge, right_edge])
        )

        layers = build_future_layers(
            state=node.state,
            dag=dag,
            theta=THETA,
        )

        evaluations = [
            evaluate_extended_candidate(
                node=node,
                edge=edge,
                layers=layers,
                dag=dag,
                hardware=hardware,
                critical_weights=critical_weights,
            )
            for edge in repair_edges
        ]

        evaluations.sort(
            key=lambda item: (
                item.score,
                -item.unlock_gain,
                item.base.delta_phi,
                item.edge,
            )
        )

        selected = evaluations[0]
        node = selected.next_node
        repair_swaps += 1

        print(
            f"repair SWAP {repair_swaps}: "
            f"{selected.edge}, "
            f"score={selected.score:.4f}, "
            f"Drain={selected.executed_now}"
        )

        if repair_swaps > hardware.num_qubits * 4:
            raise RuntimeError(
                "repair 使用 SWAP 过多，可能存在错误"
            )

    print(
        "repair 完成，目标门已执行，"
        f"共使用 {repair_swaps} 个 SWAP"
    )

    return node, repair_swaps, target_gate_id


# ============================================================
# 7. 带 repair 的完整路由器
# ============================================================

def run_router_with_repair(
    dag: CircuitDAG,
    hardware: HardwareGraph,
    initial_mapping: list[int],
    repair_threshold: int,
) -> RoutingState:
    state = make_initial_state(
        dag=dag,
        hardware=hardware,
        initial_mapping=initial_mapping,
    )

    drain(
        state=state,
        dag=dag,
        hardware=hardware,
    )

    current = RouterNode(
        state=state,
        recent_penalty=[
            1.0 for _ in range(hardware.num_qubits)
        ],
        last_swap=None,
    )

    critical_weights = compute_critical_weights(
        dag=dag,
        alpha=ALPHA,
    )

    # stable_node 始终保存“上一次真实双比特门进展后”的状态。
    stable_node = copy.deepcopy(current)

    history: deque[
        tuple[tuple[int, ...], tuple[int, ...]]
    ] = deque(maxlen=HISTORY_LENGTH)

    no_progress_count = 0
    route_step = 0

    one_step_calls = 0
    two_step_calls = 0
    three_step_calls = 0

    repair_calls = 0
    repair_swap_total = 0

    while (
        len(current.state.executed_gates)
        < len(dag.gates)
    ):
        signature = state_signature(current.state)

        occurrences = signature_occurrences(
            history=history,
            signature=signature,
        )

        repeated_state = occurrences >= 2
        repeated_three_times = occurrences >= 3

        # repair 在新一轮决策前触发：
        # 这时可以完整撤销自上次进展以来的无效后缀。
        if (
            no_progress_count >= repair_threshold
            or repeated_three_times
        ):
            current, repair_swaps, _ = (
                repair_until_progress(
                    stable_node=stable_node,
                    dag=dag,
                    hardware=hardware,
                    critical_weights=critical_weights,
                )
            )

            repair_calls += 1
            repair_swap_total += repair_swaps

            no_progress_count = 0
            stable_node = copy.deepcopy(current)

            history.clear()
            current.state.assert_valid(
                dag=dag,
                hardware=hardware,
            )

            continue

        history.append(signature)

        one_step = evaluate_all_candidates(
            node=current,
            dag=dag,
            hardware=hardware,
            critical_weights=critical_weights,
        )

        if not one_step:
            raise RuntimeError(
                "线路未完成，但没有候选 SWAP"
            )

        gap = normalized_gap(one_step)

        depth = choose_search_depth(
            one_step=one_step,
            gap=gap,
            no_progress_count=no_progress_count,
            repeated_state=repeated_state,
        )

        if depth == 1:
            one_step_calls += 1

            selected = one_step[0]
            retained_paths = [
                SearchPath(
                    evaluations=[selected],
                    cumulative_score=selected.score,
                    total_unlock=selected.unlock_gain,
                    final_depth=(
                        selected.next_node.state.current_depth()
                    ),
                )
            ]

        elif depth == 2:
            two_step_calls += 1

            retained_paths = beam_search(
                root=current,
                dag=dag,
                hardware=hardware,
                critical_weights=critical_weights,
                widths=TWO_STEP_WIDTHS,
            )

            if all(
                path.total_unlock <= 0.0
                for path in retained_paths
            ):
                two_step_calls -= 1
                three_step_calls += 1
                depth = 3

                retained_paths = beam_search(
                    root=current,
                    dag=dag,
                    hardware=hardware,
                    critical_weights=critical_weights,
                    widths=THREE_STEP_WIDTHS,
                )

            selected = retained_paths[0].first

        else:
            three_step_calls += 1

            retained_paths = beam_search(
                root=current,
                dag=dag,
                hardware=hardware,
                critical_weights=critical_weights,
                widths=THREE_STEP_WIDTHS,
            )

            # 三步搜索仍完全不能在预览中产生进展，
            # 下一轮直接触发 repair。
            if all(
                path.total_unlock <= 0.0
                for path in retained_paths
            ):
                no_progress_count = max(
                    no_progress_count,
                    repair_threshold,
                )
                continue

            selected = retained_paths[0].first

        route_step += 1

        print()
        print(f"----- 第 {route_step} 次真实决策 -----")
        print("状态签名出现次数：", occurrences)
        print("候选差距 G_t：", f"{gap:.4f}")
        print("搜索深度：", depth)

        for path in retained_paths[:3]:
            print(
                f"  edges={path.edges}, "
                f"J={path.cumulative_score:.4f}, "
                f"unlock={path.total_unlock:.3f}, "
                f"depth={path.final_depth}"
            )

        print("实际只提交第一步：", selected.edge)
        print(
            "recent penalty：",
            f"{selected.recent_cost:.4f}",
        )
        print(
            "第一步解锁收益：",
            f"{selected.unlock_gain:.3f}",
        )
        print(
            "第一步后的 Drain：",
            selected.executed_now,
        )

        current = selected.next_node

        if selected.unlock_gain > 0.0:
            no_progress_count = 0

            # 出现真实双比特门进展后，更新稳定回滚点。
            stable_node = copy.deepcopy(current)
            history.clear()
        else:
            no_progress_count += 1

        print("无进展计数：", no_progress_count)
        print(
            "当前 rho：",
            [
                round(value, 3)
                for value in current.recent_penalty
            ],
        )

        current.state.assert_valid(
            dag=dag,
            hardware=hardware,
        )

        if route_step > MAX_ROUTE_STEPS:
            raise RuntimeError(
                "超过最大路由步数"
            )

    print()
    print("===== 模块调用统计 =====")
    print("一步决策次数：", one_step_calls)
    print("两步 Beam 次数：", two_step_calls)
    print("三步 Beam 次数：", three_step_calls)
    print("repair 次数：", repair_calls)
    print("repair 内 SWAP 总数：", repair_swap_total)

    return current.state


# ============================================================
# 8. 两个实验实例
# ============================================================

def build_normal_example() -> tuple[
    CircuitDAG,
    HardwareGraph,
    list[int],
]:
    """
    与前四阶段相同的五比特实例。
    正常情况下 repair 不应频繁触发。
    """

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

    return dag, hardware, [0, 1, 2, 3, 4]


def build_repair_demo() -> tuple[
    CircuitDAG,
    HardwareGraph,
    list[int],
]:
    """
    专门验证回滚 repair：

        物理拓扑：0--1--2--3--4
        逻辑线路：CX(q0,q4)

    两端初始距离为 4，任意一个 SWAP 都不能立即解锁门。
    演示模式把 repair 阈值设为 1：
    第一个无进展 SWAP 后，下一轮回滚，再沿最短路径推进。
    """

    gates = [
        Gate(0, "cx", (0, 4)),
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

    return dag, hardware, [0, 1, 2, 3, 4]


# ============================================================
# 9. 主程序
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "第五阶段：近期惩罚、循环检测和回滚式 repair"
        )
    )

    parser.add_argument(
        "--repair-demo",
        action="store_true",
        help="运行专门触发 repair 的小实例",
    )

    args = parser.parse_args()

    if args.repair_demo:
        print("===== repair 强制演示模式 =====")
        dag, hardware, initial_mapping = (
            build_repair_demo()
        )

        threshold = 1
    else:
        print("===== 正常五比特实例 =====")
        dag, hardware, initial_mapping = (
            build_normal_example()
        )

        threshold = S_REPAIR

    state = run_router_with_repair(
        dag=dag,
        hardware=hardware,
        initial_mapping=initial_mapping,
        repair_threshold=threshold,
    )

    print()
    print_result(
        state=state,
        dag=dag,
    )


if __name__ == "__main__":
    main()
