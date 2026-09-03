from __future__ import annotations

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
# 1. 两步 Beam Search 参数
# ============================================================

B1 = 8
B2 = 8
ETA = 0.8
MAX_ROUTE_STEPS = 1000


# ============================================================
# 2. Beam 路径数据结构
# ============================================================

@dataclass
class BeamPath:
    """
    一条长度为 1 或 2 的候选路径。

    注意：
    真正提交到物理线路的永远只有 first，
    second 只用于帮助判断第一步是否值得执行。
    """

    first: CandidateEvaluation
    second: CandidateEvaluation | None
    cumulative_score: float
    total_unlock: float
    final_depth: int

    @property
    def first_edge(self) -> tuple[int, int]:
        return self.first.edge

    @property
    def second_edge(self) -> tuple[int, int] | None:
        if self.second is None:
            return None
        return self.second.edge

    @property
    def final_state(self) -> RoutingState:
        if self.second is None:
            return self.first.state_after
        return self.second.state_after


# ============================================================
# 3. 评价一个状态的全部单步候选
# ============================================================

def evaluate_all_candidates(
    state: RoutingState,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
    last_swap: tuple[int, int] | None,
) -> list[CandidateEvaluation]:
    """
    对当前状态：
    1. 重建未来层；
    2. 重新生成候选集；
    3. 对每个候选执行 SWAP -> Drain 完整模拟；
    4. 按单步评分排序。
    """

    layers = build_future_layers(
        state=state,
        dag=dag,
        theta=THETA,
    )

    if not layers:
        return []

    candidates = generate_swap_candidates(
        state=state,
        dag=dag,
        hardware=hardware,
        layers=layers,
        critical_weights=critical_weights,
    )

    evaluations = [
        evaluate_candidate(
            state=state,
            edge=edge,
            layers=layers,
            dag=dag,
            hardware=hardware,
            critical_weights=critical_weights,
            last_swap=last_swap,
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

    return evaluations


# ============================================================
# 4. 固定两步滚动 Beam Search
# ============================================================

def choose_first_step_by_two_step_beam(
    state: RoutingState,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
    last_swap: tuple[int, int] | None,
) -> tuple[CandidateEvaluation, list[BeamPath]]:
    """
    固定展开两层 Beam Search。

    第 1 层：
        对当前状态的所有候选进行评价，保留前 B1 个。

    第 2 层：
        对每个保留状态重新构造未来层和候选集，
        评价第二步，统一排序后保留前 B2 条路径。

    路径累计评分：
        J_2 = h(S,a_1) + ETA * h(S^(1),a_2)

    搜索结束后只执行最优路径的第一步。
    """

    first_level = evaluate_all_candidates(
        state=state,
        dag=dag,
        hardware=hardware,
        critical_weights=critical_weights,
        last_swap=last_swap,
    )

    if not first_level:
        raise RuntimeError("当前状态没有可用的一步候选")

    first_level = first_level[:B1]
    completed_paths: list[BeamPath] = []
    second_level_paths: list[BeamPath] = []

    for first_eval in first_level:
        first_state = first_eval.state_after

        # 如果第一步后线路已完成，则不再强行展开第二步。
        if len(first_state.executed_gates) == len(dag.gates):
            completed_paths.append(
                BeamPath(
                    first=first_eval,
                    second=None,
                    cumulative_score=first_eval.score,
                    total_unlock=first_eval.unlock_gain,
                    final_depth=first_state.current_depth(),
                )
            )
            continue

        second_level = evaluate_all_candidates(
            state=first_state,
            dag=dag,
            hardware=hardware,
            critical_weights=critical_weights,
            last_swap=first_eval.edge,
        )

        # 理论上未完成状态应当存在候选；若不存在，
        # 仍保留第一步路径，便于暴露问题而不是静默丢失。
        if not second_level:
            second_level_paths.append(
                BeamPath(
                    first=first_eval,
                    second=None,
                    cumulative_score=first_eval.score,
                    total_unlock=first_eval.unlock_gain,
                    final_depth=first_state.current_depth(),
                )
            )
            continue

        for second_eval in second_level[:B2]:
            cumulative_score = (
                first_eval.score
                + ETA * second_eval.score
            )

            second_level_paths.append(
                BeamPath(
                    first=first_eval,
                    second=second_eval,
                    cumulative_score=cumulative_score,
                    total_unlock=(
                        first_eval.unlock_gain
                        + second_eval.unlock_gain
                    ),
                    final_depth=(
                        second_eval.state_after.current_depth()
                    ),
                )
            )

    all_paths = completed_paths + second_level_paths

    if not all_paths:
        raise RuntimeError("两步 Beam Search 没有生成任何路径")

    all_paths.sort(
        key=lambda path: (
            path.cumulative_score,
            -path.total_unlock,
            path.final_depth,
            path.first_edge,
            path.second_edge or (-1, -1),
        )
    )

    retained_paths = all_paths[:B2]
    best_path = retained_paths[0]

    return best_path.first, retained_paths


# ============================================================
# 5. 两步滚动路由器
# ============================================================

def run_two_step_beam_router(
    dag: CircuitDAG,
    hardware: HardwareGraph,
    initial_mapping: list[int],
) -> RoutingState:
    state = make_initial_state(
        dag=dag,
        hardware=hardware,
        initial_mapping=initial_mapping,
    )

    critical_weights = compute_critical_weights(
        dag=dag,
        alpha=ALPHA,
    )

    print("===== 开始固定两步滚动 Beam Search =====")
    print(f"参数：B1={B1}, B2={B2}, ETA={ETA}")
    print()

    initially_executed = drain(
        state=state,
        dag=dag,
        hardware=hardware,
    )

    if initially_executed:
        print(
            "初始 Drain 执行逻辑门：",
            initially_executed,
        )

    last_swap: tuple[int, int] | None = None
    route_step = 0
    no_progress_count = 0

    while len(state.executed_gates) < len(dag.gates):
        first_eval, retained_paths = (
            choose_first_step_by_two_step_beam(
                state=state,
                dag=dag,
                hardware=hardware,
                critical_weights=critical_weights,
                last_swap=last_swap,
            )
        )

        route_step += 1

        print()
        print(f"----- 第 {route_step} 次真实决策 -----")
        print("保留路径中评分最优的前三条：")

        for path in retained_paths[:3]:
            print(
                f"  path=({path.first_edge}, {path.second_edge}), "
                f"J={path.cumulative_score:.4f}, "
                f"unlock={path.total_unlock:.3f}, "
                f"final_depth={path.final_depth}"
            )

        print(
            "实际只提交第一步：",
            first_eval.edge,
        )

        # 只提交最佳两步路径的第一步状态。
        state = first_eval.state_after
        last_swap = first_eval.edge

        if first_eval.unlock_gain > 0:
            no_progress_count = 0
        else:
            no_progress_count += 1

        print(
            "第一步单步评分：",
            f"{first_eval.score:.4f}",
        )

        print(
            "第一步解锁收益：",
            f"{first_eval.unlock_gain:.3f}",
        )

        print(
            "第一步后的 Drain：",
            first_eval.executed_now,
        )

        print(
            "当前逻辑到物理映射：",
            state.logical_to_physical,
        )

        print(
            "无进展计数：",
            no_progress_count,
        )

        state.assert_valid(
            dag=dag,
            hardware=hardware,
        )

        if route_step > MAX_ROUTE_STEPS:
            raise RuntimeError(
                "超过最大路由步数，可能出现循环；"
                "后续阶段将加入自适应触发和 repair"
            )

    return state


# ============================================================
# 6. 当前实验实例
# ============================================================

def build_example() -> tuple[
    CircuitDAG,
    HardwareGraph,
    list[int],
]:
    """
    与前两阶段使用完全相同的五比特实例，
    便于公平比较。
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

    initial_mapping = [0, 1, 2, 3, 4]

    return dag, hardware, initial_mapping


# ============================================================
# 7. 主程序
# ============================================================

def main() -> None:
    dag, hardware, initial_mapping = build_example()

    state = run_two_step_beam_router(
        dag=dag,
        hardware=hardware,
        initial_mapping=initial_mapping,
    )

    print()
    print_result(
        state=state,
        dag=dag,
    )


if __name__ == "__main__":
    main()
