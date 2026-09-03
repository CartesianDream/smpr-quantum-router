from __future__ import annotations

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
# 1. 自适应搜索参数
# ============================================================

G_MIN = 0.10

S_TWO_STEP = 2
S_THREE_STEP = 4

TWO_STEP_WIDTHS = (8, 8)
THREE_STEP_WIDTHS = (5, 3, 2)

ETA = 0.8
HISTORY_LENGTH = 32
MAX_ROUTE_STEPS = 1000


# ============================================================
# 2. Beam 路径
# ============================================================

@dataclass
class SearchPath:
    evaluations: list[CandidateEvaluation]
    cumulative_score: float
    total_unlock: float
    final_depth: int

    @property
    def first(self) -> CandidateEvaluation:
        if not self.evaluations:
            raise RuntimeError("空搜索路径没有第一步")
        return self.evaluations[0]

    @property
    def last_state(self) -> RoutingState:
        if not self.evaluations:
            raise RuntimeError("空搜索路径没有末状态")
        return self.evaluations[-1].state_after

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(item.edge for item in self.evaluations)


# ============================================================
# 3. 单步候选评价
# ============================================================

def evaluate_all_candidates(
    state: RoutingState,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
    last_swap: tuple[int, int] | None,
) -> list[CandidateEvaluation]:
    """
    在给定状态上重新构造未来层和候选集，
    对每个候选执行 SWAP -> Drain 完整模拟。
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


def normalized_gap(
    evaluations: list[CandidateEvaluation],
) -> float:
    """
    G_t=(h^(2)-h^(1))/(1+|h^(1)|)。

    只有一个候选时，差距视为正无穷。
    """

    if not evaluations:
        raise ValueError("候选集合为空")

    if len(evaluations) == 1:
        return float("inf")

    best = evaluations[0].score
    second = evaluations[1].score

    return (second - best) / (1.0 + abs(best))


# ============================================================
# 4. 通用滚动 Beam Search
# ============================================================

def beam_search(
    state: RoutingState,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    critical_weights: list[float],
    last_swap: tuple[int, int] | None,
    widths: tuple[int, ...],
) -> list[SearchPath]:
    """
    widths 的长度就是搜索深度。

    例如：
        (8,8)   表示两步 Beam；
        (5,3,2) 表示三步窄 Beam。

    每一层将上一层保留路径全部展开，
    按累计代价进行全局排序，仅保留 B_k 条。
    """

    if not widths:
        raise ValueError("Beam 宽度序列不能为空")

    frontier: list[SearchPath] = [
        SearchPath(
            evaluations=[],
            cumulative_score=0.0,
            total_unlock=0.0,
            final_depth=state.current_depth(),
        )
    ]

    completed: list[SearchPath] = []

    for level, width in enumerate(widths):
        expanded: list[SearchPath] = []

        for path in frontier:
            if path.evaluations:
                current_state = path.last_state
                current_last_swap = path.evaluations[-1].edge
            else:
                current_state = state
                current_last_swap = last_swap

            if len(current_state.executed_gates) == len(dag.gates):
                completed.append(path)
                continue

            candidates = evaluate_all_candidates(
                state=current_state,
                dag=dag,
                hardware=hardware,
                critical_weights=critical_weights,
                last_swap=current_last_swap,
            )

            if not candidates:
                # 未完成但无候选属于异常状态。
                continue

            for candidate in candidates:
                new_evaluations = path.evaluations + [candidate]

                new_path = SearchPath(
                    evaluations=new_evaluations,
                    cumulative_score=(
                        path.cumulative_score
                        + ETA ** level * candidate.score
                    ),
                    total_unlock=(
                        path.total_unlock
                        + candidate.unlock_gain
                    ),
                    final_depth=(
                        candidate.state_after.current_depth()
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
        raise RuntimeError("Beam Search 没有生成任何有效路径")

    all_paths.sort(
        key=lambda path: (
            path.cumulative_score,
            -path.total_unlock,
            path.final_depth,
            path.edges,
        )
    )

    return all_paths


# ============================================================
# 5. 状态签名与自适应深度
# ============================================================

def state_signature(
    state: RoutingState,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    状态签名必须同时包含映射与线路进度。

    只记录映射会错误地把不同执行进度的状态
    当成同一个状态。
    """

    return (
        tuple(state.logical_to_physical),
        tuple(sorted(state.executed_gates)),
    )


def choose_search_depth(
    one_step: list[CandidateEvaluation],
    gap: float,
    no_progress_count: int,
    repeated_state: bool,
) -> int:
    """
    按当前决策不确定性选择 1、2 或 3 步搜索。
    """

    best = one_step[0]
    all_fail_to_unlock = all(
        candidate.unlock_gain <= 0.0
        for candidate in one_step
    )

    # 决策很明确，且最佳候选马上带来真实进展。
    if (
        best.unlock_gain > 0.0
        and gap >= G_MIN
        and no_progress_count < S_TWO_STEP
        and not repeated_state
    ):
        return 1

    # 长时间停滞或状态重复，使用三步窄搜索。
    if (
        no_progress_count >= S_THREE_STEP
        or repeated_state
    ):
        return 3

    # 候选接近、所有候选都没有真实进展，
    # 或已连续若干步没有进展时，使用两步搜索。
    if (
        all_fail_to_unlock
        or gap < G_MIN
        or no_progress_count >= S_TWO_STEP
    ):
        return 2

    return 1


# ============================================================
# 6. 自适应滚动路由器
# ============================================================

def run_adaptive_router(
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

    history: deque[
        tuple[tuple[int, ...], tuple[int, ...]]
    ] = deque(maxlen=HISTORY_LENGTH)

    last_swap: tuple[int, int] | None = None
    no_progress_count = 0
    route_step = 0

    one_step_calls = 0
    two_step_calls = 0
    three_step_calls = 0

    print("===== 开始自适应滚动 Beam Search =====")
    print(
        "参数："
        f"G_MIN={G_MIN}, "
        f"s2={S_TWO_STEP}, "
        f"s3={S_THREE_STEP}, "
        f"ETA={ETA}"
    )

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

    while len(state.executed_gates) < len(dag.gates):
        signature = state_signature(state)
        repeated_state = signature in history
        history.append(signature)

        one_step = evaluate_all_candidates(
            state=state,
            dag=dag,
            hardware=hardware,
            critical_weights=critical_weights,
            last_swap=last_swap,
        )

        if not one_step:
            raise RuntimeError("线路未完成，但没有候选 SWAP")

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
            selected_path = SearchPath(
                evaluations=[selected],
                cumulative_score=selected.score,
                total_unlock=selected.unlock_gain,
                final_depth=selected.state_after.current_depth(),
            )

            retained_paths = [selected_path]

        elif depth == 2:
            two_step_calls += 1

            retained_paths = beam_search(
                state=state,
                dag=dag,
                hardware=hardware,
                critical_weights=critical_weights,
                last_swap=last_swap,
                widths=TWO_STEP_WIDTHS,
            )

            # 若两步搜索的最佳保留路径仍完全无进展，
            # 立即升级成三步窄搜索。
            if all(
                path.total_unlock <= 0.0
                for path in retained_paths
            ):
                depth = 3
                two_step_calls -= 1
                three_step_calls += 1

                retained_paths = beam_search(
                    state=state,
                    dag=dag,
                    hardware=hardware,
                    critical_weights=critical_weights,
                    last_swap=last_swap,
                    widths=THREE_STEP_WIDTHS,
                )

            selected = retained_paths[0].first

        else:
            three_step_calls += 1

            retained_paths = beam_search(
                state=state,
                dag=dag,
                hardware=hardware,
                critical_weights=critical_weights,
                last_swap=last_swap,
                widths=THREE_STEP_WIDTHS,
            )

            selected = retained_paths[0].first

        route_step += 1

        print()
        print(f"----- 第 {route_step} 次真实决策 -----")
        print("归一化候选差距 G_t：", f"{gap:.4f}")
        print("无进展计数：", no_progress_count)
        print("当前状态是否重复：", repeated_state)
        print("本轮搜索深度：", depth)

        print("本轮最优路径：")
        for path in retained_paths[:3]:
            print(
                f"  edges={path.edges}, "
                f"J={path.cumulative_score:.4f}, "
                f"unlock={path.total_unlock:.3f}, "
                f"final_depth={path.final_depth}"
            )

        print("实际只提交第一步：", selected.edge)
        print(
            "第一步评分：",
            f"{selected.score:.4f}",
        )
        print(
            "第一步解锁收益：",
            f"{selected.unlock_gain:.3f}",
        )
        print(
            "第一步后的 Drain：",
            selected.executed_now,
        )

        # 滚动搜索：只提交最优路径的第一步。
        state = selected.state_after
        last_swap = selected.edge

        if selected.unlock_gain > 0.0:
            no_progress_count = 0
        else:
            no_progress_count += 1

        print(
            "当前逻辑到物理映射：",
            state.logical_to_physical,
        )

        state.assert_valid(
            dag=dag,
            hardware=hardware,
        )

        if route_step > MAX_ROUTE_STEPS:
            raise RuntimeError(
                "超过最大路由步数，可能陷入循环；"
                "下一阶段将加入回滚式 repair"
            )

    print()
    print("===== 搜索模式统计 =====")
    print("一步决策次数：", one_step_calls)
    print("两步 Beam 次数：", two_step_calls)
    print("三步 Beam 次数：", three_step_calls)

    return state


# ============================================================
# 7. 测试实例
# ============================================================

def build_example() -> tuple[
    CircuitDAG,
    HardwareGraph,
    list[int],
]:
    """
    仍使用同一五比特实例。

    这样可以确认：
    自适应机制没有破坏前面已经得到的 4-SWAP 结果。
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
# 8. 主程序
# ============================================================

def main() -> None:
    dag, hardware, initial_mapping = build_example()

    state = run_adaptive_router(
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
