from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


# ============================================================
# 1. 逻辑门
# ============================================================

@dataclass(frozen=True)
class Gate:
    """一个逻辑量子门。"""

    gate_id: int
    name: str
    qubits: tuple[int, ...]

    @property
    def is_single_qubit(self) -> bool:
        return len(self.qubits) == 1

    @property
    def is_two_qubit(self) -> bool:
        return len(self.qubits) == 2


# ============================================================
# 2. 硬件耦合图
# ============================================================

class HardwareGraph:
    """无向硬件耦合图。"""

    def __init__(
        self,
        num_qubits: int,
        edges: list[tuple[int, int]],
    ) -> None:
        if num_qubits <= 0:
            raise ValueError("物理量子比特数必须为正整数")

        self.num_qubits = num_qubits
        self.edges = edges
        self.adjacency: list[set[int]] = [
            set() for _ in range(num_qubits)
        ]
        # 大图路由会反复询问同一对节点的最短路。缓存路径可避免在
        # Willow/Heron 规模上为每个候选 SWAP 重复执行 BFS。
        self._shortest_path_cache: dict[
            tuple[int, int], tuple[int, ...]
        ] = {}

        for u, v in edges:
            if not (
                0 <= u < num_qubits
                and 0 <= v < num_qubits
            ):
                raise ValueError(f"非法硬件边：{(u, v)}")

            if u == v:
                raise ValueError("硬件图不能包含自环")

            self.adjacency[u].add(v)
            self.adjacency[v].add(u)

    def adjacent(self, u: int, v: int) -> bool:
        """判断两个物理节点是否相邻。"""
        return v in self.adjacency[u]

    def shortest_path(
        self,
        start: int,
        goal: int,
    ) -> list[int]:
        """用 BFS 求一条最短路径。"""

        cached = self._shortest_path_cache.get((start, goal))
        if cached is not None:
            return list(cached)

        queue: deque[int] = deque([start])
        parent: dict[int, int | None] = {
            start: None
        }

        while queue:
            current = queue.popleft()

            if current == goal:
                break

            for neighbor in sorted(
                self.adjacency[current]
            ):
                if neighbor not in parent:
                    parent[neighbor] = current
                    queue.append(neighbor)

        if goal not in parent:
            raise ValueError(
                f"物理点 {start} 与 {goal} 不连通"
            )

        path: list[int] = []
        current: int | None = goal

        while current is not None:
            path.append(current)
            current = parent[current]

        path.reverse()
        frozen = tuple(path)
        self._shortest_path_cache[(start, goal)] = frozen
        self._shortest_path_cache[(goal, start)] = tuple(reversed(frozen))
        return path


# ============================================================
# 3. 逻辑门依赖 DAG
# ============================================================

class CircuitDAG:
    """
    根据每个逻辑量子比特上的门顺序，
    自动建立门依赖关系。
    """

    def __init__(
        self,
        num_logical_qubits: int,
        gates: list[Gate],
    ) -> None:
        self.num_logical_qubits = num_logical_qubits
        self.gates = gates

        self.predecessors: list[set[int]] = [
            set() for _ in gates
        ]

        self.successors: list[set[int]] = [
            set() for _ in gates
        ]

        self._build_dependencies()

    def _build_dependencies(self) -> None:
        """
        一个门依赖于其每个操作量子比特上
        最近出现的前一个门。
        """

        last_gate_on_qubit: list[int | None] = [
            None
            for _ in range(
                self.num_logical_qubits
            )
        ]

        for gate in self.gates:
            predecessors: set[int] = set()

            for logical_qubit in gate.qubits:
                previous_gate = (
                    last_gate_on_qubit[
                        logical_qubit
                    ]
                )

                if previous_gate is not None:
                    predecessors.add(previous_gate)

            self.predecessors[
                gate.gate_id
            ] = predecessors

            for predecessor in predecessors:
                self.successors[
                    predecessor
                ].add(gate.gate_id)

            for logical_qubit in gate.qubits:
                last_gate_on_qubit[
                    logical_qubit
                ] = gate.gate_id


# ============================================================
# 4. 路由状态
# ============================================================

@dataclass
class RoutingState:
    """
    路由状态。

    logical_to_physical[q]：
        逻辑比特 q 当前所在的物理节点。

    physical_to_logical[p]：
        物理节点 p 当前承载的逻辑比特。
    """

    logical_to_physical: list[int]
    physical_to_logical: list[int | None]

    remaining_predecessors: list[int]
    executed_gates: set[int] = field(
        default_factory=set
    )

    physical_operations: list[tuple] = field(
        default_factory=list
    )

    physical_depth: list[int] = field(
        default_factory=list
    )

    def __deepcopy__(
        self,
        memo: dict[int, object],
    ) -> "RoutingState":
        """
        高速复制路由状态。

        所有字段都是独立的可变容器；
        physical_operations 的元素只包含
        str、int、None 和不可变 tuple，
        因而只需复制外层 list。
        """
        existing = memo.get(id(self))

        if existing is not None:
            return existing  # type: ignore[return-value]

        cloned = RoutingState(
            logical_to_physical=(
                self.logical_to_physical.copy()
            ),
            physical_to_logical=(
                self.physical_to_logical.copy()
            ),
            remaining_predecessors=(
                self.remaining_predecessors.copy()
            ),
            executed_gates=(
                self.executed_gates.copy()
            ),
            physical_operations=(
                self.physical_operations.copy()
            ),
            physical_depth=(
                self.physical_depth.copy()
            ),
        )

        memo[id(self)] = cloned
        return cloned

    def current_depth(self) -> int:
        return max(
            self.physical_depth,
            default=0,
        )

    def assert_valid(
        self,
        dag: CircuitDAG,
        hardware: HardwareGraph,
    ) -> None:
        """
        检查映射、DAG 顺序和物理双比特门是否合法。
        """

        # 检查正映射与逆映射一致。
        occupied: set[int] = set()

        for logical, physical in enumerate(
            self.logical_to_physical
        ):
            if physical in occupied:
                raise AssertionError(
                    "逻辑到物理映射不是单射"
                )

            occupied.add(physical)

            if (
                self.physical_to_logical[
                    physical
                ]
                != logical
            ):
                raise AssertionError(
                    "正映射和逆映射不一致"
                )

        for physical, logical in enumerate(
            self.physical_to_logical
        ):
            if logical is not None:
                if (
                    self.logical_to_physical[
                        logical
                    ]
                    != physical
                ):
                    raise AssertionError(
                        "逆映射和正映射不一致"
                    )

        # 检查已经执行的门满足依赖关系。
        for gate_id in self.executed_gates:
            for predecessor in (
                dag.predecessors[gate_id]
            ):
                if (
                    predecessor
                    not in self.executed_gates
                ):
                    raise AssertionError(
                        "出现违反 DAG 顺序的门"
                    )

        # 检查所有物理双比特操作合法。
        for operation in (
            self.physical_operations
        ):
            operation_name = operation[0]

            if operation_name in {
                "cx",
                "swap",
            }:
                u = operation[1]
                v = operation[2]

                if not hardware.adjacent(u, v):
                    raise AssertionError(
                        "存在不满足硬件拓扑的"
                        f"双比特操作：{operation}"
                    )

    def append_single_qubit_gate(
        self,
        name: str,
        physical_qubit: int,
        source_gate_id: int,
    ) -> None:
        """
        将一个单比特门加入物理线路，
        并更新调度深度。
        """

        self.physical_depth[
            physical_qubit
        ] += 1

        self.physical_operations.append(
            (
                name,
                physical_qubit,
                source_gate_id,
            )
        )

    def append_two_qubit_gate(
        self,
        name: str,
        u: int,
        v: int,
        source_gate_id: int | None,
        hardware: HardwareGraph,
        duration: int = 1,
    ) -> None:
        """
        将一个物理双比特门加入线路。

        普通 CX 的 duration=1。
        抽象 SWAP 按 3 层双比特门估计，
        因此 duration=3。
        """

        if not hardware.adjacent(u, v):
            raise ValueError(
                f"物理节点 {u} 和 {v} 不相邻"
            )

        new_depth = (
            max(
                self.physical_depth[u],
                self.physical_depth[v],
            )
            + duration
        )

        self.physical_depth[u] = new_depth
        self.physical_depth[v] = new_depth

        self.physical_operations.append(
            (
                name,
                u,
                v,
                source_gate_id,
            )
        )

    def apply_swap(
        self,
        u: int,
        v: int,
        hardware: HardwareGraph,
    ) -> None:
        """
        在物理边 (u,v) 上执行 SWAP，
        并同步更新正映射和逆映射。
        """

        self.append_two_qubit_gate(
            name="swap",
            u=u,
            v=v,
            source_gate_id=None,
            hardware=hardware,
            duration=3,
        )

        logical_u = (
            self.physical_to_logical[u]
        )

        logical_v = (
            self.physical_to_logical[v]
        )

        # 交换物理节点上承载的逻辑比特。
        self.physical_to_logical[u] = (
            logical_v
        )

        self.physical_to_logical[v] = (
            logical_u
        )

        # 同步更新逻辑到物理映射。
        if logical_u is not None:
            self.logical_to_physical[
                logical_u
            ] = v

        if logical_v is not None:
            self.logical_to_physical[
                logical_v
            ] = u


# ============================================================
# 5. Drain
# ============================================================

def drain(
    state: RoutingState,
    dag: CircuitDAG,
    hardware: HardwareGraph,
    *,
    validate: bool = True,
) -> list[int]:
    """
    反复执行所有当前可以执行的门：

    1. 剩余前驱数为 0 的单比特门；
    2. 剩余前驱数为 0，且两个物理端点
       相邻的双比特门。

    返回本次 Drain 执行的逻辑门编号。
    """

    executed_now: list[int] = []

    made_progress = True

    while made_progress:
        made_progress = False

        for gate in dag.gates:
            gate_id = gate.gate_id

            if gate_id in state.executed_gates:
                continue

            if (
                state.remaining_predecessors[
                    gate_id
                ]
                != 0
            ):
                continue

            if gate.is_single_qubit:
                logical_qubit = gate.qubits[0]

                physical_qubit = (
                    state.logical_to_physical[
                        logical_qubit
                    ]
                )

                state.append_single_qubit_gate(
                    name=gate.name,
                    physical_qubit=(
                        physical_qubit
                    ),
                    source_gate_id=gate_id,
                )

            elif gate.is_two_qubit:
                logical_a = gate.qubits[0]
                logical_b = gate.qubits[1]

                physical_a = (
                    state.logical_to_physical[
                        logical_a
                    ]
                )

                physical_b = (
                    state.logical_to_physical[
                        logical_b
                    ]
                )

                if not hardware.adjacent(
                    physical_a,
                    physical_b,
                ):
                    continue

                state.append_two_qubit_gate(
                    name=gate.name,
                    u=physical_a,
                    v=physical_b,
                    source_gate_id=gate_id,
                    hardware=hardware,
                    duration=1,
                )

            else:
                raise ValueError(
                    "当前程序只支持一比特门"
                    "和二比特门"
                )

            state.executed_gates.add(
                gate_id
            )

            executed_now.append(gate_id)

            for successor in (
                dag.successors[gate_id]
            ):
                state.remaining_predecessors[
                    successor
                ] -= 1

                if (
                    state.remaining_predecessors[
                        successor
                    ]
                    < 0
                ):
                    raise AssertionError(
                        "剩余前驱数出现负值"
                    )

            made_progress = True

    if validate:
        state.assert_valid(dag, hardware)

    return executed_now


# ============================================================
# 6. 当前前沿层
# ============================================================

def blocked_front_layer(
    state: RoutingState,
    dag: CircuitDAG,
) -> list[int]:
    """
    Drain 后，前驱全部完成但因物理距离
    仍无法执行的双比特门集合。
    """

    front: list[int] = []

    for gate in dag.gates:
        gate_id = gate.gate_id

        if gate_id in state.executed_gates:
            continue

        if (
            state.remaining_predecessors[
                gate_id
            ]
            == 0
            and gate.is_two_qubit
        ):
            front.append(gate_id)

    return front


# ============================================================
# 7. 第一阶段的简单路由策略
# ============================================================

def run_shortest_path_baseline(
    dag: CircuitDAG,
    hardware: HardwareGraph,
    initial_mapping: list[int],
) -> RoutingState:
    """
    教学用基线：

    每次 Drain 后，选择前沿层中的第一个阻塞门，
    将它的第一个逻辑比特沿最短路径向第二个移动。

    这不是最终 OA-ABR，只用于验证底层逻辑。
    """

    physical_to_logical: list[int | None] = [
        None for _ in range(
            hardware.num_qubits
        )
    ]

    for logical, physical in enumerate(
        initial_mapping
    ):
        physical_to_logical[
            physical
        ] = logical

    state = RoutingState(
        logical_to_physical=(
            list(initial_mapping)
        ),
        physical_to_logical=(
            physical_to_logical
        ),
        remaining_predecessors=[
            len(
                dag.predecessors[
                    gate.gate_id
                ]
            )
            for gate in dag.gates
        ],
        physical_depth=[
            0 for _ in range(
                hardware.num_qubits
            )
        ],
    )

    state.assert_valid(dag, hardware)

    route_step = 0

    print("===== 开始路由 =====")

    while (
        len(state.executed_gates)
        < len(dag.gates)
    ):
        executed_now = drain(
            state,
            dag,
            hardware,
        )

        if executed_now:
            print(
                "Drain 执行逻辑门：",
                executed_now,
            )

        if (
            len(state.executed_gates)
            == len(dag.gates)
        ):
            break

        front = blocked_front_layer(
            state,
            dag,
        )

        if not front:
            raise RuntimeError(
                "线路未完成，但前沿层为空；"
                "DAG 或 Drain 更新存在错误"
            )

        selected_gate_id = front[0]
        selected_gate = (
            dag.gates[selected_gate_id]
        )

        logical_a = (
            selected_gate.qubits[0]
        )

        logical_b = (
            selected_gate.qubits[1]
        )

        physical_a = (
            state.logical_to_physical[
                logical_a
            ]
        )

        physical_b = (
            state.logical_to_physical[
                logical_b
            ]
        )

        path = hardware.shortest_path(
            physical_a,
            physical_b,
        )

        if len(path) < 2:
            raise RuntimeError(
                "最短路径长度异常"
            )

        swap_u = path[0]
        swap_v = path[1]

        state.apply_swap(
            swap_u,
            swap_v,
            hardware,
        )

        route_step += 1

        state.assert_valid(
            dag,
            hardware,
        )

        print(
            f"第 {route_step} 次 SWAP："
            f"物理边 ({swap_u}, {swap_v})"
        )

        print(
            "当前逻辑到物理映射：",
            state.logical_to_physical,
        )

        if route_step > 1000:
            raise RuntimeError(
                "SWAP 次数异常，疑似出现循环"
            )

    state.assert_valid(dag, hardware)

    return state


# ============================================================
# 8. 打印实验结果
# ============================================================

def print_result(
    state: RoutingState,
    dag: CircuitDAG,
) -> None:
    swap_count = sum(
        1
        for operation
        in state.physical_operations
        if operation[0] == "swap"
    )

    logical_two_qubit_gate_count = sum(
        1
        for gate in dag.gates
        if gate.is_two_qubit
    )

    added_two_qubit_count = (
        3 * swap_count
    )

    print()
    print("===== 路由完成 =====")

    print(
        "已执行逻辑门数：",
        len(state.executed_gates),
    )

    print(
        "逻辑门总数：",
        len(dag.gates),
    )

    print(
        "逻辑双比特门数：",
        logical_two_qubit_gate_count,
    )

    print(
        "SWAP 数：",
        swap_count,
    )

    print(
        "估计附加双比特门数：",
        added_two_qubit_count,
    )

    print(
        "估计物理线路深度：",
        state.current_depth(),
    )

    print(
        "最终逻辑到物理映射：",
        state.logical_to_physical,
    )

    print()
    print("物理操作序列：")

    for index, operation in enumerate(
        state.physical_operations
    ):
        print(
            f"{index:2d}: {operation}"
        )


# ============================================================
# 9. 主程序
# ============================================================

def main() -> None:
    # 逻辑线路：
    #
    # H(0)
    # CX(0,4)
    # CX(1,3)
    # CX(4,2)
    # CX(0,3)

    gates = [
        Gate(
            gate_id=0,
            name="h",
            qubits=(0,),
        ),
        Gate(
            gate_id=1,
            name="cx",
            qubits=(0, 4),
        ),
        Gate(
            gate_id=2,
            name="cx",
            qubits=(1, 3),
        ),
        Gate(
            gate_id=3,
            name="cx",
            qubits=(4, 2),
        ),
        Gate(
            gate_id=4,
            name="cx",
            qubits=(0, 3),
        ),
    ]

    dag = CircuitDAG(
        num_logical_qubits=5,
        gates=gates,
    )

    # 物理拓扑：
    #
    # 0 -- 1 -- 2 -- 3 -- 4

    hardware = HardwareGraph(
        num_qubits=5,
        edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
        ],
    )

    # 固定恒等初始映射：
    #
    # q_0 -> 0
    # q_1 -> 1
    # q_2 -> 2
    # q_3 -> 3
    # q_4 -> 4

    initial_mapping = [
        0,
        1,
        2,
        3,
        4,
    ]

    state = run_shortest_path_baseline(
        dag=dag,
        hardware=hardware,
        initial_mapping=initial_mapping,
    )

    print_result(state, dag)


if __name__ == "__main__":
    main()
