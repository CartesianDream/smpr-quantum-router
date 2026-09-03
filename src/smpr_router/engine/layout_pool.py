from __future__ import annotations

import argparse
import csv
import io
import math
import time
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

try:
    from qiskit import transpile
except ModuleNotFoundError:
    # 映射池本身不依赖 Qiskit；仅 Qiskit 对照函数需要它。
    transpile = None

from routing_model import (
    CircuitDAG,
    Gate,
    HardwareGraph,
    RoutingState,
)

from repair_policy import run_router_with_repair

from qiskit_adapter import (
    TestCase,
    build_coupling_map,
    build_qiskit_circuit,
    build_quick_suite,
    count_logical_two_qubit_gates,
    weighted_qiskit_depth,
)


# ============================================================
# 1. 初始映射参数
# ============================================================

LOCAL_IMPROVEMENT_STEPS = 2000
REPAIR_THRESHOLD = 8


# ============================================================
# 2. 数据结构
# ============================================================

@dataclass(frozen=True)
class MappingCandidate:
    name: str
    mapping: tuple[int, ...]
    generation_ms: float


@dataclass
class MappingRunResult:
    instance: str
    method: str
    seed: int | None
    mapping: str
    swap_count: int | None
    added_two_qubit_count: int | None
    weighted_depth: int | None
    generation_ms: float
    routing_ms: float
    total_ms: float
    valid: bool
    error: str

    def score_key(self) -> tuple[int, int, float]:
        added = (
            self.added_two_qubit_count
            if self.added_two_qubit_count is not None
            else 10**9
        )
        depth = (
            self.weighted_depth
            if self.weighted_depth is not None
            else 10**9
        )
        return added, depth, self.total_ms


# ============================================================
# 3. 线路拓扑层与时间加权交互矩阵
# ============================================================

def compute_topological_levels(
    dag: CircuitDAG,
) -> list[int]:
    """
    门的拓扑层数：

        l(g)=0，若 g 无前驱；
        l(g)=1+max l(p)，p 为 g 的前驱。
    """

    levels = [0] * len(dag.gates)

    for gate in dag.gates:
        gate_id = gate.gate_id
        predecessors = dag.predecessors[gate_id]

        if predecessors:
            levels[gate_id] = (
                1 + max(levels[p] for p in predecessors)
            )

    return levels


def make_zero_matrix(n: int) -> list[list[float]]:
    return [
        [0.0 for _ in range(n)]
        for _ in range(n)
    ]


def build_interaction_matrices(
    dag: CircuitDAG,
) -> dict[str, list[list[float]]]:
    """
    构造三类逻辑交互权重：

        M_all   ：整条线路交互总次数；
        M_early ：强调线路前部；
        M_late  ：强调线路后部。
    """

    n = dag.num_logical_qubits
    levels = compute_topological_levels(dag)

    level_max = max(levels, default=0)
    tau = max(1.0, 0.2 * max(1, level_max))

    matrices = {
        "all": make_zero_matrix(n),
        "early": make_zero_matrix(n),
        "late": make_zero_matrix(n),
    }

    for gate in dag.gates:
        if not gate.is_two_qubit:
            continue

        q_i, q_j = gate.qubits
        level = levels[gate.gate_id]

        weights = {
            "all": 1.0,
            "early": math.exp(-level / tau),
            "late": math.exp(
                -(level_max - level) / tau
            ),
        }

        for name, weight in weights.items():
            matrices[name][q_i][q_j] += weight
            matrices[name][q_j][q_i] += weight

    return matrices


# ============================================================
# 4. 硬件距离与布局代价
# ============================================================

def hardware_distance(
    hardware: HardwareGraph,
    u: int,
    v: int,
) -> int:
    return len(hardware.shortest_path(u, v)) - 1


def physical_center_cost(
    hardware: HardwareGraph,
    physical: int,
) -> int:
    return sum(
        hardware_distance(hardware, physical, other)
        for other in range(hardware.num_qubits)
    )


def weighted_logical_degree(
    matrix: list[list[float]],
    logical: int,
) -> float:
    return sum(matrix[logical])


def layout_cost(
    mapping: list[int] | tuple[int, ...],
    matrix: list[list[float]],
    hardware: HardwareGraph,
) -> float:
    """
    C_layout(pi)
      = sum_{i<j} M_ij (d_H(pi(i),pi(j))-1).
    """

    n = len(mapping)
    cost = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            weight = matrix[i][j]

            if weight == 0.0:
                continue

            distance = hardware_distance(
                hardware,
                mapping[i],
                mapping[j],
            )

            cost += weight * (distance - 1)

    return cost


# ============================================================
# 5. 从交互图贪心构造布局
# ============================================================

def greedy_mapping_from_matrix(
    matrix: list[list[float]],
    hardware: HardwareGraph,
) -> list[int]:
    """
    1. 最活跃逻辑比特放到硬件中心；
    2. 每次选择与已放置集合交互最强的逻辑比特；
    3. 放到已放置加权距离最小的空闲物理点。
    """

    num_logical = len(matrix)

    if num_logical > hardware.num_qubits:
        raise ValueError(
            "逻辑比特数不能超过物理比特数"
        )

    logical_degrees = [
        weighted_logical_degree(matrix, q)
        for q in range(num_logical)
    ]

    first_logical = min(
        range(num_logical),
        key=lambda q: (
            -logical_degrees[q],
            q,
        ),
    )

    first_physical = min(
        range(hardware.num_qubits),
        key=lambda p: (
            physical_center_cost(hardware, p),
            -len(hardware.adjacency[p]),
            p,
        ),
    )

    mapping = [-1] * num_logical
    mapping[first_logical] = first_physical

    placed_logical = {first_logical}
    used_physical = {first_physical}

    while len(placed_logical) < num_logical:
        unplaced = [
            q
            for q in range(num_logical)
            if q not in placed_logical
        ]

        next_logical = min(
            unplaced,
            key=lambda q: (
                -sum(
                    matrix[q][placed]
                    for placed in placed_logical
                ),
                -logical_degrees[q],
                q,
            ),
        )

        free_physical = [
            p
            for p in range(hardware.num_qubits)
            if p not in used_physical
        ]

        next_physical = min(
            free_physical,
            key=lambda p: (
                sum(
                    matrix[next_logical][placed]
                    * hardware_distance(
                        hardware,
                        p,
                        mapping[placed],
                    )
                    for placed in placed_logical
                ),
                physical_center_cost(hardware, p),
                -len(hardware.adjacency[p]),
                p,
            ),
        )

        mapping[next_logical] = next_physical
        placed_logical.add(next_logical)
        used_physical.add(next_physical)

    return mapping


# ============================================================
# 6. 静态布局局部改进
# ============================================================

def local_improve_mapping(
    initial_mapping: list[int],
    matrix: list[list[float]],
    hardware: HardwareGraph,
    max_steps: int = LOCAL_IMPROVEMENT_STEPS,
) -> list[int]:
    """
    确定性 best-improvement：

    - 交换两个逻辑比特的位置；
    - 若有空闲物理点，把一个逻辑比特移到空点；
    - 只有布局代价严格下降时才接受。
    """

    mapping = list(initial_mapping)
    current_cost = layout_cost(
        mapping,
        matrix,
        hardware,
    )

    tolerance = 1e-12

    for _ in range(max_steps):
        best_mapping: list[int] | None = None
        best_cost = current_cost

        num_logical = len(mapping)

        # 交换两个逻辑比特的位置。
        for q_i in range(num_logical):
            for q_j in range(q_i + 1, num_logical):
                candidate = list(mapping)
                candidate[q_i], candidate[q_j] = (
                    candidate[q_j],
                    candidate[q_i],
                )

                cost = layout_cost(
                    candidate,
                    matrix,
                    hardware,
                )

                if cost < best_cost - tolerance:
                    best_cost = cost
                    best_mapping = candidate

        # 逻辑比特移动到空闲物理节点。
        occupied = set(mapping)
        empty_nodes = [
            p
            for p in range(hardware.num_qubits)
            if p not in occupied
        ]

        for logical in range(num_logical):
            for physical in empty_nodes:
                candidate = list(mapping)
                candidate[logical] = physical

                cost = layout_cost(
                    candidate,
                    matrix,
                    hardware,
                )

                if cost < best_cost - tolerance:
                    best_cost = cost
                    best_mapping = candidate

        if best_mapping is None:
            break

        mapping = best_mapping
        current_cost = best_cost

    return mapping


# ============================================================
# 7. 映射池
# ============================================================

def deduplicate_candidates(
    candidates: list[MappingCandidate],
) -> list[MappingCandidate]:
    """
    相同映射只保留最先生成的那个名称。
    """

    seen: set[tuple[int, ...]] = set()
    unique: list[MappingCandidate] = []

    for candidate in candidates:
        if candidate.mapping in seen:
            continue

        seen.add(candidate.mapping)
        unique.append(candidate)

    return unique


def generate_base_mapping_pool(
    case: TestCase,
) -> list[MappingCandidate]:
    """
    基础映射池：

        identity
        all
        early
        late
    """

    dag, hardware, identity = case.build()
    matrices = build_interaction_matrices(dag)

    candidates: list[MappingCandidate] = [
        MappingCandidate(
            name="identity",
            mapping=tuple(identity),
            generation_ms=0.0,
        )
    ]

    for matrix_name in ("all", "early", "late"):
        start = time.perf_counter()

        raw_mapping = greedy_mapping_from_matrix(
            matrices[matrix_name],
            hardware,
        )

        improved_mapping = local_improve_mapping(
            raw_mapping,
            matrices[matrix_name],
            hardware,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        candidates.append(
            MappingCandidate(
                name=matrix_name,
                mapping=tuple(improved_mapping),
                generation_ms=elapsed_ms,
            )
        )

    return deduplicate_candidates(candidates)


# ============================================================
# 8. 正反虚拟路由
# ============================================================

def reverse_test_case(
    case: TestCase,
) -> TestCase:
    """
    将门顺序反转，并由 CircuitDAG 重新构造反向依赖。
    这里只用于生成更好的映射，不保留物理输出。
    """

    return TestCase(
        name=f"{case.name}_reversed",
        num_qubits=case.num_qubits,
        edges=case.edges,
        gate_specs=tuple(reversed(case.gate_specs)),
        initial_mapping=case.initial_mapping,
        physical_qubits=case.physical_qubits,
    )


def quiet_full_route(
    case: TestCase,
    initial_mapping: tuple[int, ...],
    verbose: bool = False,
) -> RoutingState:
    dag, hardware, _ = case.build()

    output_buffer = io.StringIO()
    context = (
        nullcontext()
        if verbose
        else redirect_stdout(output_buffer)
    )

    with context:
        state = run_router_with_repair(
            dag=dag,
            hardware=hardware,
            initial_mapping=list(initial_mapping),
            repair_threshold=REPAIR_THRESHOLD,
        )

    return state


def forward_backward_polish(
    case: TestCase,
    candidate: MappingCandidate,
    verbose: bool,
) -> MappingCandidate:
    """
    pi_0
      --正向虚拟路由--> pi_f
      --反向虚拟路由--> pi_b

    最后返回 pi_b 作为真实正向路由起点。
    """

    start = time.perf_counter()

    forward_state = quiet_full_route(
        case,
        candidate.mapping,
        verbose=verbose,
    )

    reversed_case = reverse_test_case(case)

    backward_state = quiet_full_route(
        reversed_case,
        tuple(forward_state.logical_to_physical),
        verbose=verbose,
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    return MappingCandidate(
        name=f"{candidate.name}_fb",
        mapping=tuple(
            backward_state.logical_to_physical
        ),
        generation_ms=(
            candidate.generation_ms + elapsed_ms
        ),
    )


def generate_complete_mapping_pool(
    case: TestCase,
    use_forward_backward: bool,
    verbose: bool,
) -> list[MappingCandidate]:
    base = generate_base_mapping_pool(case)

    if not use_forward_backward:
        return base

    polished = [
        forward_backward_polish(
            case,
            candidate,
            verbose=verbose,
        )
        for candidate in base
    ]

    return deduplicate_candidates(base + polished)


# ============================================================
# 9. 运行我们的完整路由器
# ============================================================

def run_mapping_candidate(
    case: TestCase,
    candidate: MappingCandidate,
    verbose: bool,
) -> MappingRunResult:
    dag, hardware, _ = case.build()

    output_buffer = io.StringIO()
    context = (
        nullcontext()
        if verbose
        else redirect_stdout(output_buffer)
    )

    start = time.perf_counter()

    try:
        with context:
            state = run_router_with_repair(
                dag=dag,
                hardware=hardware,
                initial_mapping=list(candidate.mapping),
                repair_threshold=REPAIR_THRESHOLD,
            )

        routing_ms = (
            time.perf_counter() - start
        ) * 1000.0

        state.assert_valid(dag, hardware)

        if len(state.executed_gates) != len(dag.gates):
            raise AssertionError(
                "路由结束后仍有逻辑门未执行"
            )

        swap_count = sum(
            operation[0] == "swap"
            for operation in state.physical_operations
        )

        return MappingRunResult(
            instance=case.name,
            method=f"independent rollout:{candidate.name}",
            seed=None,
            mapping=str(list(candidate.mapping)),
            swap_count=swap_count,
            added_two_qubit_count=3 * swap_count,
            weighted_depth=state.current_depth(),
            generation_ms=candidate.generation_ms,
            routing_ms=routing_ms,
            total_ms=(
                candidate.generation_ms + routing_ms
            ),
            valid=True,
            error="",
        )

    except Exception as exc:
        routing_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return MappingRunResult(
            instance=case.name,
            method=f"independent rollout:{candidate.name}",
            seed=None,
            mapping=str(list(candidate.mapping)),
            swap_count=None,
            added_two_qubit_count=None,
            weighted_depth=None,
            generation_ms=candidate.generation_ms,
            routing_ms=routing_ms,
            total_ms=(
                candidate.generation_ms + routing_ms
            ),
            valid=False,
            error=f"{type(exc).__name__}: {exc}",
        )


# ============================================================
# 10. Qiskit SABRE 自动初始映射基线
# ============================================================

def run_qiskit_auto_layout(
    case: TestCase,
    seed: int,
) -> MappingRunResult:
    circuit = build_qiskit_circuit(case)
    coupling_map = build_coupling_map(case)

    start = time.perf_counter()

    try:
        routed = transpile(
            circuit,
            coupling_map=coupling_map,
            layout_method="sabre",
            routing_method="sabre",
            optimization_level=0,
            seed_transpiler=seed,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        operations = dict(routed.count_ops())
        swap_count = int(operations.get("swap", 0))

        return MappingRunResult(
            instance=case.name,
            method="QiskitSABRE:auto",
            seed=seed,
            mapping=str(routed.layout.initial_layout),
            swap_count=swap_count,
            added_two_qubit_count=3 * swap_count,
            weighted_depth=weighted_qiskit_depth(routed),
            generation_ms=0.0,
            routing_ms=elapsed_ms,
            total_ms=elapsed_ms,
            valid=True,
            error="",
        )

    except Exception as exc:
        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return MappingRunResult(
            instance=case.name,
            method="QiskitSABRE:auto",
            seed=seed,
            mapping="",
            swap_count=None,
            added_two_qubit_count=None,
            weighted_depth=None,
            generation_ms=0.0,
            routing_ms=elapsed_ms,
            total_ms=elapsed_ms,
            valid=False,
            error=f"{type(exc).__name__}: {exc}",
        )


# ============================================================
# 11. CSV 与汇总
# ============================================================

CSV_FIELDS = (
    "instance",
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


def save_csv(
    results: list[MappingRunResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        for result in results:
            row = {
                field: getattr(result, field)
                for field in CSV_FIELDS
            }

            for field in (
                "generation_ms",
                "routing_ms",
                "total_ms",
            ):
                row[field] = f"{getattr(result, field):.3f}"

            writer.writerow(row)


def best_valid(
    rows: list[MappingRunResult],
) -> MappingRunResult:
    valid_rows = [row for row in rows if row.valid]

    if not valid_rows:
        return rows[0]

    return min(
        valid_rows,
        key=lambda row: (
            row.score_key(),
            row.seed if row.seed is not None else -1,
        ),
    )


def print_summary(
    cases: list[TestCase],
    results: list[MappingRunResult],
) -> None:
    print()
    print("=" * 102)
    print("端到端初始映射池实验")
    print("完整结果比较：附加双比特门优先，平局时加权深度优先")
    print("=" * 102)

    for case in cases:
        print()
        print(f"[{case.name}]")

        rows = [
            row
            for row in results
            if row.instance == case.name
        ]

        independent_rollout_rows = [
            row
            for row in rows
            if row.method.startswith("independent rollout:")
        ]

        qiskit_rows = [
            row
            for row in rows
            if row.method == "QiskitSABRE:auto"
        ]

        print(
            f"{'方法':<25}"
            f"{'seed':>7}"
            f"{'SWAP':>8}"
            f"{'附加2q':>10}"
            f"{'加权深度':>12}"
            f"{'生成/ms':>11}"
            f"{'路由/ms':>11}"
            f"{'总计/ms':>11}"
        )

        print("-" * 102)

        for row in independent_rollout_rows:
            print(
                f"{row.method:<25}"
                f"{'-':>7}"
                f"{str(row.swap_count):>8}"
                f"{str(row.added_two_qubit_count):>10}"
                f"{str(row.weighted_depth):>12}"
                f"{row.generation_ms:>11.3f}"
                f"{row.routing_ms:>11.3f}"
                f"{row.total_ms:>11.3f}"
            )

            if not row.valid:
                print("  错误：", row.error)

        if independent_rollout_rows:
            pool_best = best_valid(independent_rollout_rows)

            print("-" * 102)
            print(
                f"{'independent rollout:pool_best':<25}"
                f"{'-':>7}"
                f"{str(pool_best.swap_count):>8}"
                f"{str(pool_best.added_two_qubit_count):>10}"
                f"{str(pool_best.weighted_depth):>12}"
                f"{pool_best.generation_ms:>11.3f}"
                f"{pool_best.routing_ms:>11.3f}"
                f"{pool_best.total_ms:>11.3f}"
            )

        if qiskit_rows:
            qiskit_best = best_valid(qiskit_rows)

            print(
                f"{'QiskitSABRE:auto(best)':<25}"
                f"{str(qiskit_best.seed):>7}"
                f"{str(qiskit_best.swap_count):>8}"
                f"{str(qiskit_best.added_two_qubit_count):>10}"
                f"{str(qiskit_best.weighted_depth):>12}"
                f"{qiskit_best.generation_ms:>11.3f}"
                f"{qiskit_best.routing_ms:>11.3f}"
                f"{qiskit_best.total_ms:>11.3f}"
            )


# ============================================================
# 12. 主程序
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "第七阶段：all/early/late 初始映射池、"
            "局部改进、正反虚拟路由与端到端比较"
        )
    )

    parser.add_argument(
        "--qiskit-seeds",
        type=int,
        default=5,
        help="Qiskit SABRE 自动布局随机种子数，默认 5",
    )

    parser.add_argument(
        "--skip-forward-backward",
        action="store_true",
        help="快速测试时跳过正反虚拟路由",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示虚拟路由和真实路由的逐步日志",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/layout_pool.csv"
        ),
        help="CSV 输出路径",
    )

    args = parser.parse_args()

    if args.qiskit_seeds <= 0:
        raise ValueError("--qiskit-seeds 必须为正整数")

    cases = build_quick_suite()
    results: list[MappingRunResult] = []

    use_forward_backward = (
        not args.skip_forward_backward
    )

    print("===== 第七阶段：端到端初始映射池实验 =====")
    print("实例数：", len(cases))
    print("使用正反虚拟路由：", use_forward_backward)
    print("Qiskit SABRE 种子数：", args.qiskit_seeds)

    for index, case in enumerate(cases, start=1):
        print()
        print(
            f"[{index}/{len(cases)}] "
            f"正在运行：{case.name}"
        )

        pool = generate_complete_mapping_pool(
            case=case,
            use_forward_backward=use_forward_backward,
            verbose=args.verbose,
        )

        print("  生成的不同映射数：", len(pool))

        for candidate in pool:
            print(
                f"  - independent rollout:{candidate.name} "
                f"{list(candidate.mapping)}"
            )

            results.append(
                run_mapping_candidate(
                    case=case,
                    candidate=candidate,
                    verbose=args.verbose,
                )
            )

        print("  - QiskitSABRE:auto")

        for seed in range(args.qiskit_seeds):
            results.append(
                run_qiskit_auto_layout(
                    case=case,
                    seed=seed,
                )
            )

    save_csv(
        results=results,
        output_path=args.output,
    )

    print_summary(
        cases=cases,
        results=results,
    )

    print()
    print("CSV 已保存到：", args.output.resolve())


if __name__ == "__main__":
    main()
