from __future__ import annotations

import argparse
import csv
import io
import time
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.transpiler import CouplingMap
except ModuleNotFoundError:
    # historical development 的纯 independent rollout 诊断不调用 Qiskit。允许在只安装
    # Python 标准库的环境中导入 TestCase 和评价工具。
    QuantumCircuit = None
    transpile = None
    CouplingMap = None

from routing_model import (
    CircuitDAG,
    Gate,
    HardwareGraph,
    RoutingState,
    run_shortest_path_baseline,
)

from one_step_policy import run_one_step_router
from two_step_beam import run_two_step_beam_router
from adaptive_beam import run_adaptive_router
from repair_policy import run_router_with_repair


# ============================================================
# 1. 测试实例
# ============================================================

@dataclass(frozen=True)
class TestCase:
    name: str
    num_qubits: int
    edges: tuple[tuple[int, int], ...]
    gate_specs: tuple[tuple, ...]
    initial_mapping: tuple[int, ...]
    # 旧数据中逻辑比特数与物理节点数相同。大型芯片实验必须允许
    # logical_qubits << physical_qubits；None 保持所有旧脚本兼容。
    physical_qubits: int | None = None

    @property
    def num_physical_qubits(self) -> int:
        return (
            self.num_qubits
            if self.physical_qubits is None
            else self.physical_qubits
        )

    def build(self) -> tuple[CircuitDAG, HardwareGraph, list[int]]:
        gates = [
            Gate(
                gate_id=index,
                name=str(spec[0]),
                qubits=tuple(int(q) for q in spec[1:]),
            )
            for index, spec in enumerate(self.gate_specs)
        ]

        dag = CircuitDAG(
            num_logical_qubits=self.num_qubits,
            gates=gates,
        )

        hardware = HardwareGraph(
            num_qubits=self.num_physical_qubits,
            edges=list(self.edges),
        )

        return dag, hardware, list(self.initial_mapping)


def line_edges(n: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, i + 1) for i in range(n - 1))


def ring_edges(n: int) -> tuple[tuple[int, int], ...]:
    return line_edges(n) + ((n - 1, 0),)


def build_quick_suite() -> list[TestCase]:
    return [
        TestCase(
            name="paper_example_line5",
            num_qubits=5,
            edges=line_edges(5),
            gate_specs=(
                ("h", 0),
                ("cx", 0, 4),
                ("cx", 1, 3),
                ("cx", 4, 2),
                ("cx", 0, 3),
            ),
            initial_mapping=(0, 1, 2, 3, 4),
        ),
        TestCase(
            name="far_interactions_line5",
            num_qubits=5,
            edges=line_edges(5),
            gate_specs=(
                ("cx", 0, 4),
                ("cx", 0, 3),
                ("cx", 1, 4),
                ("cx", 2, 0),
                ("cx", 3, 1),
            ),
            initial_mapping=(0, 1, 2, 3, 4),
        ),
        TestCase(
            name="parallel_front_line6",
            num_qubits=6,
            edges=line_edges(6),
            gate_specs=(
                ("cx", 0, 5),
                ("cx", 1, 4),
                ("cx", 2, 3),
                ("cx", 0, 1),
                ("cx", 4, 5),
                ("cx", 2, 4),
                ("cx", 1, 3),
            ),
            initial_mapping=(0, 1, 2, 3, 4, 5),
        ),
        TestCase(
            name="alternating_line6",
            num_qubits=6,
            edges=line_edges(6),
            gate_specs=(
                ("h", 0),
                ("h", 5),
                ("cx", 0, 5),
                ("cx", 1, 4),
                ("cx", 0, 2),
                ("cx", 3, 5),
                ("cx", 1, 2),
                ("cx", 4, 5),
                ("cx", 0, 3),
            ),
            initial_mapping=(0, 1, 2, 3, 4, 5),
        ),
        TestCase(
            name="opposite_pairs_ring6",
            num_qubits=6,
            edges=ring_edges(6),
            gate_specs=(
                ("cx", 0, 3),
                ("cx", 1, 4),
                ("cx", 2, 5),
                ("cx", 0, 1),
                ("cx", 3, 4),
                ("cx", 2, 4),
                ("cx", 1, 5),
            ),
            initial_mapping=(0, 1, 2, 3, 4, 5),
        ),
    ]


# ============================================================
# 2. 统一评价指标
# ============================================================

@dataclass
class RunResult:
    instance: str
    algorithm: str
    seed: int | None
    num_qubits: int
    logical_gate_count: int
    logical_two_qubit_count: int
    swap_count: int | None
    added_two_qubit_count: int | None
    weighted_depth: int | None
    total_physical_operations: int | None
    runtime_ms: float
    valid: bool
    error: str

    def score_key(self) -> tuple[int, int]:
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
        return added, depth


def count_logical_two_qubit_gates(case: TestCase) -> int:
    return sum(len(spec) == 3 for spec in case.gate_specs)


def weighted_qiskit_depth(circuit: QuantumCircuit) -> int:
    depths = [0] * circuit.num_qubits

    for instruction in circuit.data:
        operation = instruction.operation
        qubits = instruction.qubits

        indices = [
            circuit.find_bit(qubit).index
            for qubit in qubits
        ]

        duration = 3 if operation.name == "swap" else 1

        if len(indices) == 1:
            q = indices[0]
            depths[q] += duration

        elif len(indices) == 2:
            u, v = indices
            new_depth = max(depths[u], depths[v]) + duration
            depths[u] = new_depth
            depths[v] = new_depth

        elif len(indices) == 0:
            continue

        else:
            raise ValueError(
                f"暂不支持 {len(indices)} 比特物理门："
                f"{operation.name}"
            )

    return max(depths, default=0)


# ============================================================
# 3. 自实现路由器包装
# ============================================================

OwnRunner = Callable[
    [CircuitDAG, HardwareGraph, list[int]],
    RoutingState,
]


def run_own_algorithm(
    case: TestCase,
    algorithm_name: str,
    runner: OwnRunner,
    verbose: bool,
) -> RunResult:
    dag, hardware, initial_mapping = case.build()

    output_buffer = io.StringIO()
    context = nullcontext() if verbose else redirect_stdout(output_buffer)

    start = time.perf_counter()

    try:
        with context:
            state = runner(
                dag,
                hardware,
                initial_mapping,
            )

        runtime_ms = (time.perf_counter() - start) * 1000.0

        state.assert_valid(dag, hardware)

        if len(state.executed_gates) != len(dag.gates):
            raise AssertionError("算法结束时仍有逻辑门未执行")

        swap_count = sum(
            operation[0] == "swap"
            for operation in state.physical_operations
        )

        return RunResult(
            instance=case.name,
            algorithm=algorithm_name,
            seed=None,
            num_qubits=case.num_qubits,
            logical_gate_count=len(case.gate_specs),
            logical_two_qubit_count=(
                count_logical_two_qubit_gates(case)
            ),
            swap_count=swap_count,
            added_two_qubit_count=3 * swap_count,
            weighted_depth=state.current_depth(),
            total_physical_operations=len(
                state.physical_operations
            ),
            runtime_ms=runtime_ms,
            valid=True,
            error="",
        )

    except Exception as exc:
        runtime_ms = (time.perf_counter() - start) * 1000.0

        return RunResult(
            instance=case.name,
            algorithm=algorithm_name,
            seed=None,
            num_qubits=case.num_qubits,
            logical_gate_count=len(case.gate_specs),
            logical_two_qubit_count=(
                count_logical_two_qubit_gates(case)
            ),
            swap_count=None,
            added_two_qubit_count=None,
            weighted_depth=None,
            total_physical_operations=None,
            runtime_ms=runtime_ms,
            valid=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def repair_policy_runner(
    dag: CircuitDAG,
    hardware: HardwareGraph,
    initial_mapping: list[int],
) -> RoutingState:
    return run_router_with_repair(
        dag=dag,
        hardware=hardware,
        initial_mapping=initial_mapping,
        repair_threshold=8,
    )


OWN_ALGORITHMS: tuple[tuple[str, OwnRunner], ...] = (
    ("ShortestPath", run_shortest_path_baseline),
    ("OneStepPotential", run_one_step_router),
    ("TwoStepBeam", run_two_step_beam_router),
    ("AdaptiveBeam", run_adaptive_router),
    ("AdaptiveBeamRepair", repair_policy_runner),
)


# ============================================================
# 4. Qiskit SABRE 固定布局基线
# ============================================================

def build_qiskit_circuit(case: TestCase) -> QuantumCircuit:
    circuit = QuantumCircuit(case.num_qubits)

    for spec in case.gate_specs:
        name = spec[0]

        if name == "h":
            circuit.h(spec[1])
        elif name == "x":
            circuit.x(spec[1])
        elif name == "cx":
            circuit.cx(spec[1], spec[2])
        else:
            raise ValueError(f"不支持的逻辑门类型：{name}")

    return circuit


def build_coupling_map(case: TestCase) -> CouplingMap:
    directed_edges: list[tuple[int, int]] = []

    for u, v in case.edges:
        directed_edges.append((u, v))
        directed_edges.append((v, u))

    return CouplingMap(directed_edges)


def run_qiskit_sabre(
    case: TestCase,
    seed: int,
) -> RunResult:
    circuit = build_qiskit_circuit(case)
    coupling_map = build_coupling_map(case)

    start = time.perf_counter()

    try:
        routed = transpile(
            circuit,
            coupling_map=coupling_map,
            initial_layout=list(case.initial_mapping),
            routing_method="sabre",
            optimization_level=0,
            seed_transpiler=seed,
        )

        runtime_ms = (time.perf_counter() - start) * 1000.0

        operations = dict(routed.count_ops())
        swap_count = int(operations.get("swap", 0))

        return RunResult(
            instance=case.name,
            algorithm="QiskitSABRE",
            seed=seed,
            num_qubits=case.num_qubits,
            logical_gate_count=len(case.gate_specs),
            logical_two_qubit_count=(
                count_logical_two_qubit_gates(case)
            ),
            swap_count=swap_count,
            added_two_qubit_count=3 * swap_count,
            weighted_depth=weighted_qiskit_depth(routed),
            total_physical_operations=routed.size(),
            runtime_ms=runtime_ms,
            valid=True,
            error="",
        )

    except Exception as exc:
        runtime_ms = (time.perf_counter() - start) * 1000.0

        return RunResult(
            instance=case.name,
            algorithm="QiskitSABRE",
            seed=seed,
            num_qubits=case.num_qubits,
            logical_gate_count=len(case.gate_specs),
            logical_two_qubit_count=(
                count_logical_two_qubit_gates(case)
            ),
            swap_count=None,
            added_two_qubit_count=None,
            weighted_depth=None,
            total_physical_operations=None,
            runtime_ms=runtime_ms,
            valid=False,
            error=f"{type(exc).__name__}: {exc}",
        )


# ============================================================
# 5. CSV 和终端汇总
# ============================================================

CSV_FIELDS = (
    "instance",
    "algorithm",
    "seed",
    "num_qubits",
    "logical_gate_count",
    "logical_two_qubit_count",
    "swap_count",
    "added_two_qubit_count",
    "weighted_depth",
    "total_physical_operations",
    "runtime_ms",
    "valid",
    "error",
)


def save_csv(
    results: list[RunResult],
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

            row["runtime_ms"] = (
                f"{result.runtime_ms:.3f}"
            )

            writer.writerow(row)


def best_result(
    rows: list[RunResult],
) -> RunResult:
    valid_rows = [row for row in rows if row.valid]

    if not valid_rows:
        return rows[0]

    return min(
        valid_rows,
        key=lambda row: (
            row.score_key(),
            row.runtime_ms,
            row.seed if row.seed is not None else -1,
        ),
    )


def print_summary(
    cases: list[TestCase],
    results: list[RunResult],
) -> None:
    print()
    print("=" * 86)
    print("固定初始映射实验汇总")
    print("比较顺序：附加双比特门数优先，平局时加权深度优先")
    print("=" * 86)

    for case in cases:
        print()
        print(f"[{case.name}]")

        rows_for_case = [
            row
            for row in results
            if row.instance == case.name
        ]

        algorithm_names = [
            name for name, _ in OWN_ALGORITHMS
        ] + ["QiskitSABRE"]

        print(
            f"{'算法':<22}"
            f"{'seed':>7}"
            f"{'SWAP':>8}"
            f"{'附加2q':>10}"
            f"{'加权深度':>12}"
            f"{'时间/ms':>12}"
            f"{'合法':>8}"
        )

        print("-" * 86)

        for algorithm_name in algorithm_names:
            algorithm_rows = [
                row
                for row in rows_for_case
                if row.algorithm == algorithm_name
            ]

            if not algorithm_rows:
                continue

            row = best_result(algorithm_rows)

            seed_text = (
                str(row.seed)
                if row.seed is not None
                else "-"
            )

            swap_text = (
                str(row.swap_count)
                if row.swap_count is not None
                else "-"
            )

            added_text = (
                str(row.added_two_qubit_count)
                if row.added_two_qubit_count is not None
                else "-"
            )

            depth_text = (
                str(row.weighted_depth)
                if row.weighted_depth is not None
                else "-"
            )

            print(
                f"{algorithm_name:<22}"
                f"{seed_text:>7}"
                f"{swap_text:>8}"
                f"{added_text:>10}"
                f"{depth_text:>12}"
                f"{row.runtime_ms:>12.3f}"
                f"{str(row.valid):>8}"
            )

            if not row.valid:
                print(f"  错误：{row.error}")


# ============================================================
# 6. 主程序
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "第六阶段：固定初始映射批量比较 "
            "ShortestPath、势函数路由、Beam、repair 和 Qiskit SABRE"
        )
    )

    parser.add_argument(
        "--qiskit-seeds",
        type=int,
        default=10,
        help="每个实例运行多少个 Qiskit SABRE 随机种子，默认 10",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示自实现算法的完整逐步决策日志",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/qiskit_adapter.csv"
        ),
        help="CSV 输出路径",
    )

    args = parser.parse_args()

    if args.qiskit_seeds <= 0:
        raise ValueError("--qiskit-seeds 必须为正整数")

    cases = build_quick_suite()
    results: list[RunResult] = []

    print("===== 第六阶段：固定初始映射批量实验 =====")
    print("实例数：", len(cases))
    print("Qiskit SABRE 种子数：", args.qiskit_seeds)

    for case_index, case in enumerate(cases, start=1):
        print()
        print(
            f"[{case_index}/{len(cases)}] "
            f"正在运行：{case.name}"
        )

        for algorithm_name, runner in OWN_ALGORITHMS:
            print(f"  - {algorithm_name}")

            result = run_own_algorithm(
                case=case,
                algorithm_name=algorithm_name,
                runner=runner,
                verbose=args.verbose,
            )

            results.append(result)

        print("  - QiskitSABRE")

        for seed in range(args.qiskit_seeds):
            results.append(
                run_qiskit_sabre(
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
