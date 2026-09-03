from __future__ import annotations

import argparse
import json
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path


# ============================================================
# 1. 数据结构
# ============================================================

@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    split: str
    seed: int
    num_qubits: int
    topology: str
    circuit_mode: str
    edges: tuple[tuple[int, int], ...]
    gate_specs: tuple[tuple, ...]
    initial_mapping: tuple[int, ...]


# ============================================================
# 2. 硬件拓扑
# ============================================================

def normalize_edges(
    edges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    normalized = {
        tuple(sorted((int(u), int(v))))
        for u, v in edges
        if int(u) != int(v)
    }

    return tuple(sorted(normalized))


def line_edges(n: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, i + 1) for i in range(n - 1))


def ring_edges(n: int) -> tuple[tuple[int, int], ...]:
    return normalize_edges(
        list(line_edges(n)) + [(n - 1, 0)]
    )


def grid_edges(rows: int, cols: int) -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []

    def node(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                edges.append(
                    (node(r, c), node(r + 1, c))
                )

            if c + 1 < cols:
                edges.append(
                    (node(r, c), node(r, c + 1))
                )

    return normalize_edges(edges)


def random_connected_edges(
    n: int,
    rng: random.Random,
    extra_edge_ratio: float = 0.35,
) -> tuple[tuple[int, int], ...]:
    """
    先随机生成一棵树保证连通，再加入额外边。
    """

    vertices = list(range(n))
    rng.shuffle(vertices)

    edges: set[tuple[int, int]] = set()

    for index in range(1, n):
        child = vertices[index]
        parent = vertices[rng.randrange(index)]
        edges.add(tuple(sorted((child, parent))))

    all_possible = [
        (u, v)
        for u in range(n)
        for v in range(u + 1, n)
        if (u, v) not in edges
    ]

    rng.shuffle(all_possible)

    extra_count = max(
        1,
        round(extra_edge_ratio * n),
    )

    for edge in all_possible[:extra_count]:
        edges.add(edge)

    return tuple(sorted(edges))


def make_topology(
    topology: str,
    n: int,
    rng: random.Random,
) -> tuple[tuple[int, int], ...]:
    if topology == "line":
        return line_edges(n)

    if topology == "ring":
        return ring_edges(n)

    if topology == "grid":
        if n % 2 != 0:
            raise ValueError(
                "当前 grid 生成器要求偶数个量子比特"
            )

        return grid_edges(2, n // 2)

    if topology == "random":
        return random_connected_edges(n, rng)

    raise ValueError(f"未知拓扑：{topology}")


# ============================================================
# 3. 图工具
# ============================================================

def adjacency_list(
    n: int,
    edges: tuple[tuple[int, int], ...],
) -> list[set[int]]:
    adjacency = [set() for _ in range(n)]

    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    return adjacency


def is_connected(
    n: int,
    edges: tuple[tuple[int, int], ...],
) -> bool:
    adjacency = adjacency_list(n, edges)

    visited = {0}
    queue: deque[int] = deque([0])

    while queue:
        current = queue.popleft()

        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == n


def all_pairs_distances(
    n: int,
    edges: tuple[tuple[int, int], ...],
) -> list[list[int]]:
    adjacency = adjacency_list(n, edges)
    distances = [[10**9] * n for _ in range(n)]

    for source in range(n):
        distances[source][source] = 0
        queue: deque[int] = deque([source])

        while queue:
            current = queue.popleft()

            for neighbor in adjacency[current]:
                if distances[source][neighbor] == 10**9:
                    distances[source][neighbor] = (
                        distances[source][current] + 1
                    )
                    queue.append(neighbor)

    return distances


# ============================================================
# 4. 线路生成
# ============================================================

def add_initial_h_gates(
    gates: list[tuple],
    n: int,
    rng: random.Random,
) -> None:
    """
    少量单比特门用于检查 Drain，
    不让线路完全退化为纯 CX 序列。
    """

    count = max(1, n // 4)
    selected = rng.sample(range(n), count)

    for q in selected:
        gates.append(("h", q))


def choose_pair_uniform(
    n: int,
    rng: random.Random,
) -> tuple[int, int]:
    q_i, q_j = rng.sample(range(n), 2)
    return q_i, q_j


def choose_pair_by_distance(
    distances: list[list[int]],
    rng: random.Random,
    prefer_far: bool,
) -> tuple[int, int]:
    n = len(distances)
    pairs = [
        (distances[i][j], i, j)
        for i in range(n)
        for j in range(i + 1, n)
    ]

    pairs.sort()

    band = max(1, len(pairs) // 3)

    if prefer_far:
        candidates = pairs[-band:]
    else:
        candidates = pairs[:band]

    _, q_i, q_j = rng.choice(candidates)

    if rng.random() < 0.5:
        return q_i, q_j

    return q_j, q_i


def generate_gate_specs(
    n: int,
    num_two_qubit_gates: int,
    circuit_mode: str,
    edges: tuple[tuple[int, int], ...],
    rng: random.Random,
) -> tuple[tuple, ...]:
    distances = all_pairs_distances(n, edges)
    gates: list[tuple] = []

    add_initial_h_gates(gates, n, rng)

    if circuit_mode == "uniform":
        for _ in range(num_two_qubit_gates):
            q_i, q_j = choose_pair_uniform(n, rng)
            gates.append(("cx", q_i, q_j))

    elif circuit_mode == "far":
        for _ in range(num_two_qubit_gates):
            q_i, q_j = choose_pair_by_distance(
                distances,
                rng,
                prefer_far=True,
            )
            gates.append(("cx", q_i, q_j))

    elif circuit_mode == "phase_shift":
        half = num_two_qubit_gates // 2

        # 前半段偏好远距离交互。
        for _ in range(half):
            q_i, q_j = choose_pair_by_distance(
                distances,
                rng,
                prefer_far=True,
            )
            gates.append(("cx", q_i, q_j))

        # 后半段偏好局部交互。
        for _ in range(num_two_qubit_gates - half):
            q_i, q_j = choose_pair_by_distance(
                distances,
                rng,
                prefer_far=False,
            )
            gates.append(("cx", q_i, q_j))

    elif circuit_mode == "alternating":
        for index in range(num_two_qubit_gates):
            q_i, q_j = choose_pair_by_distance(
                distances,
                rng,
                prefer_far=(index % 2 == 0),
            )
            gates.append(("cx", q_i, q_j))

    elif circuit_mode == "parallel_layers":
        """
        每层尽量使用互不相交的逻辑比特，
        制造较宽的 DAG 前沿。
        """
        produced = 0

        while produced < num_two_qubit_gates:
            vertices = list(range(n))
            rng.shuffle(vertices)

            layer_pairs = [
                (vertices[i], vertices[i + 1])
                for i in range(0, n - 1, 2)
            ]

            rng.shuffle(layer_pairs)

            for q_i, q_j in layer_pairs:
                gates.append(("cx", q_i, q_j))
                produced += 1

                if produced >= num_two_qubit_gates:
                    break

    else:
        raise ValueError(
            f"未知线路模式：{circuit_mode}"
        )

    return tuple(gates)


# ============================================================
# 5. 单个 split 生成
# ============================================================

TOPOLOGIES = (
    "line",
    "ring",
    "grid",
    "random",
)

CIRCUIT_MODES = (
    "uniform",
    "far",
    "phase_shift",
    "alternating",
    "parallel_layers",
)

QUBIT_SIZES = (6, 8, 10)
GATE_COUNTS = (18, 30, 45)


def generate_split(
    split: str,
    count: int,
    master_seed: int,
) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []

    for index in range(count):
        case_seed = (
            master_seed
            + 100_000 * {
                "dev": 1,
                "validation": 2,
                "test": 3,
            }[split]
            + index
        )

        rng = random.Random(case_seed)

        n = QUBIT_SIZES[index % len(QUBIT_SIZES)]
        topology = TOPOLOGIES[index % len(TOPOLOGIES)]

        circuit_mode = CIRCUIT_MODES[
            (index // len(TOPOLOGIES))
            % len(CIRCUIT_MODES)
        ]

        gate_count = GATE_COUNTS[
            (index // (
                len(TOPOLOGIES)
                * len(CIRCUIT_MODES)
            ))
            % len(GATE_COUNTS)
        ]

        edges = make_topology(
            topology,
            n,
            rng,
        )

        gates = generate_gate_specs(
            n=n,
            num_two_qubit_gates=gate_count,
            circuit_mode=circuit_mode,
            edges=edges,
            rng=rng,
        )

        case = BenchmarkCase(
            name=(
                f"{split}_{index:03d}_"
                f"{topology}_n{n}_"
                f"{circuit_mode}_g{gate_count}"
            ),
            split=split,
            seed=case_seed,
            num_qubits=n,
            topology=topology,
            circuit_mode=circuit_mode,
            edges=edges,
            gate_specs=gates,
            initial_mapping=tuple(range(n)),
        )

        validate_case(case)
        cases.append(case)

    return cases


# ============================================================
# 6. 验证与保存
# ============================================================

def validate_case(case: BenchmarkCase) -> None:
    n = case.num_qubits

    if n <= 1:
        raise ValueError("量子比特数必须大于 1")

    if not is_connected(n, case.edges):
        raise ValueError(
            f"{case.name} 的硬件图不连通"
        )

    for u, v in case.edges:
        if not (
            0 <= u < n
            and 0 <= v < n
            and u != v
        ):
            raise ValueError(
                f"{case.name} 存在非法硬件边 {(u, v)}"
            )

    if len(case.initial_mapping) != n:
        raise ValueError(
            f"{case.name} 初始映射长度错误"
        )

    if set(case.initial_mapping) != set(range(n)):
        raise ValueError(
            f"{case.name} 初始映射不是排列"
        )

    for gate in case.gate_specs:
        if gate[0] not in {"h", "x", "cx"}:
            raise ValueError(
                f"{case.name} 存在不支持的门 {gate[0]}"
            )

        expected_length = 2 if gate[0] in {"h", "x"} else 3

        if len(gate) != expected_length:
            raise ValueError(
                f"{case.name} 门格式错误：{gate}"
            )

        qubits = gate[1:]

        if any(
            not 0 <= int(q) < n
            for q in qubits
        ):
            raise ValueError(
                f"{case.name} 门量子比特越界：{gate}"
            )

        if len(qubits) == 2 and qubits[0] == qubits[1]:
            raise ValueError(
                f"{case.name} 存在自作用双比特门：{gate}"
            )


def case_to_json(case: BenchmarkCase) -> dict:
    data = asdict(case)

    data["edges"] = [
        list(edge)
        for edge in case.edges
    ]

    data["gate_specs"] = [
        list(gate)
        for gate in case.gate_specs
    ]

    data["initial_mapping"] = list(
        case.initial_mapping
    )

    return data


def save_split(
    cases: list[BenchmarkCase],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 1,
        "case_count": len(cases),
        "cases": [
            case_to_json(case)
            for case in cases
        ],
    }

    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def print_summary(
    all_cases: list[BenchmarkCase],
) -> None:
    print()
    print("=" * 80)
    print("Benchmark 数据集汇总")
    print("=" * 80)

    for field_name, title in (
        ("split", "数据划分"),
        ("topology", "硬件拓扑"),
        ("circuit_mode", "线路模式"),
        ("num_qubits", "量子比特数"),
    ):
        counter = Counter(
            getattr(case, field_name)
            for case in all_cases
        )

        print(f"{title}：{dict(sorted(counter.items(), key=lambda x: str(x[0])))}")

    print()
    print("前 5 个实例：")

    for case in all_cases[:5]:
        two_qubit_count = sum(
            gate[0] == "cx"
            for gate in case.gate_specs
        )

        print(
            f"  {case.name}: "
            f"n={case.num_qubits}, "
            f"|E_H|={len(case.edges)}, "
            f"CX={two_qubit_count}"
        )


# ============================================================
# 7. 主程序
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "historical development：生成可复现的开发集、验证集和测试集"
        )
    )

    parser.add_argument(
        "--master-seed",
        type=int,
        default=20260715,
        help="主随机种子",
    )

    parser.add_argument(
        "--dev-count",
        type=int,
        default=20,
        help="开发集实例数，默认 20",
    )

    parser.add_argument(
        "--validation-count",
        type=int,
        default=10,
        help="验证集实例数，默认 10",
    )

    parser.add_argument(
        "--test-count",
        type=int,
        default=10,
        help="测试集实例数，默认 10",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmarks"),
        help="输出目录",
    )

    args = parser.parse_args()

    if min(
        args.dev_count,
        args.validation_count,
        args.test_count,
    ) <= 0:
        raise ValueError(
            "三个数据划分的实例数都必须为正整数"
        )

    dev_cases = generate_split(
        split="dev",
        count=args.dev_count,
        master_seed=args.master_seed,
    )

    validation_cases = generate_split(
        split="validation",
        count=args.validation_count,
        master_seed=args.master_seed,
    )

    test_cases = generate_split(
        split="test",
        count=args.test_count,
        master_seed=args.master_seed,
    )

    save_split(
        dev_cases,
        args.output_dir / "dev.json",
    )

    save_split(
        validation_cases,
        args.output_dir / "validation.json",
    )

    save_split(
        test_cases,
        args.output_dir / "test.json",
    )

    all_cases = (
        dev_cases
        + validation_cases
        + test_cases
    )

    print("生成完成：")
    print(
        "  ",
        (args.output_dir / "dev.json").resolve(),
    )
    print(
        "  ",
        (
            args.output_dir
            / "validation.json"
        ).resolve(),
    )
    print(
        "  ",
        (args.output_dir / "test.json").resolve(),
    )

    print_summary(all_cases)


if __name__ == "__main__":
    main()
