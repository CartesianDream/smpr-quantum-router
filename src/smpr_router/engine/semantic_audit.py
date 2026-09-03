from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, qasm3, qpy
from qiskit.quantum_info import Clifford


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def load_source_cases(path: Path) -> list[dict[str, Any]]:
    root = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(root, list):
        return root

    if isinstance(root, dict) and isinstance(root.get("cases"), list):
        return root["cases"]

    raise RuntimeError("无法在输入 JSON 中找到 cases")


def build_source_circuit(case: dict[str, Any]) -> QuantumCircuit:
    circuit = QuantumCircuit(int(case["num_qubits"]))

    for spec in case["gate_specs"]:
        name = str(spec[0]).lower()

        if name == "h":
            circuit.h(int(spec[1]))
        elif name == "cx":
            circuit.cx(int(spec[1]), int(spec[2]))
        else:
            raise RuntimeError(
                f"{case['name']} 出现不支持的源门：{spec}"
            )

    return circuit


def validate_mapping(
    mapping: list[int],
    logical_qubits: int,
    physical_qubits: int,
    label: str,
) -> None:
    if len(mapping) != logical_qubits:
        raise RuntimeError(
            f"{label} 长度错误：{len(mapping)} != {logical_qubits}"
        )

    if len(set(mapping)) != len(mapping):
        raise RuntimeError(f"{label} 不是单射：{mapping}")

    if any(
        physical < 0 or physical >= physical_qubits
        for physical in mapping
    ):
        raise RuntimeError(f"{label} 包含越界物理比特：{mapping}")


def append_mapping_permutation(
    circuit: QuantumCircuit,
    source_mapping: list[int],
    target_mapping: list[int],
) -> None:
    """
    把位于 source_mapping[logical] 的逻辑状态，
    移动到 target_mapping[logical]。
    """
    logical_qubits = len(source_mapping)

    token_at: list[int | None] = [None] * circuit.num_qubits
    position_of: list[int] = list(source_mapping)

    for logical, physical in enumerate(source_mapping):
        token_at[physical] = logical

    desired_at: list[int | None] = [None] * circuit.num_qubits

    for logical, physical in enumerate(target_mapping):
        desired_at[physical] = logical

    for physical in range(circuit.num_qubits):
        desired = desired_at[physical]

        if desired is None:
            continue

        current = token_at[physical]

        if current == desired:
            continue

        other = position_of[desired]

        circuit.swap(physical, other)

        token_at[physical], token_at[other] = (
            token_at[other],
            token_at[physical],
        )

        if token_at[physical] is not None:
            position_of[token_at[physical]] = physical

        if token_at[other] is not None:
            position_of[token_at[other]] = other

    if position_of[:logical_qubits] != target_mapping:
        raise RuntimeError(
            "映射置换构造失败："
            f"{position_of[:logical_qubits]} != {target_mapping}"
        )


def expected_physical_circuit(
    source: QuantumCircuit,
    initial_mapping: list[int],
    final_mapping: list[int],
    physical_qubits: int,
) -> QuantumCircuit:
    expected = QuantumCircuit(physical_qubits)
    identity = list(range(source.num_qubits))

    # 物理输入布局 -> 规范逻辑线序。
    append_mapping_permutation(
        expected,
        initial_mapping,
        identity,
    )

    expected.compose(
        source,
        qubits=list(range(source.num_qubits)),
        inplace=True,
    )

    # 规范逻辑线序 -> 最终物理布局。
    append_mapping_permutation(
        expected,
        identity,
        final_mapping,
    )

    return expected


def mapping_from_layout(
    transpile_layout: Any,
    method_name: str,
    logical_qubits: int,
    physical_qubits: int,
) -> list[int]:
    method = getattr(transpile_layout, method_name)
    layout = method()
    input_mapping = transpile_layout.input_qubit_mapping

    result: list[int | None] = [None] * logical_qubits

    for physical, virtual in layout.get_physical_bits().items():
        logical = input_mapping.get(virtual)

        if logical is None:
            continue

        logical = int(logical)

        if logical < logical_qubits:
            result[logical] = int(physical)

    if any(value is None for value in result):
        raise RuntimeError(
            f"{method_name} 提取不完整：{result}"
        )

    mapping = [int(value) for value in result]
    validate_mapping(
        mapping,
        logical_qubits,
        physical_qubits,
        method_name,
    )
    return mapping


def weighted_depth(circuit: QuantumCircuit) -> int:
    depths = [0] * circuit.num_qubits

    for instruction in circuit.data:
        name = instruction.operation.name.lower()

        if name == "barrier":
            continue

        qubits = [
            circuit.find_bit(qubit).index
            for qubit in instruction.qubits
        ]

        if not qubits:
            continue

        if len(qubits) > 2:
            raise RuntimeError(
                f"不支持的 {len(qubits)} 比特门：{name}"
            )

        duration = 3 if name == "swap" else 1
        finish = max(depths[index] for index in qubits) + duration

        for index in qubits:
            depths[index] = finish

    return max(depths, default=0)


def topology_errors(
    circuit: QuantumCircuit,
    edges: list[list[int]],
) -> list[str]:
    allowed = {
        tuple(sorted((int(left), int(right))))
        for left, right in edges
    }
    errors: list[str] = []

    for position, instruction in enumerate(circuit.data):
        name = instruction.operation.name.lower()

        if name == "barrier":
            continue

        qubits = [
            circuit.find_bit(qubit).index
            for qubit in instruction.qubits
        ]

        if len(qubits) == 2:
            edge = tuple(sorted(qubits))

            if edge not in allowed:
                errors.append(
                    f"operation {position}: {name}{tuple(qubits)}"
                )

    return errors


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmarks_v5/final_test.json"),
    )
    parser.add_argument(
        "--independent_rollout-cases",
        type=Path,
        default=Path(
            "results/backend_export_executable/"
            "independent_rollout_winner_cases.csv"
        ),
    )
    parser.add_argument(
        "--lightsabre-cases",
        type=Path,
        default=Path(
            "results/backend_export_executable/"
            "lightsabre_winner_cases.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/semantic_audit"),
    )
    args = parser.parse_args()

    source_cases = load_source_cases(args.input)
    source_by_name = {
        str(case["name"]): case
        for case in source_cases
    }

    independent_rollout_rows = [
        row for row in read_csv(args.independent_rollout_cases)
        if row["status"] == "exported"
    ]
    lightsabre_rows = [
        row for row in read_csv(args.lightsabre_cases)
        if row["status"] == "exported"
    ]

    selected: dict[str, tuple[str, dict[str, str]]] = {}

    for row in independent_rollout_rows:
        selected[row["instance"]] = ("independent_rollout", row)

    for row in lightsabre_rows:
        instance = row["instance"]

        if instance in selected:
            raise RuntimeError(f"后端重复选择：{instance}")

        selected[instance] = ("lightsabre", row)

    expected_names = set(source_by_name)

    if set(selected) != expected_names:
        missing = sorted(expected_names - set(selected))
        extra = sorted(set(selected) - expected_names)
        raise RuntimeError(
            f"实例集合不完整；missing={missing}, extra={extra}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit_rows: list[dict[str, Any]] = []
    manifest_cases: list[dict[str, Any]] = []
    artifact_paths: set[Path] = set()

    for case_index, source_case in enumerate(source_cases):
        instance = str(source_case["name"])
        backend, row = selected[instance]

        logical_qubits = int(source_case["num_qubits"])
        source = build_source_circuit(source_case)

        qasm_path = Path(row["qasm"]).resolve()
        routed = qasm3.load(str(qasm_path))
        physical_qubits = routed.num_qubits

        qpy_path: Path | None = None
        sidecar_path: Path | None = None
        qasm_qpy_equivalent: bool | None = None

        if backend == "independent_rollout":
            sidecar_path = qasm_path.parent / "manifest.json"

            if not sidecar_path.exists():
                raise FileNotFoundError(sidecar_path)

            sidecar = json.loads(
                sidecar_path.read_text(encoding="utf-8")
            )

            initial_mapping = [
                int(value)
                for value in sidecar["initial_mapping"]
            ]
            final_mapping = [
                int(value)
                for value in sidecar["final_mapping"]
            ]

            expected_swap = int(
                sidecar["abstract_swap_count"]
            )
            expected_depth = int(sidecar["weighted_depth"])

            source_cx = sum(
                1
                for spec in source_case["gate_specs"]
                if str(spec[0]).lower() == "cx"
            )
            actual_expanded_cx = int(
                routed.count_ops().get("cx", 0)
            )
            expanded_cx_valid = (
                actual_expanded_cx
                == source_cx + 3 * expected_swap
            )

        else:
            qpy_path = Path(row["qpy"]).resolve()

            with qpy_path.open("rb") as handle:
                qpy_circuit = qpy.load(handle)[0]

            if qpy_circuit.layout is None:
                raise RuntimeError(
                    f"{instance} 的 QPY 缺少 TranspileLayout"
                )

            initial_mapping = mapping_from_layout(
                qpy_circuit.layout,
                "initial_virtual_layout",
                logical_qubits,
                physical_qubits,
            )
            final_mapping = mapping_from_layout(
                qpy_circuit.layout,
                "final_virtual_layout",
                logical_qubits,
                physical_qubits,
            )

            expected_swap = int(row["swap"])
            expected_depth = int(row["depth"])
            actual_expanded_cx = None
            expanded_cx_valid = None

            qasm_qpy_equivalent = bool(
                Clifford(routed) == Clifford(qpy_circuit)
            )

        validate_mapping(
            initial_mapping,
            logical_qubits,
            physical_qubits,
            f"{instance} initial_mapping",
        )
        validate_mapping(
            final_mapping,
            logical_qubits,
            physical_qubits,
            f"{instance} final_mapping",
        )

        expected = expected_physical_circuit(
            source,
            initial_mapping,
            final_mapping,
            physical_qubits,
        )

        semantic_equivalent = bool(
            Clifford(routed) == Clifford(expected)
        )

        edge_errors = topology_errors(
            routed,
            source_case["edges"],
        )
        topology_valid = not edge_errors

        actual_depth = weighted_depth(routed)
        depth_valid = actual_depth == expected_depth

        if backend == "lightsabre":
            actual_swap = int(
                routed.count_ops().get("swap", 0)
            )
        else:
            actual_swap = expected_swap

        swap_valid = actual_swap == expected_swap

        gate_names = sorted(
            str(name)
            for name in routed.count_ops()
        )
        unexpected_gates = sorted(
            set(gate_names) - {"h", "cx", "swap"}
        )
        gate_set_valid = not unexpected_gates

        qpy_valid = (
            True
            if backend == "independent_rollout"
            else bool(qasm_qpy_equivalent)
        )

        passed = all((
            semantic_equivalent,
            topology_valid,
            depth_valid,
            swap_valid,
            gate_set_valid,
            qpy_valid,
            (
                True
                if expanded_cx_valid is None
                else expanded_cx_valid
            ),
        ))

        audit_row = {
            "case_index": case_index,
            "instance": instance,
            "backend": backend,
            "swap": expected_swap,
            "expected_depth": expected_depth,
            "actual_depth": actual_depth,
            "initial_mapping": json.dumps(
                initial_mapping,
                separators=(",", ":"),
            ),
            "final_mapping": json.dumps(
                final_mapping,
                separators=(",", ":"),
            ),
            "semantic_equivalent": semantic_equivalent,
            "topology_valid": topology_valid,
            "depth_valid": depth_valid,
            "swap_valid": swap_valid,
            "gate_set_valid": gate_set_valid,
            "qasm_qpy_equivalent": qasm_qpy_equivalent,
            "expanded_cx_valid": expanded_cx_valid,
            "passed": passed,
            "topology_errors": " | ".join(edge_errors),
            "unexpected_gates": ",".join(unexpected_gates),
            "qasm": str(qasm_path),
            "qpy": "" if qpy_path is None else str(qpy_path),
            "sidecar": (
                ""
                if sidecar_path is None
                else str(sidecar_path)
            ),
            "qasm_sha256": sha256(qasm_path),
            "qpy_sha256": (
                ""
                if qpy_path is None
                else sha256(qpy_path)
            ),
            "sidecar_sha256": (
                ""
                if sidecar_path is None
                else sha256(sidecar_path)
            ),
        }
        audit_rows.append(audit_row)

        manifest_cases.append({
            "case_index": case_index,
            "instance": instance,
            "backend": backend,
            "swap": expected_swap,
            "weighted_depth": expected_depth,
            "logical_qubits": logical_qubits,
            "physical_qubits": physical_qubits,
            "initial_mapping": initial_mapping,
            "final_mapping": final_mapping,
            "qasm": str(qasm_path),
            "qasm_sha256": audit_row["qasm_sha256"],
            "qpy": (
                None if qpy_path is None else str(qpy_path)
            ),
            "qpy_sha256": (
                None
                if qpy_path is None
                else audit_row["qpy_sha256"]
            ),
            "semantic_equivalent": semantic_equivalent,
            "topology_valid": topology_valid,
            "metric_valid": (
                depth_valid
                and swap_valid
                and (
                    expanded_cx_valid
                    if expanded_cx_valid is not None
                    else True
                )
            ),
            "passed": passed,
        })

        print(
            f"[{case_index + 1:02d}/60] {instance}: "
            f"{backend}, swap={expected_swap}, "
            f"semantic={semantic_equivalent}, "
            f"topology={topology_valid}, "
            f"passed={passed}"
        )

        artifact_paths.add(qasm_path)

        if qpy_path is not None:
            artifact_paths.add(qpy_path)

        if sidecar_path is not None:
            artifact_paths.add(sidecar_path)

    cases_output = (
        args.output_dir / "semantic_audit_cases.csv"
    )
    manifest_output = (
        args.output_dir / "baseline_rollout_safe_executable_manifest.json"
    )
    hash_output = args.output_dir / "SHA256SUMS.txt"

    write_csv(cases_output, audit_rows)

    passed_count = sum(
        bool(row["passed"])
        for row in audit_rows
    )
    semantic_count = sum(
        bool(row["semantic_equivalent"])
        for row in audit_rows
    )
    topology_count = sum(
        bool(row["topology_valid"])
        for row in audit_rows
    )
    total_swap = sum(
        int(row["swap"])
        for row in audit_rows
    )
    backend_counts = Counter(
        str(row["backend"])
        for row in audit_rows
    )

    manifest = {
        "schema_version": 1,
        "component": "semantic_audit",
        "name": "safe-dual executable semantic audit",
        "semantic_method": (
            "exact Clifford tableau equality with explicit "
            "initial and final logical-to-physical mappings"
        ),
        "cases": manifest_cases,
        "case_count": len(manifest_cases),
        "backend_counts": dict(backend_counts),
        "total_swap": total_swap,
        "lightsabre_baseline_swap": 1691,
        "gain_vs_lightsabre": 1691 - total_swap,
        "semantic_equivalent_count": semantic_count,
        "topology_valid_count": topology_count,
        "passed_count": passed_count,
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "independent_rollout_cases": str(args.independent_rollout_cases.resolve()),
        "lightsabre_cases": str(
            args.lightsabre_cases.resolve()
        ),
    }

    manifest_output.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    artifact_paths.update({
        args.input.resolve(),
        args.independent_rollout_cases.resolve(),
        args.lightsabre_cases.resolve(),
        cases_output.resolve(),
        manifest_output.resolve(),
    })

    hash_output.write_text(
        "".join(
            f"{sha256(path)}  {path}\n"
            for path in sorted(
                artifact_paths,
                key=lambda item: str(item).lower(),
            )
        ),
        encoding="utf-8",
    )

    print("\n===== historical development semantic audit =====")
    print("cases:", len(audit_rows))
    print("backends:", dict(backend_counts))
    print("total SWAP:", total_swap)
    print("gain vs LightSABRE:", 1691 - total_swap)
    print(
        "semantic equivalent:",
        f"{semantic_count}/{len(audit_rows)}",
    )
    print(
        "topology valid:",
        f"{topology_count}/{len(audit_rows)}",
    )
    print(
        "fully passed:",
        f"{passed_count}/{len(audit_rows)}",
    )
    print("cases output:", cases_output.resolve())
    print("manifest:", manifest_output.resolve())
    print("hashes:", hash_output.resolve())

    failed = [
        row for row in audit_rows
        if not row["passed"]
    ]

    if failed:
        print("\n===== Failed cases =====")

        for row in failed:
            print(
                row["instance"],
                {
                    "backend": row["backend"],
                    "semantic": row[
                        "semantic_equivalent"
                    ],
                    "topology": row["topology_valid"],
                    "depth": row["depth_valid"],
                    "swap": row["swap_valid"],
                    "qpy": row[
                        "qasm_qpy_equivalent"
                    ],
                    "expanded_cx": row[
                        "expanded_cx_valid"
                    ],
                    "unexpected": row[
                        "unexpected_gates"
                    ],
                },
            )

        raise SystemExit(1)

    if total_swap != 1670:
        raise RuntimeError(
            f"总 SWAP 回归失败：{total_swap} != 1670"
        )


if __name__ == "__main__":
    main()
