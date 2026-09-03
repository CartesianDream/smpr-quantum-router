from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from qiskit import qasm3, qpy, transpile

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from case_io import load_cases
from qiskit_adapter import (
    build_coupling_map,
    build_qiskit_circuit,
    weighted_qiskit_depth,
)


def load_csv_by_id(
    path: Path,
) -> dict[str, dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    return {
        row["instance"]: row
        for row in rows
    }


def parse_indices(
    value: str | None,
) -> set[int] | None:
    if value is None:
        return None

    return {
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    }


def validate_topology(
    circuit,
    edges: tuple[tuple[int, int], ...],
) -> None:
    allowed = {
        tuple(sorted((int(u), int(v))))
        for u, v in edges
    }

    for instruction in circuit.data:
        name = instruction.operation.name

        if name not in {"cx", "swap"}:
            continue

        physical = [
            circuit.find_bit(qubit).index
            for qubit in instruction.qubits
        ]

        if len(physical) != 2:
            raise AssertionError(
                f"{name} 不是二比特操作"
            )

        edge = tuple(sorted(physical))

        if edge not in allowed:
            raise AssertionError(
                f"操作 {name}{tuple(physical)} "
                "不满足硬件拓扑"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "historical development: replay frozen LightSABRE "
            "best seeds and export QASM3/QPY"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/benchmarks_v5/final_test.json"
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "results/profiling_input_rust_port/"
            "full_validation_skip_cases.csv"
        ),
    )
    parser.add_argument(
        "--portfolio",
        type=Path,
        default=Path(
            "results/safe_portfolio_input_baseline_rollout_safe/"
            "baseline_rollout_safe_cases.csv"
        ),
    )
    parser.add_argument(
        "--case-indices",
        type=str,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/backend_export_executable/"
            "lightsabre_winners"
        ),
    )
    parser.add_argument(
        "--case-output",
        type=Path,
        default=Path(
            "results/backend_export_executable/"
            "lightsabre_winner_cases.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "results/backend_export_executable/"
            "lightsabre_winner_summary.json"
        ),
    )
    args = parser.parse_args()

    cases = load_cases(args.input, None)
    reference = load_csv_by_id(
        args.reference
    )
    portfolio = load_csv_by_id(
        args.portfolio
    )
    selected_indices = parse_indices(
        args.case_indices
    )

    rows: list[dict[str, object]] = []

    for index, case in enumerate(cases):
        if (
            selected_indices is not None
            and index not in selected_indices
        ):
            continue

        ref = reference[case.name]
        selected = portfolio[case.name]

        if (
            selected["selected_backend"]
            != "lightsabre"
        ):
            rows.append(
                {
                    "case_index": index,
                    "instance": case.name,
                    "status": "skipped_independent_rollout",
                    "seed": "",
                    "swap": "",
                    "depth": "",
                    "qasm": "",
                    "qpy": "",
                    "error": "",
                }
            )
            continue

        try:
            seed = int(
                ref["lightsabre_best_seed"]
            )
            expected_swap = int(
                ref["lightsabre_swap"]
            )
            expected_depth = int(
                ref["lightsabre_depth"]
            )

            logical = build_qiskit_circuit(
                case.test_case
            )
            coupling_map = build_coupling_map(
                case.test_case
            )

            routed = transpile(
                logical,
                coupling_map=coupling_map,
                layout_method="sabre",
                routing_method="sabre",
                optimization_level=0,
                seed_transpiler=seed,
            )

            operations = dict(
                routed.count_ops()
            )
            swap = int(
                operations.get("swap", 0)
            )
            depth = int(
                weighted_qiskit_depth(routed)
            )

            if (
                swap != expected_swap
                or depth != expected_depth
            ):
                raise AssertionError(
                    "冻结种子重放结果不一致："
                    f"actual={swap}/{depth}, "
                    f"expected="
                    f"{expected_swap}/"
                    f"{expected_depth}"
                )

            validate_topology(
                routed,
                case.test_case.edges,
            )

            case_dir = (
                args.output_dir /
                f"{index:03d}_{case.name}"
            )
            case_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            qasm_path = (
                case_dir /
                "lightsabre_physical.qasm"
            )
            qpy_path = (
                case_dir /
                "lightsabre_physical.qpy"
            )
            manifest_path = (
                case_dir /
                "manifest.json"
            )

            qasm_path.write_text(
                qasm3.dumps(routed),
                encoding="utf-8",
            )

            with qpy_path.open("wb") as handle:
                qpy.dump(routed, handle)

            parsed = qasm3.loads(
                qasm_path.read_text(
                    encoding="utf-8"
                )
            )
            parsed_operations = dict(
                parsed.count_ops()
            )

            if int(
                parsed_operations.get("swap", 0)
            ) != swap:
                raise AssertionError(
                    "QASM3 往返后的 SWAP 数不一致"
                )

            manifest = {
                "case_index": index,
                "instance": case.name,
                "backend": "lightsabre",
                "seed": seed,
                "swap_count": swap,
                "weighted_depth": depth,
                "source_cx_count": (
                    case.two_qubit_gate_count
                ),
                "physical_qubits": (
                    case.test_case
                    .num_physical_qubits
                ),
                "layout": str(routed.layout),
                "qasm": str(
                    qasm_path.resolve()
                ),
                "qpy": str(
                    qpy_path.resolve()
                ),
            }

            manifest_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            rows.append(
                {
                    "case_index": index,
                    "instance": case.name,
                    "status": "exported",
                    "seed": seed,
                    "swap": swap,
                    "depth": depth,
                    "qasm": str(
                        qasm_path.resolve()
                    ),
                    "qpy": str(
                        qpy_path.resolve()
                    ),
                    "error": "",
                }
            )

            print(
                f"[{index}] {case.name}: "
                f"exported seed={seed}, "
                f"{swap}/{depth}"
            )

        except Exception as error:
            rows.append(
                {
                    "case_index": index,
                    "instance": case.name,
                    "status": "failed",
                    "seed": ref.get(
                        "lightsabre_best_seed",
                        "",
                    ),
                    "swap": "",
                    "depth": "",
                    "qasm": "",
                    "qpy": "",
                    "error": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }
            )

            print(
                f"[{index}] {case.name}: "
                f"FAILED: {error}"
            )

    args.case_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.case_output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    status: dict[str, int] = {}
    for row in rows:
        name = str(row["status"])
        status[name] = status.get(name, 0) + 1

    summary = {
        "processed": len(rows),
        "status_counts": status,
        "exported_swap_total": sum(
            int(row["swap"])
            for row in rows
            if row["status"] == "exported"
        ),
        "case_output": str(
            args.case_output.resolve()
        ),
    }

    args.summary_output.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n===== historical development LightSABRE export ====="
    )
    print("processed:", len(rows))
    print("status:", status)
    print(
        "exported SWAP total:",
        summary["exported_swap_total"],
    )
    print(
        "cases:",
        args.case_output.resolve(),
    )
    print(
        "summary:",
        args.summary_output.resolve(),
    )


if __name__ == "__main__":
    main()