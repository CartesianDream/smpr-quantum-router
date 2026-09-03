from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mapping_portfolio as mapping_portfolio
import rollout_policy as rollout_policy
import local_layout_search as local_layout_search
import frozen_policy as frozen_policy
import routing_rules as routing_rules

from case_io import load_cases


SINGLE_QUBIT_GATES = {
    "h",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
}


def load_csv_by_id(
    path: Path,
) -> dict[str, dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    key = (
        "instance"
        if "instance" in rows[0]
        else "id"
    )
    return {row[key]: row for row in rows}


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


def mapping_budget(
    text: str,
) -> tuple[str, int]:
    mapping_mode, top_text = text.rsplit("/", 1)

    if not top_text.startswith("k"):
        raise ValueError(
            f"无法解析 mapping budget：{text}"
        )

    return mapping_mode, int(top_text[1:])


def export_qasm(
    state,
    hardware,
    path: Path,
) -> tuple[int, int, int]:
    """
    返回：
      原始逻辑 CX 数、
      抽象 SWAP 数、
      展开后的物理 CX 数。
    """
    logical_cx = 0
    swap_count = 0
    expanded_cx = 0

    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{hardware.num_qubits}] q;",
    ]

    for operation in state.physical_operations:
        name = str(operation[0])

        if name == "cx":
            _, u, v, _source_gate = operation

            if not hardware.adjacent(u, v):
                raise AssertionError(
                    f"非法物理 CX：{operation}"
                )

            lines.append(f"cx q[{u}], q[{v}];")
            logical_cx += 1
            expanded_cx += 1
            continue

        if name == "swap":
            _, u, v, _source_gate = operation

            if not hardware.adjacent(u, v):
                raise AssertionError(
                    f"非法物理 SWAP：{operation}"
                )

            # SWAP(u,v) = CX(u,v) CX(v,u) CX(u,v)
            lines.extend(
                [
                    f"cx q[{u}], q[{v}];",
                    f"cx q[{v}], q[{u}];",
                    f"cx q[{u}], q[{v}];",
                ]
            )
            swap_count += 1
            expanded_cx += 3
            continue

        if name not in SINGLE_QUBIT_GATES:
            raise ValueError(
                f"暂不支持的单比特门：{operation}"
            )

        _, physical, _source_gate = operation
        lines.append(f"{name} q[{physical}];")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return logical_cx, swap_count, expanded_cx


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "historical development: rerun selected historical development independent rollout "
            "routes and export physical QASM3"
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
        "--independent_rollout-reference",
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
            "independent_rollout_reference"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "results/backend_export_executable/"
            "independent_rollout_reference_summary.json"
        ),
    )
    parser.add_argument(
        "--case-output",
        type=Path,
        default=Path(
            "results/backend_export_executable/"
            "independent_rollout_reference_cases.csv"
        ),
    )
    args = parser.parse_args()

    cases = load_cases(args.input, None)
    selected_indices = parse_indices(
        args.case_indices
    )

    reference = load_csv_by_id(
        args.independent_rollout_reference
    )
    portfolio = load_csv_by_id(
        args.portfolio
    )
    config = routing_rules.configure_router()

    rows: list[dict[str, object]] = []

    for index, case in enumerate(cases):
        if (
            selected_indices is not None
            and index not in selected_indices
        ):
            continue

        portfolio_row = portfolio[case.name]
        reference_row = reference[case.name]
        backend = portfolio_row["selected_backend"]
        variant = routing_rules.frozen_variant(case)

        if backend != "independent_rollout":
            rows.append(
                {
                    "case_index": index,
                    "instance": case.name,
                    "variant": variant,
                    "status": "skipped_lightsabre",
                    "mapping": "",
                    "swap": "",
                    "depth": "",
                    "logical_cx": "",
                    "expanded_cx": "",
                    "qasm": "",
                    "error": "",
                }
            )
            continue

        if variant not in {
            "adaptive_mapping",
            "local_layout_search_ring_n10_far",
        }:
            rows.append(
                {
                    "case_index": index,
                    "instance": case.name,
                    "variant": variant,
                    "status": "deferred_special_variant",
                    "mapping": reference_row[
                        "selected_mapping"
                    ],
                    "swap": reference_row[
                        "router_swap"
                    ],
                    "depth": reference_row[
                        "router_depth"
                    ],
                    "logical_cx": "",
                    "expanded_cx": "",
                    "qasm": "",
                    "error": "",
                }
            )
            continue

        try:
            selected_name = reference_row[
                "selected_mapping"
            ]

            if variant == "adaptive_mapping":
                mapping_mode, top_k = mapping_budget(
                    reference_row["mapping_budget"]
                )
                mappings = mapping_portfolio.make_mappings(
                    case,
                    mapping_mode,
                )
                selected_mapping = next(
                    item
                    for item in mappings
                    if item.name == selected_name
                )
            else:
                (
                    _mapping_rows,
                    candidate_by_mapping,
                ) = local_layout_search.search_mapping_candidates(
                    case,
                    config,
                    frozen_policy.BEAM_WIDTH,
                    frozen_policy.SEARCH_ROUNDS,
                    frozen_policy.USE_FORWARD_BACKWARD,
                )
                selected_mapping = next(
                    item
                    for item
                    in candidate_by_mapping.values()
                    if item.name == selected_name
                )
                top_k = frozen_policy.DEFAULT_TOP_K

            dag, hardware, _ = (
                case.test_case.build()
            )

            state, stats = (
                rollout_policy.run_rollout_router(
                    dag=dag,
                    hardware=hardware,
                    initial_mapping=list(
                        selected_mapping.mapping
                    ),
                    base_config=config,
                    top_k=top_k,
                )
            )

            state.assert_valid(dag, hardware)

            if len(state.executed_gates) != len(
                dag.gates
            ):
                raise AssertionError(
                    "路由结束后仍有未执行逻辑门"
                )

            swap = sum(
                operation[0] == "swap"
                for operation
                in state.physical_operations
            )
            depth = state.current_depth()

            expected_swap = int(
                reference_row["router_swap"]
            )
            expected_depth = int(
                reference_row["router_depth"]
            )

            if (
                swap != expected_swap
                or depth != expected_depth
            ):
                raise AssertionError(
                    "重放质量与冻结结果不一致："
                    f"actual={swap}/{depth}, "
                    f"expected="
                    f"{expected_swap}/"
                    f"{expected_depth}"
                )

            case_dir = (
                args.output_dir /
                f"{index:03d}_{case.name}"
            )
            qasm_path = (
                case_dir /
                "independent_rollout_physical.qasm"
            )

            (
                logical_cx,
                exported_swap,
                expanded_cx,
            ) = export_qasm(
                state,
                hardware,
                qasm_path,
            )

            if logical_cx != (
                case.two_qubit_gate_count
            ):
                raise AssertionError(
                    "导出线路的逻辑 CX 数不一致："
                    f"{logical_cx} != "
                    f"{case.two_qubit_gate_count}"
                )

            if exported_swap != swap:
                raise AssertionError(
                    "导出 SWAP 计数不一致"
                )

            expected_expanded = (
                logical_cx + 3 * swap
            )
            if expanded_cx != expected_expanded:
                raise AssertionError(
                    "展开后的 CX 数不一致"
                )

            manifest = {
                "case_index": index,
                "instance": case.name,
                "variant": variant,
                "mapping_name": (
                    selected_mapping.name
                ),
                "initial_mapping": list(
                    selected_mapping.mapping
                ),
                "final_mapping": list(
                    state.logical_to_physical
                ),
                "physical_qubits": (
                    hardware.num_qubits
                ),
                "logical_qubits": (
                    case.num_qubits
                ),
                "source_gate_count": len(
                    dag.gates
                ),
                "source_cx_count": (
                    case.two_qubit_gate_count
                ),
                "abstract_swap_count": swap,
                "weighted_depth": depth,
                "expanded_cx_count": (
                    expanded_cx
                ),
                "route_decisions": (
                    stats.route_decisions
                ),
                "changed_decisions": (
                    stats.changed_decisions
                ),
                "qasm": str(
                    qasm_path.resolve()
                ),
            }

            manifest_path = (
                case_dir /
                "manifest.json"
            )
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
                    "variant": variant,
                    "status": "exported",
                    "mapping": (
                        selected_mapping.name
                    ),
                    "swap": swap,
                    "depth": depth,
                    "logical_cx": logical_cx,
                    "expanded_cx": expanded_cx,
                    "qasm": str(
                        qasm_path.resolve()
                    ),
                    "error": "",
                }
            )

            print(
                f"[{index}] {case.name}: "
                f"exported {swap}/{depth}, "
                f"CX={expanded_cx}"
            )

        except Exception as error:
            rows.append(
                {
                    "case_index": index,
                    "instance": case.name,
                    "variant": variant,
                    "status": "failed",
                    "mapping": reference_row.get(
                        "selected_mapping",
                        "",
                    ),
                    "swap": "",
                    "depth": "",
                    "logical_cx": "",
                    "expanded_cx": "",
                    "qasm": "",
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

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

    summary = {
        "processed": len(rows),
        "status_counts": status_counts,
        "exported_swap_total": sum(
            int(row["swap"])
            for row in rows
            if row["status"] == "exported"
        ),
        "source": str(args.input.resolve()),
        "independent_rollout_reference": str(
            args.independent_rollout_reference.resolve()
        ),
        "portfolio": str(
            args.portfolio.resolve()
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

    print("\n===== historical development independent rollout export =====")
    print("processed:", len(rows))
    print("status:", status_counts)
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
