from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil
from qiskit import transpile


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "smpr_router" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import case_io
import lightsabre_backend
import qiskit_adapter
import routing_rules
import smpr_portfolio


BUDGET_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0)


@dataclass
class PeakRSS:
    interval: float = 0.01

    def __post_init__(self) -> None:
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.wait(self.interval):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> "PeakRSS":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_manifest() -> str:
    manifest_path = ROOT / "v7" / "FROZEN_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("freeze V7 before running any result-producing job")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for row in payload.get("files", []):
        relative = Path(str(row["path"]))
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            failures.append(f"outside repository: {relative}")
        elif not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256(path) != str(row["sha256"]):
            failures.append(f"hash mismatch: {relative}")
    if failures:
        raise RuntimeError(
            "V7 frozen-input verification failed:\n" + "\n".join(failures)
        )
    return sha256(manifest_path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def warmup(case: case_io.DevCase) -> None:
    circuit = qiskit_adapter.build_qiskit_circuit(case.test_case)
    coupling = qiskit_adapter.build_coupling_map(case.test_case)
    transpile(
        circuit,
        coupling_map=coupling,
        layout_method="sabre",
        routing_method="sabre",
        optimization_level=0,
        seed_transpiler=0,
    )


def audit_selected_candidate(
    case: case_io.DevCase,
    selected: smpr_portfolio.RouteCandidate,
) -> tuple[bool, bool]:
    """Validate the selected route after the timed region.

    V7 deliberately audits the normalized routing skeleton rather than claiming
    full-unitary equivalence for the original non-Clifford source program.
    """
    if selected.backend == "lightsabre":
        if selected.circuit is None:
            return False, False
        try:
            lightsabre_backend.validate_topology(
                selected.circuit,
                case.test_case.edges,
            )
        except Exception:
            return False, False
        logical_cx = case.two_qubit_gate_count
        routed_cx = int(selected.circuit.count_ops().get("cx", 0))
        routed_swaps = int(selected.circuit.count_ops().get("swap", 0))
        trace_valid = (
            routed_cx == logical_cx
            and routed_swaps == selected.swap
            and len(selected.initial_mapping) == case.num_qubits
            and len(selected.final_mapping) == case.num_qubits
        )
        return trace_valid, True

    if selected.state is None:
        return False, False
    dag, hardware, _ = case.test_case.build()
    try:
        selected.state.assert_valid(dag, hardware)
    except Exception:
        return False, False
    executed_ids = sorted(int(gate_id) for gate_id in selected.state.executed_gates)
    trace_valid = (
        executed_ids == list(range(len(dag.gates)))
        and tuple(selected.state.logical_to_physical) == selected.final_mapping
    )
    topology_valid = all(
        hardware.adjacent(int(operation[1]), int(operation[2]))
        for operation in selected.state.physical_operations
        if operation[0] in {"cx", "swap"}
    )
    return trace_valid, topology_valid


def run_smpr(case: case_io.DevCase) -> tuple[dict[str, Any], list[Any]]:
    config = routing_rules.configure_router()
    smpr_portfolio.SELECTION_MODE = "deterministic"
    start = time.perf_counter()
    with PeakRSS() as memory:
        ls_best, ls_trials, ls_budget_ms = smpr_portfolio.run_lightsabre_trials(
            case, 20, 1, False
        )
        independent_best, independent_trials, variant = (
            smpr_portfolio.run_native_independent_rollout(case, config)
        )
        cross_trials = smpr_portfolio.run_cross_layout_candidates(
            case=case,
            qiskit_candidates=ls_trials,
            config=config,
            maximum_layouts=0,
            neighbors_per_layout=4,
            proxy_decay=0.95,
        )
        selected = min(
            [ls_best, independent_best, *cross_trials],
            key=smpr_portfolio.candidate_key,
        )
    wall_ms = (time.perf_counter() - start) * 1000.0
    attempts = [*ls_trials, *independent_trials, *cross_trials]
    routing_trace_valid, topology_valid = audit_selected_candidate(case, selected)
    result = {
        "smpr_wall_ms": wall_ms,
        "smpr_peak_rss_bytes": memory.peak,
        "smpr_swap": selected.swap,
        "smpr_depth": selected.depth,
        "smpr_branch": selected.backend,
        "smpr_label": selected.label,
        "smpr_attempts": len(attempts),
        "internal_lightsabre_swap": ls_best.swap,
        "internal_lightsabre_depth": ls_best.depth,
        "internal_lightsabre_budget_ms": ls_budget_ms,
        "independent_rollout_swap": independent_best.swap,
        "independent_rollout_depth": independent_best.depth,
        "cross_layout_attempts": len(cross_trials),
        "variant": variant,
        "routing_trace_valid": routing_trace_valid,
        "topology_valid": topology_valid,
        "safe_quality_floor": (
            (selected.swap, selected.depth) <= (ls_best.swap, ls_best.depth)
        ),
    }
    return result, attempts


def time_matched_lightsabre(
    case: case_io.DevCase,
    target_ms: float,
) -> dict[float, dict[str, Any]]:
    circuit = qiskit_adapter.build_qiskit_circuit(case.test_case)
    coupling = qiskit_adapter.build_coupling_map(case.test_case)
    thresholds = {
        multiplier: target_ms * multiplier
        for multiplier in BUDGET_MULTIPLIERS
    }
    best: tuple[int, int, int] | None = None
    records: dict[float, dict[str, Any]] = {}
    seed = 0
    start = time.perf_counter()
    with PeakRSS() as memory:
        while seed < 10_000:
            routed = transpile(
                circuit,
                coupling_map=coupling,
                layout_method="sabre",
                routing_method="sabre",
                optimization_level=0,
                seed_transpiler=seed,
            )
            quality = (
                int(routed.count_ops().get("swap", 0)),
                int(qiskit_adapter.weighted_qiskit_depth(routed)),
                seed,
            )
            if best is None or quality < best:
                best = quality
            seed += 1
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            for multiplier, threshold in thresholds.items():
                if (
                    multiplier not in records
                    and seed >= 20
                    and elapsed_ms >= threshold
                ):
                    records[multiplier] = {
                        "budget_multiplier": multiplier,
                        "lightsabre_elapsed_ms": elapsed_ms,
                        "lightsabre_trials": seed,
                        "lightsabre_swap": best[0],
                        "lightsabre_depth": best[1],
                        "lightsabre_best_seed": best[2],
                    }
            if len(records) == len(thresholds):
                break
    if best is None:
        raise RuntimeError("LightSABRE produced no candidates")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    for multiplier in BUDGET_MULTIPLIERS:
        records.setdefault(
            multiplier,
            {
                "budget_multiplier": multiplier,
                "lightsabre_elapsed_ms": elapsed_ms,
                "lightsabre_trials": seed,
                "lightsabre_swap": best[0],
                "lightsabre_depth": best[1],
                "lightsabre_best_seed": best[2],
            },
        )
        records[multiplier]["lightsabre_peak_rss_bytes"] = memory.peak
    return records


def environment_record(repeat: int, frozen_manifest_sha256: str) -> dict[str, Any]:
    process = psutil.Process()
    return {
        "repeat": repeat,
        "frozen_manifest_sha256": frozen_manifest_sha256,
        "python": sys.version,
        "executable": sys.executable,
        "qiskit": metadata.version("qiskit"),
        "qiskit_qasm3_import": metadata.version("qiskit-qasm3-import"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "memory_total_bytes": psutil.virtual_memory().total,
        "process_affinity": (
            process.cpu_affinity() if hasattr(process, "cpu_affinity") else None
        ),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "RAYON_NUM_THREADS",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="run one paired V7 SMPR/LightSABRE comparison"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "v7" / "external_cases.json",
    )
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frozen_manifest_sha256 = verify_frozen_manifest()
    if args.repeat not in {1, 2, 3}:
        raise ValueError("repeat must be 1, 2, or 3")

    cases = case_io.load_cases(args.input, None)
    output_rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        warmup(case)
        smpr, _attempts = run_smpr(case)
        frontiers = time_matched_lightsabre(case, smpr["smpr_wall_ms"])
        for multiplier in BUDGET_MULTIPLIERS:
            light = frontiers[multiplier]
            output_rows.append(
                {
                    "repeat": args.repeat,
                    "frozen_manifest_sha256": frozen_manifest_sha256,
                    "case_index": index,
                    "instance": case.name,
                    "num_logical_qubits": case.num_qubits,
                    "num_physical_qubits": case.test_case.num_physical_qubits,
                    "topology": case.topology,
                    "routing_cx_count": case.two_qubit_gate_count,
                    **smpr,
                    **light,
                    "swap_gap_smpr_minus_lightsabre": (
                        smpr["smpr_swap"] - light["lightsabre_swap"]
                    ),
                }
            )
        print(
            f"[{index + 1}/{len(cases)}] {case.name}: "
            f"SMPR={smpr['smpr_swap']}/{smpr['smpr_depth']}, "
            f"wall={smpr['smpr_wall_ms']:.1f} ms"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = args.output_dir / "paired_frontier.csv"
    environment_path = args.output_dir / "environment.json"
    write_csv(paired_path, output_rows)
    environment_path.write_text(
        json.dumps(
            environment_record(args.repeat, frozen_manifest_sha256),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.output_dir / "SHA256SUMS.txt").write_text(
        f"{sha256(environment_path)}  environment.json\n"
        f"{sha256(paired_path)}  paired_frontier.csv\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
