from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

from qiskit import QuantumCircuit, qasm2, qasm3, qpy, transpile
from qiskit.transpiler import CouplingMap


COMMIT_BASE = (
    "https://raw.githubusercontent.com/pnnl/QASMBench/"
    "357b942396d5c2b7cbc1c229c585a6ef5ccaebac"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    path.write_bytes(payload)


def strip_terminal_io(circuit: QuantumCircuit) -> QuantumCircuit:
    first_measurement: int | None = None
    for index, instruction in enumerate(circuit.data):
        operation = instruction.operation
        name = operation.name
        if getattr(operation, "blocks", None) is not None:
            raise ValueError("control flow is outside the V7 static-routing scope")
        if name == "measure" and first_measurement is None:
            first_measurement = index
        if name == "reset":
            raise ValueError("reset is outside the V7 static-routing scope")
        if first_measurement is not None and name not in {"measure", "barrier"}:
            raise ValueError("mid-circuit measurement is not allowed")

    stripped = QuantumCircuit(circuit.num_qubits)
    for instruction in circuit.data:
        name = instruction.operation.name
        if name in {"measure", "barrier"}:
            continue
        if getattr(instruction.operation, "condition", None) is not None:
            raise ValueError("classically conditioned operations are not allowed")
        stripped.append(
            instruction.operation,
            [stripped.qubits[circuit.find_bit(q).index] for q in instruction.qubits],
        )
    return stripped


def normalize(circuit: QuantumCircuit) -> QuantumCircuit:
    return transpile(
        strip_terminal_io(circuit),
        basis_gates=["u", "cx"],
        optimization_level=0,
        seed_transpiler=0,
    )


def routing_skeleton(circuit: QuantumCircuit) -> list[list[object]]:
    gates: list[list[object]] = []
    for instruction in circuit.data:
        indices = [
            circuit.find_bit(qubit).index for qubit in instruction.qubits
        ]
        if len(indices) == 1:
            gates.append(["h", indices[0]])
        elif len(indices) == 2 and instruction.operation.name == "cx":
            gates.append(["cx", indices[0], indices[1]])
        elif len(indices) == 0:
            continue
        else:
            raise ValueError(
                f"normalization produced unsupported operation "
                f"{instruction.operation.name}/{len(indices)}"
            )
    return gates


def heavy_hex_case(
    item: dict[str, object],
    source_path: Path,
    normalized: QuantumCircuit,
) -> dict[str, object]:
    logical_qubits = int(item["qubits"])
    if normalized.num_qubits != logical_qubits:
        raise ValueError(
            f"{item['id']}: expected {logical_qubits} qubits, "
            f"loaded {normalized.num_qubits}"
        )
    distance = 3 if logical_qubits <= 19 else 5
    coupling = CouplingMap.from_heavy_hex(distance, bidirectional=True)
    undirected = sorted(
        {tuple(sorted((int(u), int(v)))) for u, v in coupling.get_edges()}
    )
    gates = routing_skeleton(normalized)
    return {
        "name": f"v7_qasmbench_{item['id']}_heavy_hex_d{distance}",
        "num_qubits": logical_qubits,
        "physical_qubits": coupling.size(),
        "edges": [list(edge) for edge in undirected],
        "gate_specs": gates,
        "initial_mapping": list(range(logical_qubits)),
        "split": "external_validation",
        "topology": f"heavy_hex_d{distance}",
        "circuit_mode": str(item["id"]),
        "seed": 0,
        "source": {
            "collection": "QASMBench",
            "commit": "357b942396d5c2b7cbc1c229c585a6ef5ccaebac",
            "path": str(item["path"]),
            "source_sha256": sha256(source_path),
            "normalized_gate_count": len(normalized.data),
            "routing_gate_count": len(gates),
            "routing_cx_count": sum(gate[0] == "cx" for gate in gates),
        },
    }


def write_config(path: Path, dataset_hash: str) -> None:
    text = f"""schema_version = 1
name = "v7_external_validation"
dataset = "V7"
input = "data/v7/external_cases.json"
input_sha256 = "{dataset_hash}"

[portfolio]
qiskit_seeds = 20
qiskit_repeats = 1
cross_layout_layouts = 0
cross_layout_neighbors = 4
proxy_decay = 0.95
selection_mode = "deterministic"

[execution]
workers = 1
schedule = "contiguous"
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def output_targets(root: Path, include_sources: bool) -> list[Path]:
    targets = [
        root / "v7" / "normalized",
        root / "data" / "v7",
        root / "configs" / "v7_external.toml",
    ]
    if include_sources:
        targets.insert(0, root / "v7" / "sources")
    return targets


def require_fresh_targets(root: Path, include_sources: bool) -> None:
    existing = [
        path for path in output_targets(root, include_sources) if path.exists()
    ]
    if existing:
        rendered = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "V7 preparation targets already exist. Use a fresh release "
            "directory; do not merge partial preparation outputs:\n"
            + rendered
        )


def publish_outputs(
    staging_root: Path,
    root: Path,
    include_sources: bool,
) -> None:
    relative_paths = [
        Path("v7/normalized"),
        Path("data/v7"),
        Path("configs/v7_external.toml"),
    ]
    if include_sources:
        relative_paths.insert(0, Path("v7/sources"))

    published: list[Path] = []
    try:
        for relative in relative_paths:
            source = staging_root / relative
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            published.append(destination)
    except Exception:
        for path in reversed(published):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="prepare the frozen V7 inputs")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    manifest_path = root / "v7" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if (root / "v7" / "FROZEN_MANIFEST.json").exists():
        raise FileExistsError(
            "V7 is already frozen; prepare into a fresh release instead"
        )
    include_sources = not args.offline
    require_fresh_targets(root, include_sources=include_sources)
    if args.offline and not (root / "v7" / "sources").is_dir():
        raise FileNotFoundError(
            "offline preparation requires a complete v7/sources directory"
        )

    v7_root = root / "v7"
    v7_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".prepare-v7-",
        dir=v7_root,
    ) as temporary:
        staging_root = Path(temporary)
        source_root = (
            staging_root / "v7" / "sources"
            if include_sources
            else root / "v7" / "sources"
        )
        normalized_root = staging_root / "v7" / "normalized"

        license_files = {
            "QASMBench_LICENSE.txt": "LICENSE",
            "QASMBench_NOTICE.txt": "NOTICE",
        }
        for local_name, remote_name in license_files.items():
            path = source_root / local_name
            if include_sources:
                download(f"{COMMIT_BASE}/{remote_name}", path)
            if not path.is_file():
                raise FileNotFoundError(path)

        resolved_sources: list[
            tuple[dict[str, object], Path, str]
        ] = []
        for item in manifest["circuits"]:
            source_path = source_root / str(item["path"])
            if include_sources:
                download(f"{COMMIT_BASE}/{item['path']}", source_path)
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            actual = sha256(source_path)
            if actual != item["sha256"]:
                raise RuntimeError(
                    f"{item['id']}: source hash mismatch; "
                    f"expected {item['sha256']}, got {actual}"
                )
            resolved_sources.append((item, source_path, actual))

        cases: list[dict[str, object]] = []
        source_records: list[dict[str, object]] = []
        normalized_root.mkdir(parents=True, exist_ok=True)
        for item, source_path, actual in resolved_sources:
            logical = qasm2.load(str(source_path))
            normalized = normalize(logical)
            qpy_path = normalized_root / f"{item['id']}.qpy"
            with qpy_path.open("wb") as handle:
                qpy.dump(normalized, handle)
            qasm_path = normalized_root / f"{item['id']}.qasm"
            qasm_path.write_text(
                qasm3.dumps(normalized),
                encoding="utf-8",
                newline="\n",
            )
            case = heavy_hex_case(item, source_path, normalized)
            cases.append(case)
            source_records.append(
                {
                    "id": item["id"],
                    "source_sha256": actual,
                    "normalized_qpy_sha256": sha256(qpy_path),
                    "normalized_qasm_sha256": sha256(qasm_path),
                }
            )
            print(
                f"{item['id']}: q={case['num_qubits']}, "
                f"physical={case['physical_qubits']}, "
                f"CX={case['source']['routing_cx_count']}"
            )

        output = staging_root / "data" / "v7" / "external_cases.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 7,
            "case_count": len(cases),
            "design": manifest,
            "source_records": source_records,
            "cases": cases,
        }
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        dataset_hash = sha256(output)
        write_config(
            staging_root / "configs" / "v7_external.toml",
            dataset_hash,
        )
        publish_outputs(
            staging_root,
            root,
            include_sources=include_sources,
        )

    final_output = root / "data" / "v7" / "external_cases.json"
    print(f"dataset: {final_output}")
    print(f"sha256: {sha256(final_output)}")


if __name__ == "__main__":
    main()
