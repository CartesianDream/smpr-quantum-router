from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import stage16_generate_benchmark_v3 as stage16


SCHEMA_VERSION = 6
REPLICATES = 2
QUBIT_OFFSETS = (1, 2)
SEED_DERIVATION_TAG = "quantum-router/v6-blind/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_master_seed(frozen_manifest: Path) -> tuple[int, str, str]:
    manifest_sha256 = sha256(frozen_manifest)
    material = f"{SEED_DERIVATION_TAG}:{manifest_sha256}".encode("ascii")
    derivation_sha256 = hashlib.sha256(material).hexdigest()
    # 31-bit positive seed keeps the value portable across tooling while the
    # complete digest remains recorded in the dataset manifest.
    master_seed = int(derivation_sha256[:8], 16) & 0x7FFFFFFF
    if master_seed == 0:
        master_seed = 1
    return master_seed, manifest_sha256, derivation_sha256


def build_profiles() -> list[tuple[int, stage16.StratifiedProfile]]:
    profiles: list[tuple[int, stage16.StratifiedProfile]] = []
    global_id = 0

    for replicate, qubit_offset in enumerate(QUBIT_OFFSETS):
        for topology_index, topology in enumerate(stage16.TOPOLOGIES):
            for mode_index, circuit_mode in enumerate(stage16.CIRCUIT_MODES):
                for factor_index, length_factor in enumerate(
                    stage16.LENGTH_FACTORS
                ):
                    qubit_index = (
                        mode_index + factor_index + qubit_offset
                    ) % len(stage16.QUBIT_SIZES)
                    profiles.append(
                        (
                            replicate,
                            stage16.StratifiedProfile(
                                profile_id=global_id,
                                topology_index=topology_index,
                                circuit_mode_index=mode_index,
                                length_factor_index=factor_index,
                                topology=topology,
                                circuit_mode=circuit_mode,
                                length_factor=length_factor,
                                num_qubits=stage16.QUBIT_SIZES[qubit_index],
                                split="test",
                            ),
                        )
                    )
                    global_id += 1

    return profiles


def build_cases(
    profiles: list[tuple[int, stage16.StratifiedProfile]],
    master_seed: int,
) -> list[stage16.StratifiedCase]:
    cases: list[stage16.StratifiedCase] = []

    for index, (replicate, profile) in enumerate(profiles):
        generated = stage16.build_case(profile, index, master_seed)
        cases.append(
            replace(
                generated,
                name=(
                    f"blind_v6_{index:03d}_r{replicate}_"
                    f"{profile.topology}_n{profile.num_qubits}_"
                    f"{profile.circuit_mode}_f{profile.length_factor}_"
                    f"cx{generated.two_qubit_gate_count}"
                ),
            )
        )

    return cases


def validate_design(
    profiles: list[tuple[int, stage16.StratifiedProfile]],
    cases: list[stage16.StratifiedCase],
) -> None:
    expected_count = 60 * REPLICATES
    if len(profiles) != expected_count or len(cases) != expected_count:
        raise ValueError(
            f"V6 must contain {expected_count} cases, got {len(cases)}"
        )
    if len({case.name for case in cases}) != expected_count:
        raise ValueError("V6 case names are not unique")
    if len({case.profile_id for case in cases}) != expected_count:
        raise ValueError("V6 profile IDs are not unique")
    if len({case.seed for case in cases}) != expected_count:
        raise ValueError("V6 case seeds are not unique")

    expected_margins = {
        "topology": 30,
        "circuit_mode": 24,
        "length_factor": 40,
        "num_qubits": 40,
    }
    for field, expected in expected_margins.items():
        counts = Counter(getattr(case, field) for case in cases)
        if set(counts.values()) != {expected}:
            raise ValueError(f"unbalanced V6 {field}: {dict(counts)}")

    cells = Counter(
        (case.topology, case.circuit_mode, case.length_factor)
        for case in cases
    )
    if len(cells) != 60 or set(cells.values()) != {REPLICATES}:
        raise ValueError("V6 factorial cells are not exactly duplicated")

    for case in cases:
        stage16.validate_case(case)


def save_dataset(
    output: Path,
    frozen_manifest: Path,
    profiles: list[tuple[int, stage16.StratifiedProfile]],
    cases: list[stage16.StratifiedCase],
    master_seed: int,
    manifest_sha256: str,
    derivation_sha256: str,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "master_seed": master_seed,
        "case_count": len(cases),
        "design": {
            "purpose": "blind holdout after the Stage 53/V5 freeze",
            "seed_derivation_tag": SEED_DERIVATION_TAG,
            "frozen_manifest": str(frozen_manifest.resolve()),
            "frozen_manifest_sha256": manifest_sha256,
            "seed_derivation_sha256": derivation_sha256,
            "replicates": REPLICATES,
            "qubit_offsets": list(QUBIT_OFFSETS),
            "topologies": list(stage16.TOPOLOGIES),
            "circuit_modes": list(stage16.CIRCUIT_MODES),
            "length_factors": list(stage16.LENGTH_FACTORS),
            "qubit_sizes": list(stage16.QUBIT_SIZES),
            "profile_rule": (
                "two independently seeded realizations of every "
                "topology x mode x length-factor cell"
            ),
            "qubit_rule": (
                "(mode_index + factor_index + qubit_offset[replicate]) mod 3"
            ),
            "two_qubit_gate_rule": "CX = length_factor * num_qubits",
            "frozen_evaluation": {
                "qiskit_seeds": 20,
                "qiskit_repeats": 1,
                "hybrid_layouts": 0,
                "hybrid_neighbors": 4,
                "proxy_decay": 0.95,
                "selection": (
                    "min(SWAP, weighted_depth), LightSABRE on exact tie"
                ),
            },
            "preregistered_acceptance": {
                "semantic_topology_metric": "120/120",
                "per_case_swap_regressions": 0,
                "selected_total_swap": "<= LightSABRE total SWAP",
                "primary_inference": (
                    "upper endpoint of paired bootstrap 95% CI for mean "
                    "(selected_swap-lightsabre_swap)/CX is below zero"
                ),
                "bootstrap_samples": 10000,
                "bootstrap_seed": 20260723,
            },
        },
        "profiles": [
            {"replicate": replicate, **asdict(profile)}
            for replicate, profile in profiles
        ],
        "cases": [stage16.case_to_dict(case) for case in cases],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def print_counts(cases: list[stage16.StratifiedCase]) -> None:
    for field in (
        "topology",
        "circuit_mode",
        "length_factor",
        "num_qubits",
    ):
        counts = Counter(getattr(case, field) for case in cases)
        print(f"{field}: {dict(sorted(counts.items(), key=str))}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 55: deterministically generate the blind V6 holdout"
    )
    parser.add_argument(
        "--frozen-manifest",
        type=Path,
        default=Path(
            "results/stage54_frozen_stage53/FROZEN_SHA256SUMS.txt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks_v6/blind_test.json"),
    )
    args = parser.parse_args()

    if not args.frozen_manifest.is_file():
        raise FileNotFoundError(args.frozen_manifest)

    master_seed, manifest_sha256, derivation_sha256 = derive_master_seed(
        args.frozen_manifest
    )
    profiles = build_profiles()
    cases = build_cases(profiles, master_seed)
    validate_design(profiles, cases)
    save_dataset(
        args.output,
        args.frozen_manifest,
        profiles,
        cases,
        master_seed,
        manifest_sha256,
        derivation_sha256,
    )

    print("===== Stage 55 blind V6 generated =====")
    print("frozen manifest:", args.frozen_manifest.resolve())
    print("frozen manifest SHA256:", manifest_sha256)
    print("seed derivation SHA256:", derivation_sha256)
    print("master seed:", master_seed)
    print("cases:", len(cases))
    print_counts(cases)
    print("output:", args.output.resolve())
    print("output SHA256:", sha256(args.output))


if __name__ == "__main__":
    main()
