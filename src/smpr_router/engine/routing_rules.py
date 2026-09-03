from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import structural_policy as structural_policy
import mapping_portfolio as mapping_portfolio
import frozen_policy as frozen_policy
import layout_proxy as layout_proxy

from case_io import DevCase, load_cases
from validation_support import compare_quality, run_qiskit_multiseed
from statistical_support import bootstrap_mean_ci


PROXY_DECAY = 0.95
PROXY_TOP = 8
ROLLOUT_TOP = 2


@dataclass
class RouteOutcome:
    variant: str
    mapping_budget: str
    selected_mapping: str
    mapping_candidates_scored: int
    rollout_mappings: int
    swap: int
    depth: int
    runtime_ms: float


@dataclass
class FinalCaseResult:
    instance: str
    split: str
    topology: str
    circuit_mode: str
    num_qubits: int
    length_factor: int
    cx_count: int

    router_variant: str
    mapping_budget: str
    selected_mapping: str
    mapping_candidates_scored: int
    rollout_mappings: int
    router_swap: int
    router_depth: int
    router_runtime_ms: float

    lightsabre_best_seed: int
    lightsabre_swap: int
    lightsabre_depth: int
    lightsabre_budget_ms: float

    swap_gap: int
    depth_gap: int
    normalized_swap_gap: float
    outcome: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_metadata(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(case["name"]): case for case in payload["cases"]}


def configure_router() -> structural_policy.StructuralConfig:
    structural_policy.one_step_policy.THETA = 5
    structural_policy.one_step_policy.W_FUTURE = 0.5
    structural_policy.one_step_policy.GAMMA = 0.5
    structural_policy.one_step_policy.ALPHA = 0.5
    structural_policy.one_step_policy.LAMBDA_DEPTH = 0.0
    return structural_policy.CONFIG_BY_NAME["ordered_potential"]


def frozen_variant(case: DevCase) -> str:
    if (
        case.topology == "ring"
        and case.num_qubits == 10
        and case.circuit_mode == "far"
    ):
        return "local_layout_search_ring_n10_far"
    if (
        case.topology == "grid"
        and case.num_qubits >= 8
        and case.circuit_mode == "uniform"
    ):
        return "layout_proxy_grid_uniform_k2"
    if (
        case.topology == "grid"
        and case.num_qubits == 10
        and case.circuit_mode == "alternating"
    ):
        return "grid_alternating_policy_grid_n10_alternating_k4"
    return "adaptive_mapping"


def run_frozen_router(
    case: DevCase,
    config: structural_policy.StructuralConfig,
    lightsabre_swap: int,
    lightsabre_depth: int,
) -> RouteOutcome:
    variant = frozen_variant(case)

    if variant == "local_layout_search_ring_n10_far":
        (
            _variant,
            mapping_budget,
            selected_mapping,
            candidates,
            rollout_mappings,
            _old_swap,
            _old_depth,
            swap,
            depth,
            runtime_ms,
        ) = frozen_policy.run_special_case_with_depths(case, config)
        return RouteOutcome(
            variant=variant,
            mapping_budget=mapping_budget,
            selected_mapping=selected_mapping,
            mapping_candidates_scored=candidates,
            rollout_mappings=rollout_mappings,
            swap=swap,
            depth=depth,
            runtime_ms=runtime_ms,
        )

    if variant in {
        "layout_proxy_grid_uniform_k2",
        "grid_alternating_policy_grid_n10_alternating_k4",
    }:
        top_k = 2 if variant == "layout_proxy_grid_uniform_k2" else 4
        row, _mapping_rows = layout_proxy.run_case(
            case=case,
            config=config,
            reference={
                "lightsabre_swap": str(lightsabre_swap),
                "lightsabre_depth": str(lightsabre_depth),
            },
            proxy_decay=PROXY_DECAY,
            proxy_top=PROXY_TOP,
            rollout_top=ROLLOUT_TOP,
            top_k=top_k,
            use_forward_backward=True,
        )
        return RouteOutcome(
            variant=variant,
            mapping_budget=f"proxy-d095-top8/rollout2-k{top_k}",
            selected_mapping=row.searched_mapping,
            mapping_candidates_scored=row.mapping_candidates_scored,
            rollout_mappings=row.rollout_mappings,
            swap=row.searched_swap,
            depth=row.searched_depth,
            runtime_ms=row.runtime_ms,
        )

    (
        mapping_budget,
        selected_mapping,
        candidates,
        rollout_mappings,
        _adaptive_mapping_swap,
        _adaptive_mapping_depth,
        swap,
        depth,
        runtime_ms,
    ) = frozen_policy.run_adaptive_mapping_case(case, config)
    return RouteOutcome(
        variant="adaptive_mapping",
        mapping_budget=mapping_budget,
        selected_mapping=selected_mapping,
        mapping_candidates_scored=candidates,
        rollout_mappings=rollout_mappings,
        swap=swap,
        depth=depth,
        runtime_ms=runtime_ms,
    )


def save_csv(path: Path, rows: list[FinalCaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FinalCaseResult.__annotations__.keys(),
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def summarize_group(rows: list[FinalCaseResult]) -> dict:
    router_swap = sum(row.router_swap for row in rows)
    lightsabre_swap = sum(row.lightsabre_swap for row in rows)
    outcomes = Counter(row.outcome for row in rows)
    normalized = [row.normalized_swap_gap for row in rows]
    return {
        "case_count": len(rows),
        "total_cx": sum(row.cx_count for row in rows),
        "router_swap": router_swap,
        "lightsabre_swap": lightsabre_swap,
        "swap_gap": router_swap - lightsabre_swap,
        "relative_swap_gap": (
            (router_swap - lightsabre_swap) / lightsabre_swap
            if lightsabre_swap
            else 0.0
        ),
        "router_depth": sum(row.router_depth for row in rows),
        "lightsabre_depth": sum(row.lightsabre_depth for row in rows),
        "depth_gap": sum(row.depth_gap for row in rows),
        "wins": outcomes["independent rollout胜"],
        "ties": outcomes["平局"],
        "losses": outcomes["Qiskit胜"],
        "mean_normalized_swap_gap": statistics.mean(normalized),
        "median_normalized_swap_gap": statistics.median(normalized),
        "router_runtime_ms": sum(row.router_runtime_ms for row in rows),
        "lightsabre_budget_ms": sum(
            row.lightsabre_budget_ms for row in rows
        ),
    }


def grouped_summary(
    rows: list[FinalCaseResult],
    attribute: str,
) -> dict[str, dict]:
    groups: dict[str, list[FinalCaseResult]] = defaultdict(list)
    for row in rows:
        groups[str(getattr(row, attribute))].append(row)
    return {
        value: summarize_group(group)
        for value, group in sorted(groups.items(), key=lambda item: item[0])
    }


def summarize(
    input_path: Path,
    rows: list[FinalCaseResult],
    bootstrap_samples: int,
    bootstrap_seed: int,
    qiskit_seeds: int,
    qiskit_repeats: int,
) -> dict:
    result = summarize_group(rows)
    normalized = [row.normalized_swap_gap for row in rows]
    low, high = bootstrap_mean_ci(
        normalized,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    result.update(
        {
            "dataset": str(input_path.resolve()),
            "dataset_sha256": sha256_file(input_path),
            "router": "routing_rules_frozen_final",
            "qiskit_version_note": "Qiskit SABRE auto-layout + SABRE routing",
            "qiskit_seeds": qiskit_seeds,
            "qiskit_repeats": qiskit_repeats,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_ci95_low": low,
            "bootstrap_ci95_high": high,
            "frozen_rules": {
                "ring_n10_far": "historical development beam2/round1/rollout2/k2",
                "grid_uniform_n_ge_8": "historical development proxy decay .95/top8/k2",
                "grid_n10_alternating": "historical development proxy decay .95/top8/k4",
                "otherwise": "historical development adaptive mapping/full rollout k2",
            },
            "variant_counts": dict(
                Counter(row.router_variant for row in rows)
            ),
            "by_topology": grouped_summary(rows, "topology"),
            "by_mode": grouped_summary(rows, "circuit_mode"),
            "by_num_qubits": grouped_summary(rows, "num_qubits"),
            "by_length_factor": grouped_summary(rows, "length_factor"),
            "by_router_variant": grouped_summary(rows, "router_variant"),
        }
    )
    return result


def write_summary(
    path: Path,
    input_path: Path,
    rows: list[FinalCaseResult],
    args: argparse.Namespace,
) -> dict:
    result = summarize(
        input_path=input_path,
        rows=rows,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        qiskit_seeds=args.qiskit_seeds,
        qiskit_repeats=args.qiskit_repeats,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="historical development：冻结统一路由器在 V5 最终留出集上对比 LightSABRE"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmarks_v5/final_test.json"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-indices", type=str, default=None)
    parser.add_argument("--qiskit-seeds", type=int, default=20)
    parser.add_argument("--qiskit-repeats", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    parser.add_argument(
        "--case-output",
        type=Path,
        default=Path("results/routing_rules_v5_final_cases.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/routing_rules_v5_final_summary.json"),
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"找不到输入：{args.input.resolve()}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须为正整数")
    if min(
        args.qiskit_seeds,
        args.qiskit_repeats,
        args.bootstrap_samples,
    ) <= 0:
        raise ValueError("Qiskit 与 bootstrap 参数必须为正整数")

    cases = mapping_portfolio.choose_cases(
        load_cases(args.input, limit=None),
        mapping_portfolio.parse_indices(args.case_indices),
        args.limit,
    )
    metadata = load_metadata(args.input)
    config = configure_router()

    print("===== historical development：V5 冻结最终测试 =====")
    print(f"输入：{args.input.resolve()}")
    print(f"实例数：{len(cases)}")
    print("模型参数与条件规则：全部冻结")
    print(
        f"LightSABRE：{args.qiskit_seeds} seeds x "
        f"{args.qiskit_repeats} repeats"
    )

    rows: list[FinalCaseResult] = []
    for index, case in enumerate(cases, start=1):
        variant = frozen_variant(case)
        print(
            f"[{index}/{len(cases)}] {case.name} "
            f"[{variant}] ...",
            flush=True,
        )
        lightsabre, lightsabre_budget_ms = run_qiskit_multiseed(
            case=case,
            seeds=args.qiskit_seeds,
            repeats=args.qiskit_repeats,
        )
        lightsabre_swap = int(lightsabre.swap_count)
        lightsabre_depth = int(lightsabre.weighted_depth)
        routed = run_frozen_router(
            case=case,
            config=config,
            lightsabre_swap=lightsabre_swap,
            lightsabre_depth=lightsabre_depth,
        )
        outcome = compare_quality(
            swap_count=routed.swap,
            depth=routed.depth,
            qiskit_swap=lightsabre_swap,
            qiskit_depth=lightsabre_depth,
        )
        case_meta = metadata[case.name]
        row = FinalCaseResult(
            instance=case.name,
            split=case.split,
            topology=case.topology,
            circuit_mode=case.circuit_mode,
            num_qubits=case.num_qubits,
            length_factor=int(case_meta["length_factor"]),
            cx_count=case.two_qubit_gate_count,
            router_variant=routed.variant,
            mapping_budget=routed.mapping_budget,
            selected_mapping=routed.selected_mapping,
            mapping_candidates_scored=routed.mapping_candidates_scored,
            rollout_mappings=routed.rollout_mappings,
            router_swap=routed.swap,
            router_depth=routed.depth,
            router_runtime_ms=routed.runtime_ms,
            lightsabre_best_seed=int(lightsabre.seed),
            lightsabre_swap=lightsabre_swap,
            lightsabre_depth=lightsabre_depth,
            lightsabre_budget_ms=lightsabre_budget_ms,
            swap_gap=routed.swap - lightsabre_swap,
            depth_gap=routed.depth - lightsabre_depth,
            normalized_swap_gap=(
                (routed.swap - lightsabre_swap) / case.two_qubit_gate_count
            ),
            outcome=outcome,
        )
        rows.append(row)
        save_csv(args.case_output, rows)
        write_summary(args.summary_output, args.input, rows, args)

        print(
            f"  router={routed.swap}/{routed.depth}, "
            f"LightSABRE={lightsabre_swap}/{lightsabre_depth} "
            f"(seed={lightsabre.seed}), gap={row.swap_gap:+d}, "
            f"{outcome}, budget="
            f"{routed.runtime_ms:.1f}/{lightsabre_budget_ms:.1f} ms"
        )

    result = write_summary(args.summary_output, args.input, rows, args)
    save_csv(args.case_output, rows)

    print()
    print("===== historical development 最终汇总 =====")
    print(f"冻结路由器 total SWAP：{result['router_swap']}")
    print(f"LightSABRE total SWAP：{result['lightsabre_swap']}")
    print(f"SWAP gap：{result['swap_gap']:+d}")
    print(f"相对差：{result['relative_swap_gap']:+.2%}")
    print(
        f"逐实例胜/平/负："
        f"{result['wins']}/{result['ties']}/{result['losses']}"
    )
    print(
        "平均归一化 SWAP gap 的 bootstrap 95% CI："
        f"[{result['bootstrap_ci95_low']:+.4f}, "
        f"{result['bootstrap_ci95_high']:+.4f}]"
    )
    print(
        f"总预算 router/LightSABRE："
        f"{result['router_runtime_ms']:.1f}/"
        f"{result['lightsabre_budget_ms']:.1f} ms"
    )
    print(f"逐实例结果：{args.case_output.resolve()}")
    print(f"汇总 JSON：{args.summary_output.resolve()}")


if __name__ == "__main__":
    main()
