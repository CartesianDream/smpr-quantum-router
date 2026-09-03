from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import structural_policy as structural_policy
import mapping_portfolio as mapping_portfolio
import rollout_policy as rollout_policy
import adaptive_mapping as adaptive_mapping
import local_layout_search as local_layout_search

from case_io import DevCase, load_cases
from validation_support import compare_quality, run_qiskit_multiseed


# historical development 在 V4 dev、V4b validation、V4c test 上冻结出的唯一专门规则。
SPECIAL_TOPOLOGY = "ring"
SPECIAL_NUM_QUBITS = 10
SPECIAL_CIRCUIT_MODE = "far"

# 所有搜索预算均已冻结；historical development 不再做参数选择。
POOL_MODES = {"parallel_layers"}
DEFAULT_MAPPING = "v1_best"
DEFAULT_TOP_K = 2
POOL_TOP_K = 2
BEAM_WIDTH = 2
SEARCH_ROUNDS = 1
ROLLOUT_TOP = 2
USE_FORWARD_BACKWARD = True


@dataclass
class CaseResult:
    instance: str
    split: str
    topology: str
    circuit_mode: str
    num_qubits: int
    cx_count: int

    router_variant: str
    mapping_budget: str
    selected_mapping: str
    mapping_candidates_scored: int
    rollout_mappings: int

    adaptive_mapping_swap: int
    adaptive_mapping_depth: int
    adaptive_swap: int
    adaptive_depth: int
    improvement_vs_adaptive_mapping: int
    adaptive_runtime_ms: float

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


def configure_frozen_router() -> structural_policy.StructuralConfig:
    structural_policy.one_step_policy.THETA = 5
    structural_policy.one_step_policy.W_FUTURE = 0.5
    structural_policy.one_step_policy.GAMMA = 0.5
    structural_policy.one_step_policy.ALPHA = 0.5
    structural_policy.one_step_policy.LAMBDA_DEPTH = 0.0
    return structural_policy.CONFIG_BY_NAME["ordered_potential"]


def use_special_mapping_search(case: DevCase) -> bool:
    return (
        case.topology == SPECIAL_TOPOLOGY
        and case.num_qubits == SPECIAL_NUM_QUBITS
        and case.circuit_mode == SPECIAL_CIRCUIT_MODE
    )


def run_adaptive_mapping_case(
    case: DevCase,
    config: structural_policy.StructuralConfig,
) -> tuple[
    str,
    str,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
]:
    mapping_mode, top_k = adaptive_mapping.budget_for_case(
        case=case,
        pool_modes=POOL_MODES,
        default_mapping=DEFAULT_MAPPING,
        default_top_k=DEFAULT_TOP_K,
        pool_top_k=POOL_TOP_K,
    )
    mappings = mapping_portfolio.make_mappings(case, mapping_mode)
    attempts = [
        rollout_policy.run_attempt(case, mapping, config, top_k)
        for mapping in mappings
    ]
    row = rollout_policy.build_case_row(
        case=case,
        mappings=mappings,
        attempts=attempts,
        base_config=config,
        top_k=top_k,
    )
    return (
        f"{mapping_mode}/k{top_k}",
        row.rollout_mapping,
        len(mappings),
        len(mappings),
        row.rollout_swap,
        row.rollout_depth,
        row.rollout_swap,
        row.rollout_depth,
        row.total_runtime_ms,
    )


def run_adaptive_case(
    case: DevCase,
    config: structural_policy.StructuralConfig,
) -> tuple[
    str,
    str,
    str,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
]:
    (
        budget,
        selected_mapping,
        candidates,
        rollout_mappings,
        adaptive_mapping_swap,
        adaptive_mapping_depth,
        adaptive_swap,
        adaptive_depth,
        runtime_ms,
    ) = run_adaptive_mapping_case(case, config)
    return (
        "adaptive_mapping",
        budget,
        selected_mapping,
        candidates,
        rollout_mappings,
        adaptive_mapping_swap,
        adaptive_mapping_depth,
        adaptive_swap,
        adaptive_depth,
        runtime_ms,
    )


def run_special_case_with_depths(
    case: DevCase,
    config: structural_policy.StructuralConfig,
) -> tuple[
    str,
    str,
    str,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
]:
    """historical development 路径；同时恢复旧 historical development 的可比较深度。"""
    summary, mapping_rows = local_layout_search.run_case(
        case=case,
        config=config,
        beam_width=BEAM_WIDTH,
        search_rounds=SEARCH_ROUNDS,
        rollout_top=ROLLOUT_TOP,
        top_k=DEFAULT_TOP_K,
        use_forward_backward=USE_FORWARD_BACKWARD,
    )
    selected_row = next(
        row
        for row in mapping_rows
        if row.mapping_name == summary.selected_mapping
        and row.selected_for_rollout
        and row.rollout_swap == summary.selected_rollout_swap
    )
    old_row = next(
        row
        for row in mapping_rows
        if row.mapping_name == summary.old_mapping
        and row.selected_for_rollout
    )
    if selected_row.rollout_depth is None or old_row.rollout_depth is None:
        raise RuntimeError(f"{case.name} 缺少 historical development rollout 深度")
    return (
        "local_layout_search_ring_n10_far",
        "local-search-b2-r1/rollout2-k2",
        summary.selected_mapping,
        summary.mapping_candidates_scored,
        summary.rollout_mappings,
        summary.old_rollout_swap,
        int(old_row.rollout_depth),
        summary.selected_rollout_swap,
        int(selected_row.rollout_depth),
        summary.total_runtime_ms,
    )


def route_case(
    case: DevCase,
    config: structural_policy.StructuralConfig,
) -> tuple[
    str,
    str,
    str,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
]:
    if use_special_mapping_search(case):
        return run_special_case_with_depths(case, config)
    return run_adaptive_case(case, config)


def save_csv(path: Path, rows: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CaseResult.__annotations__.keys(),
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def summarize_group(rows: list[CaseResult]) -> dict:
    adaptive_swap = sum(row.adaptive_swap for row in rows)
    lightsabre_swap = sum(row.lightsabre_swap for row in rows)
    adaptive_mapping_swap = sum(row.adaptive_mapping_swap for row in rows)
    outcomes = Counter(row.outcome for row in rows)
    return {
        "case_count": len(rows),
        "adaptive_mapping_swap": adaptive_mapping_swap,
        "adaptive_swap": adaptive_swap,
        "improvement_vs_adaptive_mapping": adaptive_mapping_swap - adaptive_swap,
        "lightsabre_swap": lightsabre_swap,
        "swap_gap": adaptive_swap - lightsabre_swap,
        "relative_swap_gap": (
            (adaptive_swap - lightsabre_swap) / lightsabre_swap
            if lightsabre_swap
            else 0.0
        ),
        "adaptive_depth": sum(row.adaptive_depth for row in rows),
        "lightsabre_depth": sum(row.lightsabre_depth for row in rows),
        "wins": outcomes["independent rollout胜"],
        "ties": outcomes["平局"],
        "losses": outcomes["Qiskit胜"],
        "adaptive_runtime_ms": sum(row.adaptive_runtime_ms for row in rows),
        "lightsabre_budget_ms": sum(row.lightsabre_budget_ms for row in rows),
    }


def summarize(path: Path, rows: list[CaseResult]) -> dict:
    grouped_topology: dict[str, list[CaseResult]] = defaultdict(list)
    grouped_mode: dict[str, list[CaseResult]] = defaultdict(list)
    for row in rows:
        grouped_topology[row.topology].append(row)
        grouped_mode[row.circuit_mode].append(row)

    result = summarize_group(rows)
    result.update(
        {
            "dataset": str(path.resolve()),
            "dataset_sha256": sha256_file(path),
            "router": "frozen_policy_frozen_adaptive",
            "special_rule": "ring && n==10 && mode==far",
            "special_cases": sum(
                row.router_variant == "local_layout_search_ring_n10_far" for row in rows
            ),
            "by_topology": {
                key: summarize_group(group)
                for key, group in sorted(grouped_topology.items())
            },
            "by_mode": {
                key: summarize_group(group)
                for key, group in sorted(grouped_mode.items())
            },
        }
    )
    return result


def write_summary(path: Path, input_path: Path, rows: list[CaseResult]) -> dict:
    result = summarize(input_path, rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "historical development：冻结的自适应路由器与多种子 Qiskit LightSABRE 比较"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmarks_v4c/test.json"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-indices", type=str, default=None)
    parser.add_argument("--qiskit-seeds", type=int, default=20)
    parser.add_argument("--qiskit-repeats", type=int, default=2)
    parser.add_argument(
        "--case-output",
        type=Path,
        default=Path("results/frozen_policy_vs_lightsabre_cases.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/frozen_policy_vs_lightsabre_summary.json"),
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"找不到输入：{args.input.resolve()}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须为正整数")
    if args.qiskit_seeds <= 0 or args.qiskit_repeats <= 0:
        raise ValueError("Qiskit seeds/repeats 必须为正整数")

    cases = mapping_portfolio.choose_cases(
        load_cases(args.input, limit=None),
        mapping_portfolio.parse_indices(args.case_indices),
        args.limit,
    )
    config = configure_frozen_router()

    print("===== historical development：冻结路由器 vs LightSABRE =====")
    print(f"输入：{args.input.resolve()}")
    print(f"实例数：{len(cases)}")
    print("普通实例：historical development")
    print("专门规则：ring && n==10 && mode==far -> historical development")
    print(
        f"LightSABRE：Qiskit SABRE auto-layout, "
        f"{args.qiskit_seeds} seeds x {args.qiskit_repeats} repeats"
    )

    rows: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        (
            variant,
            mapping_budget,
            selected_mapping,
            mapping_candidates,
            rollout_mappings,
            adaptive_mapping_swap,
            adaptive_mapping_depth,
            adaptive_swap,
            adaptive_depth,
            adaptive_runtime_ms,
        ) = route_case(case, config)

        lightsabre, lightsabre_budget_ms = run_qiskit_multiseed(
            case=case,
            seeds=args.qiskit_seeds,
            repeats=args.qiskit_repeats,
        )
        lightsabre_swap = int(lightsabre.swap_count)
        lightsabre_depth = int(lightsabre.weighted_depth)
        outcome = compare_quality(
            swap_count=adaptive_swap,
            depth=adaptive_depth,
            qiskit_swap=lightsabre_swap,
            qiskit_depth=lightsabre_depth,
        )
        row = CaseResult(
            instance=case.name,
            split=case.split,
            topology=case.topology,
            circuit_mode=case.circuit_mode,
            num_qubits=case.num_qubits,
            cx_count=case.two_qubit_gate_count,
            router_variant=variant,
            mapping_budget=mapping_budget,
            selected_mapping=selected_mapping,
            mapping_candidates_scored=mapping_candidates,
            rollout_mappings=rollout_mappings,
            adaptive_mapping_swap=adaptive_mapping_swap,
            adaptive_mapping_depth=adaptive_mapping_depth,
            adaptive_swap=adaptive_swap,
            adaptive_depth=adaptive_depth,
            improvement_vs_adaptive_mapping=adaptive_mapping_swap - adaptive_swap,
            adaptive_runtime_ms=adaptive_runtime_ms,
            lightsabre_best_seed=int(lightsabre.seed),
            lightsabre_swap=lightsabre_swap,
            lightsabre_depth=lightsabre_depth,
            lightsabre_budget_ms=lightsabre_budget_ms,
            swap_gap=adaptive_swap - lightsabre_swap,
            depth_gap=adaptive_depth - lightsabre_depth,
            normalized_swap_gap=(
                (adaptive_swap - lightsabre_swap) / case.two_qubit_gate_count
            ),
            outcome=outcome,
        )
        rows.append(row)
        save_csv(args.case_output, rows)
        write_summary(args.summary_output, args.input, rows)

        print(
            f"[{index}/{len(cases)}] {case.name}: "
            f"router={adaptive_swap}/{adaptive_depth} ({variant}), "
            f"LightSABRE={lightsabre_swap}/{lightsabre_depth} "
            f"(seed={lightsabre.seed}), gap={row.swap_gap:+d}, {outcome}, "
            f"budget={adaptive_runtime_ms:.1f}/{lightsabre_budget_ms:.1f} ms"
        )

    result = write_summary(args.summary_output, args.input, rows)
    save_csv(args.case_output, rows)

    print()
    print("===== historical development 汇总 =====")
    print(f"historical development total SWAP：{result['adaptive_mapping_swap']}")
    print(f"冻结路由器 total SWAP：{result['adaptive_swap']}")
    print(
        "historical development 条件规则额外减少："
        f"{result['improvement_vs_adaptive_mapping']}"
    )
    print(f"LightSABRE total SWAP：{result['lightsabre_swap']}")
    print(f"相对 LightSABRE gap：{result['swap_gap']:+d}")
    print(f"相对差：{result['relative_swap_gap']:+.2%}")
    print(
        f"逐实例胜/平/负：{result['wins']}/"
        f"{result['ties']}/{result['losses']}"
    )
    print(
        f"总预算 router/LightSABRE："
        f"{result['adaptive_runtime_ms']:.1f}/"
        f"{result['lightsabre_budget_ms']:.1f} ms"
    )
    print(f"逐实例结果：{args.case_output.resolve()}")
    print(f"汇总 JSON：{args.summary_output.resolve()}")


if __name__ == "__main__":
    main()
