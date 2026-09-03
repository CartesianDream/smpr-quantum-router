from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import structural_policy as structural_policy
import mapping_portfolio as mapping_portfolio
import local_layout_search as local_layout_search

from case_io import load_cases
from validation_support import compare_quality


@dataclass
class CaseResult:
    instance: str
    circuit_mode: str
    num_qubits: int

    old_mapping: str
    old_swap: int
    old_depth: int
    searched_mapping: str
    searched_swap: int
    searched_depth: int
    improvement: int

    lightsabre_swap: int
    lightsabre_depth: int
    old_gap: int
    searched_gap: int
    old_outcome: str
    searched_outcome: str

    mapping_candidates_scored: int
    rollout_mappings: int
    runtime_ms: float


def configure_router() -> structural_policy.StructuralConfig:
    structural_policy.one_step_policy.THETA = 5
    structural_policy.one_step_policy.W_FUTURE = 0.5
    structural_policy.one_step_policy.GAMMA = 0.5
    structural_policy.one_step_policy.ALPHA = 0.5
    structural_policy.one_step_policy.LAMBDA_DEPTH = 0.0
    return structural_policy.CONFIG_BY_NAME["ordered_potential"]


def load_lightsabre_reference(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 historical development 逐实例 CSV：{path.resolve()}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {
        "instance",
        "lightsabre_swap",
        "lightsabre_depth",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"{path} 不是 historical development case CSV，缺少字段："
            f"{sorted(required - set(rows[0] if rows else []))}"
        )
    return {row["instance"]: row for row in rows}


def rollout_depth(
    rows: list[local_layout_search.MappingScore],
    mapping_name: str,
    swap_count: int,
) -> int:
    matches = [
        row
        for row in rows
        if row.selected_for_rollout
        and row.mapping_name == mapping_name
        and row.rollout_swap == swap_count
        and row.rollout_depth is not None
    ]
    if not matches:
        raise RuntimeError(
            f"找不到 mapping={mapping_name}, SWAP={swap_count} 的 rollout 深度"
        )
    return int(matches[0].rollout_depth)


def save_csv(path: Path, rows: list, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def group_summary(rows: list[CaseResult]) -> dict:
    old_swap = sum(row.old_swap for row in rows)
    searched_swap = sum(row.searched_swap for row in rows)
    lightsabre_swap = sum(row.lightsabre_swap for row in rows)
    old_outcomes = Counter(row.old_outcome for row in rows)
    searched_outcomes = Counter(row.searched_outcome for row in rows)
    return {
        "case_count": len(rows),
        "old_swap": old_swap,
        "searched_swap": searched_swap,
        "lightsabre_swap": lightsabre_swap,
        "improvement": old_swap - searched_swap,
        "old_gap": old_swap - lightsabre_swap,
        "searched_gap": searched_swap - lightsabre_swap,
        "improved_cases": sum(row.improvement > 0 for row in rows),
        "old_wins_ties_losses": {
            "wins": old_outcomes["independent rollout胜"],
            "ties": old_outcomes["平局"],
            "losses": old_outcomes["Qiskit胜"],
        },
        "searched_wins_ties_losses": {
            "wins": searched_outcomes["independent rollout胜"],
            "ties": searched_outcomes["平局"],
            "losses": searched_outcomes["Qiskit胜"],
        },
        "runtime_ms": sum(row.runtime_ms for row in rows),
    }


def summarize(rows: list[CaseResult], args: argparse.Namespace) -> dict:
    by_mode: dict[str, list[CaseResult]] = defaultdict(list)
    for row in rows:
        by_mode[row.circuit_mode].append(row)
    result = group_summary(rows)
    result.update(
        {
            "input": str(args.input.resolve()),
            "lightsabre_cases": str(args.lightsabre_cases.resolve()),
            "beam_width": args.beam_width,
            "search_rounds": args.search_rounds,
            "rollout_top": args.rollout_top,
            "top_k": args.top_k,
            "forward_backward": not args.no_forward_backward,
            "by_mode": {
                mode: group_summary(group)
                for mode, group in sorted(by_mode.items())
            },
        }
    )
    return result


def write_summary(
    path: Path,
    rows: list[CaseResult],
    args: argparse.Namespace,
) -> dict:
    result = summarize(rows, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "historical development：在 grid 开发实例上探测路由成本驱动的初始映射搜索"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmarks_v4/dev.json"),
    )
    parser.add_argument(
        "--case-indices",
        type=str,
        default="14,15,17,19",
        help="默认探测 V4 dev 的 grid uniform/far/alternating 残差实例",
    )
    parser.add_argument(
        "--lightsabre-cases",
        type=Path,
        default=Path("results/layout_probe_v4_dev_cases.csv"),
    )
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--search-rounds", type=int, default=1)
    parser.add_argument("--rollout-top", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--no-forward-backward", action="store_true")
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=Path("results/layout_probe_grid_mapping_candidates.csv"),
    )
    parser.add_argument(
        "--case-output",
        type=Path,
        default=Path("results/layout_probe_grid_mapping_cases.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/layout_probe_grid_mapping_summary.json"),
    )
    args = parser.parse_args()

    if args.beam_width <= 0 or args.search_rounds < 0:
        raise ValueError("beam-width 必须为正，search-rounds 必须非负")
    if args.rollout_top <= 0 or args.top_k < 0:
        raise ValueError("rollout-top 必须为正，top-k 必须非负")

    cases = mapping_portfolio.choose_cases(
        load_cases(args.input, limit=None),
        mapping_portfolio.parse_indices(args.case_indices),
        None,
    )
    for case in cases:
        if case.topology != "grid":
            raise ValueError(f"historical development 当前只探测 grid：{case.name}")
        if case.circuit_mode == "parallel_layers":
            raise ValueError(
                f"{case.name} 的 historical development 使用完整 mapping pool；"
                "当前探针先不混入该模式"
            )

    reference = load_lightsabre_reference(args.lightsabre_cases)
    missing = [case.name for case in cases if case.name not in reference]
    if missing:
        raise ValueError(
            "LightSABRE CSV 缺少实例：" + ", ".join(missing)
        )
    config = configure_router()

    print("===== historical development：grid 初始映射搜索探针 =====")
    print(f"输入：{args.input.resolve()}")
    print(f"实例数：{len(cases)}")
    print(f"LightSABRE 参考：{args.lightsabre_cases.resolve()}")
    print(
        f"搜索：beam={args.beam_width}, rounds={args.search_rounds}, "
        f"rollout-top={args.rollout_top}, top-k={args.top_k}, "
        f"forward/backward={not args.no_forward_backward}"
    )

    all_mapping_rows: list[local_layout_search.MappingScore] = []
    rows: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        summary, mapping_rows = local_layout_search.run_case(
            case=case,
            config=config,
            beam_width=args.beam_width,
            search_rounds=args.search_rounds,
            rollout_top=args.rollout_top,
            top_k=args.top_k,
            use_forward_backward=not args.no_forward_backward,
        )
        old_depth = rollout_depth(
            mapping_rows,
            summary.old_mapping,
            summary.old_rollout_swap,
        )
        searched_depth = rollout_depth(
            mapping_rows,
            summary.selected_mapping,
            summary.selected_rollout_swap,
        )
        ref = reference[case.name]
        lightsabre_swap = int(ref["lightsabre_swap"])
        lightsabre_depth = int(ref["lightsabre_depth"])
        old_outcome = compare_quality(
            summary.old_rollout_swap,
            old_depth,
            lightsabre_swap,
            lightsabre_depth,
        )
        searched_outcome = compare_quality(
            summary.selected_rollout_swap,
            searched_depth,
            lightsabre_swap,
            lightsabre_depth,
        )
        row = CaseResult(
            instance=case.name,
            circuit_mode=case.circuit_mode,
            num_qubits=case.num_qubits,
            old_mapping=summary.old_mapping,
            old_swap=summary.old_rollout_swap,
            old_depth=old_depth,
            searched_mapping=summary.selected_mapping,
            searched_swap=summary.selected_rollout_swap,
            searched_depth=searched_depth,
            improvement=summary.improvement_vs_old_rollout,
            lightsabre_swap=lightsabre_swap,
            lightsabre_depth=lightsabre_depth,
            old_gap=summary.old_rollout_swap - lightsabre_swap,
            searched_gap=summary.selected_rollout_swap - lightsabre_swap,
            old_outcome=old_outcome,
            searched_outcome=searched_outcome,
            mapping_candidates_scored=summary.mapping_candidates_scored,
            rollout_mappings=summary.rollout_mappings,
            runtime_ms=summary.total_runtime_ms,
        )
        rows.append(row)
        all_mapping_rows.extend(mapping_rows)

        local_layout_search.write_csv(
            args.mapping_output,
            all_mapping_rows,
            local_layout_search.MappingScore.__annotations__.keys(),
        )
        save_csv(args.case_output, rows, CaseResult.__annotations__.keys())
        write_summary(args.summary_output, rows, args)

        print(
            f"[{index}/{len(cases)}] {case.name}: "
            f"old={row.old_swap}/{row.old_depth} ({row.old_mapping}), "
            f"search={row.searched_swap}/{row.searched_depth} "
            f"({row.searched_mapping}), LightSABRE="
            f"{row.lightsabre_swap}/{row.lightsabre_depth}, "
            f"gap={row.old_gap:+d}->{row.searched_gap:+d}, "
            f"DeltaSWAP=-{row.improvement}, {row.searched_outcome}, "
            f"{row.runtime_ms:.1f} ms"
        )

    result = write_summary(args.summary_output, rows, args)
    save_csv(args.case_output, rows, CaseResult.__annotations__.keys())

    print()
    print("===== historical development 汇总 =====")
    print(f"旧 historical development total SWAP：{result['old_swap']}")
    print(f"搜索后 total SWAP：{result['searched_swap']}")
    print(f"LightSABRE total SWAP：{result['lightsabre_swap']}")
    print(f"搜索减少 SWAP：{result['improvement']}")
    print(
        f"相对 LightSABRE gap："
        f"{result['old_gap']:+d}->{result['searched_gap']:+d}"
    )
    print(
        f"严格改善实例：{result['improved_cases']}/"
        f"{result['case_count']}"
    )
    print(f"总运行时间：{result['runtime_ms']:.1f} ms")
    print(f"映射明细：{args.mapping_output.resolve()}")
    print(f"逐实例结果：{args.case_output.resolve()}")
    print(f"汇总 JSON：{args.summary_output.resolve()}")


if __name__ == "__main__":
    main()
