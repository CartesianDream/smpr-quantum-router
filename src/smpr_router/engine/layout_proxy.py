from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import structural_policy as structural_policy
import mapping_portfolio as mapping_portfolio
import rollout_policy as rollout_policy
import local_layout_search as local_layout_search
import layout_probe as layout_probe

from layout_pool import (
    MappingCandidate,
    generate_complete_mapping_pool,
)
from case_io import DevCase, load_cases
from validation_support import compare_quality


@dataclass
class MappingScore:
    instance: str
    mapping_name: str
    source: str
    mapping: str
    proxy_score: float
    selected_by_proxy: bool
    base_swap: int | None
    base_depth: int | None
    base_runtime_ms: float
    selected_for_rollout: bool
    rollout_swap: int | None
    rollout_depth: int | None
    rollout_runtime_ms: float
    valid: bool
    error: str


def configure_router() -> structural_policy.StructuralConfig:
    structural_policy.one_step_policy.THETA = 5
    structural_policy.one_step_policy.W_FUTURE = 0.5
    structural_policy.one_step_policy.GAMMA = 0.5
    structural_policy.one_step_policy.ALPHA = 0.5
    structural_policy.one_step_policy.LAMBDA_DEPTH = 0.0
    return structural_policy.CONFIG_BY_NAME["ordered_potential"]


def mapping_key(mapping: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in mapping)


def hardware_distances(case: DevCase) -> list[list[int]]:
    _dag, hardware, _mapping = case.test_case.build()
    return [
        [
            len(hardware.shortest_path(left, right)) - 1
            for right in range(case.num_qubits)
        ]
        for left in range(case.num_qubits)
    ]


def proxy_score(
    case: DevCase,
    mapping: tuple[int, ...],
    distances: list[list[int]],
    decay: float,
) -> float:
    score = 0.0
    cx_index = 0
    for gate in case.test_case.gate_specs:
        if gate[0] != "cx":
            continue
        logical_left = int(gate[1])
        logical_right = int(gate[2])
        physical_left = mapping[logical_left]
        physical_right = mapping[logical_right]
        score += (decay**cx_index) * distances[physical_left][physical_right]
        cx_index += 1
    return score


def build_proxy_shortlist(
    case: DevCase,
    config: structural_policy.StructuralConfig,
    proxy_decay: float,
    proxy_top: int,
    use_forward_backward: bool,
) -> tuple[
    list[MappingScore],
    dict[tuple[int, ...], MappingCandidate],
]:
    initial = generate_complete_mapping_pool(
        case.test_case,
        use_forward_backward=use_forward_backward,
        verbose=False,
    )
    candidates: dict[tuple[int, ...], MappingCandidate] = {
        item.mapping: item for item in initial
    }
    source = {
        item.mapping: (
            "forward_backward" if item.name.endswith("_fb") else "base"
        )
        for item in initial
    }
    proxy_selected = {item.mapping: False for item in initial}
    cache: dict[tuple[int, ...], tuple[int, int, float, str]] = {}

    def evaluate(mapping: tuple[int, ...]) -> tuple[int, int, float, str]:
        if mapping in cache:
            return cache[mapping]
        item = candidates[mapping]
        try:
            swaps, depth, elapsed = local_layout_search.score_base_mapping(
                case, item, config
            )
            cache[mapping] = (swaps, depth, elapsed, "")
        except Exception as error:
            cache[mapping] = (
                10**9,
                10**9,
                0.0,
                f"{type(error).__name__}: {error}",
            )
        return cache[mapping]

    for mapping in list(candidates):
        evaluate(mapping)

    valid_initial = [
        mapping for mapping in candidates if evaluate(mapping)[0] < 10**9
    ]
    if not valid_initial:
        raise RuntimeError(f"{case.name} 的初始映射池全部失败")
    parent = min(
        valid_initial,
        key=lambda mapping: (
            evaluate(mapping)[0],
            evaluate(mapping)[1],
            mapping,
        ),
    )

    distances = hardware_distances(case)
    all_neighbors = local_layout_search.swapped_neighbors(parent)
    ranked_neighbors = sorted(
        enumerate(all_neighbors),
        key=lambda item: (
            proxy_score(case, item[1], distances, proxy_decay),
            item[0],
        ),
    )
    for neighbor_index, neighbor in ranked_neighbors[:proxy_top]:
        if neighbor in candidates:
            proxy_selected[neighbor] = True
            continue
        candidates[neighbor] = MappingCandidate(
            name=f"proxy_r1_n{neighbor_index}",
            mapping=neighbor,
            generation_ms=0.0,
        )
        source[neighbor] = "proxy_local_search"
        proxy_selected[neighbor] = True
        evaluate(neighbor)

    rows: list[MappingScore] = []
    for mapping, item in candidates.items():
        swaps, depth, elapsed, error = evaluate(mapping)
        rows.append(
            MappingScore(
                instance=case.name,
                mapping_name=item.name,
                source=source[mapping],
                mapping=mapping_key(mapping),
                proxy_score=proxy_score(
                    case, mapping, distances, proxy_decay
                ),
                selected_by_proxy=proxy_selected[mapping],
                base_swap=None if swaps >= 10**9 else swaps,
                base_depth=None if depth >= 10**9 else depth,
                base_runtime_ms=elapsed,
                selected_for_rollout=False,
                rollout_swap=None,
                rollout_depth=None,
                rollout_runtime_ms=0.0,
                valid=not error,
                error=error,
            )
        )
    return rows, candidates


def run_case(
    case: DevCase,
    config: structural_policy.StructuralConfig,
    reference: dict[str, str],
    proxy_decay: float,
    proxy_top: int,
    rollout_top: int,
    top_k: int,
    use_forward_backward: bool,
) -> tuple[layout_probe.CaseResult, list[MappingScore]]:
    start = time.perf_counter()
    rows, candidates = build_proxy_shortlist(
        case=case,
        config=config,
        proxy_decay=proxy_decay,
        proxy_top=proxy_top,
        use_forward_backward=use_forward_backward,
    )
    valid = [row for row in rows if row.valid]
    if not valid:
        raise RuntimeError(f"{case.name} 没有合法候选映射")

    by_name = {item.name: item for item in candidates.values()}
    old_mapping = mapping_portfolio.make_mappings(case, "v1_best")[0]
    old_attempt = rollout_policy.run_attempt(case, old_mapping, config, top_k)
    if not old_attempt.valid:
        raise RuntimeError(f"旧 historical development 映射失败：{old_attempt.error}")

    ranked = sorted(
        valid,
        key=lambda row: (
            int(row.base_swap),
            int(row.base_depth),
            row.mapping_name,
        ),
    )
    selected_names: list[str] = []
    old_pool_name = next(
        (
            item.name
            for item in candidates.values()
            if item.mapping == old_mapping.mapping
        ),
        None,
    )
    if old_pool_name is None:
        raise RuntimeError(f"{case.name} 的旧映射不在候选池")
    selected_names.append(old_pool_name)
    for row in ranked:
        if len(selected_names) >= rollout_top:
            break
        if row.mapping_name not in selected_names:
            selected_names.append(row.mapping_name)

    attempts: list[rollout_policy.AttemptRow] = []
    for name in selected_names:
        attempt = rollout_policy.run_attempt(case, by_name[name], config, top_k)
        attempts.append(attempt)
        target = next(row for row in rows if row.mapping_name == name)
        target.selected_for_rollout = True
        target.rollout_runtime_ms = attempt.rollout_runtime_ms
        if attempt.valid:
            target.rollout_swap = int(attempt.rollout_swap)
            target.rollout_depth = int(attempt.rollout_depth)
        else:
            target.valid = False
            target.error = attempt.error

    valid_attempts = [attempt for attempt in attempts if attempt.valid]
    if not valid_attempts:
        raise RuntimeError(f"{case.name} 所有 rollout 都失败")
    best = min(valid_attempts, key=rollout_policy.rollout_key)

    old_swap = int(old_attempt.rollout_swap)
    old_depth = int(old_attempt.rollout_depth)
    searched_swap = int(best.rollout_swap)
    searched_depth = int(best.rollout_depth)
    lightsabre_swap = int(reference["lightsabre_swap"])
    lightsabre_depth = int(reference["lightsabre_depth"])

    return (
        layout_probe.CaseResult(
            instance=case.name,
            circuit_mode=case.circuit_mode,
            num_qubits=case.num_qubits,
            old_mapping=old_attempt.mapping,
            old_swap=old_swap,
            old_depth=old_depth,
            searched_mapping=best.mapping,
            searched_swap=searched_swap,
            searched_depth=searched_depth,
            improvement=old_swap - searched_swap,
            lightsabre_swap=lightsabre_swap,
            lightsabre_depth=lightsabre_depth,
            old_gap=old_swap - lightsabre_swap,
            searched_gap=searched_swap - lightsabre_swap,
            old_outcome=compare_quality(
                old_swap, old_depth, lightsabre_swap, lightsabre_depth
            ),
            searched_outcome=compare_quality(
                searched_swap,
                searched_depth,
                lightsabre_swap,
                lightsabre_depth,
            ),
            mapping_candidates_scored=len(rows),
            rollout_mappings=len(valid_attempts),
            runtime_ms=(time.perf_counter() - start) * 1000.0,
        ),
        rows,
    )


def save_csv(path: Path, rows: list, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def group_summary(rows: list[layout_probe.CaseResult]) -> dict:
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
        "mapping_candidates_scored": sum(
            row.mapping_candidates_scored for row in rows
        ),
        "runtime_ms": sum(row.runtime_ms for row in rows),
    }


def summarize(
    rows: list[layout_probe.CaseResult],
    args: argparse.Namespace,
) -> dict:
    by_mode: dict[str, list[layout_probe.CaseResult]] = defaultdict(list)
    for row in rows:
        by_mode[row.circuit_mode].append(row)
    result = group_summary(rows)
    result.update(
        {
            "input": str(args.input.resolve()),
            "lightsabre_cases": str(args.lightsabre_cases.resolve()),
            "proxy_decay": args.proxy_decay,
            "proxy_top": args.proxy_top,
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
    rows: list[layout_probe.CaseResult],
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
        description="historical development：廉价代理筛选的快速 grid 初始映射搜索"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmarks_v4/dev.json"),
    )
    parser.add_argument("--case-indices", type=str, default="14,15,19")
    parser.add_argument(
        "--lightsabre-cases",
        type=Path,
        default=Path("results/layout_probe_v4_dev_cases.csv"),
    )
    parser.add_argument("--proxy-decay", type=float, default=0.95)
    parser.add_argument("--proxy-top", type=int, default=8)
    parser.add_argument("--rollout-top", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--no-forward-backward", action="store_true")
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=Path("results/layout_proxy_proxy_candidates.csv"),
    )
    parser.add_argument(
        "--case-output",
        type=Path,
        default=Path("results/layout_proxy_proxy_cases.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/layout_proxy_proxy_summary.json"),
    )
    args = parser.parse_args()

    if not 0.0 < args.proxy_decay <= 1.0:
        raise ValueError("proxy-decay 必须位于 (0, 1]")
    if args.proxy_top <= 0 or args.rollout_top <= 0 or args.top_k < 0:
        raise ValueError("proxy-top/rollout-top 必须为正，top-k 必须非负")

    cases = mapping_portfolio.choose_cases(
        load_cases(args.input, limit=None),
        mapping_portfolio.parse_indices(args.case_indices),
        None,
    )
    for case in cases:
        if case.topology != "grid":
            raise ValueError(f"historical development 当前只允许 grid：{case.name}")
        if case.circuit_mode not in {"uniform", "alternating"}:
            raise ValueError(
                f"historical development 冻结模式仅为 uniform/alternating：{case.name}"
            )

    reference = layout_probe.load_lightsabre_reference(args.lightsabre_cases)
    missing = [case.name for case in cases if case.name not in reference]
    if missing:
        raise ValueError("LightSABRE CSV 缺少：" + ", ".join(missing))
    config = configure_router()

    print("===== historical development：代理筛选快速 grid 映射搜索 =====")
    print(f"输入：{args.input.resolve()}")
    print(f"实例数：{len(cases)}")
    print(f"LightSABRE 参考：{args.lightsabre_cases.resolve()}")
    print(
        f"代理：decay={args.proxy_decay}, top={args.proxy_top}; "
        f"rollout-top={args.rollout_top}, top-k={args.top_k}; "
        f"forward/backward={not args.no_forward_backward}"
    )

    case_rows: list[layout_probe.CaseResult] = []
    mapping_rows: list[MappingScore] = []
    for index, case in enumerate(cases, start=1):
        row, case_mapping_rows = run_case(
            case=case,
            config=config,
            reference=reference[case.name],
            proxy_decay=args.proxy_decay,
            proxy_top=args.proxy_top,
            rollout_top=args.rollout_top,
            top_k=args.top_k,
            use_forward_backward=not args.no_forward_backward,
        )
        case_rows.append(row)
        mapping_rows.extend(case_mapping_rows)
        save_csv(
            args.mapping_output,
            mapping_rows,
            MappingScore.__annotations__.keys(),
        )
        save_csv(
            args.case_output,
            case_rows,
            layout_probe.CaseResult.__annotations__.keys(),
        )
        write_summary(args.summary_output, case_rows, args)

        print(
            f"[{index}/{len(cases)}] {case.name}: "
            f"old={row.old_swap}/{row.old_depth} ({row.old_mapping}), "
            f"proxy={row.searched_swap}/{row.searched_depth} "
            f"({row.searched_mapping}), LightSABRE="
            f"{row.lightsabre_swap}/{row.lightsabre_depth}, "
            f"gap={row.old_gap:+d}->{row.searched_gap:+d}, "
            f"DeltaSWAP=-{row.improvement}, candidates="
            f"{row.mapping_candidates_scored}/{row.rollout_mappings}, "
            f"{row.runtime_ms:.1f} ms"
        )

    result = write_summary(args.summary_output, case_rows, args)
    print()
    print("===== historical development 汇总 =====")
    print(f"旧 historical development total SWAP：{result['old_swap']}")
    print(f"代理搜索 total SWAP：{result['searched_swap']}")
    print(f"LightSABRE total SWAP：{result['lightsabre_swap']}")
    print(f"减少 SWAP：{result['improvement']}")
    print(
        f"相对 LightSABRE gap："
        f"{result['old_gap']:+d}->{result['searched_gap']:+d}"
    )
    print(f"完整评分候选总数：{result['mapping_candidates_scored']}")
    print(f"总运行时间：{result['runtime_ms']:.1f} ms")
    print(f"映射明细：{args.mapping_output.resolve()}")
    print(f"逐实例结果：{args.case_output.resolve()}")
    print(f"汇总 JSON：{args.summary_output.resolve()}")


if __name__ == "__main__":
    main()
