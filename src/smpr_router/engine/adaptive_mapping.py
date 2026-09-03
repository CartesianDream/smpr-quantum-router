from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import structural_policy as structural_policy
import mapping_portfolio as mapping_portfolio
import rollout_policy as rollout_policy

from case_io import DevCase, load_cases


def parse_modes(text: str) -> set[str]:
    return {value.strip() for value in text.split(",") if value.strip()}


def budget_for_case(
    case: DevCase,
    pool_modes: set[str],
    default_mapping: str,
    default_top_k: int,
    pool_top_k: int,
) -> tuple[str, int]:
    if case.circuit_mode in pool_modes:
        return "pool", pool_top_k
    return default_mapping, default_top_k


def main() -> None:
    parser = argparse.ArgumentParser(
        description="historical development：按线路模式自适应分配初始映射搜索预算"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmarks_v3/dev.json"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-indices", type=str, default=None)
    parser.add_argument(
        "--pool-modes",
        type=str,
        default="parallel_layers",
        help="这些 circuit_mode 使用完整 mapping pool，逗号分隔",
    )
    parser.add_argument(
        "--default-mapping",
        choices=("identity", "v1_best"),
        default="v1_best",
    )
    parser.add_argument("--default-top-k", type=int, default=2)
    parser.add_argument("--pool-top-k", type=int, default=2)
    parser.add_argument(
        "--base-policy",
        choices=tuple(sorted(structural_policy.CONFIG_BY_NAME)),
        default="ordered_potential",
    )
    parser.add_argument(
        "--attempt-output",
        type=Path,
        default=Path("results/adaptive_mapping_adaptive_attempts.csv"),
    )
    parser.add_argument(
        "--case-output",
        type=Path,
        default=Path("results/adaptive_mapping_adaptive_cases.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/adaptive_mapping_adaptive_summary.json"),
    )
    args = parser.parse_args()

    if args.default_top_k < 0 or args.pool_top_k < 0:
        raise ValueError("top-k 必须是非负整数")

    structural_policy.one_step_policy.THETA = 5
    structural_policy.one_step_policy.W_FUTURE = 0.5
    structural_policy.one_step_policy.GAMMA = 0.5
    structural_policy.one_step_policy.ALPHA = 0.5
    structural_policy.one_step_policy.LAMBDA_DEPTH = 0.0

    cases = mapping_portfolio.choose_cases(
        load_cases(args.input, limit=None),
        mapping_portfolio.parse_indices(args.case_indices),
        args.limit,
    )
    pool_modes = parse_modes(args.pool_modes)
    base_config = structural_policy.CONFIG_BY_NAME[args.base_policy]

    print("===== historical development：自适应映射预算 + 完整 rollout =====")
    print(f"输入：{args.input.resolve()}")
    print(f"实例数：{len(cases)}")
    print(f"基线策略：{base_config.name}")
    print(f"mapping pool 模式：{sorted(pool_modes)}")
    print(
        f"默认预算：mapping={args.default_mapping}, "
        f"top-k={args.default_top_k}"
    )
    print(f"pool 预算：mapping=pool, top-k={args.pool_top_k}")

    all_attempts: list[rollout_policy.AttemptRow] = []
    case_rows: list[rollout_policy.CaseRow] = []
    allocation_counts: Counter[str] = Counter()

    for index, case in enumerate(cases, start=1):
        mapping_mode, top_k = budget_for_case(
            case=case,
            pool_modes=pool_modes,
            default_mapping=args.default_mapping,
            default_top_k=args.default_top_k,
            pool_top_k=args.pool_top_k,
        )
        allocation_counts[f"{mapping_mode}/k{top_k}"] += 1
        mappings = mapping_portfolio.make_mappings(case, mapping_mode)
        attempts = [
            rollout_policy.run_attempt(case, mapping, base_config, top_k)
            for mapping in mappings
        ]
        row = rollout_policy.build_case_row(
            case=case,
            mappings=mappings,
            attempts=attempts,
            base_config=base_config,
            top_k=top_k,
        )
        all_attempts.extend(attempts)
        case_rows.append(row)

        print(
            f"[{index}/{len(cases)}] {case.name}: "
            f"budget={mapping_mode}/k{top_k}, "
            f"base={row.base_swap}/{row.base_depth} ({row.base_mapping}), "
            f"rollout={row.rollout_swap}/{row.rollout_depth} "
            f"({row.rollout_mapping}), "
            f"DeltaSWAP=-{row.swap_improvement}, "
            f"changed={row.changed_decisions}, "
            f"{row.total_runtime_ms:.1f} ms"
        )

        rollout_policy.write_csv(
            args.attempt_output,
            all_attempts,
            rollout_policy.AttemptRow.__annotations__.keys(),
        )
        rollout_policy.write_csv(
            args.case_output,
            case_rows,
            rollout_policy.CaseRow.__annotations__.keys(),
        )
        summary = rollout_policy.summarize(case_rows)
        summary["allocation_counts"] = dict(allocation_counts)
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = rollout_policy.summarize(case_rows)
    summary["allocation_counts"] = dict(allocation_counts)
    rollout_policy.write_csv(
        args.attempt_output,
        all_attempts,
        rollout_policy.AttemptRow.__annotations__.keys(),
    )
    rollout_policy.write_csv(
        args.case_output,
        case_rows,
        rollout_policy.CaseRow.__annotations__.keys(),
    )
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("===== historical development 汇总 =====")
    print(f"预算分配：{dict(allocation_counts)}")
    print(f"基线 total SWAP：{summary['base_swap']}")
    print(f"historical development total SWAP：{summary['rollout_swap']}")
    print(f"减少 SWAP：{summary['swap_improvement']}")
    print(f"相对改善：{summary['relative_improvement']:.2%}")
    print(
        f"严格改善实例：{summary['improved_cases']}/"
        f"{summary['case_count']}"
    )
    print(f"总运行时间：{summary['total_runtime_ms']:.1f} ms")
    print()
    print(f"尝试明细：{args.attempt_output.resolve()}")
    print(f"逐实例结果：{args.case_output.resolve()}")
    print(f"汇总 JSON：{args.summary_output.resolve()}")


if __name__ == "__main__":
    main()
