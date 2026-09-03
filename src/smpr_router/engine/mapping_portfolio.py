from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import structural_policy as structural_policy

from layout_pool import MappingCandidate, generate_base_mapping_pool
from case_io import DevCase, load_cases


# ============================================================
# 1. Portfolio 配置
# ============================================================

DEFAULT_POLICIES = ("v1", "ordered_potential", "front_sum")


@dataclass
class AttemptRow:
    instance: str
    topology: str
    circuit_mode: str
    num_qubits: int
    cx_count: int
    policy: str
    mapping: str
    swap_count: int | None
    depth: int | None
    runtime_ms: float
    valid: bool
    error: str


@dataclass
class CaseRow:
    instance: str
    topology: str
    circuit_mode: str
    num_qubits: int
    cx_count: int
    mapping_count: int
    policy_count: int

    v1_mapping: str
    v1_swap: int
    v1_depth: int

    ordered_frontier_mapping: str
    ordered_frontier_swap: int
    ordered_frontier_depth: int

    selected_policy: str
    selected_mapping: str
    portfolio_swap: int
    portfolio_depth: int

    swap_improvement_vs_v1: int
    depth_improvement_on_swap_tie: int
    total_runtime_ms: float


def score_key(row: AttemptRow) -> tuple:
    if not row.valid or row.swap_count is None or row.depth is None:
        return (10**9, 10**9, 10**9, row.policy, row.mapping)
    return (
        row.swap_count,
        row.depth,
        row.runtime_ms,
        row.policy,
        row.mapping,
    )


def parse_indices(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def choose_cases(
    cases: list[DevCase],
    indices: list[int] | None,
    limit: int | None,
) -> list[DevCase]:
    if indices is not None:
        selected: list[DevCase] = []
        for index in indices:
            if not 0 <= index < len(cases):
                raise IndexError(f"case index 越界：{index}")
            selected.append(cases[index])
        return selected
    return cases if limit is None else cases[:limit]


def parse_policies(text: str | None) -> list[structural_policy.StructuralConfig]:
    names = DEFAULT_POLICIES if not text else tuple(
        name.strip() for name in text.split(",") if name.strip()
    )
    unknown = [name for name in names if name not in structural_policy.CONFIG_BY_NAME]
    if unknown:
        raise ValueError(
            f"未知策略：{unknown}；可选值：{sorted(structural_policy.CONFIG_BY_NAME)}"
        )
    if "v1" not in names or "ordered_potential" not in names:
        raise ValueError("historical development 必须包含 v1 和 ordered_potential")
    return [structural_policy.CONFIG_BY_NAME[name] for name in names]


def make_mappings(
    case: DevCase,
    mapping_mode: str,
) -> list[MappingCandidate]:
    if mapping_mode == "identity":
        return [
            MappingCandidate(
                name="identity",
                mapping=tuple(range(case.num_qubits)),
                generation_ms=0.0,
            )
        ]

    pool = generate_base_mapping_pool(case.test_case)
    if mapping_mode == "pool":
        return pool

    if mapping_mode != "v1_best":
        raise ValueError(f"未知 mapping mode：{mapping_mode}")

    v1 = structural_policy.CONFIG_BY_NAME["v1"]
    rows = [structural_policy.run_mapping(case, mapping, v1) for mapping in pool]
    valid = [row for row in rows if row.valid]
    if not valid:
        raise RuntimeError(f"{case.name} 的 v1 映射池全部失败")
    best = min(
        valid,
        key=lambda row: (
            row.swap_count,
            row.depth,
            row.runtime_ms,
            row.mapping_name,
        ),
    )
    return [next(mapping for mapping in pool if mapping.name == best.mapping_name)]


def run_attempts(
    case: DevCase,
    mappings: list[MappingCandidate],
    policies: list[structural_policy.StructuralConfig],
) -> list[AttemptRow]:
    rows: list[AttemptRow] = []

    for policy in policies:
        for mapping in mappings:
            result = structural_policy.run_mapping(case, mapping, policy)
            rows.append(
                AttemptRow(
                    instance=case.name,
                    topology=case.topology,
                    circuit_mode=case.circuit_mode,
                    num_qubits=case.num_qubits,
                    cx_count=case.two_qubit_gate_count,
                    policy=policy.name,
                    mapping=mapping.name,
                    swap_count=result.swap_count,
                    depth=result.depth,
                    runtime_ms=result.runtime_ms,
                    valid=result.valid,
                    error=result.error,
                )
            )

    return rows


def best_for_policy(rows: list[AttemptRow], policy: str) -> AttemptRow:
    candidates = [row for row in rows if row.policy == policy and row.valid]
    if not candidates:
        errors = " | ".join(row.error for row in rows if row.policy == policy)
        raise RuntimeError(f"策略 {policy} 没有合法结果：{errors}")
    return min(candidates, key=score_key)


def build_case_row(
    case: DevCase,
    mappings: list[MappingCandidate],
    policies: list[structural_policy.StructuralConfig],
    attempts: list[AttemptRow],
) -> CaseRow:
    v1 = best_for_policy(attempts, "v1")
    ordered_frontier = best_for_policy(attempts, "ordered_potential")
    portfolio = min((row for row in attempts if row.valid), key=score_key)

    depth_improvement = 0
    if portfolio.swap_count == v1.swap_count:
        depth_improvement = int(v1.depth) - int(portfolio.depth)

    return CaseRow(
        instance=case.name,
        topology=case.topology,
        circuit_mode=case.circuit_mode,
        num_qubits=case.num_qubits,
        cx_count=case.two_qubit_gate_count,
        mapping_count=len(mappings),
        policy_count=len(policies),
        v1_mapping=v1.mapping,
        v1_swap=int(v1.swap_count),
        v1_depth=int(v1.depth),
        ordered_frontier_mapping=ordered_frontier.mapping,
        ordered_frontier_swap=int(ordered_frontier.swap_count),
        ordered_frontier_depth=int(ordered_frontier.depth),
        selected_policy=portfolio.policy,
        selected_mapping=portfolio.mapping,
        portfolio_swap=int(portfolio.swap_count),
        portfolio_depth=int(portfolio.depth),
        swap_improvement_vs_v1=int(v1.swap_count) - int(portfolio.swap_count),
        depth_improvement_on_swap_tie=depth_improvement,
        total_runtime_ms=sum(row.runtime_ms for row in attempts),
    )


def write_csv(path: Path, rows: list, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def summarize(rows: list[CaseRow]) -> dict:
    by_topology: dict[str, list[CaseRow]] = defaultdict(list)
    by_mode: dict[str, list[CaseRow]] = defaultdict(list)
    for row in rows:
        by_topology[row.topology].append(row)
        by_mode[row.circuit_mode].append(row)

    def group_summary(group: list[CaseRow]) -> dict:
        v1_swap = sum(row.v1_swap for row in group)
        portfolio_swap = sum(row.portfolio_swap for row in group)
        return {
            "case_count": len(group),
            "v1_swap": v1_swap,
            "portfolio_swap": portfolio_swap,
            "swap_improvement": v1_swap - portfolio_swap,
            "relative_improvement": (
                (v1_swap - portfolio_swap) / v1_swap if v1_swap else 0.0
            ),
            "improved_cases": sum(
                row.portfolio_swap < row.v1_swap for row in group
            ),
            "swap_ties_with_better_depth": sum(
                row.portfolio_swap == row.v1_swap
                and row.portfolio_depth < row.v1_depth
                for row in group
            ),
        }

    overall = group_summary(rows)
    selected_policy_counts = dict(
        sorted(
            (policy, sum(row.selected_policy == policy for row in rows))
            for policy in {row.selected_policy for row in rows}
        )
    )
    overall.update(
        {
            "v1_depth": sum(row.v1_depth for row in rows),
            "portfolio_depth": sum(row.portfolio_depth for row in rows),
            "selected_v1": sum(row.selected_policy == "v1" for row in rows),
            "selected_ordered_potential": sum(
                row.selected_policy == "ordered_potential" for row in rows
            ),
            "selected_policy_counts": selected_policy_counts,
            "total_runtime_ms": sum(row.total_runtime_ms for row in rows),
            "mean_runtime_ms": statistics.mean(
                row.total_runtime_ms for row in rows
            ) if rows else 0.0,
        }
    )

    return {
        "overall": overall,
        "by_topology": {
            name: group_summary(group) for name, group in sorted(by_topology.items())
        },
        "by_circuit_mode": {
            name: group_summary(group) for name, group in sorted(by_mode.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="historical development：v1 + ordered potential + front-sum 多策略 portfolio"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmarks_v3/dev.json"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-indices", type=str, default=None)
    parser.add_argument(
        "--mapping",
        choices=("pool", "v1_best", "identity"),
        default="pool",
        help="正式 portfolio 推荐 pool；复现 historical development 使用 v1_best",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default=None,
        help="默认 v1,ordered_potential,front_sum",
    )
    parser.add_argument(
        "--attempt-output",
        type=Path,
        default=Path("results/mapping_portfolio_portfolio_attempts.csv"),
    )
    parser.add_argument(
        "--case-output",
        type=Path,
        default=Path("results/mapping_portfolio_portfolio_cases.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/mapping_portfolio_portfolio_summary.json"),
    )
    args = parser.parse_args()

    # 和 historical development/20 保持完全一致的公共参数。
    structural_policy.one_step_policy.THETA = 5
    structural_policy.one_step_policy.W_FUTURE = 0.5
    structural_policy.one_step_policy.GAMMA = 0.5
    structural_policy.one_step_policy.ALPHA = 0.5
    structural_policy.one_step_policy.LAMBDA_DEPTH = 0.0

    cases = choose_cases(
        load_cases(args.input, limit=None),
        parse_indices(args.case_indices),
        args.limit,
    )
    policies = parse_policies(args.policies)

    print("===== historical development：多策略 Portfolio =====")
    print(f"输入：{args.input.resolve()}")
    print(f"实例数：{len(cases)}")
    print(f"策略：{', '.join(policy.name for policy in policies)}")
    print(f"映射：{args.mapping}")

    all_attempts: list[AttemptRow] = []
    case_rows: list[CaseRow] = []

    for index, case in enumerate(cases, start=1):
        mappings = make_mappings(case, args.mapping)
        attempts = run_attempts(case, mappings, policies)
        row = build_case_row(case, mappings, policies, attempts)
        all_attempts.extend(attempts)
        case_rows.append(row)

        print(
            f"[{index}/{len(cases)}] {case.name}: "
            f"v1={row.v1_swap}/{row.v1_depth}, "
            f"ORDERED={row.ordered_frontier_swap}/{row.ordered_frontier_depth}, "
            f"portfolio={row.portfolio_swap}/{row.portfolio_depth} "
            f"({row.selected_policy}, {row.selected_mapping}), "
            f"DeltaSWAP=-{row.swap_improvement_vs_v1}"
        )

        # 每个实例完成后立即落盘，长实验中断也能保留进度。
        write_csv(
            args.attempt_output,
            all_attempts,
            AttemptRow.__annotations__.keys(),
        )
        write_csv(
            args.case_output,
            case_rows,
            CaseRow.__annotations__.keys(),
        )
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summarize(case_rows), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = summarize(case_rows)
    overall = summary["overall"]
    print()
    print("===== historical development 汇总 =====")
    print(f"v1 total SWAP：{overall['v1_swap']}")
    print(f"portfolio total SWAP：{overall['portfolio_swap']}")
    print(f"减少 SWAP：{overall['swap_improvement']}")
    print(f"相对改善：{100.0 * overall['relative_improvement']:.2f}%")
    print(f"SWAP 严格改善实例：{overall['improved_cases']}/{overall['case_count']}")
    print(
        "SWAP 相同但深度改善实例："
        f"{overall['swap_ties_with_better_depth']}/{overall['case_count']}"
    )
    print("策略被选次数：", overall["selected_policy_counts"])
    print(f"总运行时间：{overall['total_runtime_ms']:.1f} ms")
    print()
    print(f"尝试明细：{args.attempt_output.resolve()}")
    print(f"逐实例结果：{args.case_output.resolve()}")
    print(f"汇总 JSON：{args.summary_output.resolve()}")


if __name__ == "__main__":
    main()
