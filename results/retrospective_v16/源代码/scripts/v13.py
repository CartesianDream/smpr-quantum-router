"""V13/V14：V5+ 精进方案（正式命名）。V5/V6 布局库 + 多通道路由，每 case 取两者更优。

- 不指定 --channels 时：保持 V13 原 dual（ordered_potential+rollout ∪ t2_2x2）。
- 指定 --channels <json> 时：V14 多通道，每个 V5 布局跑 N 个发现策略，per-case min。
- 每 case 取 min(V5 原始结果, 多通道结果)，保证绝不倒退。

用法（项目根目录）:
    python scripts/v13.py --bench data/benchmarks_v5/final_test.json \
        --attempts results/v5/attempts.csv \
        [--channels results/v14_discovery/selected_channels.json] [--workers 20]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import multiprocessing
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "smpr_router" / "engine"))

import one_step_policy
from case_io import load_cases
from layout_pool import MappingCandidate
import structural_policy
import rollout_policy
from structural_policy import StructuralConfig


INF = 10**9

# V13 原双通道（--channels 缺省时的行为）
_DEFAULT_CHANNELS = [
    {
        "name": "ordered_rollout",
        "config": {
            "name": "ordered_rollout",
            "beta_unlock": 2.0, "undo_weight": 0.0,
            "potential_mode": "ordered_potential", "candidate_mode": "ordered_frontier",
            "search_mode": "one_step", "beam_width_1": 8, "beam_width_2": 8,
        },
        "globals": {"theta": 5, "w_future": 0.5, "gamma": 0.5, "alpha": 0.5, "ordered_decay": 0.5},
        "rollout": True,
    },
    {
        "name": "t2_2x2",
        "config": {
            "name": "t2_2x2",
            "beta_unlock": 2.0, "undo_weight": 0.0,
            "potential_mode": "front_sum", "candidate_mode": "ordered_frontier",
            "search_mode": "terminal_two_step", "beam_width_1": 2, "beam_width_2": 2,
        },
        "globals": {"theta": 5, "w_future": 0.5, "gamma": 0.5, "alpha": 0.5, "ordered_decay": 0.5},
        "rollout": False,
    },
]


def apply_channel_globals(g: dict) -> None:
    one_step_policy.THETA = g.get("theta", 5)
    one_step_policy.W_FUTURE = g.get("w_future", 0.5)
    one_step_policy.GAMMA = g.get("gamma", 0.5)
    one_step_policy.ALPHA = g.get("alpha", 0.5)
    structural_policy.ORDERED_DECAY = g.get("ordered_decay", 0.5)


def multi_score(dc, mapping: tuple[int, ...], channels: list[dict]) -> int:
    """对单个布局跑所有通道，返回 best SWAP。"""
    best = INF
    cand = MappingCandidate(name="v5p", mapping=mapping, generation_ms=0.0)
    for ch in channels:
        apply_channel_globals(ch.get("globals", {}))
        cfg = StructuralConfig(**ch["config"])
        if ch.get("rollout"):
            try:
                att = rollout_policy.run_attempt(dc, cand, cfg, top_k=2)
                if att.valid:
                    best = min(best, att.rollout_swap)
            except Exception:
                pass
        else:
            try:
                row = structural_policy.run_mapping(dc, cand, cfg)
                if row.valid:
                    best = min(best, row.swap_count)
            except Exception:
                pass
    return best


# ============================================================
# 并行 worker：每进程加载一次数据
# ============================================================

_W_CASES: dict = {}
_W_LAYOUTS: dict[str, set[tuple]] = {}
_W_V5BEST: dict[str, int] = {}
_W_CHANNELS: list[dict] = []


def init_worker(bench_path: str, attempts_path: str, channels: list[dict]) -> None:
    global _W_CASES, _W_LAYOUTS, _W_V5BEST, _W_CHANNELS
    _W_CASES = {c.name: c for c in load_cases(Path(bench_path), None)}
    layouts: dict[str, set[tuple]] = collections.defaultdict(set)
    v5_best: dict[str, int] = {}
    with open(attempts_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            inst = r["instance"]
            swap = int(float(r["swap"]))
            layouts[inst].add(tuple(int(x) for x in r["initial_mapping"].split(",")))
            if inst not in v5_best or swap < v5_best[inst]:
                v5_best[inst] = swap
    _W_LAYOUTS = dict(layouts)
    _W_V5BEST = v5_best
    _W_CHANNELS = channels


def compute_case(case_name: str) -> dict:
    dc = _W_CASES[case_name]
    best_multi = INF
    for mt in _W_LAYOUTS.get(case_name, []):
        best_multi = min(best_multi, multi_score(dc, mt, _W_CHANNELS))
    v5v = _W_V5BEST.get(case_name, INF)
    return {
        "case": case_name,
        "v5": v5v,
        "multi": best_multi,
        "merged": min(v5v, best_multi),
        "improved": best_multi < v5v,
    }


def run_all(
    bench_path: Path,
    attempts_path: Path,
    channels: list[dict],
    workers: int,
) -> tuple[dict, list[dict]]:
    cases = load_cases(bench_path, None)
    case_names = [c.name for c in cases]

    if workers <= 1:
        init_worker(str(bench_path), str(attempts_path), channels)
        rows = [compute_case(name) for name in case_names]
    else:
        with multiprocessing.Pool(
            processes=workers,
            initializer=init_worker,
            initargs=(str(bench_path), str(attempts_path), channels),
        ) as pool:
            rows = pool.map(compute_case, case_names)

    totals = {"v5": 0, "multi": 0, "merged": 0, "improved": 0, "cases": len(rows)}
    for r in rows:
        totals["v5"] += r["v5"]
        totals["multi"] += r["multi"]
        totals["merged"] += r["merged"]
        totals["improved"] += int(r["improved"])
    return totals, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="V13/V14：V5 布局库 + 多通道路由精进")
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--channels", type=Path, default=None,
                        help="V14 发现的多通道 JSON（缺省用 V13 原 dual）")
    parser.add_argument("--workers", type=int, default=1, help="并行 worker 数")
    parser.add_argument("--case-output", type=Path, default=None,
                        help="逐 case 结果 CSV（可选）")
    args = parser.parse_args()

    if args.channels and args.channels.exists():
        channels = json.loads(args.channels.read_text(encoding="utf-8"))["channels"]
        print(f"使用 V14 多通道：{len(channels)} 个")
    else:
        channels = _DEFAULT_CHANNELS
        print("使用 V13 原 dual（2 通道）")

    start = time.perf_counter()
    totals, rows = run_all(args.bench, args.attempts, channels, args.workers)

    print("\n===== V5+ 结果 =====")
    print(f"  V5 原始    = {totals['v5']}")
    print(f"  多通道     = {totals['multi']}")
    print(f"  合并       = {totals['merged']}  (精进 {totals['v5'] - totals['merged']})")
    print(f"  改善 case  = {totals['improved']}/{totals['cases']}")
    print(f"  耗时 {time.perf_counter() - start:.0f}s")

    if args.case_output:
        with args.case_output.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case", "v5", "multi", "merged", "improved"])
            w.writeheader()
            for r in sorted(rows, key=lambda x: x["case"]):
                w.writerow(r)


if __name__ == "__main__":
    main()
