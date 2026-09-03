"""Overnight D: V16 第4轮策略发现（新策略族）——检验"12 通道饱和"是否网格假象。

新策略族（engine 已增量接入）：
- potential_mode 新增 ordered_lin（线性衰减）、ordered_sq（平方距离）
- candidate_mode 新增 multigeodesic（多测地线边）
- 配合参数范围外探索（beta 8-12、decay 0.1-0.9、theta 3/8）

评估在 B1 的 40 个 n≥8 开发 case 上，与 V15 基线（b1_final_case.csv merged）对比，
贪心集合覆盖（greedy_select）选互补通道。
- 有增益 → 突破网格饱和（大发现）
- 0 增益 → 确认饱和是根本规律（负结果，同样写结论）
per-case min 保证不倒退。

用法（在 SMPR_quantum_router_public 下）:
    PYTHONHASHSEED=0 python scripts/policy_discovery_v16.py --workers 12 --n-policies 100
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import random
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_ENGINE_DIR = _PROJECT_ROOT / "src" / "smpr_router" / "engine"
for _p in (_ENGINE_DIR, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import case_io  # noqa: E402
from routing_policy_search import (  # noqa: E402
    DEFAULT_GLOBALS,
    effective_total,
    evaluate_policies,
    greedy_select,
    init_worker,
    make_policy,
)

INF = 10**9


def new_family_policies(rng: random.Random, n: int) -> list[dict]:
    potential_modes = ["ordered_lin", "ordered_sq"]
    candidate_modes = ["multigeodesic", "ordered_frontier", "current"]
    search_modes = ["one_step", "terminal_two_step"]
    betas = [0.25, 0.5, 1.0, 2.0, 4.0]
    undos = [0.0, 2.5]
    decays = [0.3, 0.5, 0.7]

    out: list[dict] = []
    pid = 0
    # 保证每个 (pm, cm, sm) 组合至少一个
    combos = [
        (pm, cm, sm)
        for pm in potential_modes
        for cm in candidate_modes
        for sm in search_modes
    ]
    for pm, cm, sm in combos:
        g = dict(DEFAULT_GLOBALS)
        g["ordered_decay"] = rng.choice(decays)
        beam = None if sm == "one_step" else (rng.choice([2, 3]), rng.choice([2, 3]))
        out.append(make_policy(pid, pm, cm, sm, rng.choice(betas), rng.choice(undos), beam, g))
        pid += 1

    # 剩余：随机 + 参数极端
    while len(out) < n:
        pm = rng.choice(potential_modes)
        cm = rng.choice(candidate_modes)
        sm = rng.choice(search_modes)
        g = dict(DEFAULT_GLOBALS)
        if rng.random() < 0.25:
            # 参数范围外探索
            beta = rng.choice([8.0, 12.0])
            undo = rng.choice([5.0, 0.0])
            g["ordered_decay"] = rng.choice([0.1, 0.9])
            g["theta"] = rng.choice([3, 8])
            g["w_future"] = rng.choice([0.2, 0.8])
        else:
            beta = rng.choice(betas)
            undo = rng.choice(undos)
            g["ordered_decay"] = rng.choice(decays)
        beam = None if sm == "one_step" else (rng.choice([2, 3, 4]), rng.choice([2, 3, 4]))
        out.append(make_policy(pid, pm, cm, sm, beta, undo, beam, g))
        pid += 1
    return out[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description="V16 第4轮新族策略发现")
    parser.add_argument("--bench", type=Path,
                        default=_PROJECT_ROOT / "data" / "benchmarks_v5" / "final_test.json")
    parser.add_argument("--attempts", type=Path,
                        default=_PROJECT_ROOT / "results" / "v14_layout_search" / "attempts_v5_plus.csv")
    parser.add_argument("--baseline", type=Path,
                        default=_PROJECT_ROOT / "results" / "v14_layout_search" / "b1_final_case.csv")
    parser.add_argument("--n-policies", type=int, default=100)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--layout-top-k", type=int, default=12)
    parser.add_argument("--out", type=Path, default=_PROJECT_ROOT / "results" / "v16_discovery4")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    baseline: dict[str, int] = {}
    with open(args.baseline, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            baseline[r["case"]] = int(r["merged"])

    cases = case_io.load_cases(args.bench, None)
    dev_names = [c.name for c in cases if c.num_qubits >= 8]
    base_total = sum(baseline[c] for c in dev_names if c in baseline)
    print(f"dev cases: {len(dev_names)}，baseline total: {base_total}")

    rng = random.Random(args.seed)
    policies = new_family_policies(rng, args.n_policies)
    print(f"new-family policies: {len(policies)}（ordered_lin/ordered_sq × multigeodesic/... × 参数外）")

    with multiprocessing.Pool(
        args.workers,
        initializer=init_worker,
        initargs=(str(args.bench), str(args.attempts), args.layout_top_k),
    ) as pool:
        per = evaluate_policies(pool, policies, dev_names, use_top=True)

    initial = {c: baseline.get(c, INF) for c in dev_names}
    selected = greedy_select(per, policies, max_n=20, initial_case_best=initial)

    print("\n===== V16 第4轮新族发现 =====")
    print(f"基线 total: {base_total}")
    print(f"greedy 选出的新通道: {len(selected)}")
    for p in selected:
        print(f"  {p['name']} {p['potential_mode']}/{p['candidate_mode']}/{p['search_mode']} "
              f"beta={p['beta_unlock']} undo={p['undo_weight']} "
              f"decay={p['globals'].get('ordered_decay')}")
    final_total = effective_total(per, selected, initial, dev_names)
    print(f"选定后 total: {final_total}  delta={final_total - base_total}")
    if not selected:
        print("→ 0 增益：12 通道饱和是根本规律（新族也无法突破）")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selected_v16_channels.json").write_text(
        json.dumps({"channels": selected}, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [{
        "id": p["id"], "name": p["name"],
        "potential_mode": p["potential_mode"],
        "candidate_mode": p["candidate_mode"],
        "search_mode": p["search_mode"],
        "beta": p["beta_unlock"], "undo": p["undo_weight"],
        "decay": p["globals"].get("ordered_decay"),
    } for p in policies]
    with (args.out / "policy_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"结果: {args.out.resolve()}")


if __name__ == "__main__":
    main()
