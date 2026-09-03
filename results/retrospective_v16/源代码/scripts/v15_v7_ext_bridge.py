"""Overnight B: 12 通道路由跑扩展公共 heavy-hex 电路（V7 的 10 条 → 25 条新增）。

对 v7_extended_cases.json 每个 case：
1. run_smpr(case) -> 全布局池（LS 20 种子 + native + cross）+ ls_best（LS 路由最佳）
2. 对 top-K 布局跑 V14 12 通道 multi_score -> per-case min
3. 视角 A 同布局隔离：v14_on_lsbest vs ls_best.swap（同一布局，不同路由器）
   视角 A+ 布局池：v14_min_over_layouts vs ls_best.swap
4. W/T/L + 总 SWAP + 按类型分桶

严谨性：PYTHONHASHSEED=0；以 LS 结果为地板不倒退。

用法（在 SMPR_quantum_router_public 下）:
    PYTHONHASHSEED=0 python scripts/v15_v7_ext_bridge.py \
        --input data/v7_extended_cases.json --workers 12 --out results/v15_ext_bridge
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import sys
import time
from collections import defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_ENGINE_DIR = _PROJECT_ROOT / "src" / "smpr_router" / "engine"
for _p in (_ENGINE_DIR, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import case_io
import smpr_portfolio
from v13 import multi_score
from run_v7_core import run_smpr

INF = 10**9

_W_CASES: dict = {}
_W_CHANNELS: list[dict] = []


def init_worker(bench_path: str, channels: list[dict]) -> None:
    global _W_CASES, _W_CHANNELS
    _W_CASES = {c.name: c for c in case_io.load_cases(Path(bench_path), None)}
    _W_CHANNELS = channels


def eval_layout(task: tuple) -> tuple:
    case_name, mapping = task
    case = _W_CASES[case_name]
    return case_name, mapping, multi_score(case, mapping, _W_CHANNELS)


def main() -> None:
    parser = argparse.ArgumentParser(description="V15 × 扩展公共电路桥梁")
    parser.add_argument("--input", type=Path,
                        default=_PROJECT_ROOT / "data" / "v7_extended_cases.json")
    parser.add_argument("--channels", type=Path,
                        default=_PROJECT_ROOT / "results" / "v14_discovery2" / "selected_channels_plus_v13.json")
    parser.add_argument("--layout-top-k", type=int, default=12)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--lean", action="store_true",
                        help="跳过 run_smpr 全管线，只用 run_lightsabre_trials 拿 LS 布局+ls_best")
    parser.add_argument("--max-cx", type=int, default=0, help=">0 时跳过 CX 超过该值的 case")
    parser.add_argument("--out", type=Path, default=_PROJECT_ROOT / "results" / "v15_ext_bridge")
    parser.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 个 case（调试用）")
    args = parser.parse_args()

    channels = json.loads(args.channels.read_text(encoding="utf-8"))["channels"]
    cases = case_io.load_cases(args.input, None)
    if args.max_cx:
        cases = [c for c in cases if c.two_qubit_gate_count <= args.max_cx]
    if args.limit:
        cases = cases[: args.limit]
    print(f"ext cases: {len(cases)}，12 通道，workers={args.workers}，lean={args.lean}")

    # ===== Phase 1：run_smpr 全布局池 + LS 隔离基线 =====
    start = time.perf_counter()
    ls_meta = {}
    for case in cases:
        if args.lean:
            ls_best, ls_trials, _ = smpr_portfolio.run_lightsabre_trials(
                case, seeds=args.seeds, repeats=1, warmup=False
            )
            attempts = list(ls_trials)
            smpr = {"smpr_swap": -1, "smpr_wall_ms": 0,
                    "internal_lightsabre_swap": ls_best.swap}
        else:
            smpr, attempts = run_smpr(case, cross_layout_layouts=4, cross_layout_neighbors=4)
        ls_cands = [c for c in attempts if getattr(c, "backend", "") == "lightsabre"]
        if ls_cands:
            ls_best = min(ls_cands, key=lambda c: c.swap)
        else:
            ls_best, _, _ = smpr_portfolio.run_lightsabre_trials(
                case, seeds=args.seeds, repeats=1, warmup=True
            )
        seen: dict[tuple, int] = {}
        for c in attempts:
            if c.swap < seen.get(c.initial_mapping, INF):
                seen[c.initial_mapping] = c.swap
        top = sorted(seen.items(), key=lambda kv: kv[1])[: args.layout_top_k]
        layouts = [mp for mp, _ in top]
        ls_meta[case.name] = {
            "ls_best_swap": ls_best.swap,
            "ls_best_layout": ls_best.initial_mapping,
            "ls_best_in_topk": ls_best.initial_mapping in layouts,
            "n_layouts": len(layouts),
            "layouts": layouts,
            "smpr_swap": smpr.get("smpr_swap", -1),
            "internal_ls": smpr.get("internal_lightsabre_swap", -1),
        }
        print(f"  {case.name}: ls_best={ls_best.swap} layouts={len(layouts)} smpr={smpr.get('smpr_swap')}",
              flush=True)

    # ===== Phase 2：12 通道 multi_score（并行）=====
    tasks = [(cn, mp) for cn, m in ls_meta.items() for mp in m["layouts"]]
    with multiprocessing.Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(str(args.input), channels),
    ) as pool:
        results = pool.map(eval_layout, tasks, chunksize=4)

    best_by = defaultdict(dict)
    for cn, mp, sw in results:
        best_by[cn][mp] = sw

    # ===== 汇总 =====
    rows = []
    totals = defaultdict(int)
    for case in cases:
        cn = case.name
        meta = ls_meta[cn]
        per_layout = best_by.get(cn, {})
        v14_min = min(per_layout.values()) if per_layout else INF
        v14_on_lsbest = per_layout.get(meta["ls_best_layout"], INF)
        ls_best = meta["ls_best_swap"]
        win_iso = v14_on_lsbest < ls_best
        win_pool = v14_min < ls_best
        rows.append({
            "case": cn,
            "type": cn.split("_")[2] if cn.count("_") >= 3 else "",
            "num_logical": case.num_qubits,
            "cx": case.two_qubit_gate_count,
            "ls_best_swap": ls_best,
            "v14_on_lsbest": v14_on_lsbest,
            "v14_min_over_layouts": v14_min,
            "iso_gain": ls_best - v14_on_lsbest,
            "pool_gain": ls_best - v14_min,
            "v7_smpr_swap": meta["smpr_swap"],
            "v7_internal_ls": meta["internal_ls"],
            "win_iso": win_iso,
            "win_pool": win_pool,
        })
        totals["ls"] += ls_best
        totals["v14_iso"] += v14_on_lsbest
        totals["v14_pool"] += v14_min
        totals["win_iso"] += int(win_iso)
        totals["win_pool"] += int(win_pool)

    # ===== 输出 =====
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "ext_bridge_results.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary_lines = [
        "===== V15 × 扩展公共电路桥梁 =====",
        f"cases: {len(cases)}（V7 10 条之外的扩展），12 通道，workers={args.workers}",
        "",
        "--- 逐 case ---",
    ]
    for r in rows:
        summary_lines.append(
            f"  {r['case']}: ls={r['ls_best_swap']} v14_on_lsbest={r['v14_on_lsbest']} "
            f"(iso {'✓' if r['win_iso'] else '=' if r['v14_on_lsbest']==r['ls_best_swap'] else '✗'}) "
            f"v14_pool={r['v14_min_over_layouts']}"
        )

    def wtl(key: str, base: str) -> str:
        w = sum(1 for r in rows if r[key] < r[base])
        t = sum(1 for r in rows if r[key] == r[base])
        l = sum(1 for r in rows if r[key] > r[base])
        return f"{w}/{t}/{l}"

    summary_lines += [
        "",
        "--- 汇总（A 同布局隔离）---",
        f"  总 SWAP: LS={totals['ls']} vs V14(同布局)={totals['v14_iso']}  "
        f"delta={totals['v14_iso']-totals['ls']} ({(totals['v14_iso']-totals['ls'])/totals['ls']*100:.1f}%)",
        f"  W/T/L (isolated) = {wtl('v14_on_lsbest', 'ls_best_swap')}",
        "--- 汇总（A+ 布局池）---",
        f"  总 SWAP: LS={totals['ls']} vs V14(池min)={totals['v14_pool']}  "
        f"delta={totals['v14_pool']-totals['ls']} ({(totals['v14_pool']-totals['ls'])/totals['ls']*100:.1f}%)",
        f"  W/T/L (pool) = {wtl('v14_min_over_layouts', 'ls_best_swap')}",
        f"  耗时 {time.perf_counter()-start:.0f}s",
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    (args.out / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(f"\n结果：{out_csv.resolve()}")


if __name__ == "__main__":
    main()
