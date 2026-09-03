"""V14 Phase 2：真实成本布局搜索（大 case 布局源主攻）。

对每个大 case（n>=8）：
1. 构造逻辑交互矩阵 W（CX 计数）。
2. VF2 风格种子：高交互逻辑簇 → 高连通物理区（简化 SabrePreLayout），
   生成多个种子变体（度匹配 / 贪心加权分配 / 反向）。
3. 单交换"首个改进"爬山，适应度 = 入选通道（快速 t2）的真实路由 SWAP，
   与部署目标（多通道 min）对齐。沿途收集 visited 布局。
4. 主进程用完整多通道 multi_score 对每个种子的最优候选做真实评估，
   若新布局的全通道成本 < 当前 merged 基线，则收编（per-case min，绝不倒退）。

用法（在 SMPR_quantum_router_public 下）:
    PYTHONHASHSEED=0 python scripts/layout_search_realcost.py \
        --bench data/benchmarks_v5/final_test.json \
        --attempts results/v5/attempts.csv \
        --channels results/v14_discovery2/selected_channels.json \
        --baseline results/v14_discovery2/b1_case.csv \
        --out results/v14_layout_search --workers 20
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import multiprocessing
import random
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_ENGINE_DIR = _PROJECT_ROOT / "src" / "smpr_router" / "engine"
if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

import one_step_policy
import structural_policy
import rollout_policy
from structural_policy import StructuralConfig
from layout_pool import MappingCandidate
from case_io import load_cases

INF = 10**9


def apply_globals(g: dict) -> None:
    one_step_policy.THETA = g.get("theta", 5)
    one_step_policy.W_FUTURE = g.get("w_future", 0.5)
    one_step_policy.GAMMA = g.get("gamma", 0.5)
    one_step_policy.ALPHA = g.get("alpha", 0.5)
    structural_policy.ORDERED_DECAY = g.get("ordered_decay", 0.5)


def single_route(case, mapping, ch: dict) -> int:
    """单个通道真实路由，返回 SWAP 或 INF。"""
    apply_globals(ch.get("globals", {}))
    cfg = StructuralConfig(**ch["config"])
    cand = MappingCandidate(name="search", mapping=mapping, generation_ms=0.0)
    try:
        row = structural_policy.run_mapping(case, cand, cfg)
        return row.swap_count if row.valid else INF
    except Exception:
        return INF


def fitness_score(case, mapping, fitness_channels: list[dict]) -> int:
    """搜索适应度：对与部署对齐的通道子集求 multi_score（含 rollout 通道）。"""
    return multi_score(case, mapping, fitness_channels)


def multi_score(case, mapping, channels: list[dict]) -> int:
    """完整多通道评估（与 v13.py 部署一致）。"""
    best = INF
    cand = MappingCandidate(name="search", mapping=mapping, generation_ms=0.0)
    for ch in channels:
        apply_globals(ch.get("globals", {}))
        cfg = StructuralConfig(**ch["config"])
        if ch.get("rollout"):
            try:
                att = rollout_policy.run_attempt(case, cand, cfg, top_k=2)
                if att.valid:
                    best = min(best, att.rollout_swap)
            except Exception:
                pass
        else:
            try:
                row = structural_policy.run_mapping(case, cand, cfg)
                if row.valid:
                    best = min(best, row.swap_count)
            except Exception:
                pass
    return best


# ============================================================
# 交互矩阵与 VF2 风格种子
# ============================================================

def interaction_matrix(dag, n: int) -> list[list[float]]:
    W = [[0.0] * n for _ in range(n)]
    for g in dag.gates:
        if g.is_two_qubit:
            a, b = g.qubits
            W[a][b] += 1.0
            W[b][a] += 1.0
    return W


def physical_centrality(hardware) -> list[float]:
    n = hardware.num_qubits
    deg = [len(hardware.adjacency[i]) for i in range(n)]
    return [deg[i] + 0.5 * sum(deg[j] for j in hardware.adjacency[i]) for i in range(n)]


def seed_deg_match(W, hardware) -> tuple:
    n = hardware.num_qubits
    cent = physical_centrality(hardware)
    logical_order = sorted(range(n), key=lambda q: -sum(W[q]))
    phys_order = sorted(range(n), key=lambda p: -cent[p])
    mapping = [0] * n
    for i, q in enumerate(logical_order):
        mapping[q] = phys_order[i]
    return tuple(mapping)


def seed_greedy(W, hardware) -> tuple:
    """贪心加权分配：交互重的逻辑先放，选与已放置邻居加权距离最小的物理点。"""
    n = hardware.num_qubits
    cent = physical_centrality(hardware)
    logical_order = sorted(range(n), key=lambda q: -sum(W[q]))
    dist = [
        [len(hardware.shortest_path(u, v)) - 1 for v in range(n)]
        for u in range(n)
    ]
    mapping = [-1] * n
    placed = set()
    for q in logical_order:
        best_p, best_cost = None, 1e18
        for p in range(n):
            if p in placed:
                continue
            cost = 0.0
            for q2 in range(n):
                if mapping[q2] != -1 and W[q][q2] > 0:
                    cost += W[q][q2] * dist[p][mapping[q2]]
            cost -= 0.01 * cent[p]
            if cost < best_cost:
                best_cost, best_p = cost, p
        mapping[q] = best_p
        placed.add(best_p)
    return tuple(mapping)


def seed_reverse(W, hardware) -> tuple:
    mapping = list(seed_deg_match(W, hardware))
    n = len(mapping)
    mapping.reverse()
    return tuple(mapping)


# ============================================================
# 爬山
# ============================================================

def neighbor_pairs(W, n: int) -> list[tuple]:
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pairs.sort(key=lambda ij: -W[ij[0]][ij[1]])
    return pairs


def hill_climb(case, start: tuple, W, n: int, fitness_channels: list[dict], max_steps: int = 25):
    cur = list(start)
    cur_swap = fitness_score(case, cur, fitness_channels)
    path = [tuple(cur)]
    pairs = neighbor_pairs(W, n)
    for _ in range(max_steps):
        improved = False
        for i, j in pairs:
            cand = list(cur)
            cand[i], cand[j] = cand[j], cand[i]
            cswap = fitness_score(case, cand, fitness_channels)
            if cswap < cur_swap:
                cur, cur_swap = cand, cswap
                path.append(tuple(cur))
                improved = True
                break
        if not improved:
            break
    return path, cur_swap


# ============================================================
# worker（每进程一次数据）
# ============================================================

_W_CASES: dict = {}
_W_CHANNELS: list[dict] = []
_W_FITNESS: list[dict] = []


def init_worker(bench_path: str, channels: list[dict], fitness_channels: list[dict]) -> None:
    global _W_CASES, _W_CHANNELS, _W_FITNESS
    _W_CASES = {c.name: c for c in load_cases(Path(bench_path), None)}
    _W_CHANNELS = channels
    _W_FITNESS = fitness_channels


def verify_candidate(task: tuple) -> tuple:
    """(case_name, mapping) -> (case_name, mapping, full_multi_swap)。"""
    case_name, mapping = task
    case = _W_CASES[case_name]
    return case_name, mapping, multi_score(case, mapping, _W_CHANNELS)


def search_case_seed(task: tuple) -> dict:
    """(case_name, seed_index) -> 该种子爬山的最优候选（按适应度）。"""
    case_name, seed_index = task
    case = _W_CASES[case_name]
    dag, hardware, _ = case.test_case.build()
    n = case.num_qubits
    W = interaction_matrix(dag, n)

    seeds = [
        seed_deg_match(W, hardware),
        seed_greedy(W, hardware),
        seed_reverse(W, hardware),
    ]
    base = seeds[seed_index % 3]
    if seed_index >= 3:
        # 随机扰动重启：从基础种子洗牌得到新起点，扩大搜索覆盖
        rng = random.Random(1000003 + sum(ord(ch) for ch in case_name) * 31 + seed_index)
        lst = list(base)
        rng.shuffle(lst)
        start = tuple(lst)
    else:
        start = base
    path, best_swap = hill_climb(case, start, W, n, _W_FITNESS)
    # 按适应度排序，返回 top-2（种子 + 爬山路途最优）
    scored = sorted(
        {m: fitness_score(case, m, _W_FITNESS) for m in {tuple(start), *path}}.items(),
        key=lambda kv: kv[1],
    )
    return {
        "case": case_name,
        "seed": seed_index,
        "best_fitness": best_swap,
        "candidates": [m for m, _ in scored[:2]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V14 Phase2：真实成本布局搜索")
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--channels", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="V14 部署的逐 case CSV（merged 列）")
    parser.add_argument("--out", type=Path, default=Path("results/v14_layout_search"))
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--restarts", type=int, default=1,
                        help="每个种子额外随机扰动重启次数（>1 时从扰动起点再爬山）")
    parser.add_argument("--only-large", action="store_true", default=True,
                        help="只搜 n>=8 大 case")
    parser.add_argument("--fitness-count", type=int, default=3,
                        help="搜索适应度通道数（对齐部署的强通道子集）")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    channels = json.loads(args.channels.read_text(encoding="utf-8"))["channels"]

    # 搜索适应度子集：优先 V13 rollout 冠军 + 前 N-1 个非 rollout 强通道（与部署对齐）
    rollout_ch = [c for c in channels if c.get("rollout")]
    fast_ch = [c for c in channels if not c.get("rollout")]
    fitness_channels = []
    if rollout_ch:
        fitness_channels.append(rollout_ch[0])  # 通常是 v13_ordered_rollout
    for c in fast_ch[: max(0, args.fitness_count - len(fitness_channels))]:
        fitness_channels.append(c)
    print(f"搜索适应度通道：{[c['name'] for c in fitness_channels]}")

    cases = load_cases(args.bench, None)
    if args.only_large:
        case_names = [c.name for c in cases if c.num_qubits >= 8]
    else:
        case_names = [c.name for c in cases]

    # 基线 merged（来自 V14 部署 CSV）
    baseline = {}
    with open(args.baseline, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            baseline[r["case"]] = int(r["merged"])

    print(f"Phase2 布局搜索：{len(case_names)} 个大 case，workers={args.workers}")
    start = time.perf_counter()

    with multiprocessing.Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(str(args.bench), channels, fitness_channels),
    ) as pool:
        tasks = [(cn, si) for cn in case_names for si in range(3 * args.restarts)]
        results = pool.map(search_case_seed, tasks, chunksize=4)
        print(f"搜索完成：{time.perf_counter()-start:.0f}s，开始并行验证", flush=True)

        # 聚合每 case 候选
        by_case: dict[str, list[tuple]] = collections.defaultdict(list)
        for r in results:
            by_case[r["case"]].extend(r["candidates"])
        verify_tasks = [
            (cn, mt)
            for cn in case_names
            for mt in dict.fromkeys(by_case[cn])
        ]
        verified = pool.map(verify_candidate, verify_tasks, chunksize=8)

    # 聚合验证结果
    best_by_case: dict[str, tuple[int, tuple]] = {}
    for cn, mt, sw in verified:
        cur = best_by_case.get(cn)
        if cur is None or sw < cur[0]:
            best_by_case[cn] = (sw, mt)

    rows = []
    total_gain = 0
    improved_cases = 0
    for cn in case_names:
        base = baseline.get(cn, INF)
        best_new, best_mt = best_by_case.get(cn, (INF, None))
        gain = max(0, base - best_new)
        improved = best_new < base
        rows.append({
            "case": cn, "baseline": base, "best_new": best_new,
            "gain": gain, "improved": improved,
            "mapping": ",".join(map(str, best_mt)) if best_mt else "",
        })
        if improved:
            improved_cases += 1
            total_gain += gain
        print(f"  {cn}: baseline={base} best_new={best_new} gain={gain} {'✓' if improved else ''}",
              flush=True)

    elapsed = time.perf_counter() - start
    print(f"\n===== Phase2 结果 =====")
    print(f"  改善 case：{improved_cases}/{len(case_names)}")
    print(f"  总 gain：{total_gain}")
    print(f"  耗时：{elapsed:.0f}s")

    # 保存
    out_csv = args.out / "search_results.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "baseline", "best_new", "gain", "improved", "mapping"])
        w.writeheader()
        w.writerows(rows)
    print(f"结果：{out_csv.resolve()}")


if __name__ == "__main__":
    main()
