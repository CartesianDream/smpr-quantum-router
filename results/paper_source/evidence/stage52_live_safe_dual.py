from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from qiskit import qasm3, qpy, transpile
from qiskit.quantum_info import Clifford
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import stage21_policy_portfolio as stage21
import stage25_rollout_policy_improvement as stage25
import stage26_adaptive_mapping_rollout as stage26
import stage27_ring_mapping_search as stage27
import stage28_frozen_router_vs_lightsabre as stage28
import stage30_proxy_grid_mapping as stage30
import stage32_frozen_final_benchmark as stage32
import stage50_lightsabre_winner_export as stage50_ls
import stage50_oaabr_circuit_export as stage50_oaabr
import stage51_semantic_audit as stage51

from stage7_initial_mapping_pool import MappingCandidate
from stage10_dev_benchmark import DevCase, load_cases
from stage6_fixed_layout_benchmark import (
    build_coupling_map,
    build_qiskit_circuit,
    weighted_qiskit_depth,
)


@dataclass
class RouteCandidate:
    backend: str
    label: str
    swap: int
    depth: int
    runtime_ms: float
    initial_mapping: tuple[int, ...]
    final_mapping: tuple[int, ...]
    seed: int | None = None
    repeat: int | None = None
    state: Any | None = None
    circuit: Any | None = None
    route_decisions: int = 0
    changed_decisions: int = 0
    proxy_score: float | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_indices(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def choose_cases(
    cases: list[DevCase],
    indices: list[int] | None,
    limit: int | None,
) -> list[tuple[int, DevCase]]:
    indexed = list(enumerate(cases))
    if indices is not None:
        selected: list[tuple[int, DevCase]] = []
        for index in indices:
            if not 0 <= index < len(cases):
                raise IndexError(f"case index 越界：{index}")
            selected.append((index, cases[index]))
        return selected
    return indexed if limit is None else indexed[:limit]


def candidate_key(candidate: RouteCandidate) -> tuple:
    # 完全同质量时优先保留 LightSABRE：它有原生 QPY，且通常更快。
    backend_priority = {
        "lightsabre": 0,
        "hybrid": 1,
        "oaabr": 2,
    }
    return (
        candidate.swap,
        candidate.depth,
        backend_priority.get(candidate.backend, 9),
        candidate.runtime_ms,
        candidate.label,
    )


def mapping_text(mapping: Iterable[int]) -> str:
    return ",".join(str(value) for value in mapping)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def top_k_for_case(case: DevCase) -> int:
    return (
        4
        if stage32.frozen_variant(case)
        == "stage31_grid_n10_alternating_k4"
        else 2
    )


def run_oaabr_mapping(
    case: DevCase,
    mapping: tuple[int, ...],
    label: str,
    backend: str,
    config,
    top_k: int,
    proxy_score: float | None = None,
    warm_base: bool = False,
) -> RouteCandidate:
    dag, hardware, _ = case.test_case.build()
    start = time.perf_counter()
    # 冻结的 Stage 25/26 原生路径会先在同一个 HardwareGraph 上运行
    # 一次基线策略，再执行 rollout。除了计入原始预算，这还会以完全
    # 确定的方式预热最短路缓存。原生 OAABR 必须保留此行为，才能与
    # 1843 的冻结基线逐例一致。交叉布局候选是独立的新后端，冷启动。
    if warm_base:
        base_state = stage25.stage20.run_router(
            dag=dag,
            hardware=hardware,
            initial_mapping=list(mapping),
            config=config,
        )
        base_state.assert_valid(dag, hardware)
    state, stats = stage25.run_rollout_router(
        dag=dag,
        hardware=hardware,
        initial_mapping=list(mapping),
        base_config=config,
        top_k=top_k,
    )
    runtime_ms = (time.perf_counter() - start) * 1000.0
    state.assert_valid(dag, hardware)
    if len(state.executed_gates) != len(dag.gates):
        raise AssertionError(f"{case.name}/{label} 仍有未执行逻辑门")
    return RouteCandidate(
        backend=backend,
        label=label,
        swap=stage25.count_swaps(state),
        depth=state.current_depth(),
        runtime_ms=runtime_ms,
        initial_mapping=tuple(mapping),
        final_mapping=tuple(state.logical_to_physical),
        state=state,
        route_decisions=stats.route_decisions,
        changed_decisions=stats.changed_decisions,
        proxy_score=proxy_score,
    )


def stage26_mappings(case: DevCase) -> tuple[list[MappingCandidate], int, str]:
    mapping_mode, top_k = stage26.budget_for_case(
        case=case,
        pool_modes=stage28.POOL_MODES,
        default_mapping=stage28.DEFAULT_MAPPING,
        default_top_k=stage28.DEFAULT_TOP_K,
        pool_top_k=stage28.POOL_TOP_K,
    )
    return (
        stage21.make_mappings(case, mapping_mode),
        top_k,
        f"{mapping_mode}/k{top_k}",
    )


def stage27_mappings(
    case: DevCase,
    config,
) -> tuple[list[MappingCandidate], int, str]:
    rows, candidates = stage27.search_mapping_candidates(
        case,
        config,
        stage28.BEAM_WIDTH,
        stage28.SEARCH_ROUNDS,
        stage28.USE_FORWARD_BACKWARD,
    )
    valid = [row for row in rows if row.valid]
    if not valid:
        raise RuntimeError(f"{case.name} 没有合法 Stage 27 映射")

    by_name = {item.name: item for item in candidates.values()}
    old_mapping = stage21.make_mappings(case, "v1_best")[0]
    selected_names: list[str] = []
    old_pool_name = next(
        (
            item.name
            for item in candidates.values()
            if item.mapping == old_mapping.mapping
        ),
        None,
    )
    if old_pool_name is not None:
        selected_names.append(old_pool_name)

    ranked = sorted(
        valid,
        key=lambda row: (
            int(row.base_swap),
            int(row.base_depth),
            row.mapping_name,
        ),
    )
    for row in ranked:
        if len(selected_names) >= stage28.ROLLOUT_TOP:
            break
        if row.mapping_name not in selected_names:
            selected_names.append(row.mapping_name)

    return (
        [by_name[name] for name in selected_names],
        stage28.DEFAULT_TOP_K,
        "local-search-b2-r1/rollout2-k2",
    )


def stage30_mappings(
    case: DevCase,
    config,
    top_k: int,
) -> tuple[list[MappingCandidate], int, str]:
    rows, candidates = stage30.build_proxy_shortlist(
        case=case,
        config=config,
        proxy_decay=stage32.PROXY_DECAY,
        proxy_top=stage32.PROXY_TOP,
        use_forward_backward=True,
    )
    valid = [row for row in rows if row.valid]
    if not valid:
        raise RuntimeError(f"{case.name} 没有合法 Stage 30 映射")

    by_name = {item.name: item for item in candidates.values()}
    old_mapping = stage21.make_mappings(case, "v1_best")[0]
    old_pool_name = next(
        (
            item.name
            for item in candidates.values()
            if item.mapping == old_mapping.mapping
        ),
        None,
    )
    if old_pool_name is None:
        raise RuntimeError(f"{case.name} 的旧映射不在 Stage 30 候选池")

    selected_names = [old_pool_name]
    ranked = sorted(
        valid,
        key=lambda row: (
            int(row.base_swap),
            int(row.base_depth),
            row.mapping_name,
        ),
    )
    for row in ranked:
        if len(selected_names) >= stage32.ROLLOUT_TOP:
            break
        if row.mapping_name not in selected_names:
            selected_names.append(row.mapping_name)

    return (
        [by_name[name] for name in selected_names],
        top_k,
        f"proxy-d095-top8/rollout2-k{top_k}",
    )


def run_native_oaabr(
    case: DevCase,
    config,
) -> tuple[RouteCandidate, list[RouteCandidate], str]:
    variant = stage32.frozen_variant(case)
    top_k = top_k_for_case(case)
    search_start = time.perf_counter()

    if variant == "stage27_ring_n10_far":
        mappings, top_k, budget = stage27_mappings(case, config)
    elif variant in {
        "stage30_grid_uniform_k2",
        "stage31_grid_n10_alternating_k4",
    }:
        mappings, top_k, budget = stage30_mappings(case, config, top_k)
    else:
        mappings, top_k, budget = stage26_mappings(case)

    search_ms = (time.perf_counter() - search_start) * 1000.0
    candidates = [
        run_oaabr_mapping(
            case=case,
            mapping=tuple(item.mapping),
            label=f"native:{item.name}",
            backend="oaabr",
            config=config,
            top_k=top_k,
            warm_base=True,
        )
        for item in mappings
    ]
    best = min(candidates, key=candidate_key)
    # 映射池/代理搜索也属于原生 OAABR 的在线预算。
    best.runtime_ms += search_ms
    return best, candidates, f"{variant}/{budget}"


def extract_qiskit_mappings(
    routed,
    logical_qubits: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if routed.layout is None:
        raise RuntimeError("LightSABRE 输出缺少 TranspileLayout")
    initial = stage51.mapping_from_layout(
        routed.layout,
        "initial_virtual_layout",
        logical_qubits,
        routed.num_qubits,
    )
    final = stage51.mapping_from_layout(
        routed.layout,
        "final_virtual_layout",
        logical_qubits,
        routed.num_qubits,
    )
    return tuple(initial), tuple(final)


def run_lightsabre_trials(
    case: DevCase,
    seeds: int,
    repeats: int,
    warmup: bool,
) -> tuple[RouteCandidate, list[RouteCandidate], float]:
    logical = build_qiskit_circuit(case.test_case)
    coupling_map = build_coupling_map(case.test_case)
    if warmup:
        transpile(
            logical,
            coupling_map=coupling_map,
            layout_method="sabre",
            routing_method="sabre",
            optimization_level=0,
            seed_transpiler=0,
        )

    candidates: list[RouteCandidate] = []
    total_budget_ms = 0.0
    for seed in range(seeds):
        repeated: list[tuple[Any, int, int, float, int]] = []
        for repeat in range(repeats):
            start = time.perf_counter()
            routed = transpile(
                logical,
                coupling_map=coupling_map,
                layout_method="sabre",
                routing_method="sabre",
                optimization_level=0,
                seed_transpiler=seed,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            swap = int(routed.count_ops().get("swap", 0))
            depth = int(weighted_qiskit_depth(routed))
            repeated.append((routed, swap, depth, elapsed_ms, repeat))

        qualities = {(row[1], row[2]) for row in repeated}
        if len(qualities) != 1:
            raise RuntimeError(
                f"{case.name}/seed={seed} 重复运行产生不同质量：{qualities}"
            )
        median_ms = float(statistics.median(row[3] for row in repeated))
        total_budget_ms += median_ms
        routed, swap, depth, _elapsed, repeat = repeated[0]
        initial, final = extract_qiskit_mappings(routed, case.num_qubits)
        stage50_ls.validate_topology(routed, case.test_case.edges)
        candidates.append(
            RouteCandidate(
                backend="lightsabre",
                label=f"lightsabre:seed{seed}",
                swap=swap,
                depth=depth,
                runtime_ms=median_ms,
                initial_mapping=initial,
                final_mapping=final,
                seed=seed,
                repeat=repeat,
                circuit=routed,
            )
        )

    best = min(candidates, key=candidate_key)
    return best, candidates, total_budget_ms


def layout_proxy_score(
    case: DevCase,
    mapping: tuple[int, ...],
    decay: float,
) -> float:
    _dag, hardware, _ = case.test_case.build()
    score = 0.0
    cx_index = 0
    for gate in case.test_case.gate_specs:
        if gate[0] != "cx":
            continue
        left = mapping[int(gate[1])]
        right = mapping[int(gate[2])]
        distance = len(hardware.shortest_path(left, right)) - 1
        score += (decay**cx_index) * distance
        cx_index += 1
    return score


def swapped_neighbors(mapping: tuple[int, ...]) -> list[tuple[int, ...]]:
    output: list[tuple[int, ...]] = []
    for left in range(len(mapping)):
        for right in range(left + 1, len(mapping)):
            candidate = list(mapping)
            candidate[left], candidate[right] = (
                candidate[right],
                candidate[left],
            )
            output.append(tuple(candidate))
    return output


def best_quality_layouts(
    qiskit_candidates: list[RouteCandidate],
    maximum: int,
) -> list[RouteCandidate]:
    best_quality = min((item.swap, item.depth) for item in qiskit_candidates)
    ranked = sorted(
        (
            item
            for item in qiskit_candidates
            if (item.swap, item.depth) == best_quality
        ),
        key=lambda item: (item.runtime_ms, int(item.seed or 0)),
    )
    unique: list[RouteCandidate] = []
    seen: set[tuple[int, ...]] = set()
    for item in ranked:
        if item.initial_mapping in seen:
            continue
        seen.add(item.initial_mapping)
        unique.append(item)
        if maximum > 0 and len(unique) >= maximum:
            break
    return unique


def run_hybrid_candidates(
    case: DevCase,
    qiskit_candidates: list[RouteCandidate],
    config,
    maximum_layouts: int,
    neighbors_per_layout: int,
    proxy_decay: float,
) -> list[RouteCandidate]:
    top_k = top_k_for_case(case)
    output: list[RouteCandidate] = []
    seen: set[tuple[int, ...]] = set()
    layouts = best_quality_layouts(qiskit_candidates, maximum_layouts)

    for layout in layouts:
        base = layout.initial_mapping
        proxy = layout_proxy_score(case, base, proxy_decay)
        proposed: list[tuple[str, tuple[int, ...], float]] = [
            (f"ls_seed{layout.seed}", base, proxy)
        ]
        if neighbors_per_layout > 0:
            ranked_neighbors = sorted(
                (
                    (layout_proxy_score(case, item, proxy_decay), item)
                    for item in swapped_neighbors(base)
                ),
                key=lambda row: (row[0], row[1]),
            )[:neighbors_per_layout]
            proposed.extend(
                (
                    f"ls_seed{layout.seed}_neighbor{rank}",
                    mapping,
                    score,
                )
                for rank, (score, mapping) in enumerate(
                    ranked_neighbors,
                    start=1,
                )
            )

        for label, mapping, score in proposed:
            if mapping in seen:
                continue
            seen.add(mapping)
            output.append(
                run_oaabr_mapping(
                    case=case,
                    mapping=mapping,
                    label=f"hybrid:{label}",
                    backend="hybrid",
                    config=config,
                    top_k=top_k,
                    proxy_score=score,
                )
            )
    return output


def attempt_row(
    case_index: int,
    case: DevCase,
    candidate: RouteCandidate,
) -> dict[str, Any]:
    return {
        "case_index": case_index,
        "instance": case.name,
        "backend": candidate.backend,
        "label": candidate.label,
        "seed": "" if candidate.seed is None else candidate.seed,
        "repeat": "" if candidate.repeat is None else candidate.repeat,
        "swap": candidate.swap,
        "depth": candidate.depth,
        "runtime_ms": candidate.runtime_ms,
        "initial_mapping": mapping_text(candidate.initial_mapping),
        "final_mapping": mapping_text(candidate.final_mapping),
        "route_decisions": candidate.route_decisions,
        "changed_decisions": candidate.changed_decisions,
        "proxy_score": (
            "" if candidate.proxy_score is None else candidate.proxy_score
        ),
    }


def export_and_audit(
    stage: int,
    case_index: int,
    case: DevCase,
    selected: RouteCandidate,
    attempts: list[RouteCandidate],
    case_dir: Path,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    qasm_path = case_dir / "selected_physical.qasm"
    qpy_path: Path | None = None

    if selected.backend == "lightsabre":
        if selected.circuit is None:
            raise RuntimeError("LightSABRE 候选缺少 Qiskit circuit")
        # Qiskit 的物理 QASM 使用 $0/$1/...；完全空闲的物理线
        # 可能被 dumps 省略。全宽 barrier 只用于保留物理线数量。
        qasm_circuit = selected.circuit.copy()
        qasm_circuit.barrier(*qasm_circuit.qubits)
        qasm_path.write_text(
            qasm3.dumps(qasm_circuit),
            encoding="utf-8",
        )
        qpy_path = case_dir / "selected_physical.qpy"
        with qpy_path.open("wb") as handle:
            qpy.dump(selected.circuit, handle)
    else:
        if selected.state is None:
            raise RuntimeError("OAABR 候选缺少 RoutingState")
        _dag, hardware, _ = case.test_case.build()
        logical_cx, swap, expanded_cx = stage50_oaabr.export_qasm(
            selected.state,
            hardware,
            qasm_path,
        )
        if logical_cx != case.two_qubit_gate_count or swap != selected.swap:
            raise AssertionError("OAABR 导出计数不一致")
        if expanded_cx != logical_cx + 3 * swap:
            raise AssertionError("OAABR SWAP 展开计数不一致")

    parsed = qasm3.loads(qasm_path.read_text(encoding="utf-8"))
    source = build_qiskit_circuit(case.test_case)

    preaudit = {
        "instance": case.name,
        "backend": selected.backend,
        "label": selected.label,
        "parsed_qubits": parsed.num_qubits,
        "logical_qubits": source.num_qubits,
        "physical_qubits": case.test_case.num_physical_qubits,
        "initial_mapping": list(selected.initial_mapping),
        "final_mapping": list(selected.final_mapping),
    }
    (case_dir / "preaudit_debug.json").write_text(
        json.dumps(preaudit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    stage51.validate_mapping(
        list(selected.initial_mapping),
        source.num_qubits,
        parsed.num_qubits,
        f"{case.name}/initial_mapping",
    )
    stage51.validate_mapping(
        list(selected.final_mapping),
        source.num_qubits,
        parsed.num_qubits,
        f"{case.name}/final_mapping",
    )

    expected = stage51.expected_physical_circuit(
        source=source,
        initial_mapping=list(selected.initial_mapping),
        final_mapping=list(selected.final_mapping),
        physical_qubits=parsed.num_qubits,
    )
    semantic_circuit = PassManager(RemoveBarriers()).run(parsed)
    semantic_equivalent = bool(
        Clifford(semantic_circuit) == Clifford(expected)
    )
    topology_error_list = stage51.topology_errors(
        parsed,
        [list(edge) for edge in case.test_case.edges],
    )
    topology_valid = not topology_error_list
    actual_depth = stage51.weighted_depth(parsed)
    depth_valid = actual_depth == selected.depth

    if selected.backend == "lightsabre":
        metric_valid = int(parsed.count_ops().get("swap", 0)) == selected.swap
    else:
        metric_valid = (
            int(parsed.count_ops().get("cx", 0))
            == case.two_qubit_gate_count + 3 * selected.swap
        )
    passed = semantic_equivalent and topology_valid and depth_valid and metric_valid
    if not passed:
        raise AssertionError(
            f"{case.name} 在线审计失败：semantic={semantic_equivalent}, "
            f"topology={topology_valid}, depth={depth_valid}, "
            f"metric={metric_valid}"
        )

    manifest = {
        "stage": stage,
        "case_index": case_index,
        "instance": case.name,
        "selection_rule": "min(SWAP, weighted_depth), LightSABRE on exact tie",
        "selected_backend": selected.backend,
        "selected_label": selected.label,
        "swap": selected.swap,
        "weighted_depth": selected.depth,
        "initial_mapping": list(selected.initial_mapping),
        "final_mapping": list(selected.final_mapping),
        "semantic_equivalent": semantic_equivalent,
        "topology_valid": topology_valid,
        "metric_valid": metric_valid and depth_valid,
        "qasm": str(qasm_path.resolve()),
        "qasm_sha256": sha256(qasm_path),
        "qpy": None if qpy_path is None else str(qpy_path.resolve()),
        "qpy_sha256": None if qpy_path is None else sha256(qpy_path),
        "attempts": [
            {
                "backend": item.backend,
                "label": item.label,
                "seed": item.seed,
                "swap": item.swap,
                "depth": item.depth,
                "runtime_ms": item.runtime_ms,
                "initial_mapping": list(item.initial_mapping),
                "final_mapping": list(item.final_mapping),
                "proxy_score": item.proxy_score,
            }
            for item in attempts
        ],
    }
    manifest_path = case_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "semantic_equivalent": semantic_equivalent,
        "topology_valid": topology_valid,
        "metric_valid": metric_valid and depth_valid,
        "qasm": str(qasm_path.resolve()),
        "qpy": "" if qpy_path is None else str(qpy_path.resolve()),
        "manifest": str(manifest_path.resolve()),
    }


def summary_payload(
    stage: int,
    input_path: Path,
    rows: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    qiskit_total = sum(int(row["lightsabre_swap"]) for row in rows)
    oaabr_total = sum(int(row["oaabr_swap"]) for row in rows)
    dual_total = sum(int(row["safe_dual_swap"]) for row in rows)
    selected_total = sum(int(row["selected_swap"]) for row in rows)
    return {
        "stage": stage,
        "input": str(input_path.resolve()),
        "case_count": len(rows),
        "qiskit_seeds": args.qiskit_seeds,
        "qiskit_repeats": args.qiskit_repeats,
        "hybrid_enabled": stage >= 53,
        "hybrid_layouts": args.hybrid_layouts,
        "hybrid_neighbors": args.hybrid_neighbors,
        "proxy_decay": args.proxy_decay,
        "lightsabre_total_swap": qiskit_total,
        "oaabr_total_swap": oaabr_total,
        "safe_dual_total_swap": dual_total,
        "selected_total_swap": selected_total,
        "gain_vs_lightsabre": qiskit_total - selected_total,
        "gain_vs_safe_dual": dual_total - selected_total,
        "selected_backends": dict(
            Counter(str(row["selected_backend"]) for row in rows)
        ),
        "semantic_passed": sum(
            str(row["semantic_equivalent"]).lower() == "true" for row in rows
        ),
        "topology_passed": sum(
            str(row["topology_valid"]).lower() == "true" for row in rows
        ),
        "attempt_count": len(attempts),
        "total_wall_ms": sum(float(row["wall_ms"]) for row in rows),
    }


def main(
    *,
    stage: int = 52,
    default_hybrid_layouts: int = 0,
    default_hybrid_neighbors: int = 0,
) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 52: live LightSABRE/OAABR safe portfolio"
            if stage == 52
            else "Stage 53: live cross-layout safe hybrid portfolio"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmarks_v5/final_test.json"),
    )
    parser.add_argument("--case-indices", type=str)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--qiskit-seeds", type=int, default=20)
    parser.add_argument("--qiskit-repeats", type=int, default=1)
    parser.add_argument(
        "--hybrid-layouts",
        type=int,
        default=default_hybrid_layouts,
        help="最多使用多少个并列最优 LightSABRE 布局；0 表示全部",
    )
    parser.add_argument(
        "--hybrid-neighbors",
        type=int,
        default=default_hybrid_neighbors,
        help="每个 LightSABRE 布局真实路由多少个代理最优一换位邻居",
    )
    parser.add_argument("--proxy-decay", type=float, default=0.95)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--attempt-output", type=Path)
    parser.add_argument("--case-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    if args.qiskit_seeds <= 0 or args.qiskit_repeats <= 0:
        raise ValueError("qiskit-seeds/repeats 必须为正整数")
    if args.hybrid_layouts < 0 or args.hybrid_neighbors < 0:
        raise ValueError("hybrid-layouts/neighbors 必须为非负整数")
    if not 0.0 < args.proxy_decay <= 1.0:
        raise ValueError("proxy-decay 必须位于 (0, 1]")

    root = args.output_dir or Path(
        "results/stage52_live_dual"
        if stage == 52
        else "results/stage53_cross_layout_hybrid"
    )
    attempt_output = args.attempt_output or root / "attempts.csv"
    case_output = args.case_output or root / "cases.csv"
    summary_output = args.summary_output or root / "summary.json"
    root.mkdir(parents=True, exist_ok=True)

    all_cases = load_cases(args.input, None)
    cases = choose_cases(all_cases, parse_indices(args.case_indices), args.limit)
    config = stage32.configure_router()
    attempt_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []

    print(f"===== Stage {stage} live safe router =====")
    print("input:", args.input.resolve())
    print("cases:", len(cases))
    print(
        "LightSABRE:",
        f"{args.qiskit_seeds} seeds x {args.qiskit_repeats} repeats",
    )
    print("historical winner/reference CSV: not used")
    if stage >= 53:
        print(
            "hybrid:",
            f"best-quality layouts={args.hybrid_layouts or 'all'}, "
            f"neighbors/layout={args.hybrid_neighbors}, "
            f"proxy-decay={args.proxy_decay}",
        )

    for ordinal, (case_index, case) in enumerate(cases, start=1):
        wall_start = time.perf_counter()
        ls_best, ls_trials, ls_budget_ms = run_lightsabre_trials(
            case,
            args.qiskit_seeds,
            args.qiskit_repeats,
            not args.no_warmup,
        )
        oaabr_best, oaabr_trials, oaabr_variant = run_native_oaabr(case, config)
        safe_dual = min((ls_best, oaabr_best), key=candidate_key)

        hybrid_trials: list[RouteCandidate] = []
        if stage >= 53:
            hybrid_trials = run_hybrid_candidates(
                case=case,
                qiskit_candidates=ls_trials,
                config=config,
                maximum_layouts=args.hybrid_layouts,
                neighbors_per_layout=args.hybrid_neighbors,
                proxy_decay=args.proxy_decay,
            )

        selectable = [ls_best, oaabr_best, *hybrid_trials]
        selected = min(selectable, key=candidate_key)
        attempts = [*ls_trials, *oaabr_trials, *hybrid_trials]
        attempt_rows.extend(
            attempt_row(case_index, case, item) for item in attempts
        )

        case_dir = root / f"{case_index:03d}_{case.name}"
        audit = export_and_audit(
            stage,
            case_index,
            case,
            selected,
            attempts,
            case_dir,
        )
        hybrid_best = (
            min(hybrid_trials, key=candidate_key)
            if hybrid_trials
            else None
        )
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        row = {
            "case_index": case_index,
            "instance": case.name,
            "topology": case.topology,
            "circuit_mode": case.circuit_mode,
            "num_qubits": case.num_qubits,
            "cx_count": case.two_qubit_gate_count,
            "oaabr_variant": oaabr_variant,
            "lightsabre_seed": ls_best.seed,
            "lightsabre_swap": ls_best.swap,
            "lightsabre_depth": ls_best.depth,
            "lightsabre_budget_ms": ls_budget_ms,
            "oaabr_swap": oaabr_best.swap,
            "oaabr_depth": oaabr_best.depth,
            "safe_dual_backend": safe_dual.backend,
            "safe_dual_swap": safe_dual.swap,
            "safe_dual_depth": safe_dual.depth,
            "hybrid_swap": "" if hybrid_best is None else hybrid_best.swap,
            "hybrid_depth": "" if hybrid_best is None else hybrid_best.depth,
            "hybrid_attempts": len(hybrid_trials),
            "selected_backend": selected.backend,
            "selected_label": selected.label,
            "selected_swap": selected.swap,
            "selected_depth": selected.depth,
            "gain_vs_lightsabre": ls_best.swap - selected.swap,
            "gain_vs_safe_dual": safe_dual.swap - selected.swap,
            "initial_mapping": mapping_text(selected.initial_mapping),
            "final_mapping": mapping_text(selected.final_mapping),
            "semantic_equivalent": audit["semantic_equivalent"],
            "topology_valid": audit["topology_valid"],
            "metric_valid": audit["metric_valid"],
            "wall_ms": wall_ms,
            "qasm": audit["qasm"],
            "qpy": audit["qpy"],
            "manifest": audit["manifest"],
        }
        case_rows.append(row)
        write_rows(attempt_output, attempt_rows)
        write_rows(case_output, case_rows)
        summary = summary_payload(stage, args.input, case_rows, attempt_rows, args)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{ordinal}/{len(cases)}] {case.name}: "
            f"LS={ls_best.swap}/{ls_best.depth}, "
            f"OAABR={oaabr_best.swap}/{oaabr_best.depth}, "
            + (
                ""
                if hybrid_best is None
                else f"hybrid={hybrid_best.swap}/{hybrid_best.depth}, "
            )
            + f"selected={selected.backend} {selected.swap}/{selected.depth}, "
            f"gain={ls_best.swap-selected.swap:+d}, "
            f"{wall_ms:.1f} ms"
        )

    summary = summary_payload(stage, args.input, case_rows, attempt_rows, args)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    hash_output = root / "SHA256SUMS.txt"
    artifacts = {
        Path(row["qasm"]) for row in case_rows
    } | {
        Path(row["manifest"]) for row in case_rows
    } | {attempt_output.resolve(), case_output.resolve(), summary_output.resolve()}
    artifacts |= {Path(row["qpy"]) for row in case_rows if row["qpy"]}
    hash_output.write_text(
        "".join(
            f"{sha256(path)}  {path}\n"
            for path in sorted(artifacts, key=lambda item: str(item).lower())
        ),
        encoding="utf-8",
    )

    print(f"\n===== Stage {stage} summary =====")
    print("cases:", summary["case_count"])
    print("LightSABRE total SWAP:", summary["lightsabre_total_swap"])
    print("native OAABR total SWAP:", summary["oaabr_total_swap"])
    print("live safe dual total SWAP:", summary["safe_dual_total_swap"])
    print("selected total SWAP:", summary["selected_total_swap"])
    print("gain vs LightSABRE:", summary["gain_vs_lightsabre"])
    print("gain vs safe dual:", summary["gain_vs_safe_dual"])
    print("selected backends:", summary["selected_backends"])
    print("semantic passed:", f"{summary['semantic_passed']}/{summary['case_count']}")
    print("topology passed:", f"{summary['topology_passed']}/{summary['case_count']}")
    print("cases output:", case_output.resolve())
    print("summary:", summary_output.resolve())
    print("hashes:", hash_output.resolve())


if __name__ == "__main__":
    main()
