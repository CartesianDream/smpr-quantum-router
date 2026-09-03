from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FROZEN_QISKIT_SEEDS = 20
FROZEN_QISKIT_REPEATS = 1
FROZEN_HYBRID_LAYOUTS = 0
FROZEN_HYBRID_NEIGHBORS = 4
FROZEN_PROXY_DECAY = 0.95

QUALITY_COLUMNS = (
    "lightsabre_swap",
    "lightsabre_depth",
    "oaabr_swap",
    "oaabr_depth",
    "safe_dual_backend",
    "safe_dual_swap",
    "safe_dual_depth",
    "hybrid_swap",
    "hybrid_depth",
    "hybrid_attempts",
    "selected_backend",
    "selected_swap",
    "selected_depth",
    "gain_vs_lightsabre",
    "gain_vs_safe_dual",
    "semantic_equivalent",
    "topology_valid",
    "metric_valid",
)

STAGE57_TIMING_RE = re.compile(
    r"\bcase=(?P<index>\d+)\b.*?:\s+"
    r"(?:ok|resumed),\s+(?P<seconds>\d+(?:\.\d+)?)s\s*$"
)
NAME_SIZE_RE = re.compile(r"(?:^|_)n(?P<n>\d+)(?:_|$)")
NAME_FACTOR_RE = re.compile(r"(?:^|_)f(?P<factor>\d+)(?:_|$)")
NAME_CX_RE = re.compile(r"(?:^|_)cx(?P<cx>\d+)(?:_|$)")


@dataclass(frozen=True)
class ShardTask:
    shard_id: int
    indices: tuple[int, ...]
    instances: tuple[str, ...]
    predicted_seconds: float
    child_dir: Path
    log_path: Path


@dataclass(frozen=True)
class ShardResult:
    shard_id: int
    indices: tuple[int, ...]
    child_dir: Path
    log_path: Path
    returncode: int
    elapsed_seconds: float
    resumed: bool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_indices(text: str | None, count: int) -> list[int]:
    if not text:
        return list(range(count))
    indices = [int(value.strip()) for value in text.split(",") if value.strip()]
    if len(indices) != len(set(indices)):
        raise ValueError("case-indices contains duplicates")
    for index in indices:
        if not 0 <= index < count:
            raise IndexError(f"case index out of range: {index}")
    return indices


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def decode_console_bytes(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    # Windows PowerShell 5.1 may emit UTF-16LE without a BOM through some
    # redirection pipelines. Detect the characteristic NUL-byte pattern.
    sample = raw[: min(len(raw), 4096)]
    odd_nuls = sample[1::2].count(0)
    even_nuls = sample[0::2].count(0)
    if odd_nuls > max(8, len(sample) // 8):
        return raw.decode("utf-16-le")
    if even_nuls > max(8, len(sample) // 8):
        return raw.decode("utf-16-be")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # Timing lines are ASCII, so replacement is safe for unrelated
        # locale-specific text surrounding them.
        return raw.decode("utf-8", errors="replace")


def parse_stage57_timings(path: Path | None) -> dict[int, float]:
    if path is None:
        return {}
    timings: dict[int, float] = {}
    text = decode_console_bytes(path.read_bytes())
    for line in text.splitlines():
        match = STAGE57_TIMING_RE.search(line.strip())
        if match is None:
            continue
        index = int(match.group("index"))
        seconds = float(match.group("seconds"))
        if seconds > 0:
            timings[index] = seconds
    if not timings:
        raise ValueError(f"no Stage 57 case timings found in: {path}")
    return timings


def extract_int(pattern: re.Pattern[str], text: str, fallback: int) -> int:
    match = pattern.search(text)
    return int(match.groupdict()[next(iter(match.groupdict()))]) if match else fallback


def heuristic_cost(case: dict[str, Any]) -> float:
    name = str(case.get("name", ""))
    n = int(case.get("logical_qubits") or case.get("n") or 0)
    if n <= 0:
        n = extract_int(NAME_SIZE_RE, name, 6)
    factor = int(case.get("length_factor") or case.get("factor") or 0)
    if factor <= 0:
        factor = extract_int(NAME_FACTOR_RE, name, 3)
    cx = int(case.get("cx_count") or case.get("cx") or 0)
    if cx <= 0:
        cx = extract_int(NAME_CX_RE, name, max(1, n * factor))
    # This estimate affects scheduling only. It never enters routing or selection.
    return float(max(1, cx) * max(1, n) ** 2)


def estimated_costs(
    source_cases: list[dict[str, Any]],
    indices: list[int],
    measured_timings: dict[int, float],
) -> dict[int, float]:
    measured = [
        measured_timings[index]
        for index in indices
        if index in measured_timings
    ]
    if measured:
        measured_scale = sum(measured) / len(measured)
        heuristic_values = [
            heuristic_cost(source_cases[index])
            for index in indices
            if index in measured_timings
        ]
        heuristic_scale = (
            sum(heuristic_values) / len(heuristic_values)
            if heuristic_values
            else 1.0
        )
        conversion = measured_scale / heuristic_scale
    else:
        conversion = 1.0
    return {
        index: measured_timings.get(
            index,
            heuristic_cost(source_cases[index]) * conversion,
        )
        for index in indices
    }


def build_shards(
    source_cases: list[dict[str, Any]],
    indices: list[int],
    workers: int,
    schedule: str,
    costs: dict[int, float],
    child_root: Path,
    log_root: Path,
) -> list[ShardTask]:
    shard_count = min(workers, len(indices))
    bins: list[list[int]] = [[] for _ in range(shard_count)]
    loads = [0.0] * shard_count

    if schedule == "lpt":
        ordered = sorted(indices, key=lambda index: (-costs[index], index))
        for index in ordered:
            shard_id = min(range(shard_count), key=lambda item: (loads[item], item))
            bins[shard_id].append(index)
            loads[shard_id] += costs[index]
    elif schedule == "round-robin":
        for position, index in enumerate(indices):
            shard_id = position % shard_count
            bins[shard_id].append(index)
            loads[shard_id] += costs[index]
    elif schedule == "contiguous":
        for position, index in enumerate(indices):
            shard_id = min(position * shard_count // len(indices), shard_count - 1)
            bins[shard_id].append(index)
            loads[shard_id] += costs[index]
    else:
        raise ValueError(f"unsupported schedule: {schedule}")

    tasks: list[ShardTask] = []
    for shard_id, shard_indices in enumerate(bins):
        ordered_indices = tuple(sorted(shard_indices))
        tasks.append(
            ShardTask(
                shard_id=shard_id,
                indices=ordered_indices,
                instances=tuple(
                    str(source_cases[index]["name"]) for index in ordered_indices
                ),
                predicted_seconds=loads[shard_id],
                child_dir=child_root / f"shard_{shard_id:02d}",
                log_path=log_root / f"shard_{shard_id:02d}.txt",
            )
        )
    return tasks


def validate_shard_output(task: ShardTask) -> bool:
    cases_path = task.child_dir / "cases.csv"
    summary_path = task.child_dir / "summary.json"
    attempts_path = task.child_dir / "attempts.csv"
    if not (cases_path.is_file() and summary_path.is_file() and attempts_path.is_file()):
        return False
    try:
        rows = read_rows(cases_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    row_indices = [int(row["case_index"]) for row in rows]
    row_instances = [str(row["instance"]) for row in rows]
    return (
        len(rows) == len(task.indices)
        and sorted(row_indices) == list(task.indices)
        and sorted(row_instances) == sorted(task.instances)
        and all(is_true(row.get("semantic_equivalent")) for row in rows)
        and all(is_true(row.get("topology_valid")) for row in rows)
        and all(is_true(row.get("metric_valid")) for row in rows)
        and int(summary.get("case_count", -1)) == len(task.indices)
    )


def run_shard(
    task: ShardTask,
    stage53_script: Path,
    input_path: Path,
    resume: bool,
) -> ShardResult:
    if resume and validate_shard_output(task):
        return ShardResult(
            shard_id=task.shard_id,
            indices=task.indices,
            child_dir=task.child_dir,
            log_path=task.log_path,
            returncode=0,
            elapsed_seconds=0.0,
            resumed=True,
        )
    if task.child_dir.exists():
        raise FileExistsError(
            "incomplete shard directory exists; use a fresh output directory "
            f"or move it before resuming: {task.child_dir}"
        )

    task.child_dir.mkdir(parents=True, exist_ok=False)
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(stage53_script),
        "--input",
        str(input_path),
        "--case-indices",
        ",".join(str(index) for index in task.indices),
        "--qiskit-seeds",
        str(FROZEN_QISKIT_SEEDS),
        "--qiskit-repeats",
        str(FROZEN_QISKIT_REPEATS),
        "--hybrid-layouts",
        str(FROZEN_HYBRID_LAYOUTS),
        "--hybrid-neighbors",
        str(FROZEN_HYBRID_NEIGHBORS),
        "--proxy-decay",
        str(FROZEN_PROXY_DECAY),
        "--output-dir",
        str(task.child_dir),
    ]
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    environment.setdefault("PYTHONUTF8", "1")
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        environment[variable] = "1"

    started = time.perf_counter()
    with task.log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
            env=environment,
            check=False,
        )
    return ShardResult(
        shard_id=task.shard_id,
        indices=task.indices,
        child_dir=task.child_dir,
        log_path=task.log_path,
        returncode=completed.returncode,
        elapsed_seconds=time.perf_counter() - started,
        resumed=False,
    )


def compare_quality(
    rows: list[dict[str, str]],
    reference_path: Path,
) -> list[dict[str, str]]:
    reference = {row["instance"]: row for row in read_rows(reference_path)}
    mismatches: list[dict[str, str]] = []
    for row in rows:
        old = reference.get(row["instance"])
        if old is None:
            mismatches.append(
                {
                    "instance": row["instance"],
                    "column": "<missing-reference>",
                    "reference": "",
                    "persistent": "present",
                }
            )
            continue
        for column in QUALITY_COLUMNS:
            if str(old.get(column, "")) != str(row.get(column, "")):
                mismatches.append(
                    {
                        "instance": row["instance"],
                        "column": column,
                        "reference": str(old.get(column, "")),
                        "persistent": str(row.get(column, "")),
                    }
                )
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 58: exact persistent-shard executor for frozen Stage 53"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmarks_v6/blind_test.json"),
    )
    parser.add_argument("--case-indices", type=str)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--schedule",
        choices=("lpt", "round-robin", "contiguous"),
        default="lpt",
    )
    parser.add_argument(
        "--timing-reference",
        type=Path,
        help="Stage 57 console log used only for LPT load balancing",
    )
    parser.add_argument(
        "--stage53-script",
        type=Path,
        default=Path(__file__).resolve().with_name(
            "stage53_cross_layout_safe_hybrid.py"
        ),
    )
    parser.add_argument("--quality-reference", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage58_persistent_sharding"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.stage53_script.is_file():
        raise FileNotFoundError(args.stage53_script)
    if args.quality_reference is not None and not args.quality_reference.is_file():
        raise FileNotFoundError(args.quality_reference)
    if args.timing_reference is not None and not args.timing_reference.is_file():
        raise FileNotFoundError(args.timing_reference)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    source_cases = payload.get("cases")
    if not isinstance(source_cases, list):
        raise ValueError("input JSON does not contain a cases list")
    indices = parse_indices(args.case_indices, len(source_cases))
    if args.limit is not None:
        indices = indices[: args.limit]
    if not indices:
        raise ValueError("no cases selected")

    root = args.output_dir.resolve()
    if root.exists() and not args.resume:
        raise FileExistsError(
            f"output directory already exists; use a new path or --resume: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    child_root = root / "shards"
    log_root = root / "worker_logs"
    child_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    measured_timings = parse_stage57_timings(args.timing_reference)
    costs = estimated_costs(source_cases, indices, measured_timings)
    tasks = build_shards(
        source_cases=source_cases,
        indices=indices,
        workers=args.workers,
        schedule=args.schedule,
        costs=costs,
        child_root=child_root,
        log_root=log_root,
    )

    print("===== Stage 58 persistent sharding frozen hybrid =====")
    print("input:", args.input.resolve())
    print("cases:", len(indices))
    print("workers / persistent shards:", len(tasks))
    print("schedule:", args.schedule)
    print("timing reference:", args.timing_reference)
    print("Stage 53:", args.stage53_script.resolve())
    print(
        "frozen parameters:",
        f"LS={FROZEN_QISKIT_SEEDS}x{FROZEN_QISKIT_REPEATS}, "
        f"layouts={FROZEN_HYBRID_LAYOUTS}, "
        f"neighbors={FROZEN_HYBRID_NEIGHBORS}, "
        f"decay={FROZEN_PROXY_DECAY}",
    )
    print("quality reference:", args.quality_reference)
    print("shard plan:")
    for task in tasks:
        print(
            f"  shard={task.shard_id:02d}, cases={len(task.indices)}, "
            f"predicted={task.predicted_seconds:.2f}, "
            f"indices={','.join(str(index) for index in task.indices)}"
        )

    wall_start = time.perf_counter()
    results: list[ShardResult] = []
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {
            executor.submit(
                run_shard,
                task,
                args.stage53_script.resolve(),
                args.input.resolve(),
                args.resume,
            ): task
            for task in tasks
        }
        completed_count = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed_count += 1
            status = (
                "resumed"
                if result.resumed
                else "ok"
                if result.returncode == 0
                else f"failed({result.returncode})"
            )
            print(
                f"[{completed_count}/{len(tasks)}] shard={result.shard_id:02d} "
                f"cases={len(result.indices)}: {status}, "
                f"{result.elapsed_seconds:.2f}s"
            )

    parallel_wall_seconds = time.perf_counter() - wall_start
    failed = sorted(
        (result for result in results if result.returncode != 0),
        key=lambda item: item.shard_id,
    )
    if failed:
        print("\n===== failed shard logs =====")
        for result in failed:
            try:
                lines = result.log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except OSError:
                lines = ["<log unavailable>"]
            print(f"--- shard {result.shard_id}: {result.log_path} ---")
            print("\n".join(lines[-40:]))
        raise RuntimeError(f"{len(failed)} Stage 53 shard processes failed")

    case_rows: list[dict[str, str]] = []
    attempt_rows: list[dict[str, str]] = []
    result_by_shard = {result.shard_id: result for result in results}
    shard_rows: list[dict[str, Any]] = []
    for task in tasks:
        if not validate_shard_output(task):
            raise RuntimeError(f"invalid shard output: {task.child_dir}")
        child_cases = read_rows(task.child_dir / "cases.csv")
        case_rows.extend(child_cases)
        attempt_rows.extend(read_rows(task.child_dir / "attempts.csv"))
        result = result_by_shard[task.shard_id]
        shard_rows.append(
            {
                "shard_id": task.shard_id,
                "case_count": len(task.indices),
                "case_indices": ",".join(str(index) for index in task.indices),
                "predicted_seconds": f"{task.predicted_seconds:.6f}",
                "actual_seconds": f"{result.elapsed_seconds:.6f}",
                "resumed": result.resumed,
                "output_dir": str(task.child_dir),
                "log_path": str(task.log_path),
            }
        )

    case_rows.sort(key=lambda row: int(row["case_index"]))
    attempt_rows.sort(
        key=lambda row: (
            int(row["case_index"]),
            str(row["backend"]),
            str(row["label"]),
            str(row["seed"]),
        )
    )
    merged_indices = [int(row["case_index"]) for row in case_rows]
    if merged_indices != sorted(indices):
        raise RuntimeError(
            f"merged case indices differ from requested indices: {merged_indices}"
        )

    cases_path = root / "cases.csv"
    attempts_path = root / "attempts.csv"
    shards_path = root / "shards.csv"
    write_rows(cases_path, case_rows)
    write_rows(attempts_path, attempt_rows)
    write_rows(shards_path, shard_rows)

    mismatches: list[dict[str, str]] = []
    if args.quality_reference is not None:
        mismatches = compare_quality(case_rows, args.quality_reference)
    mismatch_path = root / "quality_mismatches.csv"
    if mismatches:
        write_rows(mismatch_path, mismatches)

    ls_total = sum(int(row["lightsabre_swap"]) for row in case_rows)
    oaabr_total = sum(int(row["oaabr_swap"]) for row in case_rows)
    dual_total = sum(int(row["safe_dual_swap"]) for row in case_rows)
    selected_total = sum(int(row["selected_swap"]) for row in case_rows)
    semantic_passed = sum(
        is_true(row["semantic_equivalent"]) for row in case_rows
    )
    topology_passed = sum(is_true(row["topology_valid"]) for row in case_rows)
    metric_passed = sum(is_true(row["metric_valid"]) for row in case_rows)
    compute_seconds = sum(result.elapsed_seconds for result in results)
    safe = all(
        (int(row["selected_swap"]), int(row["selected_depth"]))
        <= (int(row["lightsabre_swap"]), int(row["lightsabre_depth"]))
        for row in case_rows
    )
    summary = {
        "stage": 58,
        "purpose": "exact persistent-shard execution of frozen Stage 53",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "stage53_script": str(args.stage53_script.resolve()),
        "stage53_sha256": sha256(args.stage53_script),
        "case_count": len(case_rows),
        "workers": len(tasks),
        "persistent_processes": len(tasks),
        "avoided_process_restarts": len(case_rows) - len(tasks),
        "schedule": args.schedule,
        "timing_reference": (
            None
            if args.timing_reference is None
            else str(args.timing_reference.resolve())
        ),
        "measured_timing_count": sum(
            index in measured_timings for index in indices
        ),
        "frozen_parameters": {
            "qiskit_seeds": FROZEN_QISKIT_SEEDS,
            "qiskit_repeats": FROZEN_QISKIT_REPEATS,
            "hybrid_layouts": FROZEN_HYBRID_LAYOUTS,
            "hybrid_neighbors": FROZEN_HYBRID_NEIGHBORS,
            "proxy_decay": FROZEN_PROXY_DECAY,
        },
        "totals": {
            "lightsabre_swap": ls_total,
            "oaabr_swap": oaabr_total,
            "safe_dual_swap": dual_total,
            "selected_swap": selected_total,
            "gain_vs_lightsabre": ls_total - selected_total,
            "gain_vs_safe_dual": dual_total - selected_total,
        },
        "selected_backends": dict(
            Counter(row["selected_backend"] for row in case_rows)
        ),
        "semantic_passed": semantic_passed,
        "topology_passed": topology_passed,
        "metric_passed": metric_passed,
        "safe_quality_floor": safe,
        "quality_reference": (
            None
            if args.quality_reference is None
            else str(args.quality_reference.resolve())
        ),
        "quality_mismatches": len(mismatches),
        "parallel_wall_seconds": parallel_wall_seconds,
        "summed_shard_seconds": compute_seconds,
        "effective_parallelism": (
            compute_seconds / parallel_wall_seconds
            if parallel_wall_seconds
            else 0.0
        ),
        "resumed_shards": sum(result.resumed for result in results),
        "attempt_count": len(attempt_rows),
        "shards": shard_rows,
    }
    summary_path = root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    hash_paths = [cases_path, attempts_path, shards_path, summary_path]
    hash_paths.extend(sorted(log_root.glob("*.txt")))
    hash_paths.extend(sorted(child_root.glob("shard_*/SHA256SUMS.txt")))
    hash_output = root / "SHA256SUMS.txt"
    hash_output.write_text(
        "".join(
            f"{sha256(path)}  {path}\n"
            for path in sorted(hash_paths, key=lambda item: str(item).lower())
        ),
        encoding="utf-8",
    )

    print("\n===== Stage 58 summary =====")
    print("cases:", len(case_rows))
    print("persistent processes:", len(tasks))
    print("avoided process restarts:", len(case_rows) - len(tasks))
    print("LightSABRE total SWAP:", ls_total)
    print("OAABR total SWAP:", oaabr_total)
    print("safe dual total SWAP:", dual_total)
    print("selected total SWAP:", selected_total)
    print("selected backends:", summary["selected_backends"])
    print("audits:", semantic_passed, topology_passed, metric_passed)
    print("safe quality floor:", safe)
    print("quality mismatches:", len(mismatches))
    print("parallel wall seconds:", round(parallel_wall_seconds, 3))
    print("summed shard seconds:", round(compute_seconds, 3))
    print("effective parallelism:", round(summary["effective_parallelism"], 3))
    print("cases output:", cases_path)
    print("summary:", summary_path)
    print("hashes:", hash_output)

    if mismatches:
        raise RuntimeError(
            f"persistent execution has {len(mismatches)} frozen quality mismatches"
        )


if __name__ == "__main__":
    main()
