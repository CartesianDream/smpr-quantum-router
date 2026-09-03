from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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


@dataclass(frozen=True)
class CaseTask:
    index: int
    instance: str
    child_dir: Path
    log_path: Path


@dataclass(frozen=True)
class CaseTaskResult:
    index: int
    instance: str
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


def child_complete(task: CaseTask) -> bool:
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
    return (
        len(rows) == 1
        and rows[0].get("instance") == task.instance
        and is_true(rows[0].get("semantic_equivalent"))
        and is_true(rows[0].get("topology_valid"))
        and is_true(rows[0].get("metric_valid"))
        and int(summary.get("case_count", -1)) == 1
    )


def run_task(
    task: CaseTask,
    stage53_script: Path,
    input_path: Path,
    resume: bool,
) -> CaseTaskResult:
    if resume and child_complete(task):
        return CaseTaskResult(
            index=task.index,
            instance=task.instance,
            child_dir=task.child_dir,
            log_path=task.log_path,
            returncode=0,
            elapsed_seconds=0.0,
            resumed=True,
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
        str(task.index),
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
    return CaseTaskResult(
        index=task.index,
        instance=task.instance,
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
                    "parallel": "present",
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
                        "parallel": str(row.get(column, "")),
                    }
                )
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 57: exact whole-case parallel executor for frozen Stage 53"
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
        default=Path("results/stage57_parallel_frozen_hybrid"),
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
    child_root = root / "children"
    log_root = root / "worker_logs"
    child_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    tasks = [
        CaseTask(
            index=index,
            instance=str(source_cases[index]["name"]),
            child_dir=child_root / f"case_{index:03d}",
            log_path=log_root / f"case_{index:03d}.txt",
        )
        for index in indices
    ]

    print("===== Stage 57 exact parallel frozen hybrid =====")
    print("input:", args.input.resolve())
    print("cases:", len(tasks))
    print("workers:", args.workers)
    print("Stage 53:", args.stage53_script.resolve())
    print(
        "frozen parameters:",
        f"LS={FROZEN_QISKIT_SEEDS}x{FROZEN_QISKIT_REPEATS}, "
        f"layouts={FROZEN_HYBRID_LAYOUTS}, "
        f"neighbors={FROZEN_HYBRID_NEIGHBORS}, "
        f"decay={FROZEN_PROXY_DECAY}",
    )
    print("quality reference:", args.quality_reference)

    wall_start = time.perf_counter()
    results: list[CaseTaskResult] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_task,
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
                f"[{completed_count}/{len(tasks)}] case={result.index:03d} "
                f"{result.instance}: {status}, "
                f"{result.elapsed_seconds:.2f}s"
            )

    parallel_wall_seconds = time.perf_counter() - wall_start
    failed = sorted(
        (result for result in results if result.returncode != 0),
        key=lambda item: item.index,
    )
    if failed:
        print("\n===== failed child logs =====")
        for result in failed:
            try:
                lines = result.log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except OSError:
                lines = ["<log unavailable>"]
            print(f"--- case {result.index}: {result.log_path} ---")
            print("\n".join(lines[-30:]))
        raise RuntimeError(f"{len(failed)} Stage 53 child processes failed")

    case_rows: list[dict[str, str]] = []
    attempt_rows: list[dict[str, str]] = []
    for task in tasks:
        child_cases = read_rows(task.child_dir / "cases.csv")
        if len(child_cases) != 1 or child_cases[0]["instance"] != task.instance:
            raise RuntimeError(f"invalid child case output: {task.child_dir}")
        case_rows.extend(child_cases)
        attempt_rows.extend(read_rows(task.child_dir / "attempts.csv"))

    case_rows.sort(key=lambda row: int(row["case_index"]))
    attempt_rows.sort(
        key=lambda row: (
            int(row["case_index"]),
            str(row["backend"]),
            str(row["label"]),
            str(row["seed"]),
        )
    )
    cases_path = root / "cases.csv"
    attempts_path = root / "attempts.csv"
    write_rows(cases_path, case_rows)
    write_rows(attempts_path, attempt_rows)

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
        "stage": 57,
        "purpose": "exact whole-case parallel execution of frozen Stage 53",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "stage53_script": str(args.stage53_script.resolve()),
        "stage53_sha256": sha256(args.stage53_script),
        "case_count": len(case_rows),
        "workers": args.workers,
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
        "summed_child_seconds": compute_seconds,
        "effective_parallelism": (
            compute_seconds / parallel_wall_seconds
            if parallel_wall_seconds
            else 0.0
        ),
        "resumed_cases": sum(result.resumed for result in results),
        "attempt_count": len(attempt_rows),
    }
    summary_path = root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    hash_paths = [cases_path, attempts_path, summary_path]
    hash_paths.extend(sorted(log_root.glob("*.txt")))
    hash_paths.extend(
        sorted(child_root.glob("case_*/SHA256SUMS.txt"))
    )
    hash_output = root / "SHA256SUMS.txt"
    hash_output.write_text(
        "".join(
            f"{sha256(path)}  {path}\n"
            for path in sorted(hash_paths, key=lambda item: str(item).lower())
        ),
        encoding="utf-8",
    )

    print("\n===== Stage 57 summary =====")
    print("cases:", len(case_rows))
    print("LightSABRE total SWAP:", ls_total)
    print("OAABR total SWAP:", oaabr_total)
    print("safe dual total SWAP:", dual_total)
    print("selected total SWAP:", selected_total)
    print("selected backends:", summary["selected_backends"])
    print("audits:", semantic_passed, topology_passed, metric_passed)
    print("safe quality floor:", safe)
    print("quality mismatches:", len(mismatches))
    print("parallel wall seconds:", round(parallel_wall_seconds, 3))
    print("summed child seconds:", round(compute_seconds, 3))
    print("effective parallelism:", round(summary["effective_parallelism"], 3))
    print("cases output:", cases_path)
    print("summary:", summary_path)
    print("hashes:", hash_output)

    if mismatches:
        raise RuntimeError(
            f"parallel execution has {len(mismatches)} frozen quality mismatches"
        )


if __name__ == "__main__":
    main()
