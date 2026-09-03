from __future__ import annotations

import argparse
import json
import re
from importlib import metadata
from pathlib import Path
from typing import Any

from .config import FrozenConfig, load_config
from .runtime import (
    engine_script,
    find_repo_root,
    python_command,
    run_command,
)


EXPECTED_VERSIONS = {
    "qiskit": "2.4.2",
    "qiskit-qasm3-import": "0.6.0",
}
FROZEN_PORTFOLIO = (20, 1, 0, 4, 0.95)
BANNED_PUBLIC_NAMES = re.compile(
    r"(?i)" + "oa" + r"abr|hy" + r"brid|stage[ _-]*\d+"
)
IGNORED_SCAN_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "results",
}


def _root(value: str | None) -> Path:
    return find_repo_root(Path(value) if value else None)


def _resolve(value: str, root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _config(path: str, root: Path) -> FrozenConfig:
    config = load_config(_resolve(path, root), root)
    if config.schema_version != 1:
        raise ValueError(f"unsupported configuration schema: {config.schema_version}")
    config.validate_input()
    return config


def _append(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _check_frozen(config: FrozenConfig) -> None:
    values = (
        config.portfolio.qiskit_seeds,
        config.portfolio.qiskit_repeats,
        config.portfolio.cross_layout_layouts,
        config.portfolio.cross_layout_neighbors,
        config.portfolio.proxy_decay,
    )
    if values != FROZEN_PORTFOLIO:
        raise ValueError(
            f"persistent execution requires {FROZEN_PORTFOLIO}; got {values}"
        )


def build_route_command(
    *,
    root: Path,
    config: FrozenConfig,
    executor: str,
    output_dir: Path,
    case_indices: str | None,
    limit: int | None,
    workers: int | None,
    schedule: str | None,
    timing_reference: Path | None,
    quality_reference: Path | None,
    quality_policy: str,
    resume: bool,
) -> list[str]:
    if executor == "sequential":
        if resume:
            raise ValueError("sequential execution does not support --resume")
        command = python_command(
            engine_script("route"),
            "--input",
            str(config.input_path),
            "--qiskit-seeds",
            str(config.portfolio.qiskit_seeds),
            "--qiskit-repeats",
            str(config.portfolio.qiskit_repeats),
            "--cross-layout-layouts",
            str(config.portfolio.cross_layout_layouts),
            "--cross-layout-neighbors",
            str(config.portfolio.cross_layout_neighbors),
            "--proxy-decay",
            str(config.portfolio.proxy_decay),
            "--selection-mode",
            config.portfolio.selection_mode,
            "--output-dir",
            str(output_dir),
        )
        _append(command, "--case-indices", case_indices)
        _append(command, "--limit", limit)
        return command

    if executor != "persistent":
        raise ValueError(f"unknown executor: {executor}")
    _check_frozen(config)
    command = python_command(
        engine_script("parallel"),
        "--input",
        str(config.input_path),
        "--workers",
        str(workers or config.execution.workers),
        "--schedule",
        schedule or config.execution.schedule,
        "--portfolio-script",
        str(engine_script("route")),
        "--selection-mode",
        config.portfolio.selection_mode,
        "--quality-policy",
        quality_policy,
        "--output-dir",
        str(output_dir),
    )
    _append(command, "--case-indices", case_indices)
    _append(command, "--limit", limit)
    _append(command, "--timing-reference", timing_reference)
    _append(command, "--quality-reference", quality_reference)
    if resume:
        command.append("--resume")
    return command


def command_run(args: argparse.Namespace) -> int:
    root = _root(args.repo_root)
    config = _config(args.config, root)
    output = _resolve(args.output_dir, root)
    command = build_route_command(
        root=root,
        config=config,
        executor=args.executor,
        output_dir=output,
        case_indices=args.case_indices,
        limit=args.limit,
        workers=args.workers,
        schedule=args.schedule,
        timing_reference=(
            _resolve(args.timing_reference, root) if args.timing_reference else None
        ),
        quality_reference=(
            _resolve(args.quality_reference, root)
            if args.quality_reference
            else None
        ),
        quality_policy=args.quality_policy,
        resume=args.resume,
    )
    return run_command(command, repo_root=root, dry_run=args.dry_run)


def command_audit(args: argparse.Namespace) -> int:
    root = _root(args.repo_root)
    command = python_command(
        engine_script("audit"),
        "--v5-cases",
        str(_resolve(args.v5_cases, root)),
        "--v6-cases",
        str(_resolve(args.v6_cases, root)),
        "--output-dir",
        str(_resolve(args.output_dir, root)),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--bootstrap-seed",
        str(args.bootstrap_seed),
    )
    return run_command(command, repo_root=root, dry_run=args.dry_run)


def _scan_public_names(root: Path) -> list[str]:
    leaks: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        generated = any(
            part in IGNORED_SCAN_DIRECTORIES
            or part.startswith(".venv")
            or part.endswith(".egg-info")
            for part in relative.parts[:-1]
        )
        if (
            not path.is_file()
            or generated
            or path.suffix.lower() in {".pyc", ".qpy", ".png", ".pdf", ".zip"}
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if BANNED_PUBLIC_NAMES.search(line):
                leaks.append(
                    f"{path.relative_to(root).as_posix()}:{line_number}"
                )
    return leaks


def command_doctor(args: argparse.Namespace) -> int:
    root = _root(args.repo_root)
    checks: list[dict[str, str]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(
            {"name": name, "status": "pass" if passed else "fail", "detail": detail}
        )

    for name in ("frozen_v5.toml", "frozen_v6.toml"):
        try:
            config = load_config(root / "configs" / name, root)
            config.validate_input()
        except Exception as error:
            add(name, False, str(error))
        else:
            add(name, True, f"{config.dataset} input hash verified")

    for distribution, expected in EXPECTED_VERSIONS.items():
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            add(distribution, args.allow_missing_qiskit, "not installed")
        else:
            add(
                distribution,
                installed == expected,
                f"installed={installed}, expected={expected}",
            )

    engine = Path(__file__).resolve().parent / "engine"
    modules = sorted(engine.glob("*.py"))
    add("engine modules", len(modules) == 30, f"{len(modules)} Python files")
    leaks = _scan_public_names(root)
    add(
        "public naming",
        not leaks,
        "zero legacy-name leaks" if not leaks else ", ".join(leaks[:10]),
    )
    license_pending = (root / "LICENSE-REVIEW-REQUIRED.txt").is_file()
    checks.append(
        {
            "name": "release license",
            "status": "warn" if license_pending else "pass",
            "detail": (
                "copyright-holder approval is still required"
                if license_pending
                else "public license present"
            ),
        }
    )

    if args.json:
        print(json.dumps({"repository": str(root), "checks": checks}, indent=2))
    else:
        print(f"repository: {root}")
        for check in checks:
            print(
                f"[{check['status'].upper():4}] "
                f"{check['name']}: {check['detail']}"
            )
    return int(any(check["status"] == "fail" for check in checks))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smpr-router",
        description="Safe Multi-Layout Portfolio Router",
    )
    parser.add_argument("--repo-root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="verify source and environment")
    doctor.add_argument("--allow-missing-qiskit", action="store_true")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(handler=command_doctor)

    run = subparsers.add_parser("run", help="run the SMPR portfolio")
    run.add_argument("--config", required=True)
    run.add_argument(
        "--executor", choices=("sequential", "persistent"), default="persistent"
    )
    run.add_argument("--output-dir", required=True)
    run.add_argument("--case-indices")
    run.add_argument("--limit", type=int)
    run.add_argument("--workers", type=int)
    run.add_argument(
        "--schedule", choices=("lpt", "round-robin", "contiguous")
    )
    run.add_argument("--timing-reference")
    run.add_argument("--quality-reference")
    run.add_argument(
        "--quality-policy", choices=("exact", "primary"), default="exact"
    )
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(handler=command_run)

    audit = subparsers.add_parser("audit", help="run the V5/V6 reliability audit")
    audit.add_argument("--v5-cases", required=True)
    audit.add_argument("--v6-cases", required=True)
    audit.add_argument("--output-dir", required=True)
    audit.add_argument("--bootstrap-samples", type=int, default=10_000)
    audit.add_argument("--bootstrap-seed", type=int, default=20_260_723)
    audit.add_argument("--dry-run", action="store_true")
    audit.set_defaults(handler=command_audit)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        returncode = int(args.handler(args))
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
