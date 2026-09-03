from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


ENGINE_SCRIPTS = {
    "route": "smpr_portfolio.py",
    "parallel": "persistent_runner.py",
    "audit": "reliability_audit.py",
}


def find_repo_root(start: Path | None = None) -> Path:
    candidates: list[Path] = []
    if start is not None:
        candidates.extend([start.resolve(), *start.resolve().parents])
    package_path = Path(__file__).resolve()
    candidates.extend(package_path.parents)
    candidates.extend([Path.cwd().resolve(), *Path.cwd().resolve().parents])
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "data").is_dir()
            and (candidate / "src" / "smpr_router" / "engine").is_dir()
        ):
            return candidate
    raise FileNotFoundError("cannot locate the SMPR public-core repository root")


def engine_script(role: str) -> Path:
    try:
        filename = ENGINE_SCRIPTS[role]
    except KeyError as error:
        raise ValueError(f"unknown engine role: {role}") from error
    path = Path(__file__).resolve().parent / "engine" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def deterministic_environment() -> dict[str, str]:
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
    return environment


def python_command(script: Path, *arguments: str) -> list[str]:
    return [sys.executable, "-u", str(script), *arguments]


def format_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def run_command(
    command: list[str],
    *,
    repo_root: Path,
    dry_run: bool = False,
) -> int:
    print(format_command(command))
    if dry_run:
        return 0
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=deterministic_environment(),
        check=False,
    )
    return completed.returncode

