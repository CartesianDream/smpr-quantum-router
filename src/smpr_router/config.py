from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility.
    import tomli as tomllib


SCHEDULES = {"lpt", "round-robin", "contiguous"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    qiskit_seeds: int
    qiskit_repeats: int
    cross_layout_layouts: int
    cross_layout_neighbors: int
    proxy_decay: float
    selection_mode: str


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    workers: int
    schedule: str


@dataclass(frozen=True, slots=True)
class FrozenConfig:
    source: Path
    schema_version: int
    name: str
    dataset: str
    input_path: Path
    input_sha256: str
    portfolio: PortfolioConfig
    execution: ExecutionConfig

    def validate_input(self) -> None:
        if not self.input_path.is_file():
            raise FileNotFoundError(self.input_path)
        actual = sha256(self.input_path)
        if actual != self.input_sha256:
            raise RuntimeError(
                f"dataset hash mismatch for {self.input_path}: "
                f"expected {self.input_sha256}, got {actual}"
            )


def _required(
    mapping: dict[str, Any],
    key: str,
    kind: type | tuple[type, ...],
) -> Any:
    if key not in mapping:
        raise ValueError(f"missing configuration key: {key}")
    value = mapping[key]
    if not isinstance(value, kind):
        names = (
            "/".join(item.__name__ for item in kind)
            if isinstance(kind, tuple)
            else kind.__name__
        )
        raise TypeError(f"configuration key {key!r} must be {names}")
    return value


def load_config(path: Path, repo_root: Path) -> FrozenConfig:
    source = path.resolve()
    payload = tomllib.loads(source.read_text(encoding="utf-8"))
    portfolio_raw = _required(payload, "portfolio", dict)
    execution_raw = _required(payload, "execution", dict)
    input_value = _required(payload, "input", str)
    input_path = Path(input_value)
    if not input_path.is_absolute():
        input_path = repo_root / input_path

    portfolio = PortfolioConfig(
        qiskit_seeds=int(_required(portfolio_raw, "qiskit_seeds", int)),
        qiskit_repeats=int(_required(portfolio_raw, "qiskit_repeats", int)),
        cross_layout_layouts=int(_required(portfolio_raw, "cross_layout_layouts", int)),
        cross_layout_neighbors=int(_required(portfolio_raw, "cross_layout_neighbors", int)),
        proxy_decay=float(
            _required(portfolio_raw, "proxy_decay", (int, float))
        ),
        selection_mode=str(
            _required(portfolio_raw, "selection_mode", str)
        ),
    )
    execution = ExecutionConfig(
        workers=int(_required(execution_raw, "workers", int)),
        schedule=str(_required(execution_raw, "schedule", str)),
    )
    if portfolio.qiskit_seeds <= 0 or portfolio.qiskit_repeats <= 0:
        raise ValueError("qiskit seeds and repeats must be positive")
    if portfolio.cross_layout_layouts < 0 or portfolio.cross_layout_neighbors < 0:
        raise ValueError("cross_layout layout counts must be non-negative")
    if not 0.0 < portfolio.proxy_decay <= 1.0:
        raise ValueError("proxy_decay must lie in (0, 1]")
    if portfolio.selection_mode not in {"legacy", "deterministic"}:
        raise ValueError(
            f"unknown selection mode: {portfolio.selection_mode}"
        )
    if execution.workers <= 0:
        raise ValueError("workers must be positive")
    if execution.schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule: {execution.schedule}")

    return FrozenConfig(
        source=source,
        schema_version=int(_required(payload, "schema_version", int)),
        name=str(_required(payload, "name", str)),
        dataset=str(_required(payload, "dataset", str)),
        input_path=input_path.resolve(),
        input_sha256=str(_required(payload, "input_sha256", str)),
        portfolio=portfolio,
        execution=execution,
    )
