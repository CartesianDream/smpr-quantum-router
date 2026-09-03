from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, Protocol


BACKEND_PRIORITY = {
    "lightsabre": 0,
    "cross_layout": 1,
    "adaptive_rollout": 2,
    "independent_rollout": 2,
}
SelectionMode = Literal["legacy", "deterministic"]


class CandidateLike(Protocol):
    backend: str
    label: str
    swap: int
    depth: int
    runtime_ms: float
    initial_mapping: tuple[int, ...]
    seed: int | None


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Auditable result of safe portfolio selection."""

    winner: CandidateLike
    baseline: CandidateLike
    mode: SelectionMode

    @property
    def swap_gain(self) -> int:
        return self.baseline.swap - self.winner.swap

    @property
    def quality_gain(self) -> bool:
        return (self.winner.swap, self.winner.depth) < (
            self.baseline.swap,
            self.baseline.depth,
        )


def legacy_candidate_key(candidate: CandidateLike) -> tuple:
    """Frozen historical development/53 key, retained only for exact historical reproduction."""

    return (
        candidate.swap,
        candidate.depth,
        BACKEND_PRIORITY.get(candidate.backend, 9),
        candidate.runtime_ms,
        candidate.label,
    )


def deterministic_candidate_key(candidate: CandidateLike) -> tuple:
    """Quality-preserving key whose tie-breakers do not depend on wall time."""

    seed = candidate.seed if candidate.seed is not None else 2**31 - 1
    return (
        candidate.swap,
        candidate.depth,
        BACKEND_PRIORITY.get(candidate.backend, 9),
        seed,
        tuple(candidate.initial_mapping),
        candidate.label,
    )


def deterministic_layout_key(candidate: CandidateLike) -> tuple:
    """Canonical order for tied LightSABRE layouts."""

    seed = candidate.seed if candidate.seed is not None else 2**31 - 1
    return seed, tuple(candidate.initial_mapping), candidate.label


def unique_best_quality_layouts(
    candidates: Iterable[CandidateLike],
    maximum: int = 0,
) -> list[CandidateLike]:
    """Return unique tied-best layouts in deterministic order.

    ``maximum=0`` retains all tied-best layouts, matching the frozen V5/V6
    configuration.
    """

    items = list(candidates)
    if not items:
        raise ValueError("candidate collection is empty")
    if maximum < 0:
        raise ValueError("maximum must be non-negative")
    best_quality = min((item.swap, item.depth) for item in items)
    ranked = sorted(
        (
            item
            for item in items
            if (item.swap, item.depth) == best_quality
        ),
        key=deterministic_layout_key,
    )
    output: list[CandidateLike] = []
    seen: set[tuple[int, ...]] = set()
    for item in ranked:
        mapping = tuple(item.initial_mapping)
        if mapping in seen:
            continue
        seen.add(mapping)
        output.append(item)
        if maximum and len(output) >= maximum:
            break
    return output


def select_safe_candidate(
    candidates: Iterable[CandidateLike],
    *,
    mode: SelectionMode = "deterministic",
) -> SelectionResult:
    """Select a candidate while enforcing the LightSABRE quality floor.

    The deterministic mode is the historical development public specification.  Legacy mode
    exists only to reproduce the frozen V5/V6 evidence.  Both modes compare
    ``(swap, depth, backend_priority)`` before any implementation tie-breaker,
    so the quality-floor assertion is common to both.
    """

    items = list(candidates)
    if not items:
        raise ValueError("candidate collection is empty")
    baselines = [item for item in items if item.backend == "lightsabre"]
    if not baselines:
        raise ValueError("safe selection requires a LightSABRE candidate")

    key: Callable[[CandidateLike], tuple]
    if mode == "deterministic":
        key = deterministic_candidate_key
    elif mode == "legacy":
        key = legacy_candidate_key
    else:
        raise ValueError(f"unknown selection mode: {mode}")

    baseline = min(baselines, key=key)
    winner = min(items, key=key)
    if (winner.swap, winner.depth) > (baseline.swap, baseline.depth):
        raise AssertionError("selected candidate violates the quality floor")
    return SelectionResult(winner=winner, baseline=baseline, mode=mode)
