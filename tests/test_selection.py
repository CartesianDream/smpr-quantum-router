from __future__ import annotations

import unittest
from dataclasses import dataclass

from smpr_router.selection import (
    deterministic_candidate_key,
    legacy_candidate_key,
    select_safe_candidate,
    unique_best_quality_layouts,
)


@dataclass
class Candidate:
    backend: str
    label: str
    swap: int
    depth: int
    runtime_ms: float
    initial_mapping: tuple[int, ...]
    seed: int | None


class SelectionTests(unittest.TestCase):
    def test_deterministic_key_ignores_runtime(self) -> None:
        slow = Candidate("lightsabre", "seed0", 5, 12, 100.0, (0, 1), 0)
        fast = Candidate("lightsabre", "seed1", 5, 12, 1.0, (1, 0), 1)
        self.assertIs(min((slow, fast), key=deterministic_candidate_key), slow)
        self.assertIs(min((slow, fast), key=legacy_candidate_key), fast)

    def test_quality_tie_prefers_lightsabre(self) -> None:
        baseline = Candidate("lightsabre", "ls", 4, 10, 99.0, (0, 1), 5)
        transfer = Candidate(
            "cross_layout", "transfer", 4, 10, 1.0, (1, 0), None
        )
        self.assertIs(
            min((transfer, baseline), key=deterministic_candidate_key),
            baseline,
        )

    def test_layouts_are_seed_ordered_and_deduplicated(self) -> None:
        candidates = [
            Candidate("lightsabre", "seed3", 7, 20, 1.0, (1, 0), 3),
            Candidate("lightsabre", "seed0", 7, 20, 9.0, (0, 1), 0),
            Candidate("lightsabre", "seed2", 7, 20, 0.5, (0, 1), 2),
        ]
        selected = unique_best_quality_layouts(candidates)
        self.assertEqual([item.seed for item in selected], [0, 3])

    def test_safe_selection_preserves_floor(self) -> None:
        baseline = Candidate("lightsabre", "ls", 8, 20, 4.0, (0, 1), 0)
        better = Candidate(
            "cross_layout", "transfer", 7, 22, 8.0, (1, 0), None
        )
        result = select_safe_candidate((baseline, better))
        self.assertIs(result.winner, better)
        self.assertEqual(result.swap_gain, 1)


if __name__ == "__main__":
    unittest.main()

