from __future__ import annotations

import unittest
from pathlib import Path

from smpr_router.cli import build_route_command
from smpr_router.config import load_config
from smpr_router.runtime import find_repo_root


class ConfigAndCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = find_repo_root(Path(__file__).resolve())

    def test_v5_and_v6_hashes_validate(self) -> None:
        for name in ("frozen_v5.toml", "frozen_v6.toml"):
            config = load_config(self.root / "configs" / name, self.root)
            config.validate_input()

    def test_persistent_command_is_self_contained(self) -> None:
        config = load_config(
            self.root / "configs" / "frozen_v6.toml", self.root
        )
        command = build_route_command(
            root=self.root,
            config=config,
            executor="persistent",
            output_dir=self.root / "results" / "test",
            case_indices="1,39",
            limit=None,
            workers=4,
            schedule="lpt",
            timing_reference=None,
            quality_reference=self.root / "evidence" / "reference" / "v6_cases.csv",
            quality_policy="primary",
            resume=False,
        )
        joined = " ".join(command)
        self.assertIn("persistent_runner.py", joined)
        self.assertIn("smpr_portfolio.py", joined)
        self.assertIn("--selection-mode legacy", joined)

    def test_sequential_command_exposes_portfolio(self) -> None:
        config = load_config(
            self.root / "configs" / "frozen_v5.toml", self.root
        )
        command = build_route_command(
            root=self.root,
            config=config,
            executor="sequential",
            output_dir=self.root / "results" / "test",
            case_indices=None,
            limit=1,
            workers=None,
            schedule=None,
            timing_reference=None,
            quality_reference=None,
            quality_policy="exact",
            resume=False,
        )
        joined = " ".join(command)
        self.assertIn("--qiskit-seeds 20", joined)
        self.assertIn("--cross-layout-layouts 0", joined)
        self.assertIn("--cross-layout-neighbors 4", joined)


if __name__ == "__main__":
    unittest.main()

