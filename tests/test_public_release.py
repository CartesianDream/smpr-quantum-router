from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from smpr_router.cli import _scan_public_names
from smpr_router.runtime import find_repo_root


class PublicReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = find_repo_root(Path(__file__).resolve())

    def test_reference_totals(self) -> None:
        expected = {
            "v5_cases.csv": (1691, 1641, 60),
            "v6_cases.csv": (3176, 3051, 120),
        }
        for filename, (baseline, selected, count) in expected.items():
            with (
                self.root / "evidence" / "reference" / filename
            ).open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), count)
            self.assertEqual(sum(int(row["lightsabre_swap"]) for row in rows), baseline)
            self.assertEqual(sum(int(row["selected_swap"]) for row in rows), selected)

    def test_v7_manifest_is_prospective_and_over_ten_qubits(self) -> None:
        payload = json.loads(
            (self.root / "v7" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(payload["circuits"]), 10)
        self.assertTrue(all(item["qubits"] > 10 for item in payload["circuits"]))
        self.assertEqual(
            [item["qubits"] for item in payload["circuits"]],
            sorted(item["qubits"] for item in payload["circuits"]),
        )
        by_id = {item["id"]: item for item in payload["circuits"]}
        self.assertNotIn("cc_n12", by_id)
        self.assertNotIn("qec9xz_n17", by_id)
        self.assertEqual(
            by_id["dnn_n16"]["sha256"],
            "c194be8740c380fc9679ebcca2515ac52215aea1c58dc5cfc507679a88c64d79",
        )
        self.assertEqual(
            by_id["qram_n20"]["sha256"],
            "2b22f2a0a013cb91c5b59063b04e33998a1e24bbd5300a085b01edadd9ac763d",
        )
        self.assertIn("prospective", payload["status"])

    def test_v7_prepare_rejects_mid_circuit_measurement(self) -> None:
        script = self.root / "scripts" / "prepare_v7.py"
        spec = importlib.util.spec_from_file_location("prepare_v7_test", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        from qiskit import QuantumCircuit

        circuit = QuantumCircuit(2, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        circuit.cx(0, 1)
        with self.assertRaisesRegex(ValueError, "mid-circuit measurement"):
            module.strip_terminal_io(circuit)

    def test_v7_prepare_uses_staging_before_publication(self) -> None:
        source = (
            self.root / "scripts" / "prepare_v7.py"
        ).read_text(encoding="utf-8")
        self.assertIn("tempfile.TemporaryDirectory(", source)
        self.assertIn("publish_outputs(", source)
        self.assertIn("require_fresh_targets(", source)

    def test_v7_prepare_failure_publishes_nothing(self) -> None:
        script = self.root / "scripts" / "prepare_v7.py"
        spec = importlib.util.spec_from_file_location(
            "prepare_v7_transaction_test",
            script,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "release"
            upstream = base / "upstream"
            source = upstream / "medium" / "dynamic_n2" / "dynamic_n2.qasm"
            source.parent.mkdir(parents=True)
            source.write_text(
                "\n".join(
                    [
                        "OPENQASM 2.0;",
                        'include "qelib1.inc";',
                        "qreg q[2];",
                        "creg c[1];",
                        "h q[0];",
                        "measure q[0] -> c[0];",
                        "cx q[0],q[1];",
                        "",
                    ]
                ),
                encoding="utf-8",
                newline="\n",
            )
            for filename in ("LICENSE", "NOTICE"):
                (upstream / filename).write_text(
                    filename + "\n",
                    encoding="utf-8",
                    newline="\n",
                )

            manifest = {
                "schema_version": 1,
                "status": "prospective",
                "circuits": [
                    {
                        "id": "dynamic_n2",
                        "qubits": 2,
                        "path": "medium/dynamic_n2/dynamic_n2.qasm",
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ],
            }
            manifest_path = root / "v7" / "manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(manifest),
                encoding="utf-8",
                newline="\n",
            )

            with (
                mock.patch.object(module, "COMMIT_BASE", upstream.as_uri()),
                mock.patch.object(
                    sys,
                    "argv",
                    ["prepare_v7.py", "--repo-root", str(root)],
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "mid-circuit measurement",
                ):
                    module.main()

            for path in module.output_targets(root, include_sources=True):
                self.assertFalse(path.exists(), path)
            self.assertEqual(
                list((root / "v7").glob(".prepare-v7-*")),
                [],
            )

    def test_no_internal_name_leaks(self) -> None:
        pattern = re.compile(
            r"(?i)" + "oa" + r"abr|hy" + r"brid|stage[ _-]*\d+"
        )
        ignored_directories = {
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            "__pycache__",
            "build",
            "dist",
            "results",
        }
        leaks: list[str] = []
        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root)
            generated = any(
                part in ignored_directories
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
            if pattern.search(text):
                leaks.append(path.relative_to(self.root).as_posix())
        self.assertEqual(leaks, [])

    def test_portfolio_entrypoint_always_enables_cross_layout(self) -> None:
        source = (
            self.root
            / "src"
            / "smpr_router"
            / "engine"
            / "smpr_portfolio.py"
        ).read_text(encoding="utf-8")
        self.assertIn("default_cross_layout_neighbors: int = 4", source)
        self.assertIn("cross_layout_trials = run_cross_layout_candidates(", source)
        self.assertNotRegex(source, r"(?i)\bstage\b")

    def test_name_scan_ignores_generated_environments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = (
                root
                / ".venv-smpr-public"
                / "Lib"
                / "site-packages"
                / "third_party.py"
            )
            generated.parent.mkdir(parents=True)
            generated.write_text("oa" + "abr\n", encoding="utf-8")
            result = root / "results" / "worker.txt"
            result.parent.mkdir()
            result.write_text("hy" + "brid\n", encoding="utf-8")
            self.assertEqual(_scan_public_names(root), [])


if __name__ == "__main__":
    unittest.main()
