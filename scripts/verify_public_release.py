from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BANNED = re.compile(
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    python_files = sorted((ROOT / "src").rglob("*.py"))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeError) as error:
            failures.append(f"syntax: {path.relative_to(ROOT)}: {error}")

    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
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
        for number, line in enumerate(text.splitlines(), start=1):
            if BANNED.search(line):
                failures.append(
                    f"name leak: {path.relative_to(ROOT)}:{number}"
                )

    for config_name, relative_input in (
        ("frozen_v5.toml", "data/benchmarks_v5/final_test.json"),
        ("frozen_v6.toml", "data/benchmarks_v6/blind_test.json"),
    ):
        config = (ROOT / "configs" / config_name).read_text(encoding="utf-8")
        match = re.search(r'(?m)^input_sha256 = "([0-9a-f]{64})"$', config)
        actual = sha256(ROOT / relative_input)
        if match is None or match.group(1) != actual:
            failures.append(f"input hash: {config_name}")

    manifest_path = ROOT / "SOURCE_MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["files"]:
            path = ROOT / row["path"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                failures.append(f"manifest: {row['path']}")

    required = [
        ROOT / "NOTICE.md",
        ROOT / "CITATION.cff",
        ROOT / "docs" / "ALGORITHM.md",
        ROOT / "docs" / "V7_PROTOCOL.md",
        ROOT / "v7" / "manifest.json",
    ]
    failures.extend(
        f"missing: {path.relative_to(ROOT)}"
        for path in required
        if not path.is_file()
    )
    print(f"Python files: {len(python_files)}")
    print(f"verification failures: {len(failures)}")
    for failure in failures:
        print(f"[FAIL] {failure}")
    print(f"PASSED: {not failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
