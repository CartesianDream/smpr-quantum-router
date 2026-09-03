#!/usr/bin/env python3
"""Build the deterministic, GitHub-ready SMPR repository archive.

The script refreshes the full-file SHA-256 manifest and creates a ZIP with a
single top-level repository directory. Local runs and generated caches are
excluded so a test run does not change the release artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import stat
import zipfile


FIXED_ZIP_TIME = (2026, 9, 1, 0, 0, 0)
MANIFEST = Path("results/FULL_REPOSITORY_SHA256SUMS.txt")
EXCLUDED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".DS_Store"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def should_include(relative: Path) -> bool:
    if relative == MANIFEST:
        return False
    if any(part in EXCLUDED_DIR_NAMES for part in relative.parts):
        return False
    if relative.parts[:2] == ("results", "local_runs"):
        return False
    if relative.name.startswith(".openai-download-"):
        return False
    if any(relative.name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        return False
    return True


def repository_files(root: Path) -> list[Path]:
    return sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and should_include(path.relative_to(root))
    )


def write_manifest(root: Path, files: list[Path]) -> None:
    target = root / MANIFEST
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{sha256_file(root / relative)}  {relative.as_posix()}" for relative in files]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def add_file(archive: zipfile.ZipFile, root: Path, relative: Path) -> None:
    source = root / relative
    archive_name = f"{root.name}/{relative.as_posix()}"
    info = zipfile.ZipInfo(archive_name, FIXED_ZIP_TIME)
    info.create_system = 3
    executable = bool(source.stat().st_mode & stat.S_IXUSR)
    permissions = 0o755 if executable else 0o644
    info.external_attr = (permissions & 0xFFFF) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    with source.open("rb") as handle:
        archive.writestr(info, handle.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build(root: Path, output: Path) -> tuple[int, int, str]:
    root = root.resolve()
    files_without_manifest = repository_files(root)
    write_manifest(root, files_without_manifest)
    files = sorted(files_without_manifest + [MANIFEST])

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for relative in files:
            add_file(archive, root, relative)
    os.replace(temporary, output)
    return len(files), output.stat().st_size, sha256_file(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    root = arguments.repo_root.resolve()
    output = arguments.output or root.parent / f"{root.name}.zip"
    count, size, digest = build(root, output)
    print(f"archive: {output.resolve()}")
    print(f"files: {count}")
    print(f"bytes: {size}")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
