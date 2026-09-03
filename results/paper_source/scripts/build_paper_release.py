from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


RELEASE_ROOT = "SMPR_chinese_paper"
FIXED_ZIP_TIME = (2026, 7, 25, 0, 0, 0)
EXCLUDED_DIRECTORIES = {"__pycache__"}
EXCLUDED_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".pdf",
    ".pyc",
    ".toc",
    ".xdv",
    ".zip",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return (
        path.is_file()
        and not any(part in EXCLUDED_DIRECTORIES for part in relative.parts)
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
    )


def write_zip(root: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for path in root.rglob("*") if included(path, root))
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as handle:
        for path in paths:
            relative = Path(RELEASE_ROOT) / path.relative_to(root)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with path.open("rb") as source:
                handle.writestr(info, source.read(), compresslevel=9)
    print(f"archive: {archive}")
    print(f"archive sha256: {sha256(archive)}")
    print(f"source files: {len(paths)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the deterministic SMPR LaTeX source archive"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_zip(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
