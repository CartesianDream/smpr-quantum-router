from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "v7" / "FROZEN_MANIFEST.json"
    if output.exists():
        raise FileExistsError(
            "V7 is already frozen; do not overwrite the preregistration"
        )
    required = [
        root / "v7" / "manifest.json",
        root / "data" / "v7" / "external_cases.json",
        root / "configs" / "v7_external.toml",
        root / "scripts" / "prepare_v7.py",
        root / "scripts" / "freeze_v7.py",
        root / "scripts" / "run_v7_core.py",
        root / "scripts" / "analyze_v7.py",
        root / "scripts" / "run_v7.ps1",
    ]
    required.extend(sorted((root / "v7" / "sources").rglob("*")))
    required.extend(sorted((root / "v7" / "normalized").rglob("*")))
    required.extend(sorted((root / "src" / "smpr_router").rglob("*.py")))
    files = sorted({path.resolve() for path in required if path.is_file()})
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing V7 freeze inputs:\n" + "\n".join(missing))
    payload = {
        "schema_version": 1,
        "purpose": "prospective freeze before V7 result inspection",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "file_count": len(files),
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    digest = sha256(output)
    (root / "v7" / "FROZEN_MANIFEST.sha256").write_text(
        f"{digest}  FROZEN_MANIFEST.json\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"frozen files: {len(files)}")
    print(f"manifest: {output}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()

