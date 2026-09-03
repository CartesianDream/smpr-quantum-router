from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def forbidden_name(path: Path) -> bool:
    name = path.name.lower()
    if name in {".env", "credentials.json", "token.txt"}:
        return True
    if path.suffix.lower() in {".pem", ".p12", ".pfx"}:
        return True
    return "api-key" in name or "api_key" in name


def main() -> None:
    failures: list[str] = []
    checks = 0

    def require(condition: bool, label: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(label)

    expected_hashes = {
        "paper/SMPR_Chinese_Paper.pdf":
            "56d5ad60dba283cce2f5e6491eb5401112abf46b84556c312303d2d896f2e652",
        "paper/SMPR_Experiment_Report.pdf":
            "60a54d32b5c99bb89cb7b7f749dc71e389443fb32b07374f7ab57976f4572cd4",
        "paper/SMPR_LaTeX_Source.zip":
            "c62ab8b36ebab4040e41b04cd51cd1001404700ef612a6cb6c0f6821c1556d81",
        "artifacts/SMPR_public_core_1.0.0rc3_V7_validated.zip":
            "60c185186b92947144972eeb793fbebf6b04312718b0c5ac3f5a6ddde7ffe91f",
        "artifacts/SMPR_complete_research_history.zip":
            "877a8fb1c903c6d8de1b9b953ef8db8640cf6fadd5f5e9ab03b8c97706213ae9",
        "artifacts/SMPR_confirmed_evidence.zip":
            "6c7f38aecea6318b3a8407a73d8f5ae67223ea8853e8f2bc3839261c9af3a85d",
        "artifacts/SMPR_public_external_evidence.zip":
            "a8e21b91a1ae0deabeca306b4cbc841ea31a8d104c578948dfe25d9a8b9a9a75",
    }

    for relative, expected in expected_hashes.items():
        path = ROOT / relative
        require(path.is_file(), f"missing: {relative}")
        if path.is_file():
            require(sha256(path) == expected, f"sha256: {relative}")

    for relative in expected_hashes:
        if not relative.endswith(".zip"):
            continue
        path = ROOT / relative
        if not path.is_file():
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                require(archive.testzip() is None, f"zip integrity: {relative}")
        except zipfile.BadZipFile:
            require(False, f"bad zip: {relative}")

    expected_reference = {
        "v5_cases.csv": (60, 1691, 1641),
        "v6_cases.csv": (120, 3176, 3051),
    }
    for filename, (count, baseline, selected) in expected_reference.items():
        rows = csv_rows(ROOT / "evidence" / "reference" / filename)
        require(len(rows) == count, f"reference count: {filename}")
        require(
            sum(int(row["lightsabre_swap"]) for row in rows) == baseline,
            f"reference baseline: {filename}",
        )
        require(
            sum(int(row["selected_swap"]) for row in rows) == selected,
            f"reference selected: {filename}",
        )

    v7_path = ROOT / "results" / "complete_evidence" / "v7" / "summary.json"
    v7 = json.loads(v7_path.read_text(encoding="utf-8"))
    require(v7["passed"] is True, "V7 passed")
    require(v7["row_count"] == 120, "V7 row count")
    require(v7["routing_trace_passed"] == 120, "V7 routing audit")
    require(v7["topology_passed"] == 120, "V7 topology audit")
    require(v7["safe_floor_passed"] == 120, "V7 safe floor")
    expected_frontier = {
        "0.25": (990, 1113),
        "0.5": (990, 1086),
        "1.0": (990, 1068),
        "2.0": (990, 1044),
    }
    for budget, totals in expected_frontier.items():
        row = v7["frontier"][budget]
        require(
            (row["smpr_total_swap"], row["lightsabre_total_swap"]) == totals,
            f"V7 frontier: {budget}",
        )

    v16_root = ROOT / "results" / "retrospective_v16" / "实验结果"
    b1_rows = csv_rows(v16_root / "b1_15ch.csv")
    require(len(b1_rows) == 60, "V16 B1 count")
    require(sum(int(row["v5"]) for row in b1_rows) == 1641, "V16 B1 source")
    require(sum(int(row["multi"]) for row in b1_rows) == 1564, "V16 B1 channels")
    require(sum(int(row["merged"]) for row in b1_rows) == 1559, "V16 B1 merged")
    merged = {row["case"]: int(row["merged"]) for row in b1_rows}
    for row in csv_rows(v16_root / "b1_deep_search.csv"):
        case = row["case"]
        merged[case] = min(merged[case], int(row["best_new"]))
    require(sum(merged.values()) == 1556, "V16 B1 final")

    b2_gain = sum(
        int(row["gain"])
        for row in csv_rows(v16_root / "b2_deep_search.csv")
    )
    require(b2_gain == 8, "V16 B2 layout gain")
    require(2907 - b2_gain == 2899, "V16 B2 registered result")

    public_rows = csv_rows(v16_root / "ext_bridge_results.csv")
    require(len(public_rows) == 17, "V16 public count")
    require(sum(int(row["ls_best_swap"]) for row in public_rows) == 851,
            "V16 public baseline")
    require(sum(int(row["v14_on_lsbest"]) for row in public_rows) == 816,
            "V16 public same layout")
    require(sum(int(row["v14_min_over_layouts"]) for row in public_rows) == 800,
            "V16 public layout pool")

    required_analysis = [
        ROOT / "results" / "integrated_analysis" / "SMPR_整合证据报告_20260823.md",
        ROOT / "results" / "integrated_analysis" / "SMPR_整合证据工作簿_20260823.xlsx",
        ROOT / "results" / "integrated_analysis" / "PROJECT_CLOSEOUT_20260901.md",
    ]
    for path in required_analysis:
        require(path.is_file() and path.stat().st_size > 0,
                f"integrated analysis: {path.name}")

    forbidden = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and forbidden_name(path)
    ]
    require(not forbidden, "sensitive filenames: " + ", ".join(forbidden))

    print(f"checks: {checks}")
    print(f"failures: {len(failures)}")
    for failure in failures:
        print(f"[FAIL] {failure}")
    print(f"PASSED: {not failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
