"""Overnight B: 生成 v7_extended_cases.json（QASMBench 扩展公共电路）。

从 staging_qasm/raw/ 下载的 QASMBench 电路（commit 357b942，经国内镜像
ghfast.top 下载）构建与 data/v7/external_cases.json 同 schema 的扩展 case 文件，
映射到 heavy-hex（n≤19→d3，n≥20→d5）。
复用 prepare_v7.normalize / heavy_hex_case / sha256；比特数从电路实际声明推导。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
_ENGINE_DIR = _ROOT / "src" / "smpr_router" / "engine"
for _p in (_ENGINE_DIR, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from prepare_v7 import normalize, heavy_hex_case, sha256  # noqa: E402
import qiskit  # noqa: E402
from qiskit import qasm2, qasm3, qpy  # noqa: E402

RAW = _ROOT / "staging_qasm" / "raw"
COMMIT = "357b942396d5c2b7cbc1c229c585a6ef5ccaebac"

# 选定新电路（相对路径；比特数从 qasm 声明读取）。medium/large 优先。
SELECT = [
    "medium/seca_n11/seca_n11.qasm",
    "medium/cc_n12/cc_n12.qasm",
    "medium/qf21_n15/qf21_n15.qasm",
    "medium/qec9xz_n17/qec9xz_n17.qasm",
    "medium/qft_n18/qft_n18.qasm",
    "medium/square_root_n18/square_root_n18.qasm",
    "medium/swap_test_n25/swap_test_n25.qasm",
    "medium/knn_n25/knn_n25.qasm",
    "medium/ising_n26/ising_n26.qasm",
    "medium/wstate_n27/wstate_n27.qasm",
    "large/adder_n28/adder_n28.qasm",
    "large/bv_n30/bv_n30.qasm",
    "large/cat_n35/cat_n35.qasm",
    "large/cc_n32/cc_n32.qasm",
    "large/ghz_n40/ghz_n40.qasm",
    "large/ising_n34/ising_n34.qasm",
    "large/knn_n31/knn_n31.qasm",
    "large/qft_n29/qft_n29.qasm",
    "large/qugan_n39/qugan_n39.qasm",
    "large/swap_test_n41/swap_test_n41.qasm",
    "large/wstate_n36/wstate_n36.qasm",
    # small（可能有中途测量/不可映射的会被跳过）
    "small/bb84_n8/bb84_n8.qasm",
    "small/adder_n10/adder_n10.qasm",
    "small/ising_n10/ising_n10.qasm",
    "small/hhl_n7/hhl_n7.qasm",
]


def load_qasm(src: Path):
    """加载 QASM2；失败（未定义门，qiskit 2.5 include 解析问题）时内联 qelib1.inc。"""
    try:
        return qasm2.load(str(src))
    except Exception:
        std_dir = Path(qiskit.__file__).parent / "qasm" / "libs"
        inc = (std_dir / "qelib1.inc").read_text(encoding="utf-8")
        lines = [
            line
            for line in src.read_text(encoding="utf-8").splitlines()
            if not line.strip().startswith(("OPENQASM", "include"))
        ]
        merged = "OPENQASM 2.0;\n" + inc + "\n" + "\n".join(lines)
        tmp = src.parent / f"_inline_{src.stem}.qasm"
        tmp.write_text(merged, encoding="utf-8")
        try:
            return qasm2.load(str(tmp))
        finally:
            tmp.unlink(missing_ok=True)


def main() -> None:
    cases: list[dict] = []
    records: list[dict] = []
    normalized_root = _ROOT / "staging_qasm" / "normalized"
    normalized_root.mkdir(parents=True, exist_ok=True)

    skipped: list[str] = []
    for rel in SELECT:
        src = RAW / rel
        if not src.exists():
            print(f"  MISSING {rel}", flush=True)
            continue
        cid = rel.split("/")[1].split(".")[0]
        try:
            logical = load_qasm(src)
            norm = normalize(logical)
            n = int(norm.num_qubits)
            item = {"id": cid, "qubits": n, "path": rel}
            case = heavy_hex_case(item, src, norm)
        except Exception as exc:  # noqa: BLE001 —— V7 协议拒绝的电路（中途测量等）跳过
            skipped.append(f"{cid} ({type(exc).__name__}: {exc})")
            print(f"  SKIP {cid}: {type(exc).__name__}: {exc}", flush=True)
            continue
        cases.append(case)

        qpy_path = normalized_root / f"{cid}.qpy"
        qasm_path = normalized_root / f"{cid}.qasm"
        with qpy_path.open("wb") as fh:
            qpy.dump(norm, fh)
        qasm_path.write_text(qasm3.dumps(norm), encoding="utf-8", newline="\n")
        records.append({
            "id": cid,
            "source_sha256": sha256(src),
            "normalized_qpy_sha256": sha256(qpy_path),
            "normalized_qasm_sha256": sha256(qasm_path),
        })
        cx = sum(1 for g in case["gate_specs"] if g[0] == "cx")
        print(
            f"  {cid}: n={n} CX={cx} phys={case['physical_qubits']} "
            f"topo={case['topology']}",
            flush=True,
        )

    design = {
        "schema_version": 1,
        "protocol": "SMPR V15 公共电路扩展（QASMBench 追加）",
        "status": "extended public circuits beyond V7's 10",
        "source": {
            "name": "QASMBench",
            "repository": "https://github.com/pnnl/QASMBench",
            "commit": COMMIT,
            "license": "BSD",
            "citation_doi": "10.1145/3550488",
        },
        "note": (
            "经国内镜像 ghfast.top 下载（机器直连 GitHub 不通）；SHA 为下载内容哈希"
            "（镜像字节级一致）。"
        ),
    }
    out = {
        "schema_version": 7,
        "case_count": len(cases),
        "design": design,
        "source_records": records,
        "cases": cases,
    }
    out_path = _ROOT / "data" / "v7_extended_cases.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written {len(cases)} cases -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
