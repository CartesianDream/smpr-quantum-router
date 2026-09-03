# SMPR V16 成果汇总

> 2026-08-07。本目录收录 SMPR 路由器自 V5 基线以来（V13–V16）的全部改进、
> 方法、证据与边界，并附源代码与实验数据。内容以正式语体撰写，可直接用作
> 论文方法/结果章节的材料。

## 概述

SMPR（Structural-routing Multi-channel Quantum Router）以 LightSABRE 布局库为
基础，通过评估驱动的多通道策略发现与真实成本布局搜索，在合成基准与公共
heavy-hex 基准上持续降低量子电路编译的 SWAP 数量。V16 为本研究当前最优版本：

| 基准 | V5 基线 | V16 | 相对 V5 | 相对原论文 SMPR |
|------|---------|-----|---------|----------------|
| B1（合成，60 case） | 1641 | **1556** | −5.2% | — |
| B2（盲测，120 case） | 3051 | **2899** | −5.0% | −5.0%（原论文 SMPR = V5） |
| 公共 heavy-hex 10 条（同布局） | — | 356（LS 386） | — | −7.8% |
| 公共 heavy-hex 10 条（布局池） | — | 314（LS 386） | — | −18.7% |

## 目录

| 路径 | 内容 |
|------|------|
| `文档/01_V16方法.md` | V14–V16 方法：策略发现、多通道组合、布局搜索、新策略族、公共电路扩展 |
| `文档/02_V16结果.md` | V16 全部实验结果（合成、公共、z3） |
| `文档/03_相对基线提升.md` | 相对 V5/V6/V7 与原论文叙事的提升（含可直接引用的对比表） |
| `文档/04_证据与边界.md` | 证据链（最优性证明、零回归、泛化、负结果）与方法边界 |
| `源代码/` | 本版本关键脚本与引擎改动（5 个脚本 + structural_policy.py） |
| `实验结果/` | 全部实验 CSV / JSON（部署、搜索、bridge、策略发现） |

## 复现

1. 环境：`python 3.12` + `qiskit` + `psutil` + `z3-solver`。
2. 策略发现：`python scripts/policy_discovery_v16.py --workers 12`。
3. 公共电路扩展：`python scripts/prepare_v7_ext.py`；
   `python scripts/v15_v7_ext_bridge.py --lean --max-cx 500`。
4. 布局深搜：`python scripts/layout_search_realcost.py --max-steps 50 --restarts 2`。
5. 部署：`python scripts/v13.py --channels results/.../selected_channels_15.json ...`。
