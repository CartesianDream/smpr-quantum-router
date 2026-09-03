# SMPR public core

> 本目录现同时作为最终 GitHub 仓库根目录。维护代码仍位于 `src/`；完整逐例
> 证据、论文、回顾性扩展和原始归档分别位于 `results/`、`paper/` 和
> `artifacts/`，不会改变公开 Python API。

当前发布候选为 `1.0.0rc3`。该版本保留 `1.0.0rc2` 已独立复核的 V5/V6
核心，并修复 V7 预处理清单中的两条动态线路及失败后残留半成品的问题。
请勿使用 rc1；rc2 只能用于已验证的 V5/V6，不能用于准备 V7。

SMPR（Safe Multi-Layout Portfolio Router，安全多布局组合路由器）面向受限耦合
拓扑执行量子比特路由。它把多种候选显式放在同一选择器中：

1. 20 个固定种子的 LightSABRE；
2. 独立完整策略展开；
3. 从最佳 LightSABRE 初始布局及其邻域启动的跨布局展开。

最终按 `(SWAP, weighted_depth, branch_priority, deterministic_tie_key)`
字典序选优，并始终保留 LightSABRE 候选。因此，在同一次运行的候选均完成且
通过审计时，SMPR 的选中结果不会劣于本次 LightSABRE 候选集。

本目录只包含可执行公开核心、冻结 V5/V6 输入与公开字段参考结果。完整探索
历史、原始字段和证据哈希位于独立的“完整复现档案”，二者不可混合发布。

## 冻结证据边界

- V5：60 个实例，LightSABRE / SMPR 总 SWAP 为 `1691 / 1641`；
- V6：120 个预注册盲测实例，总 SWAP 为 `3176 / 3051`；
- V6 逐实例胜/平/负为 `50 / 70 / 0`；
- V6 相对减少 `3.9358%`；
- 实例级 bootstrap 95% CI 为 `[-0.024346, -0.013634]`；
- 因子单元 cluster bootstrap 95% CI 为 `[-0.024392, -0.013594]`；
- V5/V6 的语义、拓扑和指标审计全部通过。

上述结果支持“冻结测试上减少 SWAP 且逐实例无 SWAP 退化”，不支持把 SMPR
描述为普遍更快。V7 将在同一机器上单独报告质量—时间前沿。

## 安装

Windows PowerShell：

```powershell
py -3.10 -m venv .venv-smpr
.\.venv-smpr\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m smpr_router doctor
python -m unittest discover -s .\tests -v
```

Python 3.10–3.12 均可使用；冻结依赖为 Qiskit 2.4.2 和
qiskit-qasm3-import 0.6.0。

## 五例冒烟测试

```powershell
$env:PYTHONHASHSEED = "0"
python -m smpr_router run `
  --config .\configs\frozen_v5.toml `
  --executor persistent `
  --case-indices 1,25,28,53,56 `
  --workers 4 `
  --quality-reference .\evidence\reference\v5_cases.csv `
  --quality-policy exact `
  --output-dir .\results\v5_smoke
```

预期 LightSABRE / SMPR 总 SWAP 为 `191 / 182`，三项审计为 `5/5/5`，
参考结果差异数为 0。

完整 V5/V6：

```powershell
.\scripts\run_frozen_v5.ps1
.\scripts\run_frozen_v6.ps1
```

## V7 外部验证

V7 使用固定提交的 QASMBench 公共线路、Qiskit heavy-hex 距离 3/5 拓扑和
三次同机重复。必须先冻结清单，再运行：

```powershell
python .\scripts\prepare_v7.py
python .\scripts\freeze_v7.py
.\scripts\run_v7.ps1
```

`prepare_v7.py` 先在临时目录完成全部下载、源哈希校验、静态线路检查和
规范化，全部成功后才发布正式输入。若它失败，请不要冻结或复用半成品；
改用未执行过 V7 准备的新解压目录。

`run_v7.ps1` 会顺序完成三次重复并自动调用分析器；如需单独重新分析已有结果：

```powershell
python .\scripts\analyze_v7.py `
  --results-root .\results `
  --output-dir .\results\v7_analysis_recomputed
```

在 `freeze_v7.py` 产生冻结哈希之后，不得依据 V7 结果修改候选、参数、线路
清单或接受标准。详见 [docs/V7_PROTOCOL.md](docs/V7_PROTOCOL.md)。

## 公开包结构

```text
src/smpr_router/          稳定 API、CLI 与公开核心
src/smpr_router/engine/   描述性算法模块
configs/                  冻结 V5/V6 配置
data/                     冻结输入；V7 输入由脚本准备
evidence/reference/       公开字段参考结果
scripts/                  复现、V7 和发布校验入口
tests/                    核心、命名、配置和命令测试
docs/                     算法、复现与 V7 协议
```

## 发布前法律步骤

当前是审阅候选包。自动整理过程不能替版权所有者选择许可证，因此
`LICENSE-REVIEW-REQUIRED.txt` 暂不授予公开再分发权。正式发布前必须由
版权所有者完成 [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) 中的作者、
版权、许可证、仓库 URL 和 DOI 项。

## 完整仓库入口

- [论文与实验报告](paper/README.md)
- [完整实验结果和证据层次](results/README.md)
- [完整仓库最终验收](results/GITHUB_FINAL_VALIDATION.md)
- [原始验证归档](artifacts/README.md)
- [GitHub 上传与最终收尾](GITHUB_UPLOAD_GUIDE.md)
- [安全说明](SECURITY.md)

除原公开核心验证外，完整仓库还提供：

```powershell
python .\scripts\verify_full_repository.py
```

该命令检查论文与归档哈希、V5/V6 总量、V7 120 条观测和回顾性扩展中的可复算
数字。它不重新运行耗时的完整路由实验。
