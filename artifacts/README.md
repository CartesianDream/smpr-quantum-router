# Verified archives

这些 ZIP 是构建本仓库时保留的字节级原始交付物，用于复现和来源追踪。

| 文件 | 用途 |
|---|---|
| `SMPR_public_core_1.0.0rc3_V7_validated.zip` | 已验证的正式可执行核心 |
| `SMPR_complete_research_history.zip` | 完整研发历史、旧字段、实验脚本与自校验清单 |
| `SMPR_confirmed_evidence.zip` | 冻结合成实验的独立重跑、审计和环境记录 |
| `SMPR_public_external_evidence.zip` | 公共线路外部验证的冻结清单、三次重复和分析 |

根目录代码是维护入口；历史归档只用于复现和证据追踪，不是第二套公共 API。
运行 `python scripts/verify_full_repository.py` 可检查这些归档的原始 SHA-256。

