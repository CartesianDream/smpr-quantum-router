# SMPR 正式论文 LaTeX 源码

论文主文件为 `main.tex`，独立详细实验报告为
`experiment_report.tex`。两者均使用 XeLaTeX；论文另使用 BibTeX。论文题目：

> 安全多布局组合量子比特路由：完整策略展开、跨布局迁移与可审计选择

## 编译

```bash
latexmk main.tex
latexmk experiment_report.tex
```

根目录 `latexmkrc` 已固定 XeLaTeX。云端编译时必须把主文档设为
`main.tex`，编译器设为 XeLaTeX；实验报告应单独把主文档切换为
`experiment_report.tex`。详见 `BUILD_TROUBLESHOOTING.md`。

本修正版已将正文中的 TikZ/PGFPlots 图预渲染为矢量 PDF，避免每一轮
XeLaTeX 重复计算图形。可编辑图源仍保留在 `figures/`。

或手动：

```bash
xelatex -no-pdf main.tex
bibtex main
xelatex -no-pdf main.tex
xelatex -no-pdf main.tex
xdvipdfmx -E -o main.pdf main.xdv
```

## 字体

- 中文正文：Noto Serif SC 子集；
- 中文标题：Noto Sans SC 子集；
- 英文正文：Nimbus Roman；
- 英文标题：Nimbus Sans；
- 数学：标准 LaTeX 数学字体。

`fonts/` 已包含论文与实验报告所需字形、Noto OFL 许可证和 URW Base 35
字体许可证。若修改中文内容后出现
`Missing character`，应从完整 Noto CJK 字体重新生成子集。

构建不含 PDF 和 LaTeX 中间文件的确定性源码包：

```bash
python scripts/build_paper_release.py --output SMPR_chinese_paper_source_v7_final.zip
```

## 证据目录

`evidence/` 保存冻结质量、运行时间、分组表、独立复现表、历史审计脚本和
V7 冻结分析证据；其中 `evidence/v7/` 保存 V7 分析报告、摘要、三次配对
前沿、冻结清单和逐文件哈希。
公开核心中的 `evidence/reference/` 保存对外命名的逐实例参考表；完整原始证据
位于独立复现档案，不与论文源码包重复。

## 投稿前填写

当前版本使用 `Anonymous Author` 和双盲单位占位。投稿前按目标期刊要求：

1. 决定保留匿名信息还是填写作者、单位和通信邮箱；
2. 补充致谢、基金和利益冲突声明；
3. 填入仓库 URL、提交号和正式许可证；
4. 补充 CPU、内存和操作系统；
5. 若使用期刊模板，迁移正文、图表、算法和参考文献。

## 结论边界

- V6 是冻结算法后的预注册盲测，承担主要推断；
- V5 只用于冻结等价与工程效率复核；
- 3.94% 改善属于完整 SMPR 组合，不能归因于独立展开分支；
- 持久分片加速相对旧 SMPR 执行器，不代表快于单独 LightSABRE；
- V7 已完成 10 条公共线路、11--23 比特、heavy-hex 和三次同机质量--时间验证；
- V7 四档总量均有利于 SMPR，但线路聚类 95% 区间跨零，不能表述为统计显著。

## 与另一份初步路线的关系

`METHOD_COMPARISON.md` 对比了 SMPR 与《量子比特路由路线初步》中的关键路径
势函数、自适应 Beam Search 和回滚 repair。建议把该路线作为未来的候选分支，
而不是移除 SMPR 的 LightSABRE 安全下界。
