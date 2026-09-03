# LaTeX 快速编译与超时处理

## 推荐设置

- 主文档：`main.tex`
- 编译器：**XeLaTeX**
- 参考文献：BibTeX
- 实验报告需要单独将主文档切换为 `experiment_report.tex`
- 不要使用 pdfLaTeX；随包字体通过 `fontspec` 加载

本修正版把正文中的 TikZ/PGFPlots 图预渲染为 `figures/*.pdf`。主论文和
实验报告只插入这些矢量图，不再在每轮 XeLaTeX 中重算坐标。图的可编辑
源文件仍位于 `figures/*.tex`。

## Overleaf

1. 上传本 ZIP 并解压为一个新项目；
2. 在 **Menu / Settings** 中把 Compiler 设为 **XeLaTeX**；
3. 把 Main document 设为 `main.tex`；
4. 选择 **Recompile from scratch** 清除旧辅助文件；
5. 实验报告不要与主论文同时编译；需要报告时再把 Main document 改为
   `experiment_report.tex`。

根目录 `latexmkrc` 已固定 XeLaTeX、错误即停和最多 5 轮收敛。若平台不允许
项目级配置，手动选择 XeLaTeX 即可。

## 本地命令

```bash
latexmk main.tex
latexmk experiment_report.tex
```

PowerShell 也可直接运行：

```powershell
latexmk main.tex
latexmk experiment_report.tex
```

若系统没有 `latexmk`，使用：

```bash
xelatex -no-pdf main.tex
bibtex main
xelatex -no-pdf main.tex
xelatex -no-pdf main.tex
xdvipdfmx -E -o main.pdf main.xdv
```

## 已验证结果

在干净目录的完整 TeX Live 环境中，修正版从零构建成功：

- 主论文：28 页；
- 实验报告：20 页；
- 无未定义引用、未定义引文、缺字和 LaTeX 错误；
- 主论文冷构建约 5 秒，实验报告约 5 秒（机器相关）。

包内同时附带已验证 PDF。若云平台仍超时，可先使用 PDF，并在本地 TeX Live
中编译；此时问题通常是平台资源或项目设置，不是正文无法编译。

