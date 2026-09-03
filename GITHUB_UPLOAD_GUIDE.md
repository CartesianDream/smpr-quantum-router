# GitHub 上传与收尾指南

本目录已经是完整仓库根目录。正式上传前，先完成作者和许可证信息，再运行全套
验证。

## 1. 本地验证

Windows PowerShell：

```powershell
py -3.11 -m venv .venv-smpr
.\.venv-smpr\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements-frozen.txt
python -m pip install -e .

python .\scripts\verify_public_release.py
python -m unittest discover -s .\tests -v
python .\scripts\verify_full_repository.py
python -m smpr_router doctor
```

预期所有验证均通过。完整路由复现实验计算量较大，不属于 GitHub CI 的默认任务。

## 2. 补全公开信息

上传公开仓库前必须处理：

1. 在 `CITATION.cff` 中填写作者、仓库 URL 和 DOI；
2. 由版权主体选择许可证，并替换 `LICENSE-REVIEW-REQUIRED.txt`；
3. 同步修改 `pyproject.toml` 中的许可证字段；
4. 检查论文作者、单位、通信方式、资助和利益冲突声明；
5. 再次运行三个验证命令。

如果暂时只建立私有仓库，可以保留许可证占位文件，但不要把它理解为已经获得
公开再分发授权。

## 3. 创建仓库

```powershell
git init
git branch -M main
git add .
git commit -m "Release complete SMPR research repository"
git remote add origin https://github.com/OWNER/smpr-quantum-router.git
git push -u origin main
```

建议在验证通过后创建只读标签：

```powershell
git tag -a smpr-contest-final-2026 -m "Final contest release"
git push origin smpr-contest-final-2026
```

## 4. GitHub 页面检查

- Actions 中的 `verify` 工作流通过；
- `paper/` 中两份 PDF 可以直接下载；
- `results/README.md` 能正确解释证据层次；
- `artifacts/` 中四个原始归档均存在；
- Release 页面附上最终仓库 ZIP 的 SHA-256；
- 不再依据现有冻结结果修改算法后继续沿用原确认性表述。

