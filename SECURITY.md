# Security

本仓库不应包含 API key、访问令牌、私钥、`.env` 或个人机器凭据。

提交前运行：

```text
python scripts/verify_full_repository.py
```

历史归档可能保留对本地凭据文件名的代码引用，但归档验证器会拒绝实际凭据文件。
如果曾在开发目录中使用真实凭据，应在公开仓库创建前完成撤销和轮换。

