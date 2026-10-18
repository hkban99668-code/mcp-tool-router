# mcp-classifier（基于 Jev 的 LLM 快速分类）

给 Claude 的项目说明。启动后先读 `.claude/memory/MEMORY.md`（本项目记忆的备份），再读 `.collab/board.md` 看任务状态。

## 项目是什么
- `mcpcls`：把用户请求分成 need（none/single/multi）+ 能力域，再从真实工具目录（2405 个工具）里检索候选工具。tfidf/双语索引 + 软过滤 + 风险规则。
- 标签来自双路打标：第一路 Jev（commandcode 网关），第二路 DeepSeek v4.1-flash（T-017 起，替换 Haiku）。答案缓存在 `.cache/answers/`，不加 `--live` 时只读缓存、0 次调用。
- 现行口径：need v3 + taxonomy v3（14 域）+ primary 一致判定；taxonomy v4（拆 system_misc）在分支实验中（T-015/T-018）。

## 协作方式
- Claude 写任务卡/审查/定结论，CodeBuddy 实现：`.collab/`（PROTOCOL.md、CODEBUDDY.md、tasks/、reviews/、board.md），派发 `.collab/dispatch.sh T-xxx [fix|continue]`，用 nohup + setsid 脱离会话。
- 看板：`~/CodeBuddy/20260921235524/collab-dashboard`（`python3 dashboard.py`，127.0.0.1:8765）。

## 硬规则
- 真实 API 调用只由 Claude 在 Mac 本机发起（经 7890 代理）；CodeBuddy 不跑 `--live`。遇到 429/额度就停，不硬重试。
- key 不落地：不读、不打印、不写进文件或命令行。
- 合并只在本地 main，合并前先打 tag；不删数据、不改写历史；推 GitHub 前查全历史无 key；2026-10-16 起 gu 授权 Claude 自行合并、推送，只回报重大决策（改默认、改门槛、方向取舍）。
- 评估纪律：gold 只用来报告，调参看 5 折交叉验证（开发折 0–2，测试折 3–4 只报一次），探针集写好后冻结。
