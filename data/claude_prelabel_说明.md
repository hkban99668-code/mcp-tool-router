# Claude 预标说明（2026-10-13，gu 选了方案 A）

两份文件由 Claude 预标（它没参与过 Jev / Haiku / DeepSeek 的打标，判断是独立的），gu 只需要抽查"需要你看"标了"是"的那些行。**这些是强模型给的参考答案，不是纯人工金标**，报告里会按这个口径写。

## data/gold_candidates_claude.csv（200 条）
- `need` / `domains` / `actions` 三列是 Claude 的答案；原来的预填值移到了 `prefill_*` 三列。
- `需要你看 = 是` 的有 22 行，都是 need 判断不确定的。不同意就直接改那三列。
- 判断口径：v3（实时信息算需要工具，计算不算）。同一个工具调多次（比如查两个城市的天气）算 single；用到两个不同的工具才算 multi。

## data/tools_real_risk_audit_claude.csv（60 行）
- `claude_risk` 是 Claude 的判断；`需要你看 = 是` 的有 8 行。
- gu 只在不同意时填 `gu_risk`；留空就表示同意 `claude_risk`。

## 初步发现
- need：在两路一致的 100 条留出集上，Claude 和银标 97 条相同。
- 风险：60 行里只有 28 行和银标一致。银标把两路的动作取并集，偏保守，比如播放音乐、电影被判成 L3，Claude 判 L2。
