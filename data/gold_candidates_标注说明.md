# 金标候选表 · 标注速查（v3 口径 + primary 判定）

文件：`data/gold_candidates.csv`，200 行，用 Excel 直接打开（带 BOM）。
旧版（v2 口径、exact 判定）改名保留为 `data/gold_candidates_v2exact.csv`。
完整的类别定义见同目录的 `gold_candidates_README.md`（由代码生成）。

## 只填这三列，其余列都别动

| 列 | 取值 | 多值写法 |
| --- | --- | --- |
| `need` | `none` / `single` / `multi`，只能填一个 | — |
| `domains` | 14 个里选：`file` `code` `database` `search_web` `browser` `comms` `calendar_task` `knowledge` `cloud_ops` `business` `media` `weather_geo` `entertainment` `system_misc` | 英文分号 `;`，不加空格，主域写在第一个 |
| `actions` | `read` `create` `update` `delete` `execute` `send` | 英文分号 `;` |

规则：
- `need = none` 时，`domains` 和 `actions` 都留空。
- `need` 是 `single` 或 `multi` 时，`domains` 和 `actions` 至少各填一个。
- **v3 口径**：要拿实时或外部信息的（天气、比分、股价、新闻、航班、某个网页现在的内容），算需要工具，不填 `none`。纯知识问答、写作、翻译、计算才填 `none`。
- 多个能力域时，把最主要的那个写在第一个（以后算主域口径的指标要用）。

## 这几种行要特别看（看 `note` 列）

| note | 行数 | 预填从哪来 | 怎么处理 |
| --- | --- | --- | --- |
| （空） | 100 | 留出集，两路模型意见一致 | 大多是对的，只改错的 |
| `主域不一致` | 43 | need 用两路一致的判断，domains 只预填 Jev 的主域 | 重点核对 domains |
| `need 不一致` | 53 | need 用的是**本地模型**的判断，它明显偏向 `multi`（其中 43 行预填成了 multi） | **当成没填过，从头判断 need** |
| `有一路没问到答案` | 4 | 只有一路的结果 | 从头判断 |

另外有 21 行 `need` 预填成 single/multi，`domains` 却是空的（Jev 判了 none，所以没有域可预填）。如果你确认需要工具，就把 `domains` 和 `actions` 补上；可以参考 `labeler2_domains`。
