# 公开数据来源

由 `scripts/fetch_public.py` 下载到 `data/raw/`(该目录不进版本库)。每行给出下载 URL、许可、文件 sha256、下载日期与署名,换机器重跑后 sha256 应当完全一致。


| 来源 | 文件 | 许可 | 行数 | 大小(字节) | sha256 | 下载日期 |
| --- | --- | --- | --- | --- | --- | --- |
| bfcl | `bfcl/BFCL_v3_live_simple.json` | Apache-2.0 | 258 | 260128 | `b398743e6c0dfd2a…` | 2026-10-09 |
| bfcl | `bfcl/BFCL_v3_live_multiple.json` | Apache-2.0 | 1053 | 3891342 | `af2520ac5bb0b05a…` | 2026-10-09 |
| bfcl | `bfcl/BFCL_v3_live_parallel.json` | Apache-2.0 | 16 | 15649 | `6c26e9fdc3350cf5…` | 2026-10-09 |
| bfcl | `bfcl/BFCL_v3_live_parallel_multiple.json` | Apache-2.0 | 24 | 82332 | `21d4b9319c1faac4…` | 2026-10-09 |
| bfcl | `bfcl/BFCL_v3_simple.json` | Apache-2.0 | 400 | 280474 | `fbc37b2ad252bf9a…` | 2026-10-09 |
| bfcl | `bfcl/BFCL_v3_multiple.json` | Apache-2.0 | 200 | 316583 | `aef168155ebd74b7…` | 2026-10-09 |
| bfcl | `bfcl/BFCL_v3_parallel.json` | Apache-2.0 | 200 | 172094 | `fad375d68295776e…` | 2026-10-09 |
| bfcl | `bfcl/BFCL_v3_parallel_multiple.json` | Apache-2.0 | 200 | 347080 | `8863ea8433239f55…` | 2026-10-09 |
| apibank | `apibank/lv1-train.json` | MIT | 6184 | 26042358 | `b2c6b416452636b1…` | 2026-10-09 |
| apibank | `apibank/lv2-train.json` | MIT | 9279 | 26701289 | `e0cca75bb396ac01…` | 2026-10-09 |
| alpaca_zh | `alpaca_zh/alpaca_gpt4_data_zh.json` | CC-BY-4.0 | 48818 | 35100559 | `6fcb98c6d79691d1…` | 2026-10-09 |

## 署名

- Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm
- API-Bank, liminghao1630 (原作者 Minghao Li 等)
- alpaca-zh, shibing624(由 GPT-4 生成的中文指令数据)

## URL 清单

- `bfcl/BFCL_v3_live_simple.json` ← https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_live_simple.json
- `bfcl/BFCL_v3_live_multiple.json` ← https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_live_multiple.json
- `bfcl/BFCL_v3_live_parallel.json` ← https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_live_parallel.json
- `bfcl/BFCL_v3_live_parallel_multiple.json` ← https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_live_parallel_multiple.json
- `bfcl/BFCL_v3_simple.json` ← https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_simple.json
- `bfcl/BFCL_v3_multiple.json` ← https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_multiple.json
- `bfcl/BFCL_v3_parallel.json` ← https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_parallel.json
- `bfcl/BFCL_v3_parallel_multiple.json` ← https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_parallel_multiple.json
- `apibank/lv1-train.json` ← https://huggingface.co/datasets/liminghao1630/API-Bank/resolve/main/training-data/lv1-train.json
- `apibank/lv2-train.json` ← https://huggingface.co/datasets/liminghao1630/API-Bank/resolve/main/training-data/lv2-train.json
- `alpaca_zh/alpaca_gpt4_data_zh.json` ← https://huggingface.co/datasets/shibing624/alpaca-zh/resolve/main/alpaca_gpt4_data_zh.json

<!-- tool-catalog -->

## 工具目录(T-010)

本小节由 `scripts/build_real_tools.py` 维护(重跑 `scripts/fetch_public.py` 会重写上面的表格,之后需要再跑一次本脚本把这一节补回来)。


### 用到的 BFCL 文件


| 文件 | 许可 | 行数 | 大小(字节) | sha256 |
| --- | --- | --- | --- | --- |
| `bfcl/BFCL_v3_live_simple.json` | Apache-2.0 | 258 | 260128 | `b398743e6c0dfd2a…` |
| `bfcl/possible_answer/BFCL_v3_live_simple.json` | Apache-2.0 | 258 | 46971 | `4a1988780cdd8a72…` |
| `bfcl/BFCL_v3_live_multiple.json` | Apache-2.0 | 1053 | 3891342 | `af2520ac5bb0b05a…` |
| `bfcl/possible_answer/BFCL_v3_live_multiple.json` | Apache-2.0 | 1053 | 199668 | `97e90d59c5bd76c5…` |
| `bfcl/BFCL_v3_live_parallel.json` | Apache-2.0 | 16 | 15649 | `6c26e9fdc3350cf5…` |
| `bfcl/possible_answer/BFCL_v3_live_parallel.json` | Apache-2.0 | 16 | 4974 | `5e3d26c9103dd259…` |
| `bfcl/BFCL_v3_live_parallel_multiple.json` | Apache-2.0 | 24 | 82332 | `21d4b9319c1faac4…` |
| `bfcl/possible_answer/BFCL_v3_live_parallel_multiple.json` | Apache-2.0 | 24 | 7420 | `f5b5f360556c5feb…` |
| `bfcl/BFCL_v3_simple.json` | Apache-2.0 | 400 | 280474 | `fbc37b2ad252bf9a…` |
| `bfcl/possible_answer/BFCL_v3_simple.json` | Apache-2.0 | 400 | 60809 | `2911a2bc00df82c4…` |
| `bfcl/BFCL_v3_multiple.json` | Apache-2.0 | 200 | 316583 | `aef168155ebd74b7…` |
| `bfcl/possible_answer/BFCL_v3_multiple.json` | Apache-2.0 | 200 | 32254 | `244e00ce9395df94…` |
| `bfcl/BFCL_v3_parallel.json` | Apache-2.0 | 200 | 172094 | `fad375d68295776e…` |
| `bfcl/possible_answer/BFCL_v3_parallel.json` | Apache-2.0 | 200 | 65905 | `3af54bf04e5b7640…` |
| `bfcl/BFCL_v3_parallel_multiple.json` | Apache-2.0 | 200 | 347080 | `8863ea8433239f55…` |
| `bfcl/possible_answer/BFCL_v3_parallel_multiple.json` | Apache-2.0 | 200 | 74601 | `df4d2c16238c1465…` |

### 派生文件


| 文件 | 条数 | 大小(字节) | sha256 |
| --- | --- | --- | --- |
| `tools_real.jsonl` | 2405 | 1961344 | `63e784a393e2e511…` |
| `requests_real_tool_ids.jsonl` | 600 | 138402 | `caedd22bbdd6512a…` |

### 署名与许可

- Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm — Apache-2.0(问题文件与 possible_answer 同一许可)
- 工具目录是 BFCL 每条请求自带函数定义的去重汇总,`inputSchema` 原样保留,没有改写内容。
