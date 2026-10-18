# 真实工具目录统计(T-010 A 部分)

由 `scripts/build_real_tools.py` 生成,数据来自 `data/raw/bfcl/` 的 8 个问题文件与对应的 possible_answer。全部离线,没有调任何模型。


## 1. 工具目录

- 函数出现总数(去重前):6226
- 去重后工具数:**2405**
- 唯一函数名:1287
- 同名但描述或参数不同的名字数:**433**(涉及 1551 条工具记录)
- 没有描述的工具数:0

按来源文件(一个工具可能出现在多个文件里,所以这里相加大于总数):


| 来源文件 | 工具数 |
| --- | --- |
| `BFCL_v3_live_multiple.json` | 1073 |
| `BFCL_v3_parallel_multiple.json` | 482 |
| `BFCL_v3_multiple.json` | 480 |
| `BFCL_v3_simple.json` | 400 |
| `BFCL_v3_parallel.json` | 198 |
| `BFCL_v3_live_simple.json` | 154 |
| `BFCL_v3_live_parallel_multiple.json` | 82 |
| `BFCL_v3_live_parallel.json` | 18 |

同名冲突最多的 5 个名字:


| 函数名 | 变体数 | 其中一条描述 |
| --- | --- | --- |
| `get_current_weather` | 48 | Fetches the current weather data for a specified location, w |
| `Events_3_FindEvents` | 23 | Find cultural events such as concerts and plays happening in |
| `Flights_4_SearchRoundtripFlights` | 20 | Search for available roundtrip flights based on specific cri |
| `add_postgres_server` | 18 | Add a new PostgreSQL server configuration or update an exist |
| `Movies_3_FindMovies` | 14 | Retrieve a list of movies based on director, genre, and cast |

按推断的 server(前 20):


| server | 工具数 |
| --- | --- |
| `misc` | 1143 |
| `Events` | 36 |
| `Hotels` | 35 |
| `Movies` | 34 |
| `Flights` | 30 |
| `Services` | 27 |
| `math` | 23 |
| `project_api.ProjectApi` | 23 |
| `Restaurants` | 20 |
| `Music` | 18 |
| `RentalCars` | 18 |
| `badge_api.BadgeApi` | 18 |
| `Buses` | 16 |
| `kinematics` | 15 |
| `Trains` | 14 |
| `version_api.VersionApi` | 14 |
| `Travel` | 12 |
| `weather` | 12 |
| `geometry` | 11 |
| `requests` | 11 |

共 429 个不同的 server。


## 2. 请求 → 工具对应关系

- 池子里 BFCL 请求:600 条
- 有 tool_ids 的:600 条(占 100.0%)
- 跳过(没有 possible_answer 或 id 对不上):0 条
- 答案函数名映射失败的:0 条
- tool_ids 总数:693;每条请求平均 1.16 个

| 每条请求的 tool_ids 个数 | 请求数 |
| --- | --- |
| 1 | 522 |
| 2 | 63 |
| 3 | 15 |

| 候选集大小 | 请求数 |
| --- | --- |
| 1 | 295 |
| 2 | 128 |
| 3 | 93 |
| 4 | 26 |
| 5 | 29 |
| 6 | 15 |
| 7 | 5 |
| 8 | 1 |
| 9 | 5 |
| 10 | 3 |

候选集大小:最小 1,中位数 2,最大 10,平均 2.15。


## 3. 描述长度(字符数)

| 最小 | 25% | 中位数 | 75% | 最大 | 平均 |
| --- | --- | --- | --- | --- | --- |
| 20 | 67 | 86 | 112 | 545 | 94.7 |

| 长度区间 | 工具数 |
| --- | --- |
| 0–39 | 48 |
| 40–79 | 969 |
| 80–159 | 1216 |
| 160–319 | 166 |
| ≥320 | 6 |

没有描述的工具:0 个。


## 4. 粗看风险(只计数,不是标签)

函数名或描述里出现 `delete|remove|drop|send|post|pay|transfer|execute|run|create|update|write` 的工具:**228** 个(占 9.5%)。

这个正则很宽(`run`/`post` 之类会误命中),只用来给 B 部分打标时对照,不要当风险标签用。

