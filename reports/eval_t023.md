# 风险规则补两类信号:扫描外部主机、控制实体设备(T-023)

真实工具目录 2405 个,银标是 Jev / Haiku 两路一致结果。改前 = main 的 `rules.rule_risk`(T-011 最终版),改后 = 叠加 T-023 两类信号。

> 词表只按开发份(按函数名 6:4 切分)选,测试份和参考集只在最后各报一次、不参与选词:扫描类逐词选(开发份银标 L3 命中 >= 2 且非 L3 误伤 = 0),设备类是"设备名词 + 控制动词"组合信号(见 1.0)。复跑:`cli tools-real report-t023`(只读 `data/` 下已落盘文件,不联网)。


## 0. 参考集(D-005,只报一次,改后)

- 参考集 60 个工具。规则 L3 判低(真值 L3 但判低):改前 2 个 → **改后 0 个**(改前 ['Crawler/Crawler.scan', 'misc/oven_preheat']);多打扰(真值非 L3 但判 L3):改前 17 → 改后 **17**。

- 改后新增 L3 的工具:


| tool_id | 改前 | 改后 | 真值 |
| --- | --- | --- | --- |
| Crawler/Crawler.scan | L1 | L3 | L3 |
| misc/oven_preheat | L0 | L3 | L3 |

参考集准确率 56.7% → 60.0%。


## 1. 词表(只按开发份选)


### 1.0 两类工具概览(信号命中,银标分布)


| 信号 | 命中数 | 银标L3 | 银标L2 | 银标L1 | 银标L0 | 未一致 | 5 个例子 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| device_control | 7 | 7 | 0 | 0 | 0 | 0 | ControlAppliance/ControlAppliance.execute, ControlAppliance/ControlAppliance.execute#2, misc/ThinQ_Connect, misc/ThinQ_Connect#2, misc/ThinQ_Connect#3 |
| scan_host | 7 | 6 | 0 | 0 | 0 | 1 | Crawler/Crawler.scan, XSS_Scanner/XSS_Scanner.scan, hacking/hacking.query, misc/Attack, misc/nuclei_http |

### 1.1 宽候选词排序(开发份,按银标 L3 命中降序 / 非 L3 误伤升序)

两类信号的选词口径:**扫描类**按词选(开发份银标 L3 命中 >= 2 且非 L3 误伤 = 0 才留);**设备类**是"设备名词 + 控制动词"的组合信号,单个词不单独判定,词表见 1.3 / 1.4,本表用来看每个候选词自己会不会误伤。`用在规则里` = 该词进了最终词表。


| 类 | 词 | 用在规则里 | 开发份命中 | 开发L3 | 开发L2 | 开发L1 | 开发L0 | 测试命中 | 测试L3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 控制设备 | **power** | 是 | 6 | 3 | 0 | 0 | 0 | 0 | 0 |
| 控制设备 | temperature | 否 | 30 | 3 | 0 | 0 | 21 | 18 | 1 |
| 控制设备 | **appliance** | 是 | 2 | 2 | 0 | 0 | 0 | 4 | 4 |
| 控制设备 | **control** | 是 | 5 | 2 | 2 | 0 | 1 | 8 | 6 |
| 控制设备 | **oven** | 是 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 控制设备 | **preheat** | 是 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 控制设备 | **device** | 是 | 10 | 0 | 0 | 0 | 0 | 3 | 1 |
| 控制设备 | iot | 否 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 控制设备 | **thermostat** | 是 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 控制设备 | **light** | 是 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| 控制设备 | **lock** | 是 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 控制设备 | **unlock** | 是 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 控制设备 | **heater** | 是 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 控制设备 | **bulb** | 是 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 控制设备 | **lamp** | 是 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 控制设备 | turn on | 否 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 控制设备 | turn off | 否 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 控制设备 | switch | 否 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 控制设备 | camera | 否 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 控制设备 | door | 否 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 控制设备 | **garage** | 是 | 6 | 0 | 0 | 0 | 2 | 0 | 0 |
| 控制设备 | **heat** | 是 | 4 | 0 | 0 | 0 | 3 | 1 | 0 |
| 控制设备 | sensor | 否 | 6 | 0 | 0 | 0 | 5 | 7 | 0 |
| 控制设备 | vehicle | 否 | 7 | 0 | 0 | 0 | 7 | 1 | 0 |
| 扫描/探测 | **scan** | 是 | 4 | 3 | 0 | 0 | 0 | 1 | 1 |
| 扫描/探测 | **scanner** | 是 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| 扫描/探测 | **penetration** | 是 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| 扫描/探测 | vulnerabilities | 否 | 4 | 2 | 0 | 0 | 0 | 11 | 1 |
| 扫描/探测 | nuclei | 否 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 扫描/探测 | attack | 否 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 扫描/探测 | subdomain | 否 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 扫描/探测 | osint | 否 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 扫描/探测 | crawl | 否 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 扫描/探测 | crawler | 否 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| 扫描/探测 | probe | 否 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 扫描/探测 | port | 否 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 扫描/探测 | nmap | 否 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 扫描/探测 | ping | 否 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 扫描/探测 | traceroute | 否 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 扫描/探测 | whois | 否 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 扫描/探测 | enumerate | 否 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 扫描/探测 | enum | 否 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 扫描/探测 | vulnerability | 否 | 9 | 0 | 0 | 0 | 0 | 11 | 0 |
| 扫描/探测 | recon | 否 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 扫描/探测 | detect | 否 | 0 | 0 | 0 | 0 | 0 | 2 | 1 |
| 扫描/探测 | exploit | 否 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| 扫描/探测 | discover | 否 | 7 | 0 | 0 | 0 | 3 | 7 | 0 |

扫描类被排除的候选词:`crawl` / `crawler` / `probe` / `port` / `nmap` / `ping` /`traceroute` / `whois`(开发份 L3 命中 0;参考集里的 `Crawler.scan` 已由 `scan` 命中)、`nuclei` / `attack` / `subdomain` / `osint` / `detect` / `exploit` / `vulnerability` /`vulnerabilities`(开发份 L3 命中 < 2,等于把单个工具的名字记进词表;`nuclei_http` 的描述里有 `scan`,已被 `scan` 覆盖)。


设备类被排除的候选词:`vehicle` / `sensor` / `temperature`(开发份非 L3 误伤 7 / 5 / 21,都是 `calculate_emissions`、`get_sensor_readings` 这类只读查询)、`iot`(开发份 L3 命中 0)。1.2 / 1.3 表里的命中数是单个词自己的计数,设备信号只在"名词 + 动词"同时出现时才生效:`heat` 作为裸词会误伤 `Movies_*_FindMovies`,和 `oven` 组合才命中 `oven_preheat`;`set` 裸词很常见(开发份 36 个),但和 `thermostat` / `device` 这类名词组合时才是控制动作。


### 1.2 扫描 / 探测外部主机


| 词 | 来源 | 开发份命中 | 开发L3 | 开发L2 | 开发L1 | 开发L0 | 测试份命中 | 测试L3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| penetration | train | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| scan | train | 4 | 3 | 0 | 0 | 0 | 1 | 1 |
| scanner | train | 2 | 2 | 0 | 0 | 0 | 0 | 0 |

### 1.3 控制实体设备:设备名词


| 词 | 来源 | 开发份命中 | 开发L3 | 开发L2 | 开发L1 | 开发L0 | 测试份命中 | 测试L3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| appliance | train | 2 | 2 | 0 | 0 | 0 | 4 | 4 |
| bulb | prior | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| device | train | 10 | 0 | 0 | 0 | 0 | 3 | 1 |
| garage | train | 6 | 0 | 0 | 0 | 2 | 0 | 0 |
| heater | prior | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lamp | prior | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| light | train | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| lock | prior | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| oven | train | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| thermostat | prior | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### 1.4 控制实体设备:控制动词


| 词 | 来源 | 开发份命中 | 开发L3 | 开发L2 | 开发L1 | 开发L0 | 测试份命中 | 测试L3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| activate | train | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| control | train | 5 | 2 | 2 | 0 | 1 | 8 | 6 |
| deactivate | prior | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim | prior | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| heat | train | 4 | 0 | 0 | 0 | 3 | 1 | 0 |
| lock | prior | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| power | train | 6 | 3 | 0 | 0 | 0 | 0 | 0 |
| preheat | train | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| set | train | 36 | 2 | 17 | 0 | 2 | 13 | 3 |
| toggle | prior | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| unlock | prior | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

命中工具(全部,共 14 个;开发份 / 测试份按函数名划分):


| tool_id | 信号 | 划分 | 银标 | 改前 | 改后 |
| --- | --- | --- | --- | --- | --- |
| ControlAppliance/ControlAppliance.execute | device_control | 开发份 | L3 | L3 | L3 |
| ControlAppliance/ControlAppliance.execute#2 | device_control | 开发份 | L3 | L3 | L3 |
| Crawler/Crawler.scan | scan_host | 测试份 | L3 | L1 | L3 |
| XSS_Scanner/XSS_Scanner.scan | scan_host | 开发份 | L3 | L0 | L3 |
| hacking/hacking.query | scan_host | 开发份 | L3 | L0 | L3 |
| misc/Attack | scan_host | 开发份 | L3 | L0 | L3 |
| misc/ThinQ_Connect | device_control | 测试份 | L3 | L0 | L3 |
| misc/ThinQ_Connect#2 | device_control | 测试份 | L3 | L0 | L3 |
| misc/ThinQ_Connect#3 | device_control | 测试份 | L3 | L0 | L3 |
| misc/ThinQ_Connect#4 | device_control | 测试份 | L3 | L0 | L3 |
| misc/nuclei_http | scan_host | 开发份 | L3 | L0 | L3 |
| misc/oven_preheat | device_control | 开发份 | L3 | L0 | L3 |
| misc/wps_scan | scan_host | 开发份 | (未一致) | L1 | L3 |
| network_scanner/network_scanner.scan | scan_host | 开发份 | L3 | L0 | L3 |

## 2. 开发份 / 测试份规则指标(改前 / 改后)


**按函数名分组 6:4**


| 集合 | 条数 | L3 条数 | 准确率(改前/改后) | L3 召回(改前/改后) | 低报率(改前/改后) | 低判成 L3(改前/改后) |
| --- | --- | --- | --- | --- | --- | --- |
| 开发份 | 1018 | 138 | 86.1% / 86.7% | 55.1% / 59.4% | 8.8% / 8.3% | 39 / 39 |
| 测试份 | 666 | 111 | 87.2% / 88.0% | 55.0% / 59.5% | 9.2% / 8.4% | 17 / 17 |

**按 server 分组 6:4**


| 集合 | 条数 | L3 条数 | 准确率(改前/改后) | L3 召回(改前/改后) | 低报率(改前/改后) | 低判成 L3(改前/改后) |
| --- | --- | --- | --- | --- | --- | --- |
| 开发份 | 879 | 122 | 86.6% / 86.8% | 46.7% / 48.4% | 9.6% / 9.3% | 31 / 31 |
| 测试份 | 805 | 127 | 86.6% / 87.7% | 63.0% / 70.1% | 8.3% / 7.2% | 25 / 25 |

测试份(按函数名)L3 召回 55.0% → 59.5%(折合 +5.0 条);测试份(按 server)L3 召回 63.0% → 70.1%。低判成 L3 两套划分都不增加(按函数名 17→17、按 server 25→25)。


## 3. 结论

- 参考集两个漏判 L3(`Crawler/Crawler.scan`、`misc/oven_preheat`)都补上了:规则 L3 判低 2 → 0;多打扰 17 → 17。

- 测试份(两种划分)L3 召回都上升、低判成 L3 不增加;开发份同理。

- 合成工具(`data/tools.jsonl`,53 个)无一命中两类信号,`cli evaluate` 输出不变。

