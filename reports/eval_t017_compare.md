# 标签口径对照(T-009 / T-010a)

比的是 2 套口径:`data/v3-primary`(v3-primary)、`data/v3-primary-cn`(v3-primary-cn);编码器 `tfidf`。

> **各口径的留出集不是同一批标签**:标签是重新判定出来的,所以第五节的指标只能看趋势,不能逐点比较。

## 口径总表(need 说明版本 × 一致判定规则)

| 口径 | 目录 | 有效条数 | 一致率 | need 一致率 | 需要工具的样本数 | weather_geo | entertainment | 留出集 need 准确率 | need macro-F1 | 需要 MCP 召回 | none 召回 | 能力域 micro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v3-primary | `data/v3-primary` | 1396 | 83.8% | 93.8% | 399 | 69 | 42 | 84.6% | 56.4% | 88.8% | 86.4% | 64.2% |
| v3-primary-cn | `data/v3-primary-cn` | 1400 | 82.1% | 90.7% | 435 | 69 | 41 | 87.0% | 58.4% | 93.1% | 86.7% | 61.7% |

“有效条数”的分母是两路都答上来的条目;留出集那五列是 C 方案(合成 + real_train)在**各自**留出集上的数字,不同口径的留出集不是同一批标签,只能看趋势。

## 1. 数据规模与双路一致率

| 口径 | 一致集 | 待人工 | real_train | 留出集 | 有效条数 | 双路一致 | 一致率 | need 一致率 | 有一路没答 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v3-primary | 1170 | 230 | 936 | 234 | 1396 | 1170 | 83.8% | 93.8% | 4 |
| v3-primary-cn | 1150 | 250 | 920 | 230 | 1400 | 1150 | 82.1% | 90.7% | 0 |

“有一路没答”的条目既不算一致也不算不一致，已从上表的有效条数里排除。


**按来源**(有效条数、一致率)

| source | v3-primary 有效 | v3-primary 一致率 | v3-primary-cn 有效 | v3-primary-cn 一致率 |
| --- | --- | --- | --- | --- |
| alpaca_zh | 497 | 98.6% | 500 | 96.0% |
| apibank | 299 | 67.2% | 300 | 66.3% |
| bfcl | 600 | 79.8% | 600 | 78.5% |

## 2. 一致集里的 need 分布与能力域样本数

| need | v3-primary(条) | v3-primary-cn(条) |
| --- | --- | --- |
| none | 771 | 715 |
| single | 391 | 422 |
| multi | 8 | 13 |

| 能力域 | v3-primary(条) | v3-primary-cn(条) |
| --- | --- | --- |
| file | 13 | 11 |
| code | 5 | 5 |
| database | 17 | 28 |
| search_web | 93 | 111 |
| browser | 0 | 0 |
| comms | 5 | 4 |
| calendar_task | 35 | 36 |
| knowledge | 1 | 5 |
| cloud_ops | 12 | 11 |
| business | 71 | 79 |
| media | 10 | 15 |
| weather_geo | 69 | 69 |
| entertainment | 42 | 41 |
| system_misc | 26 | 20 |

weather_geo / entertainment 是 v3 起的新域:v2 口径的答案里没有这两题,v2 那一列一定是 0,只有 v3 重新打标后才可能不为 0。

## 3. 留出集里“需要工具”的样本数

| 口径 | 留出集 | 需要工具(single/multi) | none | 占比 |
| --- | --- | --- | --- | --- |
| v3-primary | 234 | 80 | 154 | 34.2% |
| v3-primary-cn | 230 | 87 | 143 | 37.8% |

## 4. need 转移矩阵(v3-primary → v3-primary-cn;共 1078 条两套口径都有一致标签;只在前一套 92 条,只在后一套 72 条)

| 分组 | none→single | single→none | none→multi | multi→none | single→multi | multi→single | 不变 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 全部 | 0 | 0 | 0 | 0 | 0 | 0 | 1078 |
| source=alpaca_zh | 0 | 0 | 0 | 0 | 0 | 0 | 477 |
| source=apibank | 0 | 0 | 0 | 0 | 0 | 0 | 169 |
| source=bfcl | 0 | 0 | 0 | 0 | 0 | 0 | 432 |
| math_like=True | 0 | 0 | 0 | 0 | 0 | 0 | 119 |
| math_like=False | 0 | 0 | 0 | 0 | 0 | 0 | 959 |

## 5. C 方案(合成 + real_train)在各自留出集上的指标

| 口径 | 训练样本 | 留出集 | need 准确率 | need macro-F1 | 需要 MCP 召回 | none 召回 | 能力域 micro/macro | 动作 micro/macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v3-primary | 1143 | 234 | 84.6% | 56.4% | 88.8% | 133/154 | 64.2% / 45.7% | 70.0% / 65.2% |
| v3-primary-cn | 1127 | 230 | 87.0% | 58.4% | 93.1% | 124/143 | 61.7% / 48.2% | 68.6% / 35.1% |

各口径留出集的标签来自各自口径下的双路一致,**不是同一批标签**:这里的差别里既有标签口径的变化,也有模型本身的变化,只能当作趋势看。

## 6. multi 的标注情况(真实数据里 multi 有多少)

| 口径 | jev 判 multi | haiku 判 multi | 两路都判 multi | 一路 multi 一路 single | 按来源拆 |
| --- | --- | --- | --- | --- | --- |
| v3-primary | 26 | 18 | 11 | 21 | apibank 4、bfcl 17 |
| v3-primary-cn | 26 | 29 | 14 | 24 | apibank 6、bfcl 18 |

这一节只有真实数据自己的答案,没有掺合成数据:multi 少就是少。合成训练集里 multi 是主力之一,所以第五节里能用上 multi 的数字,并不能证明真实请求上也判得出来。
