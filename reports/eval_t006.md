# 真实数据重训评估(T-006)

编码器:`tfidf`;合成训练集 207 条 / 合成 gold 69 条 / real_train 784 条 / 银标留出集 196 条 / 单路数据 0 条。

> 留出集的标签来自 Jev 与 Haiku 两路一致,不是人工金标,所以下面的数字偏向这两个模型,不能当成真实水平。

> 选 A/B/C 看银标留出集和下面的交叉验证,不看合成 gold。


| 训练集 | 评估集 | n | need 准确率 | need macro-F1 | 需要 MCP 召回 | none 判对 | 能力域 micro/macro | 动作 micro/macro | Recall@1/3/5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A 只用合成训练集 | 合成 gold | 69 | 92.8% | 84.3% | 98.3% | 10/11 | 74.4% / 74.1% | 64.2% / 56.1% | 53.4% / 81.9% / 88.8% |
| A 只用合成训练集 | 银标留出集 | 196 | 22.4% | 18.6% | 93.3% | 27/166 | 22.9% / 25.3% | 51.3% / 44.0% | - |
| B 只用 real_train | 合成 gold | 69 | 31.9% | 22.1% | 22.4% | 11/11 | 5.2% / 3.3% | 19.0% / 10.6% | 12.1% / 21.6% / 32.8% |
| B 只用 real_train | 银标留出集 | 196 | 88.8% | 53.1% | 73.3% | 153/166 | 48.5% / 32.9% | 49.2% / 39.4% | - |
| C 合成 + real_train | 合成 gold | 69 | 81.2% | 63.8% | 84.5% | 11/11 | 64.3% / 60.7% | 53.2% / 45.4% | 47.4% / 70.7% / 81.0% |
| C 合成 + real_train | 银标留出集 | 196 | 90.8% | 54.9% | 66.7% | 158/166 | 44.4% / 43.1% | 50.0% / 51.9% | - |

**A 只用合成训练集 在留出集上按来源拆分的 need 准确率**:alpaca_zh 20.2%、apibank 34.3%、bfcl 19.4%

**A 只用合成训练集 在留出集上按 math_like=math_like 拆分的 need 准确率**:8.3%(24 条);need 分布 multi 1、none 22、single 1

**A 只用合成训练集 在留出集上按 math_like=other 拆分的 need 准确率**:24.4%(172 条);need 分布 none 144、single 28

**B 只用 real_train 在留出集上按来源拆分的 need 准确率**:alpaca_zh 100.0%、apibank 71.4%、bfcl 83.3%

**B 只用 real_train 在留出集上按 math_like=math_like 拆分的 need 准确率**:91.7%(24 条);need 分布 multi 1、none 22、single 1

**B 只用 real_train 在留出集上按 math_like=other 拆分的 need 准确率**:88.4%(172 条);need 分布 none 144、single 28

**C 合成 + real_train 在留出集上按来源拆分的 need 准确率**:alpaca_zh 100.0%、apibank 77.1%、bfcl 86.1%

**C 合成 + real_train 在留出集上按 math_like=math_like 拆分的 need 准确率**:95.8%(24 条);need 分布 multi 1、none 22、single 1

**C 合成 + real_train 在留出集上按 math_like=other 拆分的 need 准确率**:90.1%(172 条);need 分布 none 144、single 28

**C 方案在 real_train 上的交叉验证**(4 折,`random_state=0`,784 条):need 准确率 89.9% ± 0.4%,need macro-F1 53.5% ± 1.0%。
