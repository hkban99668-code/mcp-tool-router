# 真实数据重训评估(T-006)

编码器:`tfidf`;合成训练集 207 条 / 合成 gold 69 条 / real_train 680 条 / 银标留出集 170 条 / 单路数据 0 条。

> 留出集的标签来自 Jev 与 Haiku 两路一致,不是人工金标,所以下面的数字偏向这两个模型,不能当成真实水平。

> 选 A/B/C 看银标留出集和下面的交叉验证,不看合成 gold。


| 训练集 | 评估集 | n | need 准确率 | need macro-F1 | 需要 MCP 召回 | none 判对 | 能力域 micro/macro | 动作 micro/macro | Recall@1/3/5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A 只用合成训练集 | 合成 gold | 69 | 92.8% | 84.3% | 98.3% | 10/11 | 74.4% / 74.1% | 64.2% / 56.1% | 53.4% / 81.9% / 88.8% |
| A 只用合成训练集 | 银标留出集 | 170 | 19.4% | 14.6% | 87.5% | 25/154 | 6.2% / 4.8% | 32.4% / 32.4% | - |
| B 只用 real_train | 合成 gold | 69 | 33.3% | 23.0% | 25.9% | 11/11 | 7.5% / 4.5% | 9.9% / 3.6% | 11.2% / 17.2% / 24.1% |
| B 只用 real_train | 银标留出集 | 170 | 89.4% | 65.0% | 31.2% | 147/154 | 27.3% / 11.6% | 26.1% / 18.3% | - |
| C 合成 + real_train | 合成 gold | 69 | 76.8% | 50.2% | 81.0% | 11/11 | 64.9% / 61.4% | 56.9% / 46.6% | 55.2% / 76.7% / 87.1% |
| C 合成 + real_train | 银标留出集 | 170 | 88.8% | 46.9% | 50.0% | 143/154 | 25.0% / 17.2% | 38.5% / 23.0% | - |

**B 只用 real_train**:训练集缺 multi,multi 无法预测(两段式 need 头的第二段恒定输出剩下的那一类,概率 1.0)。

**A 只用合成训练集 在留出集上按来源拆分的 need 准确率**:alpaca_zh 23.3%、apibank 17.4%、bfcl 14.0%

**A 只用合成训练集 在留出集上按 math_like=math_like 拆分的 need 准确率**:13.6%(22 条);need 分布 none 21、single 1

**A 只用合成训练集 在留出集上按 math_like=other 拆分的 need 准确率**:20.3%(148 条);need 分布 none 133、single 15

**B 只用 real_train 在留出集上按来源拆分的 need 准确率**:alpaca_zh 98.9%、apibank 73.9%、bfcl 80.7%

**B 只用 real_train 在留出集上按 math_like=math_like 拆分的 need 准确率**:90.9%(22 条);need 分布 none 21、single 1

**B 只用 real_train 在留出集上按 math_like=other 拆分的 need 准确率**:89.2%(148 条);need 分布 none 133、single 15

**C 合成 + real_train 在留出集上按来源拆分的 need 准确率**:alpaca_zh 96.7%、apibank 60.9%、bfcl 87.7%

**C 合成 + real_train 在留出集上按 math_like=math_like 拆分的 need 准确率**:100.0%(22 条);need 分布 none 21、single 1

**C 合成 + real_train 在留出集上按 math_like=other 拆分的 need 准确率**:87.2%(148 条);need 分布 none 133、single 15

**C 方案在 real_train 上的交叉验证**(5 折,`random_state=0`,680 条):need 准确率 92.8% ± 0.9%,need macro-F1 76.7% ± 2.9%。
