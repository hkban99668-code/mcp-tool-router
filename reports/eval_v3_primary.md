# 真实数据重训评估(T-006)

编码器:`tfidf`;合成训练集 207 条 / 合成 gold 69 条 / real_train 936 条 / 银标留出集 234 条 / 单路数据 0 条。

> 留出集的标签来自 Jev 与 Haiku 两路一致,不是人工金标,所以下面的数字偏向这两个模型,不能当成真实水平。

> 选 A/B/C 看银标留出集和下面的交叉验证,不看合成 gold。


| 训练集 | 评估集 | n | need 准确率 | need macro-F1 | 需要 MCP 召回 | none 判对 | 能力域 micro/macro | 动作 micro/macro | Recall@1/3/5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A 只用合成训练集 | 合成 gold | 69 | 92.8% | 84.3% | 98.3% | 10/11 | 74.4% / 74.1% | 64.2% / 56.1% | 53.4% / 81.9% / 88.8% |
| A 只用合成训练集 | 银标留出集 | 234 | 26.9% | 23.4% | 92.5% | 24/154 | 11.1% / 10.9% | 63.1% / 37.1% | - |
| B 只用 real_train | 合成 gold | 69 | 43.5% | 29.2% | 41.4% | 10/11 | 21.2% / 12.1% | 28.0% / 13.2% | 11.2% / 21.6% / 28.4% |
| B 只用 real_train | 银标留出集 | 234 | 85.5% | 56.6% | 90.0% | 131/154 | 63.7% / 44.3% | 67.1% / 43.9% | - |
| C 合成 + real_train | 合成 gold | 69 | 81.2% | 53.6% | 86.2% | 11/11 | 60.9% / 57.7% | 55.3% / 42.7% | 48.3% / 68.1% / 85.3% |
| C 合成 + real_train | 银标留出集 | 234 | 84.6% | 56.4% | 88.8% | 133/154 | 64.2% / 45.7% | 70.0% / 65.2% | - |

**A 只用合成训练集 在留出集上按来源拆分的 need 准确率**:alpaca_zh 22.7%、apibank 41.5%、bfcl 24.8%

**A 只用合成训练集 在留出集上按 math_like=math_like 拆分的 need 准确率**:9.7%(31 条);need 分布 multi 1、none 24、single 6

**A 只用合成训练集 在留出集上按 math_like=other 拆分的 need 准确率**:29.6%(203 条);need 分布 multi 1、none 130、single 72

**B 只用 real_train 在留出集上按来源拆分的 need 准确率**:alpaca_zh 97.7%、apibank 80.5%、bfcl 77.1%

**B 只用 real_train 在留出集上按 math_like=math_like 拆分的 need 准确率**:90.3%(31 条);need 分布 multi 1、none 24、single 6

**B 只用 real_train 在留出集上按 math_like=other 拆分的 need 准确率**:84.7%(203 条);need 分布 multi 1、none 130、single 72

**C 合成 + real_train 在留出集上按来源拆分的 need 准确率**:alpaca_zh 97.7%、apibank 78.0%、bfcl 76.2%

**C 合成 + real_train 在留出集上按 math_like=math_like 拆分的 need 准确率**:87.1%(31 条);need 分布 multi 1、none 24、single 6

**C 合成 + real_train 在留出集上按 math_like=other 拆分的 need 准确率**:84.2%(203 条);need 分布 multi 1、none 130、single 72

**C 方案在 real_train 上的交叉验证**(5 折,`random_state=0`,936 条):need 准确率 85.5% ± 1.9%,need macro-F1 56.3% ± 1.5%。
