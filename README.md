# mcpcls:MCP 分类扩展

在原有"基于 jev 的 LLM 快速分类模型"上并联一组 MCP 分类头,对应设计文档
[MCP 分类扩展方案](https://claude.ai/code/artifact/a82f6c27-0382-41ae-bdae-bdea1139accb)。

- **请求侧(在线)**:判断一条请求要不要调用 MCP(不需要 / 单工具 / 多工具),属于哪个能力域、什么动作,再到工具目录里取 Top-k 候选。
- **工具侧(离线)**:给每个 MCP 工具归类(能力域、动作类型),推出风险等级 L0 到 L3。L3 工具调用前必须由用户确认。

原有分类头和它的输出不受影响:编码器先冻结,只训练新增的分类头。

## 现在是什么状态

> **项目总结(2026-10-16,T-001 → T-028)**:[reports/项目总结.md](reports/项目总结.md)——默认能力、可信数字、可选重排、走不通的路、剩余问题。

线上默认(`cli train` / `cli route`)跑的是**真实工具目录**这条链:分类头和检索都用 `tfidf`,
索引里拼中文描述(双语),软过滤和按函数名去重都关,判成 `none` 时用软 need 阈值 0.7 附带候选。
完整配置、每个数字的来源和一节"和既有报告对得上"的一致性核对见
[reports/eval_current_default.md](reports/eval_current_default.md)(T-021 快照,只读已落盘的 `data/`,
离线可复跑,`cli tools-real snapshot` 出这份报告)。

下表是这套默认配置在 600 条 BFCL 真实请求上的表现。开发折 = 第 0-2 折(360 条),
测试折 = 第 3-4 折(240 条),两套都报,不做选择。

| 项目 | 当前默认 | 来源 |
| --- | --- | --- |
| 工具目录 | 真实目录 `data/tools_real.jsonl`(2405 个工具) | [eval_current_default.md](reports/eval_current_default.md) §0 |
| 请求头 / 检索编码器 | `tfidf` / `tfidf`(检索编码器默认沿用分类头) | [eval_current_default.md](reports/eval_current_default.md) §0 |
| 索引文本 | 双语(英文描述 + `data/tools_real_zh.jsonl` 的中文描述) | [eval_current_default.md](reports/eval_current_default.md) §0 |
| 能力域软过滤 / 按名去重 | 关 / 关 | [eval_current_default.md](reports/eval_current_default.md) §0 |
| 置信度阈值 / top_k | `0.55` / `5` | [eval_current_default.md](reports/eval_current_default.md) §0 |
| 软 need 阈值 | `0.7`(`tools_real.SOFT_NEED_DEFAULT`) | [eval_current_default.md](reports/eval_current_default.md) §0 |
| 请求头训练数据 | 合成训练集 + `data/v3-primary/real_train.jsonl`(Jev + Haiku 两路一致) | [eval_current_default.md](reports/eval_current_default.md) §0 |
| 中文严格 R@5(主口径,开发折 / 测试折) | 37.5% / 32.0% | [eval_current_default.md](reports/eval_current_default.md) §1 |
| 中文同名 R@5(主口径,开发折 / 测试折) | 49.0% / 43.7% | [eval_current_default.md](reports/eval_current_default.md) §1 |
| 英文严格 R@5(主口径,开发折 / 测试折) | 63.8% / 57.2% | [eval_current_default.md](reports/eval_current_default.md) §1 |
| 中文严格 R@5(上限:假设 need 全对,开发折) | 51.0% | [eval_current_default.md](reports/eval_current_default.md) §2 |
| need 需要 MCP 召回 / 精确率(中文,479 条银标子集) | 80.0% / 80.9% | [eval_current_default.md](reports/eval_current_default.md) §3;[eval_t019.md](reports/eval_t019.md) 第 1 节 |
| 软 need 触发率 / alpaca_zh 误挂率(开发折) | 21.9% / 2.5% | [eval_current_default.md](reports/eval_current_default.md) §4 |
| 端到端延迟 p50 / p95(开发折,中文) | 2.5 ms / 3.4 ms | [eval_current_default.md](reports/eval_current_default.md) §5 |
| `router.pkl` 大小 | 24.3 MiB | [eval_current_default.md](reports/eval_current_default.md) §7 |
| need 需要工具召回(参考集 200 条) | 75.5%(精确率 94.9%) | [eval_reference.md](reports/eval_reference.md) §1 |
| 能力域主域准确率(参考集 need≠none 147 条,主口径) | 40.8%(旧口径取 `domains[0]` 为 30.6%;预测 domains 为空 36 条) | [eval_reference.md](reports/eval_reference.md) §3 |
| 风险 L3 判低个数(参考集 60 个工具:规则 / 风险头 / 银标) | 2 / 0 / 0(共 28 个 L3) | [eval_reference.md](reports/eval_reference.md) §5 |

> 主口径 = 走完整 `route`,need 误判计 0(软 need 附带的那批候选也算数);上限 = 假设 need 全对,纯检索天花板。
> 从 T-012 到现在,线上默认只多了一个 `soft_need_threshold=0.7`,中文主口径严格 R@5 从 26.0% 抬到 37.5%;
> 编码器(T-016/T-019)与去重(T-013)的结论都是维持 tfidf / 关,没有改默认。逐项变化见同一份报告 §8。
> 末三行来自**参考集**(口径"Claude 预标 + gu 抽查",D-005):请求 200 条、工具 60 个,只用于报告、不调参;
> 能力域主域准确率的**主口径**是"预测 `domain_probs` 最大的域 ∈ 参考 `domains`"(`domains[0]` 口径偏低,见 §3);
> 风险侧三路(规则 / 风险头 / 银标)与参考集 `gu_risk`(空则回退 `claude_risk`)对比,数字见 [eval_reference.md](reports/eval_reference.md)。

## 怎么用

```bash
pip install -r requirements.txt
python -m mcpcls.cli train                # 默认真实工具目录(2405 个工具),存到 models/
python -m mcpcls.cli route "帮我订一张明天去上海的火车票"
python -m mcpcls.cli evaluate             # 合成数据回归基线,输出 reports/eval_report.md
python -m pytest
```

`train` 的 `--catalog {real,synth}` 默认 `real`:工具取 `data/tools_real.jsonl`,
domains/actions 取两路一致银标,索引文本拼上 `data/tools_real_zh.jsonl` 的中文描述,
软过滤关(依据 [reports/eval_t011.md](reports/eval_t011.md) 第 2.2 节)。真实输出:

```
训练完成(目录 real):1143 条请求,2405 个工具;468 个工具需人工审核。模型在 models/
```

`--catalog synth` 是手写的 53 个工具,行为与 T-011 时逐字节一致。两条路都不联网,也不加示例查询。

`route` 输出里带 `catalog` 字段和 top-k 每个工具的风险等级(`final_risk`,规则与银标 actions 取更高一档):

```json
{
  "text": "帮我订一张明天去上海的火车票",
  "need": "single",
  "domains": ["calendar_task"],
  "actions": ["create"],
  "confidence": 0.6265,
  "source": "model",
  "candidates": [
    {"tool_id": "Trains/Trains_1_GetTrainTickets#7", "score": 0.1348, "risk": "L3", "domains": ["business", "weather_geo", "system_misc"]},
    {"tool_id": "Trains/Trains_1_GetTrainTickets#6", "score": 0.1348, "risk": "L3", "domains": ["business", "weather_geo", "system_misc"]}
  ],
  "requires_confirmation": true,
  "soft_candidates": false,
  "latency_ms": 9.9,
  "catalog": "real"
}
```

（上面是真实输出,候选列表裁到 2 条。`soft_candidates` 只在软 need 触发时为 `true`,
含义见下一条;`解释一下什么是梯度下降` 这类置信度高的 none 判成 `need: "none"`、候选为空。）

当 need 头把请求判成 `none`、但给 none 的概率低于软 need 阈值 0.7 时,路由照样检索并返回候选:
`need` 仍是 `"none"`、`candidates` 非空,但把 `soft_candidates` 标成 `true`,表示"这些只是候选,
要不要用由下游 agent 自己定"。例如 `你是谁`(真实输出,候选裁到 1 条):

```json
{
  "text": "你是谁",
  "need": "none",
  "domains": [],
  "actions": [],
  "confidence": 0.5304,
  "source": "model_low_conf",
  "candidates": [
    {"tool_id": "calculate/calculate.weight_in_space", "score": 0.0234, "risk": "L0", "domains": ["system_misc"]}
  ],
  "requires_confirmation": false,
  "soft_candidates": true,
  "latency_ms": 5.3,
  "catalog": "real"
}
```

想用跨语言语义编码器(可选,**默认 tfidf 不装也能跑**):`pip install -r requirements-st.txt`
(会连带装上 torch),训练时检索编码器加 `--retrieval-encoder st:BAAI/bge-m3`:

```bash
python -m mcpcls.cli train --retrieval-encoder st:BAAI/bge-m3
```

分类头编码器与检索编码器分开配、模型名不进仓库等细节见下面"接入原有的 jev"一节。

**检索重排(T-025,可选,默认关)**:召回仍是 tfidf,只对它的 Top-N 用语义模型重排并与 tfidf 分数线性融合。需要 `requirements-st.txt` 和本地模型缓存;不加 `--rerank` 时行为与以前一致。两档配置(数字来自 [reports/eval_t025.md](reports/eval_t025.md) §2 测试折,端到端主口径,严格 R@5;延迟是单请求重排 p95,CPU,工具向量已缓存,随机器负载会变):

| 档位 | 命令参数 | 中文 R@5(基线 31.5%) | 英文 R@5(基线 55.6%) | 重排 p95 | 模型磁盘 |
| --- | --- | --- | --- | --- | --- |
| 主选 | `--rerank BAAI/bge-m3 --rerank-n 50 --rerank-fusion linear --rerank-alpha 0.5` | 39.1% | 59.5% | 约 150 ms | 约 4.6 GB |
| 轻量 | `--rerank intfloat/multilingual-e5-small --rerank-n 50 --rerank-fusion linear --rerank-alpha 0.3` | 37.8% | 58.8% | 约 20 ms | 约 470 MB |

```bash
python -m mcpcls.cli train --rerank intfloat/multilingual-e5-small --rerank-n 50 --rerank-fusion linear --rerank-alpha 0.3
```

## 实验记录(T-009 → T-028)

T-009 到 T-028 每张卡一行(做了什么、结论、报告)。默认配置的变化都能在上面
[reports/eval_current_default.md](reports/eval_current_default.md) 里追到;实验分支注明"未合并、等决定"。

| 任务 | 做了什么 | 结论 | 报告 |
| --- | --- | --- | --- |
| T-009 | 标签口径 v3:need v3 说明 + taxonomy 14 域 | 合并 main(v3 + primary) | [.collab/reports/T-009.md](.collab/reports/T-009.md);[reports/label_agreement_compare.md](reports/label_agreement_compare.md) |
| T-010 | 工具侧补课 A:真实工具目录 + 请求→工具对应(不调 API) | 合并 main | [.collab/reports/T-010.md](.collab/reports/T-010.md);[reports/real_tools_stats.md](reports/real_tools_stats.md) |
| T-010a | 双路一致按主域判定 + 重训缺类不崩 | 随 T-009 分支一起合并 | [.collab/reports/T-010a.md](.collab/reports/T-010a.md);[reports/eval_v3_exact.md](reports/eval_v3_exact.md) |
| T-010b | 工具侧补课 B:真实工具双路打标 + 风险复验 + 真实 Recall@k | 合并 main | [.collab/reports/T-010b.md](.collab/reports/T-010b.md);[reports/eval_t010.md](reports/eval_t010.md) |
| T-011 | 双语索引 + 软过滤选参 + 风险规则补漏 + 真实数据工具头 | 合并 main;双语索引把中文严格 R@5 从 10.2% 抬到 48.4%,软过滤默认关 | [.collab/reports/T-011.md](.collab/reports/T-011.md);[reports/eval_t011.md](reports/eval_t011.md) |
| T-012 | 真实工具目录设为线上默认 + 端到端评估 | 合并 main | [.collab/reports/T-012.md](.collab/reports/T-012.md);[reports/eval_e2e.md](reports/eval_e2e.md) |
| T-013 | 检索结果按函数名去重 | 合并 main;去重抬高同名口径但严格口径下降,默认**关** | [.collab/reports/T-013.md](.collab/reports/T-013.md);[reports/eval_t013.md](reports/eval_t013.md) |
| T-014 | need 口径实验 v3b:精确计算算需要工具 | **未合并、等决定**;一致率下降,只有 11 条稳定改判 | [.collab/reviews/T-014.md](.collab/reviews/T-014.md) |
| T-015 / T-018 | taxonomy v4 实验:拆分 system_misc | **不合并、保留方案**(D-006);v4 让软过滤更亏(开发折 R@5 48.9%→26.8%),端到端与 v3 持平,默认保持 v3,`MCPCLS_TAXONOMY=v4` 可切换 | [.collab/reviews/T-018.md](.collab/reviews/T-018.md) |
| T-016 | 多语言语义编码器(bge-m3 / e5-small) | 合并 main;作为**可选**,默认仍 tfidf(语义编码器会拖累 need/能力域头) | [.collab/reports/T-016.md](.collab/reports/T-016.md);[reports/eval_t016.md](reports/eval_t016.md) |
| T-017 | 第二路打标换成国产模型(选中小样本最优的 DeepSeek v4.1-flash) | 合并 main;主指标与 Haiku 参考行无法区分 | [.collab/reports/T-017.md](.collab/reports/T-017.md);[reports/eval_t017.md](reports/eval_t017.md) |
| T-019 | 检索编码器与分类头编码器分开配置 + 部署瘦身 | 合并 main;没有检索编码器同时满足门槛,默认检索编码器仍 tfidf | [.collab/reports/T-019.md](.collab/reports/T-019.md);[reports/eval_t019.md](reports/eval_t019.md) |
| T-020 | 低置信度的 none 也附带候选(软 need) | 合并 main;默认软 need 阈值 0.7(开发折选、测试折复核) | [.collab/reports/T-020.md](.collab/reports/T-020.md);[reports/eval_t020.md](reports/eval_t020.md) |
| T-021 | 当前默认配置端到端快照 + README 整理 | 合并 main;请求头训练数据维持 v3-primary | [.collab/reports/T-021.md](.collab/reports/T-021.md);[reports/eval_current_default.md](reports/eval_current_default.md) |
| T-022 | 用定稿参考集评估当前默认(只报告) | 合并 main;need 召回 75.5%、主域 40.8%、风险规则漏判 L3 2 个 | [.collab/reports/T-022.md](.collab/reports/T-022.md);[reports/eval_reference.md](reports/eval_reference.md) |
| T-023 | 风险规则补两类信号:扫描外部主机、控制实体设备 | 合并 main;测试份 L3 召回 +4.5~7.1 点、多打扰 +0,参考集漏判 2→0 | [.collab/reports/T-023.md](.collab/reports/T-023.md) |
| T-024 | need 漏判诊断与改进 | 合并 main,不改默认;bge-m3 need 头端到端英文 -18 点,阈值 0.55 无变化(软 need 已覆盖) | [.collab/reports/T-024.md](.collab/reports/T-024.md);[reports/eval_t024.md](reports/eval_t024.md) |
| T-025 | 检索两段式:tfidf 召回 Top-N + 语义重排(分数融合) | 合并 main,**可选、默认关**(D-007);测试折中文严格 R@5 31.5%→39.1%(bge-m3)/37.8%(e5-small),英文 +3~4 点 | [.collab/reports/T-025.md](.collab/reports/T-025.md);[reports/eval_t025.md](reports/eval_t025.md) |
| T-026 | 混合召回:tfidf Top-50 ∪ 语义 Top-M,再融合重排 | 合并 main,无收益;召回天花板中文 +11.3 点,但兑现到开发折仅比 T-025 +1.9 点(<2 点门槛),`--rerank-recall-m` 默认 0 | [.collab/reviews/T-026.md](.collab/reviews/T-026.md);[reports/eval_t026.md](reports/eval_t026.md) |
| T-027 | 能力域预测:分类头 + 检索 Top-k 工具域投票 | 合并 main(仅离线报告),不改默认;开发折用 BFCL 工具域当标签与投票同源(循环),参考集请求域上 -8.2 点 | [.collab/reviews/T-027.md](.collab/reviews/T-027.md);[reports/eval_t027.md](reports/eval_t027.md) |
| T-028 | 能力域投票改用请求级银标(real_train 319 条 5 折折外)重评 | 合并 main(仅离线报告),不改默认;融合只比分类头 +3.4 点(<4 点门槛、CI 重叠),纯投票 38.9% 远低于分类头 62.1%,投票线关闭 | [.collab/reviews/T-028.md](.collab/reviews/T-028.md);[reports/eval_t028.md](reports/eval_t028.md) |

## 已知局限

- **银标不是人工标注**:工具 domains/actions/风险都是 Jev 与 Haiku 两路一致的结果,偏向这两个模型;两路没谈拢的 721 个工具 domains 直接用了 Jev 的,风险可能偏低。
- **参考集已定稿(D-005)**:200 条金标候选(`data/gold_candidates_claude.csv`)和 60 行风险抽检(`data/tools_real_risk_audit_claude.csv`)是 Claude **预标**、gu 抽查通过的参考集(口径写作"Claude 预标 + gu 抽查",不是纯人工金标),说明见 [data/claude_prelabel_说明.md](data/claude_prelabel_说明.md)。现在能给风险规则和 need 判定一个不循环的评估,结果见 [reports/eval_reference.md](reports/eval_reference.md)(`cli tools-real reference-eval`,只读已落盘文件);参考集只用于报告,不参与调参。
- **风险银标偏保守**(silver label 宁可高报,会拉低 L3 精确率),真实风险分布要等人工抽检。
- **need 口径待定**:"精确计算类请求要不要工具"这条边界,Jev 与 Haiku 自己判得就不稳(T-014 里 v3b 只稳定改判 11 条)。线上默认仍是 v3 口径。
- **`system_misc` 偏大**:2405 个工具里主域是 `system_misc` 的有 1480 个;T-015/T-018 的 v4 能拆到 295,但软过滤更亏、端到端持平,按 D-006 默认保持 v3。
- **编码器是 TF-IDF**:中英文检索的差距首先归编码器(word 特征只吃 ASCII 词,中文只剩字符 n-gram);换 sentence-transformers 之后差距会小很多,但它拖累 need/能力域头,只能单独用在检索侧(T-016/T-019),默认没开。T-025 改成"tfidf 召回 + 语义重排"后中英都涨,作为可选两档写在"怎么用"里,默认仍关。

## 接入原有的 jev

编码器通过 `--encoder` 指定:

| 写法 | 含义 |
| --- | --- |
| `tfidf` | 默认值。字符 n-gram 加英文词,不用下载模型,适合跑基线 |
| `st:BAAI/bge-m3` | 跨语言语义编码器(推荐)。任意 sentence-transformers 模型都行 |
| `st:intfloat/multilingual-e5-small` | 轻量跨语言语义编码器(权重约 0.5 GB) |
| `jev:你的模块:函数` | 原方案的 jev。函数签名为 `f(list[str]) -> np.ndarray` |

```bash
python -m mcpcls.cli evaluate --encoder jev:my_project.jev:encode
```

`st:` 编码器要装 `sentence-transformers`。它不在默认的 `requirements.txt` 里,单独放在可选文件 `requirements-st.txt`(`pip install -r requirements-st.txt`,会连带装上 torch);默认的 tfidf 不装也能跑。模型权重**不进仓库**,首次用到时从 HuggingFace 下载到 `~/.cache/huggingface`。三种编码器(默认 tfidf、bge-m3、e5-small)在真实目录上的对照见 [reports/eval_t016.md](reports/eval_t016.md)。

**分类头编码器和检索编码器可以分开配**:T-016 发现语义编码器的 need / 能力域头会把大量请求判成 none,走完整路由反而退步,所以它只适合用在检索上。`--encoder` 仍只管分类头(need / 能力域 / 动作),`--retrieval-encoder` 管检索索引和查询(如 `st:BAAI/bge-m3`),两者默认相同、行为与改前一致:

```bash
python -m mcpcls.cli train --encoder tfidf --retrieval-encoder st:BAAI/bge-m3
python -m mcpcls.cli tools-real e2e --retrieval-encoder st:BAAI/bge-m3
```

对照见 [reports/eval_t019.md](reports/eval_t019.md)。带语义编码器的 router 存盘(`models/router.pkl`)时**只存模型名、不存权重**,加载时从 HuggingFace 缓存重建,部署包不会因此变成 GB 级。

原有分类函数通过 `Router(original_head=...)` 传进来,它的结果会原样放进 `RouteResult.original`。这样一次请求只调用一次路由,就能同时拿到原有结果和 MCP 结果。

## 软 need(低置信度的 none 也附带候选)

need 头把请求判成 `none` 时,路由默认不返回候选。`Router` 有这个可选开关 `soft_need_threshold`(类默认 `None` = 行为完全不变);**真实目录这条链**(`build_real_router`,也就是 `cli train` 默认的 `--catalog real`)把它设成 `tools_real.SOFT_NEED_DEFAULT = 0.7`。设了值时,若 `need == "none"` 且 need 头给 none 的概率**低于**该阈值,仍照常检索并返回 top_k 个候选,`need` 字段保持 `"none"`,只把 `RouteResult.soft_candidates` 标成 `True`,由下游 agent 自己决定用不用这些候选。

```python
Router(request_head, catalog, soft_need_threshold=0.7)
python -m mcpcls.cli tools-real soft-need-eval       # 阈值网格评估 -> reports/eval_t020.md
```

开发折上阈值 0.7 把中文主口径严格 R@5 从 26.0% 抬到 37.5%;两路一致判成 none 的 alpaca_zh 请求误挂率 2.5%,平均每条请求多 0.5 ms。阈值网格、误挂率、选择规则和测试折复核见 [reports/eval_t020.md](reports/eval_t020.md)。**T-020 已把真实目录这条链的默认设为 0.7**(开发折选、测试折复核);合成目录(`--catalog synth`)仍是关。快照里的触发率 / 误挂率见 [reports/eval_current_default.md](reports/eval_current_default.md) 第 4 节。

## Jev 兜底(低置信度时问云端决策模型)

Jev(TypeSafe AI)不输出向量,所以它不是编码器:`--encoder jev:...` 那条路还留着,但 Jev 真正的用法是**当兜底分类器**。本地模型 1~2 ms,金标上约 17% 的请求低于置信度阈值,只有这部分去问 Jev。

- 一次请求问的题数跟着 taxonomy 走:1 个 `need` 选择题 + 14 个能力域 + 6 个动作是非题(T-009 起 21 题,之前是 12 域 19 题)。问题的说明文字全部从 `taxonomy.py` 的类别定义生成,不另写一份,免得两边说法不一致。
- `need` 那一题有三个版本:`--need-version v1`(T-004 原文)、`v2`(T-005,默认)、`v3`(T-009,把"实时/最新信息需要工具"和"数学计算不需要工具"写进说明)。
- 结果来源标 `jev_fallback`;Jev 超时或报错时保留本地结果、标 `model_low_conf`,路由不抛异常(默认超时 15 秒,重试 1 次)。
- 默认走 commandcode 网关(`https://api.commandcode.ai/provider/v1/systemone`,模型 `typesafe/jev`),需要 `JEV_API_KEY`。换服务商用 `JEV_BASE_URL`(TypeSafe 原生接口的根路径,请求发到 `<根>/systemone`;OpenRouter 是 `https://openrouter.ai/api/v1`)和 `JEV_MODEL`。key 依次从 `JevClient(api_key=...)`、`JEV_API_KEY`、`OPENROUTER_API_KEY` 读,异常信息与调用记录里都不会出现它。
- 缓存按模型分开:旧的 `typesafe/jev-1.13` 答案在 `.cache/answers/jev-*`,commandcode 的 `typesafe/jev` 在 `jev_at_typesafe-jev-*`。

```bash
python -m mcpcls.cli route "给张三发封邮件说合同签了" --jev-fallback   # 单条请求
python -m mcpcls.cli evaluate --live-jev                              # 金标上跑两组对照
```

`--live-jev` 在金标上再跑两组:(a) 本地模型 + Jev 兜底,(b) 全部交给 Jev,和本地基线并排写进 `reports/eval_jev.md` 与 `reports/eval_jev.json`;`reports/eval_report.md` 不动。不加这个开关时完全不联网。环境变量里没有 key 时打印一行中文提示,退出码 2。

**两个兜底比例不是一回事**:`eval_report.md` 里的「LLM 兜底比例(低于阈值)」是按置信度算的**估计值**,不接 Jev 时没有任何实际调用;`eval_jev.md` 里的调用次数、失败次数、延迟、费用都来自每次调用的实际记录(`JevCall`)。Jev 失败时该条回退到本地预测,所以实际调用次数可能多于结果被改写的条数。

## 兜底策略与离线对比

真实 Jev 评估里,本地模型和 Jev 错的方向是互补的(本地需要 MCP 召回 98.3% 但 11 条 none 只对 4 条,Jev 相反),而"低置信度就整体替换"反而拉低了 need 准确率。所以合成方式被拆成可配置的策略和触发方式,线上路由和离线对比用同一个纯函数 `strategy.apply_strategy`,两边结果不可能对不上。

| 策略 `STRATEGIES` | 含义 |
| --- | --- |
| `local` | 恒用本地结果(基线) |
| `replace` | 云端答案整体替换(T-004 的行为,默认) |
| `need_only` | 只换 need;云端判 none 时清空域与动作,否则域与动作用本地的 |
| `both_none` | 两边都判 none 才算 none;域与动作规则同 `need_only` |

触发方式 `TRIGGERS`:`low_conf` 只有本地置信度低于阈值才问(默认),`always` 每条都问。

```bash
python -m mcpcls.cli route "给张三发封邮件" --fallback-client haiku --fallback-strategy both_none \
  --fallback-trigger low_conf --need-version v2          # 需要 JEV_API_KEY
python -m mcpcls.cli compare --dataset gold --clients jev,haiku --need-versions v1,v2
python -m mcpcls.cli compare --dataset train --clients jev,haiku --need-versions v1,v2 --live
```

`compare` 输出全组合对比表到 `reports/compare_<dataset>.md` / `.json`(每个组合 = 客户端 × 说明版本 × 策略 × 触发,外加一行纯本地基线)。**联网取答案和算指标是分开的**:`--live` 才联网,每个(客户端, 说明版本, 数据集)只问一次并存进 `.cache/answers/`;不加 `--live` 时只读缓存,缺答案的条目直接退回本地结果。所以问一次之后,所有策略都能离线反复算。

`--dataset train` 的本地结果用 5 折交叉验证的折外预测(和 `evaluate` 同一份分折),`--dataset gold` 用全部训练集训练后预测。挑策略和说明版本看 train,gold 只报告。

## 真实数据流水线(T-006)

现有的 207 条训练 + 69 条 gold 都是同一作者手写的合成数据,本地模型等于开卷考试。
这一节把公开数据集里的**真实用户请求**拉进来:Jev 与 Haiku 两路离线打标,
一致的进训练集,不一致的进人工审核队列,同时给人工准备一份 200 条的金标候选表。

```bash
.venv/bin/python scripts/fetch_public.py          # 下载三个公开数据集到 data/raw/,写 data/SOURCES.md
.venv/bin/python scripts/build_pool.py            # 取样 + 清洗 + 去重 -> data/pool.jsonl(1400 条,固定种子)
python -m mcpcls.cli realdata translate   --live  # 英文翻成中文口语请求 -> data/pool_zh.jsonl
python -m mcpcls.cli realdata label       --live  # 双路打标 -> labeled_agree.jsonl + review_queue.jsonl
python -m mcpcls.cli realdata label --single --labelers codebuddy --live  # 单路 -> labeled_single.jsonl
python -m mcpcls.cli realdata split               # 一致集按 need 分层 -> real_train.jsonl + real_holdout.jsonl
python -m mcpcls.cli realdata candidates          # -> data/gold_candidates.csv(200 行,带 BOM,Excel 直接开)
python -m mcpcls.cli evaluate-real                # 三种训练集 × 两个评估集(有单路数据再加 D) -> reports/eval_t006.md
```

数据来源(BFCL v3 Apache-2.0、API-Bank MIT、alpaca-zh CC-BY-4.0)的 URL、许可、sha256
与署名都记在 `data/SOURCES.md`;`data/raw/` 不进版本库。

几条规定,和钱有关也和结论的可靠性有关:

- **每一步都落盘、都走缓存**。缓存就是 `.cache/answers/` 里那份,和 `compare` 共用一套
  (按客户端 × 说明版本 × 数据集分文件,按文本去重),中断后重跑只补没做完的。
- **所有联网步骤共用一个累计预算** `--budget-usd 2.0`,落盘在 `data/realdata_cost.json`。
  下一次调用前发现累计已超,打印一行提示并退出码 3,已完成的答案都留在缓存里。
- **两路可替换**:`--labelers jev,haiku` 与 `--translator haiku` 都走 `jev.make_client`。
  某一路构造失败(没 key、账号没钱、命令不在 PATH)或连续失败(jev / haiku 连续 5 条,
  codebuddy 连续 3 整批)立刻停下并退出码 4。
  2026-10-09 的情况是 OpenRouter credits 为 0,Haiku 返回 402,所以第二路换成本机命令行
  `--labelers jev,codebuddy` / `--translator codebuddy`(T-007)。
- **限流与欠费立刻停**:HTTP 402(PaymentRequired)、HTTP 429(RateLimited)和 codebuddy
  命令行的"stdout 空 + stderr 以 429 开头"都不重试、不等连续计数,`stats.stopped` 写明
  "限流,<重置时间>"或"OpenRouter 余额不足(402)",退出码 4。
- **codebuddy 这一路不联网也不花钱**:跑的是本机的 `codebuddy -p`,子进程环境里删掉了
  `OPENROUTER_API_KEY`。一次调用 30~50 秒、和条数关系不大,所以它走批量接口:
  一次问 20 条(题目说明只渲染一次,按编号对回每条答案),批与批之间才用 `--workers` 并发。
  整批失败重试一次,仍失败则这批每条都算失败(连续失败按批计,批内部分失败不累加);
  某一解析不出来时只有那一条算失败。默认模型限流后可以用 `--codebuddy-model` 换模型,
  指定模型的缓存单独一份(`codebuddy@<模型>`,文件名里 `@` 写成 `_at_`),默认模型的缓存
  文件名不变。走 `jev` / `haiku` 的客户端一条一次调用,行为和以前完全一样。
- **第一路缺答案的条目单路补**:`realdata label --single --labelers <一路>` 只用一路打标,
  结果写到 `labeled_single.jsonl`(每行 `single_source: true`),不进 labeled_agree /
  real_train / holdout;双路模式下终端会单独打印"第一路缺答案"的条数。
  `evaluate-real` 有这份数据时多报一组 D(合成 + real_train + 单路),只在银标留出集上和 C 对比。
- **失败不会被吞掉**:统计失败条数,终端和报告都打印"失败 N 条,第一条错误:<脱敏后的错误>"
  (T-005 里 Haiku 的 402 被整批吞掉、报告只显示"调用 0%")。
- **留出集不是金标**:`real_holdout.jsonl` 的标签来自两路一致,会偏向这两个模型,
  `reports/eval_t006.md` 里写明这一点;选 A/B/C 训练集看它和 real_train 上的交叉验证,不看合成 gold。
- **BFCL 里像算学/计算的那部分单独报**:`scripts/build_pool.py` 给 BFCL 的 `hint` 打一个
  `math_like`(函数名或描述里出现 area / calculate / calc / math / equation / solve / convert /
  derivative / integral / factorial / prime / gcd / lcm / sqrt,1400 条里 138 条),
  打标统计和 `evaluate-real` 都按它分开报一致率与 need 分布。`hint` 只用于分析,不当标签。

### 双路一致的三种判定口径(T-010a)

`realdata label --agreement {exact,primary,primary_sym}`,默认 `primary`:

| 口径 | 判定 | 一致时的标签 domains |
| --- | --- | --- |
| `exact`(旧行为) | need 相同,且 need 非 none 时两路的域**集合完全相同** | 第一路的整份 domains |
| `primary`(默认) | need 相同;两路都 none;或者 need 非 none 时**第一路概率最高的那个域**(主域)落在第二路的域里 | `[主域]` |
| `primary_sym` | 上面那条,**或者**第二路的主域落在第一路的域里 | 命中的那个主域(两个都命中取第一路的) |

第一路就是 `--labelers` 里的第一个。Jev 一路多选偏多(v3 平均 2.86 个域 vs Haiku 1.16 个),
要求两个集合完全相同会把大量"need 其实一致、只是一个多选了"的条目打进人工队列。
actions 的合并规则(交集,空则取第一路)三种口径都一样。待审队列每行带着
`agreement_rule` 与 `primary_domain`,换口径重算后能看出这行是按哪条规则判的。

缓存里有答案就能换口径重算,不联网:`realdata label`(不加 `--live`)只读缓存。
训练集缺某个 need 类(比如一致集里一条 multi 都没有)时 `evaluate-real` 不再崩:
两段式的第二段恒定输出剩下的那一类,报告里写一行"训练集缺 multi,multi 无法预测"。

### 多套口径并排对照

一套口径一个目录(里面四份同名 jsonl),`compare-labels` 一次可以比多套:

```bash
python -m mcpcls.cli compare-labels --old data=v2-exact --new data/v3=v3-exact \
  --extra data/v2-primary=v2-primary,data/v3-primary=v3-primary \
  --report-md reports/label_agreement_compare.md --report-json reports/label_agreement_compare.json
```

`目录=标签` 里的标签就是报告里的行名(省略则用目录名)。报告开篇是一张总表
(一致率、需要工具的样本数、两个新域的样本数、留出集指标),后面六节逐个展开,
第六节专门报 multi 在两路各自的标注数量与一路 multi 一路 single 的条数。

## 真实工具目录(T-010)

工具侧此前的评估都建立在 `data/tools.jsonl` 的 53 个手写工具上。为了在真实数据上测检索,
`scripts/build_real_tools.py` 从 BFCL 每条请求自带的函数定义里抽出一份真实工具目录(离线,不调模型):

```
python scripts/build_real_tools.py
```

产出三样东西:

| 文件 | 内容 |
| --- | --- |
| `data/tools_real.jsonl` | 2405 个去重后的工具(BFCL 里出现 6226 次),字段与 `data/tools.jsonl` 对齐;`server` 由函数名推断(`.` 或 `_数字_` 分隔,都没有则 `misc`),`inputSchema` 原样保留 |
| `data/requests_real_tool_ids.jsonl` | 池子里 600 条 BFCL 请求 → 应调用的 tool_id(来自 BFCL 的 possible_answer)与它自带的候选集 tool_id |
| `reports/real_tools_stats.md` | 统计:去重前后数量、按文件/按 server 的分布、同名冲突、候选集大小、描述长度、风险词粗计数 |

同名但描述或参数不同的当成不同工具,`tool_id` 依次加 `#2`、`#3`(1287 个名字里有 433 个是这样的)。
这份目录**没有** `domains` / `actions` / 风险标签——打标属于 B 部分,还没做。

数据源与许可记在 [data/SOURCES.md](data/SOURCES.md) 的"工具目录"一节。

## 真实工具目录修复(T-011)

T-010b 的评估发现真中文查询对英文工具描述几乎检索不到。T-011 在 `tools-real` 子命令上补了四件事,
细节与改前改后对比见 [reports/eval_t011.md](reports/eval_t011.md)。

```bash
python -m mcpcls.cli tools-real translate --live   # 联网把描述翻成中文,缓存 tools_real_desc/zh
python -m mcpcls.cli tools-real translate          # 再跑一次:调用 0 次,输出逐字节相同
python -m mcpcls.cli tools-real evaluate --bilingual
```

| 开关 | 作用 |
| --- | --- |
| `tools-real translate [--live]` | 用 Haiku 把 `data/tools_real.jsonl` 的 description 翻成中文,写 `data/tools_real_zh.jsonl`;函数名/参数名/代码标识符原样 |
| `tools-real evaluate --bilingual` | 检索索引文本改成"名字 + 英文描述 + 中文描述",只在这次评估里生效,`catalog.py` 默认的 `_index_text` 不变 |
| `tools-real evaluate --grid-search` | 在开发集上跑软过滤网格选参(仅供报告,不改 `catalog.retrieve` 的默认参数) |
| `tools-real evaluate --tool-head {synth,real,both}` | 工具侧分类头的训练数据:合成 / 真实银标 / 两者(默认 `synth`,另两种是分支实验) |

四个结论:双语索引把中文严格 R@5 从 10.2% 提到 48.4%(英文不退步);真实目录下软过滤不划算
(开发集开/关无法区分,测试集过滤关更高);风险规则按开发份补词后,L3 召回从 22.5% 提到 55.0%
(合成 `evaluate` 一条不动);用真实数据训练工具头能力域 F1 大涨,但 L3 召回低于 100% 且低报率上升,
只能放在 `--tool-head` 开关后面。

## 历史指标(改前改后对照)

**当前**默认配置的数字在开头的"现在是什么状态"一节(来源 [reports/eval_current_default.md](reports/eval_current_default.md));
这一节保留的是各次改动**当时**的改前 / 改后对照,每行注明属于哪个任务,不要和最新数字混着看。

| 指标 | 数值 | 来源 |
| --- | --- | --- |
| 工具双路打标一致率(主域口径) | 1684 / 2405(70.0%) | [reports/eval_t010.md](reports/eval_t010.md) 第 1 节 |
| 中文严格 R@5(单语 → 双语) | 10.2% → 48.4% | [reports/eval_t011.md](reports/eval_t011.md) 第 1 节 |
| 英文严格 R@5(单语 → 双语) | 64.8% → 64.7%(无法区分) | [reports/eval_t011.md](reports/eval_t011.md) 第 1 节 |
| 风险规则 L3 召回(改前 → 改后,按函数名测试份) | 22.5% → 55.0% | [reports/eval_t011.md](reports/eval_t011.md) 第 3.2 节 |
| 端到端 need 召回 / 精确率(中文,T-012 时) | 84.2% / 79.6% | [reports/eval_e2e.md](reports/eval_e2e.md) 第 1 节 |
| 端到端中文严格 R@5(T-012 时,need 误判计 0) | 23.7% | [reports/eval_e2e.md](reports/eval_e2e.md) 第 2 节 |
| 上限:假设 need 全对的中文严格 R@5(T-012 时) | 48.4% | [reports/eval_e2e.md](reports/eval_e2e.md) 第 2 节 |
| 按函数名去重:开发折中文严格 R@5(关 → 开,假设 need 全对) | 51.0% → 43.9%(更差,不设默认) | [reports/eval_t013.md](reports/eval_t013.md) 第 1 节 |

> T-012 那两份 e2e 报告里的 need 列读的是**合成目录**那一行的请求头(`e2e_need_metrics` 取 `catalogs[0]`),
> 和线上真实目录头不是同一个,所以 84.2% 与快照的 80.0% 不能直接比,详见 [reports/eval_current_default.md](reports/eval_current_default.md) 第 9 节的 need 口径说明。

`cli tools-real e2e` 出端到端报告:600 条 BFCL 请求走完整 `Router.route`,请求头按 5 折折外训练,主口径分母是全部请求(判成 none 的按 0 计),另给一组"假设 need 全对"的检索上限。三行是合成目录、真实目录单语索引、真实目录双语索引。

`cli tools-real dedupe-eval` 报按函数名去重的结果(去重关 / 开 × 开发折 / 测试折 × 中英文 × 严格 / 同名 × R@1 / R@3 / R@5):真实目录里同名函数最多有 48 个变体,去重能把同名口径的 R@5 明显抬高(开发折中文上限 63.5% → 72.5%),但金标偏偏是低分变体时会被同名的更高分变体顶掉,严格口径反而下降,所以**真实目录默认不去重**(`Router.dedupe_by_name=False`,依据 [reports/eval_t013.md](reports/eval_t013.md) 第 0 节)。

`cli tools-real encoder-eval` 比三种编码器(默认 tfidf / `st:BAAI/bge-m3` / `st:intfloat/multilingual-e5-small`)× 单语 / 双语索引 × 中英文 × 严格 / 同名 × R@1 / R@5,并报端到端延迟 p50/p95 与 `router.pkl` 大小;开发折(第 0-2 折)决定是否改默认,测试折(第 3-4 折)只报一次。结果见 [reports/eval_t016.md](reports/eval_t016.md)。

合成数据上的 `cli evaluate`(`reports/eval_report.md`,工具侧 L3 召回 100%、低报率 1.9%)是**回归基线**:它用 53 个手写工具和 207 条手写请求,等于开卷考试,只用来保证代码改动没有破坏既有行为,不代表真实分布。

## 用真实数据替换种子数据

`data/` 里的数据是手写的合成种子数据,只用来跑通流程,不代表真实分布。换成真实数据分三步:

1. **拉取工具列表**:`python -m mcpcls.cli fetch-tools --config mcp.json --out data/tools_raw.jsonl`,需要先 `pip install mcp`。配置格式与常见 MCP 客户端的 `mcpServers` 相同。
2. **LLM 双路打标**:`python -m mcpcls.cli label --kind tool --in data/tools_raw.jsonl --out data/tools.jsonl`,需要 `ANTHROPIC_API_KEY`。两路结果一致的直接入库,不一致的写进 `*.review.jsonl`,交给人工审核。请求日志用 `--kind request` 打标。
3. **人工标金标集**:金标集只用于评估和校准阈值,不参与训练。

## 代码结构

| 文件 | 作用 |
| --- | --- |
| `mcpcls/taxonomy.py` | 三轴类别:14 个能力域(T-009 加了 `weather_geo` / `entertainment`)、6 种动作、4 个风险等级,以及给 LLM 打标用的类别定义卡 |
| `mcpcls/rules.py` | 规则信号:工具名动词、参数名、annotations、能力域关键词、请求侧多步骤连接词 |
| `mcpcls/encoder.py` | 编码器接口:TF-IDF、sentence-transformers、jev |
| `mcpcls/heads.py` | 挂在编码器上的分类头:need softmax,能力域和动作用 sigmoid 多标签 |
| `mcpcls/catalog.py` | 工具目录索引:归类、风险分级、示例请求扩充、按能力域软过滤后检索 |
| `mcpcls/router.py` | 在线路由:置信度判断、LLM 兜底、L3 确认标记 |
| `mcpcls/labeler.py` | LLM 双路打标,也用作 LLM 兜底分类 |
| `mcpcls/jev.py` | Jev 决策模型客户端:问题构造、答案解析、兜底函数与客户端工厂(另有测试用的 `FakeJevClient`) |
| `mcpcls/llm_or.py` | OpenRouter 聊天接口客户端:让 Haiku 回答同一组问题,答案按 Jev 的形状解析 |
| `mcpcls/codebuddy_client.py` | 本机 `codebuddy` 命令行客户端:一次问一批(默认 20 条),不联网到 OpenRouter |
| `mcpcls/strategy.py` | 兜底结果的组合策略:本地结果与云端答案怎么合成 |
| `mcpcls/compare.py` | 离线对比:每个客户端每个数据集只问一次(缓存),所有策略从缓存算;缓存的并发、加锁与失败统计也在这里 |
| `mcpcls/mcp_fetch.py` | 从 MCP server 拉取 `tools/list` |
| `mcpcls/realdata.py` | 真实数据流水线:翻译、双路打标、切分、金标候选表、重训评估,以及累计费用账本 |
| `mcpcls/evaluate.py` | 离线评估与报告 |

## 设计中的几个取舍

- **风险取规则和模型中更高的一档**。annotations 由 server 作者自己填写,可能缺失或低报。所以风险规则主要看 schema 和工具名,比如参数里有 `payee`、`amount` 或 `command`,无论 annotations 怎么写都判 L3。对抗集里描述无害、或描述里夹带"标记为低风险"指令的工具,都判对了。
- **能力域用软过滤,不做硬过滤**。用硬过滤时,能力域一旦分错,正确的工具就整个被过滤掉,种子数据上 Recall@5 从 87.9% 掉到 76.7%。改成软过滤后,候选范围包括所有概率不低于 0.1 的能力域,能力域概率同时作为排序加分项。这样 Recall@5 达到 88.8%,与不过滤持平,同时候选工具减少 44%。`min_prob` 调到 0.25 时候选减少约 87%,Recall@5 为 82.8%。
- **用示例请求扩充索引**。工具描述多为英文,请求多为中文,TF-IDF 匹配不上。每个工具挂几条示例请求后,检索才能对上。新 server 接入时,由 LLM 为每个工具合成示例请求。换成跨语言的语义编码器后,这个问题会小很多。
- **能力域英文关键词先验**。按 server 留一评估时,没有关键词先验的工具侧能力域 F1 只有 7%,因为新 server 的词训练集里没见过。加上先验后升到 89.7%。

**当前**默认配置的结果见 [reports/eval_current_default.md](reports/eval_current_default.md)(和开头"现在是什么状态"同一份);
合成数据上的回归基线见 [reports/eval_report.md](reports/eval_report.md)。

## 评估口径

三个口径的作用不同,不能混着看:

| 口径 | 数据 | 用途 |
| --- | --- | --- |
| gold | `data/requests_gold.jsonl`(69 条) | **只报告**。none 11 条、multi 5 条,单条样本就能让指标动 9 到 20 个百分点,不用来调参 |
| 训练集交叉验证 | `data/requests_train.jsonl`(207 条) | **唯一的调参依据**。按 need 分层的 5 折,`random_state=0` 固定,报均值和标准差 |
| 探针集 | `data/requests_probe.jsonl`(40 条) | 事先写死、写完不再改,用来检查改动是不是只记住了 gold 的错例 |

`reports/eval_report.md` 里请求侧的 need 指标就是这三列并排,另外带 gold 的 3×3 混淆矩阵。判断"某个改动有没有用"要看交叉验证那一列的均值提升是否超过它自己的标准差,以及需要 MCP 的召回有没有下降。

**规则词表的来源**:`rules.py` 里两个意图词表的每个词都标了来源(`KEYWORD_SOURCES`),只允许三种:`train`(训练集里出现过)、`taxonomy`(来自 `taxonomy.py` 的类别定义卡)、`prior`(凭通用语言常识写的词)。其中有 13 个词在训练集里出现 0 次、只在 gold 出现(`GOLD_ONLY_KEYWORDS`),交叉验证的任何一折都触发不了它们,所以 **`evaluate` 默认以"去掉这 13 个词"的口径作为主数字**,含词口径在同一张表里并列——两列的差就是这些词带来的全部增益。来源标注是否自洽、gold-only 集合是否变化,由 `keyword_leak_check` 和 `tests/test_mcpcls.py` 里的两条测试逐词核对。

```bash
python -m mcpcls.cli evaluate                      # 默认:need 规则特征开、两级判定开
python -m mcpcls.cli evaluate --no-need-rules      # 对照:关掉"无外部对象"规则特征
python -m mcpcls.cli evaluate --no-two-stage       # 对照:关掉两级判定
python -m mcpcls.cli evaluate --keep-gold-only-words  # 主口径改回含那 13 个 gold-only 词
python -m mcpcls.cli evaluate --folds 10 --cv-seed 1
```

`data/requests_probe.jsonl` 是冻结文件:写完单独提交,后续任何提交都不得修改它(`git log --follow` 应当只有一条记录)。

