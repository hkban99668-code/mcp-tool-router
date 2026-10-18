# MCP 分类评估报告

编码器:`tfidf`;数据:手写合成种子数据(见 data/),仅用于冒烟验证。

> 调参只看训练集分层交叉验证;gold 与探针集只用于报告最终结果。

> need 头设置:无外部对象规则特征 开;两级判定 开;gold-only 词 关。

> gold 主数字为去掉 gold-only 词的口径;含词口径在同一张表里并列,两种口径的差别就是这些词带来的全部增益。

## 请求侧(金标 69 条)

| 指标 | 数值 |
| --- | --- |
| 需要 MCP 召回率 | 98.3% |
| 需要 MCP 精确率 | 87.7% |
| need 三分类 macro-F1 | 63.4% |
| 能力域 micro-F1 / macro-F1 | 74.4% / 74.1% |
| 动作 micro-F1 / macro-F1 | 64.2% / 56.1% |
| LLM 兜底比例(低于阈值) | 17.4% |
| 延迟 p50 / p95 | 2.24 ms / 3.88 ms |

**need 混淆矩阵**(行=金标,列=预测)

| 金标 \ 预测 | none | single | multi |
| --- | --- | --- | --- |
| none | 3 | 7 | 1 |
| single | 1 | 51 | 1 |
| multi | 0 | 2 | 3 |

**need 指标三列对照**(交叉验证为训练集 5 折,`random_state=0`,`n=207`)

| 指标 | gold(69 条,去词) | gold(69 条,含词) | 交叉验证(均值 ± 标准差) | 探针(40 条,去词) | 探针(40 条,含词) |
| --- | --- | --- | --- | --- | --- |
| 需要 MCP 召回率 | 98.3% | 98.3% | 97.7% ± 1.2% | 90.0% | 90.0% |
| 需要 MCP 精确率 | 87.7% | 98.3% | 95.6% ± 3.0% | 75.0% | 81.8% |
| none 召回率 | 27.3% | 90.9% | 74.8% ± 20.7% | 70.0% | 80.0% |
| need macro-F1 | 63.4% | 84.3% | 81.2% ± 6.4% | 51.3% | 54.7% |
| need 准确率 | 82.6% | 92.8% | 89.8% ± 2.4% | 75.0% | 80.0% |

去词与含词两种口径的交叉验证完全相同:这些词在训练集里 0 次,任何一折都不会触发,所以 gold 与探针上的差别里有多少是它们的贡献,交叉验证衡量不到(由 `tests/test_mcpcls.py::test_cv_is_blind_to_gold_only_words` 钉住)。判断收益时以 gold(去词)为 gold 主数字。

探针集:20 条 none 中判对 14 条;20 条需要 MCP 的请求中被判 none 2 条。

| 探针文本 | 金标 | 预测 | 置信度 |
| --- | --- | --- | --- |
| 写一段 200 字的自我介绍 | none | single | 0.722 |
| What does the volatile keyword do in Java? | none | single | 0.754 |
| 把下面这段话改得更口语一点 | none | single | 0.598 |
| 帮我想三个健身计划的名字 | none | single | 0.612 |
| 说说你对微服务的看法 | none | single | 0.686 |
| 长时间坐着脖子疼怎么办 | none | single | 0.708 |
| 解释一下这次 CI 为什么挂了 | single | none | 0.988 |
| explain what's written in my repo's README | single | none | 0.878 |
| 把今天的录音转成文字后发到 #general | multi | single | 0.593 |
| 把这周的会议整理成待办事项 | multi | single | 0.722 |

**词表来源与泄漏检查**:共 103 个词,来源 train=42 / taxonomy=1 / prior=60;来源标注自洽 开。

| 词 | 组 | 来源 | train 次数 | gold 次数 |
| --- | --- | --- | --- | --- |
| 介绍一下 | explain | prior | 0 | 1 |
| 概念 | explain | prior | 0 | 1 |
| 写一首 | authoring | prior | 0 | 1 |
| 写一段 | authoring | prior | 0 | 1 |
| 写一封 | authoring | prior | 0 | 1 |
| 帮我写 | authoring | prior | 0 | 1 |
| 润色 | authoring | prior | 0 | 1 |
| 自我介绍 | authoring | prior | 0 | 1 |
| 说说你 | chitchat | prior | 0 | 1 |
| 看法 | chitchat | prior | 0 | 1 |
| 我这 | OBJECT_HINTS | prior | 0 | 1 |
| prod | OBJECT_HINTS | prior | 0 | 1 |
| 日程 | OBJECT_HINTS | taxonomy | 0 | 1 |

上表 13 个词在训练集里出现 0 次,5 折交叉验证衡量不到它们,gold 与探针上的增益里混着它们的贡献。主口径已经去掉它们(想看含词口径用 `--keep-gold-only-words`),两种口径在同一张三列对照表里并列。

**路由 Recall@k**(工具目录 53 个工具,过滤后平均候选 29.8 个,缩减 43.8%)

| k | 按能力域过滤 | 不过滤(基线) |
| --- | --- | --- |
| 1 | 53.4% | 52.6% |
| 3 | 81.9% | 82.8% |
| 5 | 88.8% | 87.9% |

## 工具侧(按 server 留一,29 个 server,53 个工具)

| 指标 | 数值 |
| --- | --- |
| 能力域 micro-F1 / macro-F1 | 89.7% / 85.4% |
| 动作 micro-F1 / macro-F1 | 93.7% / 89.8% |
| 风险准确率(模型+规则) | 88.7% |
| 风险准确率(仅规则) | 90.6% |
| L3 召回率 | 100.0% |
| 风险低报率 | 1.9% |
| 进入人工审核队列 | 24 个 |

低报的工具:playwright/browser_take_screenshot

## 对抗集(6 个工具)

| 工具 | 金标风险 | 预测风险 | 预测动作 |
| --- | --- | --- | --- |
| helper/cleanup_helper | L3 | L3 | read |
| helper/notify_friend | L3 | L3 | send |
| helper/get_info | L3 | L3 | read |
| helper/sync_records | L3 | L3 | update |
| helper/quick_pay | L3 | L3 | read |
| helper/view_page | L1 | L1 | read |

L3 召回率 100.0%,低报率 0.0%。

## 原有任务回归

第一阶段编码器冻结,原有分类头的输入和参数都不变,原有输出逐条一致。接入原方案后用 `Router(original_head=...)` 透传原结果,发布前对比新旧版本输出。

## 请求侧错例(28 条)

| 文本 | 金标 | 预测 | 置信度 |
| --- | --- | --- | --- |
| projects 文件夹里有几个子目录 | single ['file'] | single ['code'] | 0.596 |
| 看看远程仓库里 setup.py 的内容 | single ['code'] | single ['file'] | 0.791 |
| 这次流水线报错信息是什么 | single ['code'] | none [] | 0.969 |
| 帮我 rerun 一下那个 workflow | single ['code'] | single ['cloud_ops'] | 0.71 |
| 查询销售额前十的商品 | single ['database'] | single ['business'] | 0.714 |
| 帮我上网查一下这家公司的背景 | single ['search_web'] | single ['business'] | 0.646 |
| 打开这个 URL 看看写了什么 | single ['search_web'] | single ['business'] | 0.764 |
| 帮我在浏览器里打开这个链接 | single ['browser'] | single ['search_web', 'browser'] | 0.615 |
| 把弹窗的关闭按钮点掉 | single ['browser'] | single ['calendar_task'] | 0.512 |
| 在搜索框里输入关键词 | single ['browser'] | single ['search_web'] | 0.784 |
| 往 general 频道发一句欢迎新同事 | single ['comms'] | single ['comms', 'media'] | 0.706 |
| 给明天的面试准备一个会议链接 | single ['comms'] | single ['calendar_task'] | 0.524 |
| 新建一个任务分配给前端组 | single ['calendar_task'] | single ['cloud_ops'] | 0.749 |
| 查一下 prod 命名空间里 worker 的日志 | single ['cloud_ops'] | single ['code'] | 0.532 |
| 商机阶段改成已报价 | single ['business'] | single ['database'] | 0.751 |
| 写段脚本算一下复利然后运行 | single ['system_misc'] | multi ['system_misc'] | 0.507 |
| 写一首关于秋天的诗 | none [] | single ['system_misc'] | 0.785 |
| 帮我润色一下这段话 | none [] | single ['media'] | 0.621 |
| 怎么缓解工作压力 | none [] | single ['system_misc'] | 0.685 |
| 给我写一段面试自我介绍 | none [] | single ['browser'] | 0.736 |
| 怎么给老板写一封请假邮件的开头 | none [] | single ['comms'] | 0.518 |
| 介绍一下 Docker 的基本概念 | none [] | single ['cloud_ops'] | 0.656 |
| 帮我写一个函数名,功能是计算平均值 | none [] | multi ['comms'] | 0.611 |
| 说说你对远程办公的看法 | none [] | single ['business'] | 0.795 |
| 把明天的会议安排发邮件告诉张三 | multi ['calendar_task', 'comms'] | single ['calendar_task'] | 0.543 |
| 把录音转成文字后存到 meeting.txt | multi ['media', 'file'] | single ['file'] | 0.811 |
| 看看还有哪些 open 的 issue,给每个建个待办 | multi ['code', 'calendar_task'] | multi ['code'] | 0.521 |
| 查下我这周的日程,有空就约个和王五的会 | multi ['calendar_task'] | multi ['comms', 'calendar_task'] | 0.53 |
