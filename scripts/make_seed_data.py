"""生成种子数据:工具目录 + 请求样本。

这是手写的合成种子数据,用来跑通流程和做冒烟评估,不代表真实分布。
正式训练时应替换为:真实 MCP server 的 tools/list(mcpcls fetch-tools)+ 线上请求日志
+ Agent 调用轨迹,再用 LLM 双路打标(mcpcls label)。

每个工具写 4 条请求:前 3 条进训练集,第 4 条进金标集。
"""

import json
import pathlib
import random

OUT = pathlib.Path(__file__).resolve().parent.parent / "data"

# (server, name, description, params, annotations, domains, actions, [请求 x4])
T = [
    # ---- 文件 ----
    ("filesystem", "read_file", "Read the complete contents of a file from the file system.", ["path"],
     {"readOnlyHint": True}, ["file"], ["read"],
     ["帮我看一下 config.yaml 里写了什么", "读取 /data/report.txt 的内容", "打开 README.md 给我看看", "把 notes/todo.txt 的内容念给我听"]),
    ("filesystem", "write_file", "Create a new file or completely overwrite an existing file with new content.", ["path", "content"],
     {"destructiveHint": True}, ["file"], ["create", "delete"],
     ["把这段总结保存成 summary.md", "新建一个 hello.py 文件,内容写 print('hi')", "把结果写到 output.csv 里", "把会议记录存成 meeting.txt"]),
    ("filesystem", "list_directory", "Get a detailed listing of all files and directories in a specified path.", ["path"],
     {"readOnlyHint": True}, ["file"], ["read"],
     ["看看 downloads 文件夹里有哪些文件", "列出当前目录下的所有文件", "src 目录下都有什么", "projects 文件夹里有几个子目录"]),
    ("filesystem", "move_file", "Move or rename files and directories.", ["source", "destination"],
     {}, ["file"], ["update"],
     ["把 a.txt 改名成 b.txt", "把 draft.docx 移到 archive 文件夹", "把这几张图片挪到 images 目录", "把 old_report.pdf 重命名为 2025_report.pdf"]),
    ("gdrive", "search_files", "Search for files in Google Drive by name or content.", ["query"],
     {"readOnlyHint": True, "openWorldHint": True}, ["file"], ["read"],
     ["在云盘里找一下预算表", "Google Drive 里有没有叫季度规划的文件", "帮我搜云盘中关于合同的文档", "云盘里找找去年的年终总结"]),
    ("gdrive", "delete_file", "Permanently delete a file from Google Drive.", ["file_id"],
     {"destructiveHint": True, "openWorldHint": True}, ["file"], ["delete"],
     ["把云盘里那个旧的测试文档删掉", "删除 Drive 上的 draft_v1", "云盘里的重复文件帮我删了", "把 Google Drive 里过期的报价单删除"]),
    # ---- 代码 ----
    ("github", "create_pull_request", "Create a new pull request in a GitHub repository.", ["owner", "repo", "title", "head", "base", "body"],
     {"openWorldHint": True}, ["code"], ["create"],
     ["帮我从 feature 分支向 main 提个 PR", "给这个修复开一个 pull request", "在 web 仓库新建 PR,标题叫修复登录", "把 dev 分支的改动提 PR 合到 main"]),
    ("github", "list_issues", "List issues in a GitHub repository with filters.", ["owner", "repo", "state", "labels"],
     {"readOnlyHint": True, "openWorldHint": True}, ["code"], ["read"],
     ["看看仓库里还有哪些没关的 issue", "列出带 bug 标签的 issue", "最近有人提了什么 issue", "api 仓库里有多少个 open 的 issue"]),
    ("github", "get_file_contents", "Get the contents of a file or directory from a GitHub repository.", ["owner", "repo", "path", "branch"],
     {"readOnlyHint": True, "openWorldHint": True}, ["code"], ["read"],
     ["看下 GitHub 上 main 分支的 package.json", "拉一下仓库里 src/app.ts 的代码", "仓库根目录下的 Dockerfile 写了什么", "看看远程仓库里 setup.py 的内容"]),
    ("github", "merge_pull_request", "Merge a pull request in a GitHub repository.", ["owner", "repo", "pull_number", "merge_method"],
     {"openWorldHint": True}, ["code"], ["update"],
     ["把 #42 这个 PR 合进去", "PR 已经审完了,帮我 merge", "用 squash 方式合并 123 号 PR", "把那个修复 PR 合并到主干"]),
    ("ci", "get_workflow_run_logs", "Get logs of a CI workflow run.", ["repo", "run_id"],
     {"readOnlyHint": True}, ["code"], ["read"],
     ["CI 为什么挂了,看下日志", "拉一下最近一次构建的日志", "看看 GitHub Actions 失败的原因", "这次流水线报错信息是什么"]),
    ("ci", "rerun_workflow", "Re-run a CI workflow.", ["repo", "run_id"],
     {}, ["code"], ["execute"],
     ["把失败的 CI 重跑一遍", "重新触发一下这次构建", "流水线再跑一次", "帮我 rerun 一下那个 workflow"]),
    # ---- 数据库 ----
    ("postgres", "query", "Run a read-only SQL query against the database.", ["sql"],
     {"readOnlyHint": True}, ["database"], ["read"],
     ["查一下上个月的订单总数", "统计每个城市的用户数", "数据库里昨天新增了多少注册", "查询销售额前十的商品"]),
    ("postgres", "execute_sql", "Execute an arbitrary SQL statement including INSERT, UPDATE, DELETE and DDL.", ["sql"],
     {"destructiveHint": True}, ["database"], ["execute", "update", "delete"],
     ["把 status 为 test 的数据全部删掉", "给 users 表加一个 phone 字段", "把订单 1001 的状态改成已发货", "清空临时表 tmp_import"]),
    ("postgres", "list_tables", "List all tables in the database schema.", ["schema_name"],
     {"readOnlyHint": True}, ["database"], ["read"],
     ["数据库里有哪些表", "列出 public schema 下的所有表", "看看库里有没有日志表", "当前数据库都建了哪些表"]),
    ("sheets", "update_cells", "Update cell values in a spreadsheet.", ["spreadsheet_id", "range", "values"],
     {"openWorldHint": True}, ["database"], ["update"],
     ["把表格 B2 改成 300", "在预算表里更新三月的数字", "把这些数据填到 Sheet1 的 A 列", "修改一下销售表里张三的业绩"]),
    # ---- 搜索与网页 ----
    ("brave_search", "web_search", "Search the web and return relevant results.", ["query", "count"],
     {"readOnlyHint": True, "openWorldHint": True}, ["search_web"], ["read"],
     ["搜一下最新的 MCP 协议规范", "网上查查今年的诺贝尔奖得主", "搜索一下 Python 3.13 有什么新特性", "帮我上网查一下这家公司的背景"]),
    ("fetch", "fetch_url", "Fetch a URL and return its content as markdown.", ["url", "max_length"],
     {"readOnlyHint": True, "openWorldHint": True}, ["search_web"], ["read"],
     ["抓取一下 https://example.com 的正文", "把这个网址的内容拿下来给我总结", "读一下这篇博客链接里的文章", "打开这个 URL 看看写了什么"]),
    ("arxiv", "search_papers", "Search academic papers on arXiv.", ["query", "max_results"],
     {"readOnlyHint": True, "openWorldHint": True}, ["search_web"], ["read"],
     ["找几篇关于 RAG 的论文", "arXiv 上搜一下知识蒸馏的最新研究", "帮我查查多模态大模型的综述论文", "有没有关于工具调用评测的论文"]),
    # ---- 浏览器 ----
    ("playwright", "browser_navigate", "Navigate the browser to a URL.", ["url"],
     {"readOnlyHint": True, "openWorldHint": True}, ["browser"], ["read"],
     ["用浏览器打开公司官网", "打开淘宝首页", "浏览器跳转到后台管理页面", "帮我在浏览器里打开这个链接"]),
    ("playwright", "browser_click", "Click an element on the current page.", ["element", "ref"],
     {"openWorldHint": True}, ["browser"], ["execute"],
     ["点一下页面上的登录按钮", "帮我点提交", "在页面上点击下一页", "把弹窗的关闭按钮点掉"]),
    ("playwright", "browser_fill_form", "Fill form fields on the current page.", ["fields"],
     {"openWorldHint": True}, ["browser"], ["execute"],
     ["在表单里填上我的姓名和邮箱", "把注册页的信息填好", "帮我填一下网页上的报名表", "在搜索框里输入关键词"]),
    ("playwright", "browser_take_screenshot", "Take a screenshot of the current page.", ["filename", "full_page"],
     {"readOnlyHint": True}, ["browser"], ["read"],
     ["给当前页面截个图", "把整个网页截长图", "截一下浏览器现在的画面", "页面截图保存一下"]),
    # ---- 通讯 ----
    ("slack", "post_message", "Post a message to a Slack channel.", ["channel", "text"],
     {"openWorldHint": True}, ["comms"], ["send"],
     ["在 #dev 频道发个通知说今晚发版", "给团队群发消息说会议改到四点", "在 Slack 上提醒大家交周报", "往 general 频道发一句欢迎新同事"]),
    ("slack", "get_channel_history", "Get recent messages from a Slack channel.", ["channel", "limit"],
     {"readOnlyHint": True, "openWorldHint": True}, ["comms"], ["read"],
     ["看看 #support 频道最近聊了什么", "拉一下 Slack 群里今天的消息", "dev 频道昨天有人说什么了", "总结一下产品群最近的讨论"]),
    ("gmail", "send_email", "Send an email to recipients.", ["to", "subject", "body", "cc"],
     {"openWorldHint": True}, ["comms"], ["send"],
     ["给张三发封邮件说合同已经签了", "发邮件通知客户明天停机维护", "把报价发邮件给 li@example.com", "回一封邮件感谢面试官"]),
    ("gmail", "search_emails", "Search emails in the mailbox.", ["query", "max_results"],
     {"readOnlyHint": True, "openWorldHint": True}, ["comms"], ["read"],
     ["找一下上周财务发来的邮件", "邮箱里有没有关于发票的邮件", "搜一下来自 HR 的邮件", "看看最近有没有未读的重要邮件"]),
    ("zoom", "create_meeting", "Create a Zoom meeting and return the join link.", ["topic", "start_time", "duration"],
     {"openWorldHint": True}, ["comms"], ["create"],
     ["开一个 Zoom 会议链接", "建个线上会议,主题是需求评审", "帮我创建一个半小时的视频会议", "给明天的面试准备一个会议链接"]),
    # ---- 日程与任务 ----
    ("gcal", "create_event", "Create a calendar event.", ["title", "start_time", "end_time", "attendees"],
     {"openWorldHint": True}, ["calendar_task"], ["create"],
     ["明天下午三点约个会,叫上王五", "在日历上加一个下周一的周会", "帮我把牙医预约记到日历", "周五晚上七点安排一个聚餐"]),
    ("gcal", "list_events", "List calendar events in a time range.", ["start_time", "end_time"],
     {"readOnlyHint": True, "openWorldHint": True}, ["calendar_task"], ["read"],
     ["我明天有什么安排", "看看这周日历上有哪些会", "下周三下午我有空吗", "今天还有几个会"]),
    ("gcal", "delete_event", "Delete a calendar event.", ["event_id"],
     {"destructiveHint": True, "openWorldHint": True}, ["calendar_task"], ["delete"],
     ["把明天的周会取消掉", "删除日历里周五的聚餐", "那个评审会不开了,从日历删了", "取消下午三点的约会"]),
    ("linear", "create_issue", "Create a task issue in the project tracker.", ["title", "description", "assignee", "due_date"],
     {"openWorldHint": True}, ["calendar_task"], ["create"],
     ["建一个任务,让李四周五前修完登录 bug", "在项目里加个待办:写接口文档", "提醒我周一交报告,建个待办", "新建一个任务分配给前端组"]),
    ("linear", "update_issue", "Update status, assignee or fields of an issue.", ["issue_id", "status", "assignee"],
     {"openWorldHint": True}, ["calendar_task"], ["update"],
     ["把这个任务标记为完成", "把登录 bug 的任务转给王五", "任务 ENG-12 的状态改成进行中", "把文档任务的截止日期推到下周"]),
    # ---- 知识库 ----
    ("notion", "search_pages", "Search pages in the Notion workspace.", ["query"],
     {"readOnlyHint": True, "openWorldHint": True}, ["knowledge"], ["read"],
     ["在 Notion 里找上周的周报", "知识库里有没有入职指南", "查一下 wiki 上的发布流程", "Notion 里搜一下产品路线图"]),
    ("notion", "create_page", "Create a new page in Notion.", ["parent_id", "title", "content"],
     {"openWorldHint": True}, ["knowledge"], ["create"],
     ["把这份会议纪要记到 Notion", "在知识库新建一页写部署步骤", "把今天的复盘整理成一篇 wiki", "在 Notion 里建个页面记录需求"]),
    ("memory", "search_memory", "Search the long-term memory knowledge graph.", ["query"],
     {"readOnlyHint": True}, ["knowledge"], ["read"],
     ["你记得我上次说的项目名称吗", "查一下之前存过的客户偏好", "记忆里有没有我的生日", "我之前跟你说过喜欢什么咖啡来着"]),
    # ---- 云与运维 ----
    ("kubernetes", "get_pod_logs", "Get logs of a pod.", ["namespace", "pod", "tail_lines"],
     {"readOnlyHint": True}, ["cloud_ops"], ["read"],
     ["看下 api 服务 pod 的日志", "拉一下 payment pod 最近 100 行日志", "线上那个 pod 为什么一直重启,看下日志", "查一下 prod 命名空间里 worker 的日志"]),
    ("kubernetes", "scale_deployment", "Scale a deployment to a number of replicas.", ["namespace", "name", "replicas"],
     {}, ["cloud_ops"], ["update"],
     ["把 api 服务扩到 3 个副本", "把 worker 缩容到 1 个", "流量大了,把前端扩容一下", "把 deployment web 的副本数调到 5"]),
    ("kubernetes", "delete_pod", "Delete a pod.", ["namespace", "pod"],
     {"destructiveHint": True}, ["cloud_ops"], ["delete"],
     ["把卡住的那个 pod 删掉", "删除 staging 里的 test pod", "那个异常 pod 直接干掉重建", "kill 掉 worker-7 这个 pod"]),
    ("aws", "describe_instances", "Describe EC2 instances.", ["region", "filters"],
     {"readOnlyHint": True, "openWorldHint": True}, ["cloud_ops"], ["read"],
     ["看看 AWS 上现在有几台机器在跑", "列出东京区的 EC2 实例", "查一下那台数据库服务器的配置", "云上哪些实例是停止状态"]),
    ("aws", "deploy_stack", "Deploy or update a CloudFormation stack.", ["stack_name", "template", "region"],
     {"openWorldHint": True}, ["cloud_ops"], ["execute"],
     ["把新的基础设施模板部署上去", "部署一下 staging 的 CloudFormation", "更新生产环境的 stack", "用这个模板发布一套新环境"]),
    # ---- 业务系统 ----
    ("stripe", "create_refund", "Refund a charge to a customer.", ["charge_id", "amount", "reason"],
     {"openWorldHint": True}, ["business"], ["send"],
     ["给这个订单退款", "把客户多扣的 50 块退回去", "订单 1234 全额退款", "这笔支付退一半给用户"]),
    ("stripe", "list_payments", "List recent payments.", ["customer_id", "limit"],
     {"readOnlyHint": True, "openWorldHint": True}, ["business"], ["read"],
     ["看看今天收了多少钱", "查一下客户 A 最近的付款记录", "列出最近十笔支付", "这个月有没有支付失败的订单"]),
    ("crm", "get_customer", "Get a customer profile and follow-up history from CRM.", ["customer_id"],
     {"readOnlyHint": True}, ["business"], ["read"],
     ["查一下客户 A 的跟进记录", "看看这个客户的联系人是谁", "CRM 里这家公司的状态是什么", "调出客户王总的资料"]),
    ("crm", "update_deal", "Update a deal stage in CRM.", ["deal_id", "stage", "amount"],
     {}, ["business"], ["update"],
     ["把这个商机推进到谈判阶段", "更新一下 A 公司订单的金额", "把那个单子标记为赢单", "商机阶段改成已报价"]),
    ("zendesk", "create_ticket", "Create a support ticket.", ["subject", "description", "priority"],
     {"openWorldHint": True}, ["business"], ["create"],
     ["给这个客户问题建个工单", "提一个高优先级工单说支付失败", "开个售后工单跟进退货", "帮用户登记一个投诉工单"]),
    # ---- 多媒体 ----
    ("image_gen", "generate_image", "Generate an image from a text prompt.", ["prompt", "width", "height"],
     {"openWorldHint": True}, ["media"], ["create"],
     ["画一张猫在月球上的插画", "生成一张产品海报", "帮我做个 logo 图", "给文章配一张赛博朋克风格的图"]),
    ("whisper", "transcribe_audio", "Transcribe an audio file to text.", ["audio_url", "language"],
     {"readOnlyHint": True}, ["media"], ["read"],
     ["把这段录音转成文字", "帮我听写这个会议音频", "这个语音消息说了什么", "把采访录音整理成文字稿"]),
    ("video", "trim_video", "Trim a video to a time range.", ["video_url", "start", "end"],
     {}, ["media"], ["create"],
     ["把视频剪成前 30 秒", "截取这段视频的 1 分到 2 分", "把片头去掉只留正片", "剪一个 15 秒的预告片段"]),
    # ---- 系统与其他 ----
    ("shell", "run_command", "Run a shell command on the local machine.", ["command", "cwd"],
     {"destructiveHint": True}, ["system_misc"], ["execute"],
     ["运行 ls -la", "执行一下 npm install", "跑一下 make test", "在终端里查一下磁盘占用"]),
    ("python", "run_code", "Execute Python code in a sandbox and return the output.", ["code"],
     {}, ["system_misc"], ["execute"],
     ["跑一段代码算一下 1 到 100 的质数", "用 Python 把这个列表排序后输出", "执行代码画个正弦曲线", "写段脚本算一下复利然后运行"]),
    ("weather", "get_forecast", "Get the weather forecast for a location.", ["location", "days"],
     {"readOnlyHint": True, "openWorldHint": True}, ["system_misc"], ["read"],
     ["北京明天天气怎么样", "这周末上海会下雨吗", "查一下杭州未来三天的天气", "出门要不要带伞"]),
    ("maps", "get_directions", "Get directions between two places.", ["origin", "destination", "mode"],
     {"readOnlyHint": True, "openWorldHint": True}, ["system_misc"], ["read"],
     ["从公司到机场怎么走", "查一下去西湖的公交路线", "开车到浦东要多久", "附近最近的地铁站怎么走"]),
]

# 不需要 MCP 的请求(LLM 直接回答)
NONE = [
    "什么是 MCP 协议", "解释一下 Transformer 的注意力机制", "帮我把这句话翻译成英文:今天天气很好",
    "写一首关于秋天的诗", "Python 里列表和元组有什么区别", "给我讲个笑话", "怎么理解边际效用递减",
    "帮我润色一下这段话", "1 加 1 等于几", "推荐几本机器学习入门书", "如何写好一份周报",
    "解释一下什么是知识蒸馏", "帮我起个产品名字", "把这段代码改成更简洁的写法:for i in range(len(a)): print(a[i])",
    "什么是 RESTful API", "怎么缓解工作压力", "帮我列个学习计划大纲", "总结一下你刚才说的要点",
    "SQL 里 join 有几种", "给我写一段面试自我介绍", "如何判断一个数是不是质数", "什么是 Kubernetes",
    "帮我想三个周末活动", "解释一下 git rebase 和 merge 的区别", "这句话有没有语病:我们要提高大家的认识水平",
    "写一个冒泡排序的伪代码", "什么是 OAuth", "怎么给老板写一封请假邮件的开头", "比较一下 Redis 和 Memcached",
    "用一句话介绍你自己", "如何设计一个分类体系", "解释一下精确率和召回率", "早上好", "谢谢你的帮助",
    "帮我把这个列表按字母排序:香蕉 苹果 橙子", "介绍一下 Docker 的基本概念", "怎么做番茄炒蛋",
    "如何提高英语口语", "什么是 F1 分数", "帮我写一个函数名,功能是计算平均值", "解释一下 HTTP 状态码 404",
    "帮我把下面的要点扩写成一段话", "微服务有什么优缺点", "说说你对远程办公的看法",
]

# 需要多个工具的请求:(文本, [tool_id...], domains, actions)
MULTI = [
    ("查一下上个月的订单总数,然后发到 #sales 频道", ["postgres/query", "slack/post_message"], ["database", "comms"], ["read", "send"]),
    ("看看 CI 为什么挂了,修好后提个 PR", ["ci/get_workflow_run_logs", "github/create_pull_request"], ["code"], ["read", "create"]),
    ("搜一下竞品最新动态,整理成 Notion 页面", ["brave_search/web_search", "notion/create_page"], ["search_web", "knowledge"], ["read", "create"]),
    ("把明天的会议安排发邮件告诉张三", ["gcal/list_events", "gmail/send_email"], ["calendar_task", "comms"], ["read", "send"]),
    ("读一下 report.txt 然后发给老板", ["filesystem/read_file", "gmail/send_email"], ["file", "comms"], ["read", "send"]),
    ("pod 一直重启,看下日志再扩个容", ["kubernetes/get_pod_logs", "kubernetes/scale_deployment"], ["cloud_ops"], ["read", "update"]),
    ("查一下这个客户的付款记录,有问题就建个工单", ["stripe/list_payments", "zendesk/create_ticket"], ["business"], ["read", "create"]),
    ("把录音转成文字后存到 meeting.txt", ["whisper/transcribe_audio", "filesystem/write_file"], ["media", "file"], ["read", "create"]),
    ("查一下明天北京天气,如果下雨就把户外活动取消", ["weather/get_forecast", "gcal/delete_event"], ["system_misc", "calendar_task"], ["read", "delete"]),
    ("抓取这个网页,把正文保存成 markdown 文件", ["fetch/fetch_url", "filesystem/write_file"], ["search_web", "file"], ["read", "create"]),
    ("统计每个城市的用户数并更新到表格里", ["postgres/query", "sheets/update_cells"], ["database"], ["read", "update"]),
    ("看看还有哪些 open 的 issue,给每个建个待办", ["github/list_issues", "linear/create_issue"], ["code", "calendar_task"], ["read", "create"]),
    ("在 Notion 里找到发布流程,然后在 #dev 发出来", ["notion/search_pages", "slack/post_message"], ["knowledge", "comms"], ["read", "send"]),
    ("用浏览器打开后台,截个图发给我邮箱", ["playwright/browser_navigate", "playwright/browser_take_screenshot", "gmail/send_email"], ["browser", "comms"], ["read", "send"]),
    ("找几篇 RAG 论文,下载后列个清单存到云盘", ["arxiv/search_papers", "filesystem/write_file"], ["search_web", "file"], ["read", "create"]),
    ("查下我这周的日程,有空就约个和王五的会", ["gcal/list_events", "gcal/create_event"], ["calendar_task"], ["read", "create"]),
    ("把 status 是 test 的数据删掉,然后通知大家", ["postgres/execute_sql", "slack/post_message"], ["database", "comms"], ["delete", "send"]),
    ("看看 AWS 上停止的实例,写个清单发给运维", ["aws/describe_instances", "gmail/send_email"], ["cloud_ops", "comms"], ["read", "send"]),
    ("生成一张海报,然后发到 general 频道", ["image_gen/generate_image", "slack/post_message"], ["media", "comms"], ["create", "send"]),
    ("给客户退款后在 CRM 里把商机标记为流失", ["stripe/create_refund", "crm/update_deal"], ["business"], ["send", "update"]),
]


def tool_dict(row):
    server, name, desc, params, ann, domains, actions, _ = row
    return {
        "server": server, "name": name, "description": desc,
        "inputSchema": {"type": "object", "properties": {p: {"type": "string"} for p in params}},
        "annotations": ann, "domains": domains, "actions": actions,
    }


def main():
    OUT.mkdir(exist_ok=True)
    random.seed(7)
    tools = [tool_dict(r) for r in T]
    train, gold = [], []
    for row in T:
        server, name, _, _, _, domains, actions, reqs = row
        tid = f"{server}/{name}"
        for i, text in enumerate(reqs):
            rec = {"text": text, "need": "single", "domains": domains, "actions": actions, "tool_ids": [tid]}
            (gold if i == 3 else train).append(rec)
    for i, text in enumerate(NONE):
        rec = {"text": text, "need": "none", "domains": [], "actions": [], "tool_ids": []}
        (gold if i % 4 == 3 else train).append(rec)
    for i, (text, tids, domains, actions) in enumerate(MULTI):
        rec = {"text": text, "need": "multi", "domains": domains, "actions": actions, "tool_ids": tids}
        (gold if i % 4 == 3 else train).append(rec)
    random.shuffle(train)
    for fname, rows in (("tools.jsonl", tools), ("requests_train.jsonl", train), ("requests_gold.jsonl", gold)):
        with open(OUT / fname, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{fname}: {len(rows)} 行")


if __name__ == "__main__":
    main()
