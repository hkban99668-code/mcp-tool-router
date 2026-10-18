"""三轴类别体系:能力域、动作类型、风险等级,以及请求侧的"是否需要 MCP"。

请求侧和工具侧共用这套类别,所以请求侧模型预测出的能力域可以直接拿去过滤工具目录。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    definition: str
    examples: tuple = field(default_factory=tuple)
    boundary: str = ""  # 容易混淆的边界说明,放进 LLM 打标 prompt


DOMAINS = (
    Category("file", "文件与文档", "读写本地文件、云盘文件,解析 PDF/Office 文档",
             ("读取 report.pdf", "把这段内容保存成 notes.md"),
             "Wiki/笔记类内容归 knowledge;代码仓库里的文件归 code"),
    Category("code", "代码与开发", "Git 仓库、Issue/PR、CI、包管理",
             ("给这个仓库开个 PR", "看看 CI 为什么挂了"),
             "执行任意 shell 命令归 system_misc"),
    Category("database", "数据库与数据分析", "SQL 查询、数仓、表格计算",
             ("查一下上个月订单量", "建一张用户表")),
    Category("search_web", "搜索与网页", "网页搜索、抓取网页内容、学术检索",
             ("搜一下最新的 MCP 规范", "抓取这个网址的正文"),
             "需要点击、登录、填表的网页操作归 browser"),
    Category("browser", "浏览器与自动化", "操作浏览器、点击填表、截图、RPA",
             ("打开官网帮我点登录", "给这个页面截个图")),
    Category("comms", "通讯协作", "邮件、即时消息、会议",
             ("给张三发封邮件", "在 #dev 频道发个通知")),
    Category("calendar_task", "日程与任务", "日历、待办、项目管理任务",
             ("明天下午三点约个会", "建一个待办提醒我交报告")),
    Category("knowledge", "知识库与笔记", "Wiki、笔记、内部知识库检索",
             ("在 Notion 里找上周的周报", "把会议纪要记到知识库")),
    Category("cloud_ops", "云与运维", "云资源、容器、监控告警",
             ("把 api 服务扩到 3 个副本", "看下 pod 日志")),
    Category("business", "业务系统", "CRM、电商、支付、工单",
             ("给这个订单退款", "查一下客户 A 的跟进记录")),
    Category("media", "多媒体", "图片/音频/视频的生成与处理",
             ("画一张猫的插画", "把这段音频转成文字")),
    Category("weather_geo", "天气地图与出行", "天气预报、地图与地点查询、导航路线、航班火车与打车等出行查询和预订",
             ("北京明天天气怎么样", "查一下从机场到酒店怎么走"),
             "订酒店、买票付款的订单管理归 business;只是查信息归这里"),
    Category("entertainment", "娱乐资讯与推荐", "影视、音乐、体育、游戏、演出的信息查询与推荐(查片单、找歌、看比分、查演出)",
             ("推荐几部最近的科幻片", "湖人昨晚比分多少"),
             "生成或处理图片音视频归 media;购票付款归 business"),
    Category("system_misc", "系统与其他", "Shell、本地系统、设备控制等以上都不属于的通用工具",
             ("运行 ls -la", "把空调调到 26 度")),
)

ACTIONS = (
    Category("read", "只读查询", "只读取或检索信息,不改变任何状态"),
    Category("create", "新建", "新建一条记录、文件、资源"),
    Category("update", "修改", "修改已有的记录、文件、资源,可撤销"),
    Category("delete", "删除", "删除或覆盖数据,通常不可撤销"),
    Category("execute", "执行", "运行代码、命令、部署、脚本"),
    Category("send", "对外发送", "发消息、发邮件、发布内容、付款,对外部产生影响"),
)

RISK_LEVELS = (
    Category("L0", "只读内部", "只读,作用于本地或内部数据"),
    Category("L1", "只读外部", "只读,但会访问外部网络"),
    Category("L2", "可撤销写", "新建或修改,可以撤销"),
    Category("L3", "不可撤销", "删除、执行、对外发送等不可撤销操作,必须用户确认"),
)

NEED = (
    Category("none", "不需要", "LLM 直接回答即可,不需要调用任何 MCP 工具"),
    Category("single", "单工具", "需要调用一个 MCP 工具"),
    Category("multi", "多工具", "需要组合调用多个 MCP 工具"),
)

DOMAIN_IDS = tuple(c.id for c in DOMAINS)
ACTION_IDS = tuple(c.id for c in ACTIONS)
RISK_IDS = tuple(c.id for c in RISK_LEVELS)
NEED_IDS = tuple(c.id for c in NEED)

# 这些动作一旦出现,风险就是 L3
L3_ACTIONS = frozenset({"delete", "execute", "send"})
WRITE_ACTIONS = frozenset({"create", "update"})
# 只读时默认会访问外部网络的能力域
OPEN_WORLD_DOMAINS = frozenset({"search_web", "browser"})


def risk_rank(level: str) -> int:
    return RISK_IDS.index(level)


def max_risk(*levels: str) -> str:
    levels = [lv for lv in levels if lv]
    return max(levels, key=risk_rank) if levels else "L0"


def category_cards() -> str:
    """生成类别定义卡,作为 LLM 打标 prompt 的固定部分。"""
    out = []
    for title, cats in (("能力域 domains(多选)", DOMAINS), ("动作类型 actions(多选)", ACTIONS),
                        ("是否需要 MCP need(单选)", NEED)):
        out.append(f"## {title}")
        for c in cats:
            line = f"- {c.id}:{c.name}。{c.definition}"
            if c.examples:
                line += f" 例:{';'.join(c.examples)}"
            if c.boundary:
                line += f" 边界:{c.boundary}"
            out.append(line)
    return "\n".join(out)
