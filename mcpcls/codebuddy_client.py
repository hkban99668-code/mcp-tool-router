"""CodeBuddy 命令行客户端:把本机的 `codebuddy -p` 当第二路打标 / 翻译模型。

为什么不走 OpenRouter:2026-10-09 账户余额为 0,Haiku 全部返回 402。这条路上的调用
不联网到 OpenRouter,也不读 `OPENROUTER_API_KEY`(子进程环境里会把它删掉,见 `_env`)。

一次调用 30~50 秒、和条数关系不大,所以必须批量:一次问 `batch_size` 条,题目说明只渲染
一次,模型回一个 `{"items": [...]}`,按 `i` 对回原顺序。

失败分两层:
- 整批失败(进程超时 / 退出码非 0 / `is_error` 为 True / 找不到 JSON):按 `retries` 重试整批,
  仍失败则这批每条记 ok=False,抛 `JevError`;
- 部分失败:能解析的条目正常返回,缺失或校验不过的位置为 None,该条 ok=False。

限流是第三种:退出码 0、stdout 空、stderr 一行 "429 ..."。这不算整批失败也不重试,
直接抛 `RateLimited`(见 `_run`),上层看到就停。

题目渲染与答案校验复用 `llm_or.py`(`render_questions` / `parse_answers_obj`),不复制第二份规则。
"""

import json
import os
import shutil
import subprocess
import time

from .jev import JevCall, JevError, JevUnavailable, PaymentRequired, RateLimited
from .llm_or import _JSON_RULES, _strip_fences, parse_answers_obj, render_questions

# `-p` 后面跟 prompt,其余固定这几项:`--tools ""` 让子进程不要去用工具,
# `--no-session-persistence` 免得把打标的问题写进会话历史,
# `--output-format json` 才能拿到 result / is_error。
TAIL_ARGS = ("--tools", "", "--no-session-persistence", "--output-format", "json")


def render_batch_prompt(states, questions: dict) -> str:
    """一次问多条:题目说明只出现一次,后面列编号的请求,要求回 `{"items": [...]}`。"""
    n, m = len(states), len(questions)
    lines = [f"下面是 {n} 条用户请求。对每一条,回答下面这组问题(共 {m} 题)。",
             "",
             render_questions(questions),
             "请求列表:"]
    for i, s in enumerate(states, start=1):
        lines.append(f"{i}. {s}")
    lines += ["",
              "只输出一个 JSON 对象,不要任何别的文字:",
              '{"items": [{"i": 1, "<题名>": <答案>, ...}, ...]}',
              _JSON_RULES,
              f"- i 是上面请求列表里的编号;每条请求一个元素,共 {n} 个,一条都不能少",
              "再强调一次:只输出这个 JSON 对象,不要用 ``` 代码围栏,前后不要加任何说明。"]
    return "\n".join(lines)


# 宿主会话注入这个变量时,子进程会跑去连宿主的服务、一句话都不输出(2026-10-09 实测:
# 有它 stdout 是 0 字节、退出码 0;去掉后正常返回 17 KB 的 JSON)。客户端要的是
# "干净的一次性调用",所以连同 OPENROUTER_API_KEY 一起从子进程环境里删掉。
HOST_ENV_VARS = ("SERVER__PORT",)


def _env() -> dict:
    """子进程环境:删掉 OPENROUTER_API_KEY(免得它拿去打 OpenRouter)和宿主注入的变量。"""
    drop = {"OPENROUTER_API_KEY", "JEV_API_KEY", *HOST_ENV_VARS}
    return {k: v for k, v in os.environ.items() if k not in drop}


def _result_text(stdout: str) -> str:
    """从 `codebuddy --output-format json` 的输出里取 result 文本。

    输出是一个 JSON 数组,最后一个 `type == "result"` 的元素里才是回复;
    `is_error` 是字符串 "True"/"False"。输出偶尔会混进非 JSON 的行,
    所以整段解析失败时退回逐行找最后一个 result。
    """
    text = (stdout or "").strip()
    for rec in _iter_records(text):
        if isinstance(rec, dict) and rec.get("type") == "result":
            if str(rec.get("is_error")) == "True":
                raise JevError(f"codebuddy 返回 is_error:{str(rec)[:200]}")
            return rec.get("result") or ""
    raise JevError(f"输出里没有 type=result 的记录:{text[:200]}")


def _iter_records(text: str):
    """先试整段 JSON(数组就倒着看),不行就逐行试;都不行就是空。"""
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, list):
        yield from reversed(data)
        return
    if isinstance(data, dict):
        yield data
        return
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except ValueError:
            continue


def _first_json_object(text: str):
    """取文本里第一个完整的 JSON 对象:先剥代码围栏,再从第一个 `{` 开始按花括号配对扫。"""
    t = _strip_fences(text)
    start = t.find("{")
    if start < 0:
        raise JevError(f"输出里没有 JSON 对象:{t[:200]}")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise JevError(f"输出里的 JSON 对象没有闭合:{t[:200]}")


def _parse_batch(text: str, n: int, questions: dict) -> list:
    """把 `{"items": [...]}` 按 i 对回原顺序;缺失或校验不过的位置为 None。"""
    obj = _first_json_object(text)
    items = obj.get("items")
    if not isinstance(items, list):
        raise JevError(f"输出里没有 items 数组:{str(obj)[:200]}")
    out = [None] * n
    for raw in items:
        if not isinstance(raw, dict):
            continue
        idx = raw.get("i")
        if isinstance(idx, bool) or not isinstance(idx, int) or not 1 <= idx <= n:
            continue
        try:
            out[idx - 1] = parse_answers_obj({k: v for k, v in raw.items() if k != "i"},
                                             questions)
        except JevError:
            out[idx - 1] = None          # 这一条的答案非法,按没解析出来处理
    return out


class CodeBuddyClient:
    """`codebuddy -p` 的批量客户端,接口与 `JevClient` 对齐(decide / calls)。"""

    source = "codebuddy_fallback"
    error_source = "codebuddy_error"
    batch_size = 20
    # 一批 20 条实测 78~120 秒(调查结论见 reports/T-007.md),
    # realdata 那条默认 30 秒的超时会把每一批都打成超时,所以客户端声明自己的超时下限。
    min_timeout = 180.0

    def __init__(self, timeout=180.0, retries=1, model=None, binary="codebuddy", runner=None):
        """runner 是测试注入的假进程:`runner(argv, timeout)` 或 `runner(argv, timeout, env=...)`,
        返回 stdout 全文,或 (stdout, stderr) 元组(限流时 stderr 是那行 429)。不注入时才检查
        `binary` 在不在 PATH 里。
        """
        self.timeout = timeout
        self.retries = retries
        self.model = model
        self.binary = binary
        self.runner = runner
        self.calls: list = []
        self._runner_takes_env = _accepts_env(runner)
        if runner is None and not shutil.which(binary):
            raise JevUnavailable(f"找不到命令 {binary!r}:先装 CodeBuddy 命令行,或传 binary=完整路径")

    def decide(self, state: str, questions: dict) -> dict:
        """单条:就是一批只有一条。"""
        return self.decide_batch([state], questions)[0]

    def decide_batch(self, states, questions: dict) -> list:
        """一次问一批,返回与 states 等长、同序的 answers 列表;某条解析不出来该位置为 None。

        整批失败(超时 / 退出码非 0 / is_error / 找不到 JSON)按 `retries` 重试整批,
        仍失败则这批每条记一条 ok=False 的调用并抛 `JevError`。
        限流(RateLimited)不重试,拿到就抛。
        """
        states = list(states)
        if not states:
            return []
        argv = [self.binary, "-p", self._prompt(states, questions), *TAIL_ARGS]
        if self.model:
            argv += ["--model", self.model]
        env = _env()
        t0 = time.perf_counter()
        last = ""
        for _ in range(self.retries + 1):
            try:
                items = _parse_batch(_result_text(self._run(argv, env)), len(states), questions)
                return self._record(states, items, time.perf_counter() - t0)
            except (RateLimited, PaymentRequired) as e:
                # 限流 / 欠费重试没有意义,但按条记一次失败再抛
                per = (time.perf_counter() - t0) * 1000.0 / len(states)
                self.calls.extend(JevCall(per, 0.0, 0, False, str(e)[:300]) for _ in states)
                raise
            except Exception as e:      # 超时、退出码、is_error、不是 JSON
                last = f"{type(e).__name__}: {e}"
        per = (time.perf_counter() - t0) * 1000.0 / len(states)
        self.calls.extend(JevCall(per, 0.0, 0, False, last[:300]) for _ in states)
        raise JevError(last[:300])

    @property
    def cache_name(self) -> str:
        """缓存里的客户端名:默认模型就是 codebuddy(老缓存继续命中);指定模型带后缀。"""
        return f"codebuddy@{self.model}" if self.model else "codebuddy"

    # ---------------- 内部 ----------------

    def _prompt(self, states, questions) -> str:
        return render_batch_prompt(states, questions)

    def _record(self, states, items, seconds) -> list:
        """按条记调用:延迟是本批总耗时除以条数,费用按命令行调用记 0。"""
        per = seconds * 1000.0 / len(states)
        for ans in items:
            self.calls.append(JevCall(per, 0.0, 0, ans is not None,
                                      "" if ans is not None else "这条没解析出答案"))
        return items

    def _run(self, argv, env) -> str:
        """跑一次子进程,返回 stdout;限流(stdout 空 + stderr 以 429 开头)抛 RateLimited。

        真实的限流长这样:退出码 0、stdout 空、stderr 一行
        "429 您的使用量已超出频率限制,将在 <时间> 重置,..."。
        """
        if self.runner is not None:
            if self._runner_takes_env:
                out = self.runner(argv, self.timeout, env=env)
            else:
                out = self.runner(argv, self.timeout)
            if isinstance(out, tuple):
                stdout, stderr = out
            else:
                stdout, stderr = out, ""
        else:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout,
                                  stdin=subprocess.DEVNULL, env=env)
            if proc.returncode != 0:
                raise JevError(f"{self.binary} 退出码 {proc.returncode}:{(proc.stderr or '')[:200]}")
            stdout, stderr = proc.stdout, proc.stderr or ""
        if not (stdout or "").strip() and (stderr or "").strip().startswith("429"):
            raise RateLimited(f"codebuddy 被限流:{stderr.strip()[:200]}")
        return stdout


def _accepts_env(runner) -> bool:
    """runner 的签名里有没有 env 参数:有就把子进程环境传进去给测试断言(A5)。"""
    if runner is None:
        return False
    try:
        import inspect

        return "env" in inspect.signature(runner).parameters
    except (TypeError, ValueError):      # 内置函数 / C 扩展取不到签名
        return False
