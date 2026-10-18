"""从真实的 MCP server 拉取 tools/list,存成工具目录 jsonl。

配置文件格式与常见 MCP 客户端一致:
{"mcpServers": {"filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}}}

需要可选依赖:pip install mcp
"""

import asyncio
import json


async def _list_tools(name: str, cfg: dict) -> list:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=cfg["command"], args=cfg.get("args", []), env=cfg.get("env"))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
    tools = []
    for t in result.tools:
        ann = getattr(t, "annotations", None)
        tools.append({
            "server": name,
            "name": t.name,
            "description": t.description or "",
            "inputSchema": t.inputSchema or {},
            "annotations": ann.model_dump(exclude_none=True) if ann else {},
        })
    return tools


def fetch_all(config_path: str, out_path: str) -> int:
    with open(config_path, encoding="utf-8") as f:
        servers = json.load(f).get("mcpServers", {})
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for name, cfg in servers.items():
            try:
                tools = asyncio.run(_list_tools(name, cfg))
            except Exception as e:  # 单个 server 失败不影响其他 server
                print(f"[skip] {name}: {e}")
                continue
            for t in tools:
                out.write(json.dumps(t, ensure_ascii=False) + "\n")
            n += len(tools)
            print(f"[ok] {name}: {len(tools)} tools")
    return n
