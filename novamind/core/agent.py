"""ReAct Agent：工具调用循环（kb_search / calculator / memory）。作者：晨星"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .llm import LLMProvider
from .pipeline import RAGPipeline

# ---------- 安全计算器（AST 白名单，禁 eval） ----------
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("表达式含不允许的语法")


def safe_calc(expr: str) -> float:
    return _eval_node(ast.parse(expr, mode="eval"))


@dataclass
class ToolCall:
    tool: str
    input: str
    output_preview: str


@dataclass
class AgentResult:
    reply: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class ToolBox:
    """工具注册表：名称 -> (描述, 执行函数)。"""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._memory: dict[str, str] = {}
        self.tools: dict[str, tuple[str, Callable[[str], str]]] = {
            "kb_search": (
                "检索知识库，输入查询词，返回最相关的内容摘要",
                lambda q: self._kb_search(pipeline, q),
            ),
            "calculator": (
                "数学计算，输入算术表达式（如 12*(3+4)），返回数值结果",
                lambda e: str(safe_calc(e)),
            ),
            "memory_save": (
                "保存事实到会话记忆，输入 key=value",
                self._memory_save,
            ),
            "memory_read": (
                "读取会话记忆，输入 key；输入 all 返回全部",
                self._memory_read,
            ),
        }

    @staticmethod
    def _kb_search(pipeline: RAGPipeline, query: str) -> str:
        hits = pipeline.retrieve(query, top_k=3)
        if not hits:
            return "知识库未命中"
        return "\n".join(
            f"[{h.chunk.chunk_id}] {h.chunk.text[:200]}" for h in hits
        )

    def _memory_save(self, kv: str) -> str:
        if "=" not in kv:
            return "格式应为 key=value"
        k, v = kv.split("=", 1)
        self._memory[k.strip()] = v.strip()
        return f"已保存 {k.strip()}"

    def _memory_read(self, key: str) -> str:
        key = key.strip()
        if key == "all":
            return "\n".join(f"{k}={v}" for k, v in self._memory.items()) or "记忆为空"
        return self._memory.get(key, "无此记忆")


_ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\[([^\]]*)\]", re.IGNORECASE)
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)

AGENT_SYSTEM = (
    "你是 ReAct 智能体。每轮输出以下三种之一：\n"
    "1) 调用工具：Action: 工具名[输入]\n"
    "2) 最终回答：Final Answer: 内容\n"
    "可用工具：\n{tool_desc}\n"
    "规则：需要查知识库用 kb_search；需要计算必须 calculator，禁止心算；"
    "最多 {max_steps} 步。"
)


class ReActAgent:
    """Thought-Action-Observation 循环。LLM 不可用时退化为规则直答。"""

    def __init__(self, llm: LLMProvider, pipeline: RAGPipeline, max_steps: int = 6) -> None:
        self.llm = llm
        self.pipeline = pipeline
        self.toolbox = ToolBox(pipeline)
        self.max_steps = max_steps

    def _system_prompt(self) -> str:
        desc = "\n".join(f"- {name}: {info[0]}" for name, info in self.toolbox.tools.items())
        return AGENT_SYSTEM.format(tool_desc=desc, max_steps=self.max_steps)

    def _rule_based(self, message: str) -> AgentResult:
        """离线降级：计算意图直走 calculator，其余直走 kb_search。"""
        calls: list[ToolCall] = []
        expr_m = re.search(r"[\d][\d\s+\-*/().%]+[\d)]", message)
        if expr_m:
            expr = expr_m.group(0).strip()
            try:
                out = self.toolbox.tools["calculator"][1](expr)
                calls.append(ToolCall("calculator", expr, out))
                return AgentResult(reply=f"计算结果：{out}", tool_calls=calls)
            except ValueError:
                pass
        out = self.toolbox.tools["kb_search"][1](message)
        calls.append(ToolCall("kb_search", message, out[:200]))
        return AgentResult(
            reply=f"知识库检索结果：\n{out}", tool_calls=calls
        )

    def run(self, message: str) -> AgentResult:
        name, _error = self.llm.status()
        if name == "mock-template":
            # MockLLM 无推理能力 → 规则路径
            return self._rule_based(message)

        calls: list[ToolCall] = []
        conversation = message
        for _ in range(self.max_steps):
            out = self.llm.generate(
                [
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": conversation},
                ],
                max_tokens=512,
            )
            final = _FINAL_RE.search(out)
            if final:
                return AgentResult(reply=final.group(1).strip(), tool_calls=calls)
            action = _ACTION_RE.search(out)
            if not action:
                return AgentResult(reply=out.strip() or "（无输出）", tool_calls=calls)
            tool_name, tool_input = action.group(1), action.group(2)
            tool = self.toolbox.tools.get(tool_name)
            if tool is None:
                observation = f"未知工具 {tool_name}"
            else:
                try:
                    observation = tool[1](tool_input)
                except Exception as exc:  # noqa: BLE001
                    observation = f"工具错误: {type(exc).__name__}: {exc}"
            calls.append(ToolCall(tool_name, tool_input, observation[:200]))
            conversation += (
                f"\nAssistant: {out}\nObservation: {observation}\n请继续。"
            )
        return AgentResult(reply="已达最大步数，未能收敛。", tool_calls=calls)
