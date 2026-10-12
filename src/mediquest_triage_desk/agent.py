"""The triage agent: a LangGraph graph over two MCP servers.

    intake ─► score_vitals ─► extract ⇄ (match_discriminators tool loop, self-correcting) ─► propose ─► gate ─► commit
       │                                                                                       │        (nurse)
       └─► handoff (child / no adult chart)                                     needs_vitals ◄─┘

* `triage-desk` (ours): SATS scoring tools. `time` (borrowed, modelcontextprotocol/servers): facility-local clock.
* The open-weights model (Ollama) does ONE job — read the complaint and quote the SATS discriminators it supports —
  through the `match_discriminators` tool, retrying when the server rejects a quote. It never sees commit_priority.
* The colour comes from deterministic tools. The gate is a LangGraph interrupt: nothing reaches the queue until a
  named nurse confirms, overrides (with reason) or rejects.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import AsyncExitStack
from typing import Any, Callable, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from . import store

FACILITY_TZ = os.environ.get("TRIAGE_FACILITY_TZ", "Africa/Nairobi")
MODEL = os.environ.get("TRIAGE_MODEL", "qwen2.5:3b")
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
TEMPERATURE = float(os.environ.get("TRIAGE_TEMPERATURE", "0"))
MAX_EXTRACT_TURNS = 3


class MCPHub:
    """Holds one stdio session per MCP server and logs every call (inputs, output, timestamp, latency)."""

    def __init__(self, on_call: Callable[[dict[str, Any]], None] | None = None):
        self._stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.on_call = on_call

    async def __aenter__(self) -> "MCPHub":
        servers = {
            "triage-desk": StdioServerParameters(command=sys.executable, args=["-m", "mediquest_triage_desk.server"],
                                                 env={**os.environ}),
            "time": StdioServerParameters(command="uvx", args=["mcp-server-time", f"--local-timezone={FACILITY_TZ}"],
                                          env={**os.environ}),
        }
        errlog = self._stack.enter_context((store.data_dir() / "mcp-servers.log").open("a"))
        for name, params in servers.items():
            read, write = await self._stack.enter_async_context(stdio_client(params, errlog=errlog))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[name] = session
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    async def tool_schema(self, server: str, tool: str) -> dict[str, Any]:
        for t in (await self.sessions[server].list_tools()).tools:
            if t.name == tool:
                return {"type": "function", "function": {"name": t.name, "description": t.description or "",
                                                         "parameters": t.inputSchema}}
        raise KeyError(f"{server} has no tool {tool}")

    async def call(self, server: str, tool: str, args: dict[str, Any], caller: str) -> dict[str, Any]:
        started = time.perf_counter()
        res = await self.sessions[server].call_tool(tool, args)
        ms = round((time.perf_counter() - started) * 1000)
        text = "".join(getattr(c, "text", "") for c in res.content)
        payload = res.structuredContent if res.structuredContent is not None else _maybe_json(text)
        if isinstance(payload, dict) and set(payload) == {"result"}:
            payload = payload["result"]
        record = store.audit("mcp_call", server=server, tool=tool, caller=caller, inputs=args,
                             output=payload, is_error=bool(res.isError), latency_ms=ms)
        if self.on_call:
            self.on_call(record)
        if res.isError:
            raise ToolError(text or f"{tool} failed")
        return payload


class ToolError(RuntimeError):
    pass


def _maybe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


class TriageState(TypedDict, total=False):
    patient_id: str
    age_years: int
    complaint: str
    vitals: dict[str, Any]
    tews: dict[str, Any]
    candidates: list[dict[str, Any]]
    extract_turns: int
    extract_note: str
    proposal: dict[str, Any]
    decision: dict[str, Any]
    outcome: str


SYSTEM = """You assist a triage nurse using the South African Triage Scale (SATS, adult chart).
Your ONLY job: read the presenting complaint and pick the SATS clinical discriminators it explicitly supports.

Rules:
- Use only ids from this list:
{discriminators}
- For each one, `quote` must be words copied EXACTLY from the complaint (a short phrase, not a paraphrase).
  Use the patient's own wording, not the chart label (e.g. "coughed up blood", not "Coughing blood").
- The complaint may be written in Kiswahili or another language. Match its meaning to the chart, but quote the
  original-language words exactly as written.
- Only pick what the complaint actually states. Do not infer conditions, do not diagnose, do not guess severity.
  Pain counts as "severe" or "moderate" only if the complaint says so or gives a pain score (severe: 8-10, moderate: 5-7).
- Vitals are scored separately by a calculator; do not use them to pick discriminators.
- If nothing on the list is supported, call the tool with an empty candidates list.
- Call `match_discriminators` with the complaint and your candidates. If it rejects any, fix or drop them and call it
  again. When everything you submitted is accepted, reply with the single word DONE."""


def build_graph(hub: MCPHub, llm: Any | None = None, checkpointer: Any | None = None):
    llm = llm or ChatOllama(model=MODEL, base_url=OLLAMA_URL, temperature=TEMPERATURE, num_ctx=8192)

    async def intake(state: TriageState) -> TriageState:
        return {"extract_turns": 0, "candidates": []}

    def route_intake(state: TriageState) -> str:
        return "handoff" if int(state.get("age_years") or 0) < 13 else "score_vitals"

    async def handoff(state: TriageState) -> TriageState:
        store.audit("handoff", patient_id=state["patient_id"], reason="under 13: no adult SATS chart; nurse triages")
        return {"outcome": "handoff_child"}

    async def score_vitals(state: TriageState) -> TriageState:
        tews = await hub.call("triage-desk", "compute_tews", {"vitals": state["vitals"]}, caller="agent")
        return {"tews": tews}

    async def extract(state: TriageState) -> TriageState:
        listing = await hub.call("triage-desk", "list_discriminators", {}, caller="agent")
        lines = "\n".join(f'  {d["id"]}: {d["label"]}' for d in listing["discriminators"])
        tool = await hub.tool_schema("triage-desk", "match_discriminators")
        bound = llm.bind_tools([tool])
        messages: list[Any] = [SystemMessage(SYSTEM.format(discriminators=lines)),
                               HumanMessage(f"Presenting complaint:\n{state['complaint']}")]
        accepted: list[dict[str, Any]] = []
        note = "model never called the tool"
        for turn in range(1, MAX_EXTRACT_TURNS + 1):
            reply: AIMessage = await bound.ainvoke(messages)
            messages.append(reply)
            if not reply.tool_calls:
                note = "model finished" if turn > 1 else note
                break
            for call in reply.tool_calls:
                args = dict(call.get("args") or {})
                args["complaint"] = state["complaint"]  # the model cannot validate against a different text
                try:
                    result = await hub.call("triage-desk", "match_discriminators", args, caller="model")
                    accepted = result["accepted"]
                    content = json.dumps(result)
                    note = "accepted" if not result["rejected"] else f"{len(result['rejected'])} rejected"
                except ToolError as exc:
                    content = f"ERROR: {exc}. Fix the arguments and call again."
                    note = f"tool error: {exc}"
                messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
            if note == "accepted":
                break
        store.audit("extract_done", patient_id=state["patient_id"], turns=turn, note=note,
                    accepted=[a["id"] for a in accepted])
        return {"candidates": [{"id": a["id"], "quote": a["quote"]} for a in accepted],
                "extract_turns": turn, "extract_note": note}

    async def propose(state: TriageState) -> TriageState:
        proposal = await hub.call("triage-desk", "propose_priority", {
            "patient_id": state["patient_id"], "age_years": int(state["age_years"]), "complaint": state["complaint"],
            "vitals": state["vitals"], "candidates": state.get("candidates", [])}, caller="agent")
        return {"proposal": proposal}

    def route_proposal(state: TriageState) -> str:
        p = state["proposal"]
        if p.get("status") == "refused":
            return "handoff"
        return "gate" if p.get("proposal_id") else "needs_vitals"

    async def needs_vitals(state: TriageState) -> TriageState:
        return {"outcome": "needs_vitals"}

    async def gate(state: TriageState) -> TriageState:
        # Pauses the graph. Resumes only with a named person's decision.
        decision = interrupt({"proposal": state["proposal"], "extract_note": state.get("extract_note")})
        store.audit("gate_decision", patient_id=state["patient_id"], proposal_id=state["proposal"]["proposal_id"],
                    **decision)
        return {"decision": decision}

    async def commit(state: TriageState) -> TriageState:
        d = state["decision"]
        clock = await hub.call("time", "get_current_time", {"timezone": FACILITY_TZ}, caller="gate")
        result = await hub.call("triage-desk", "commit_priority", {
            "proposal_id": state["proposal"]["proposal_id"], "nurse_name": d["nurse_name"],
            "decision": d["decision"], "confirmed_at_local": clock["datetime"],
            "override_colour": d.get("override_colour"), "reason": d.get("reason")}, caller="gate")
        return {"outcome": result["status"], "decision": {**d, **result}}

    g = StateGraph(TriageState)
    for name, fn in [("intake", intake), ("handoff", handoff), ("score_vitals", score_vitals), ("extract", extract),
                     ("propose", propose), ("needs_vitals", needs_vitals), ("gate", gate), ("commit", commit)]:
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_intake, ["handoff", "score_vitals"])
    g.add_edge("score_vitals", "extract")
    g.add_edge("extract", "propose")
    g.add_conditional_edges("propose", route_proposal, ["gate", "needs_vitals", "handoff"])
    g.add_edge("gate", "commit")
    g.add_edge("commit", END)
    g.add_edge("needs_vitals", END)
    g.add_edge("handoff", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
