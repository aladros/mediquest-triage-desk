"""Notebook front end (Kaggle / Jupyter). Same graph, same MCP servers, same gate as the CLI — but the nurse's decision
is its own cell, so the pause is visible: the graph stops at the gate and nothing moves until `decide(...)` runs.

    desk = await TriageDesk.open()
    await desk.triage(patient)                  # prints tool calls + the cited proposal, then PAUSES at the gate
    await desk.decide("Nurse Wanjiku", "confirm")
    await desk.decide("Nurse Wanjiku", "override", colour="red", reason="looks grey and clammy")
    await desk.recheck("OPD-0413", vitals)       # deterioration flag -> pauses for decide(...)
    await desk.queue()
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from langgraph.types import Command

from .agent import FACILITY_TZ, MODEL, MCPHub, build_graph
from .cli import BOLD, RESET, chip, print_call, print_proposal, print_queue


class TriageDesk:
    def __init__(self, hub: MCPHub):
        self.hub = hub
        self.graph = build_graph(hub)
        self.pending: dict[str, Any] | None = None

    _current: "TriageDesk | None" = None

    @classmethod
    async def open(cls) -> "TriageDesk":
        # The MCP stdio connections are anyio contexts: they must be opened and closed in the SAME task. Notebook cells
        # run as separate tasks, so one long-lived task owns the hub and the cells only send requests through it.
        if cls._current is not None:
            await cls._current.close()
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        stop = asyncio.Event()

        async def own_hub() -> None:
            try:
                async with MCPHub(on_call=print_call) as hub:
                    ready.set_result(hub)
                    await stop.wait()
            except Exception as exc:  # surface start-up failures to the cell that asked
                if not ready.done():
                    ready.set_exception(exc)

        owner = asyncio.create_task(own_hub())
        hub = await ready
        desk = cls(hub)
        desk._stop, desk._owner = stop, owner
        cls._current = desk
        tools = {name: [t.name for t in (await s.list_tools()).tools] for name, s in hub.sessions.items()}
        print(f"model={MODEL} (open-weights, Ollama)  facility tz={FACILITY_TZ}")
        for name, names in tools.items():
            print(f"MCP server {name}: {', '.join(names)}")
        return desk

    async def triage(self, patient: dict[str, Any]) -> None:
        if self.pending:
            print("A proposal is still waiting at the gate — decide(...) first.")
            return
        print(f"{BOLD}━━ {patient['patient_id']} · {patient['age_years']}y · \"{patient['complaint']}\"{RESET}")
        config = {"configurable": {"thread_id": f"{patient['patient_id']}-{uuid.uuid4().hex[:6]}"}}
        state = await self.graph.ainvoke(patient, config)
        if "__interrupt__" in state:
            payload = state["__interrupt__"][0].value
            print_proposal(payload["proposal"])
            print(f"\n  ⏸  PAUSED at the nurse gate — nothing has changed in the queue. Run decide(...).")
            self.pending = {"kind": "graph", "config": config}
        elif state.get("outcome") == "handoff_child":
            print("  ↪ handed to the nurse: under 13, the adult SATS chart does not apply")
        elif state.get("outcome") == "needs_vitals":
            print(f"  ↪ {state['proposal'].get('message')}")

    async def recheck(self, patient_id: str, vitals: dict[str, Any]) -> None:
        flag = await self.hub.call("triage-desk", "flag_recheck", {"patient_id": patient_id, "vitals": vitals},
                                   caller="agent")
        if not flag.get("flag"):
            print(f"  no flag: {flag.get('reason')}")
            return
        print(f"\n  {BOLD}Deterioration flag:{RESET} {chip(flag['previous_colour'])} → {chip(flag['colour'])} "
              f"(TEWS {flag['tews_total']}) — second look requested, patient NOT moved")
        for c in flag["citations"]:
            print(f"    • [{c['cite']}] {c['detail']}")
        print(f"\n  ⏸  PAUSED at the nurse gate. Run decide(...).")
        self.pending = {"kind": "recheck", "proposal_id": flag["proposal_id"]}

    async def decide(self, nurse_name: str, decision: str, colour: str | None = None,
                     reason: str | None = None) -> None:
        if not self.pending:
            print("Nothing is waiting at the gate.")
            return
        # Same rules commit_priority enforces server-side; checked here too so a bad entry keeps the patient at the gate.
        problem = ("a named nurse or clinician must make this decision" if not nurse_name.strip() else
                   "decision must be confirm, override or reject" if decision not in ("confirm", "override", "reject")
                   else "an override needs the new colour and a reason" if decision == "override" and not (colour and reason)
                   else "a rejection needs a reason" if decision == "reject" and not reason else None)
        if problem:
            print(f"  ✗ gate refused: {problem}. Still waiting at the gate.")
            return
        d = {"nurse_name": nurse_name, "decision": decision, "override_colour": colour, "reason": reason}
        pending, self.pending = self.pending, None
        if pending["kind"] == "graph":
            state = await self.graph.ainvoke(Command(resume=d), pending["config"])
            res = state["decision"]
        else:
            clock = await self.hub.call("time", "get_current_time", {"timezone": FACILITY_TZ}, caller="gate")
            res = await self.hub.call("triage-desk", "commit_priority", {
                "proposal_id": pending["proposal_id"], "confirmed_at_local": clock["datetime"], **d}, caller="gate")
        print(f"  ✔ {res['status']} by {res['decided_by']} → {chip(res.get('final_colour'))} see by {res.get('see_by')}")

    async def queue(self) -> None:
        print_queue(await self.hub.call("triage-desk", "get_queue", {}, caller="agent"))

    async def close(self) -> None:
        self._stop.set()
        await self._owner
        if TriageDesk._current is self:
            TriageDesk._current = None
