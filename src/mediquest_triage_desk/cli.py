"""Command line for the triage desk.

  triage-desk demo     run the synthetic morning queue; YOU are the nurse at the gate
  triage-desk recheck  repeat vitals for a waiting patient -> deterioration flag -> nurse gate
  triage-desk queue    print the confirmed queue and what is awaiting a nurse
  triage-desk eval     run evals/cases.yaml N times and write evals/results.json
  triage-desk serve    run the triage-desk MCP server on stdio (for any MCP client)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from langgraph.types import Command

from . import store
from .agent import FACILITY_TZ, MODEL, MCPHub, build_graph

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
COLOURS = {"red": "\033[41;97m", "orange": "\033[48;5;208;30m", "yellow": "\033[43;30m", "green": "\033[42;30m"}


def chip(colour: str | None) -> str:
    return f"{COLOURS.get(colour or '', '')} {(colour or 'NO COLOUR').upper()} {RESET}"


def short(obj: Any, limit: int = 160) -> str:
    text = json.dumps(obj, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def print_call(rec: dict[str, Any]) -> None:
    who = {"model": "model", "agent": "agent", "gate": "gate "}.get(rec["caller"], rec["caller"])
    flag = " ERROR" if rec["is_error"] else ""
    print(f"  {DIM}[{who}] → {rec['server']}.{rec['tool']} {short(rec['inputs'], 120)}{RESET}")
    print(f"  {DIM}        ← {rec['latency_ms']}ms{flag} {short(rec['output'])}{RESET}")


def print_proposal(p: dict[str, Any]) -> None:
    print(f"\n  {BOLD}Proposed priority for {p['patient_id']}:{RESET} {chip(p['colour'])}  "
          f"TEWS {p.get('tews_total')} · driven by {p.get('driver')} · see within {p.get('target_minutes')} min")
    if p.get("status") == "proposed_vitals_incomplete":
        print(f"  ⚠ vitals incomplete ({', '.join(p['missing_vitals'])}) — colour is from a discriminator only")
    print("  Why (cited):")
    for c in p.get("citations", []):
        print(f"    • [{c['cite']}] {c['detail']}")
    for u in p.get("unverified", []):
        print(f"    ⚠ UNVERIFIED — model suggested [{u['id']}] {u['label']} ({u['colour']}) but could not cite it "
              f"(tried \"{u['quote']}\"). Not in the colour. Check with the patient.")
    for r in p.get("rejected_candidates", []):
        print(f"    {DIM}✗ dropped [{r['id']}] \"{r['quote']}\" — {r['reason']}{RESET}")


def ask_nurse(nurse: str) -> dict[str, Any]:
    while True:
        ans = input(f"\n  {BOLD}{nurse}{RESET}: [c]onfirm / [o]verride / [r]eject ? ").strip().lower()
        if ans in ("c", "confirm"):
            return {"nurse_name": nurse, "decision": "confirm"}
        if ans in ("o", "override"):
            colour = input("  new colour (red/orange/yellow/green): ").strip().lower()
            reason = input("  reason: ").strip()
            if colour in COLOURS and reason:
                return {"nurse_name": nurse, "decision": "override", "override_colour": colour, "reason": reason}
            print("  an override needs a valid colour and a reason")
        if ans in ("r", "reject"):
            reason = input("  reason: ").strip()
            if reason:
                return {"nurse_name": nurse, "decision": "reject", "reason": reason}


async def run_patient(graph, patient: dict[str, Any], decide) -> dict[str, Any]:
    config = {"configurable": {"thread_id": f"{patient['patient_id']}-{uuid.uuid4().hex[:6]}"}}
    state = await graph.ainvoke(patient, config)
    while "__interrupt__" in state:
        payload = state["__interrupt__"][0].value
        decision = await decide(payload)
        state = await graph.ainvoke(Command(resume=decision), config)
    return state


async def demo(args) -> None:
    patients = json.loads(Path(args.patients).read_text())
    nurse = args.nurse or input("Name of the triage nurse at the desk: ").strip()
    if not nurse:
        sys.exit("A named nurse is required — the agent proposes, the nurse decides.")
    print(f"{DIM}model={MODEL} (open-weights, local)  facility tz={FACILITY_TZ}  audit log={store.data_dir()}/audit.jsonl{RESET}")
    async with MCPHub(on_call=print_call) as hub:
        graph = build_graph(hub)

        async def decide(payload):
            print_proposal(payload["proposal"])
            return ask_nurse(nurse)

        for p in patients:
            print(f"\n{BOLD}━━ {p['patient_id']} · {p['age_years']}y · \"{p['complaint']}\"{RESET}")
            state = await run_patient(graph, p, decide)
            outcome = state.get("outcome")
            if outcome == "handoff_child":
                print("  ↪ handed to the nurse: under 13, the adult SATS chart does not apply")
            elif outcome == "needs_vitals":
                print(f"  ↪ {state['proposal'].get('message')}")
            else:
                d = state["decision"]
                print(f"  ✔ {outcome} by {d['decided_by']} → {chip(d.get('final_colour'))} see by {d.get('see_by')}")
        q = await hub.call("triage-desk", "get_queue", {}, caller="agent")
    print_queue(q)


async def recheck(args) -> None:
    spec = json.loads(Path(args.file).read_text())
    nurse = args.nurse or input("Name of the triage nurse at the desk: ").strip()
    print(f"\n{BOLD}━━ Re-check {spec['patient_id']}{RESET} — {spec.get('note', '')}")
    async with MCPHub(on_call=print_call) as hub:
        flag = await hub.call("triage-desk", "flag_recheck",
                              {"patient_id": spec["patient_id"], "vitals": spec["vitals"]}, caller="agent")
        if not flag.get("flag"):
            print(f"  no flag: {flag.get('reason')}")
            return
        print(f"\n  {BOLD}Deterioration flag:{RESET} {chip(flag['previous_colour'])} → {chip(flag['colour'])} "
              f"(TEWS {flag['tews_total']}) — second look requested, patient NOT moved")
        for c in flag["citations"]:
            print(f"    • [{c['cite']}] {c['detail']}")
        d = ask_nurse(nurse)
        clock = await hub.call("time", "get_current_time", {"timezone": FACILITY_TZ}, caller="gate")
        res = await hub.call("triage-desk", "commit_priority", {
            "proposal_id": flag["proposal_id"], "confirmed_at_local": clock["datetime"], **d}, caller="gate")
        print(f"  ✔ {res['status']} by {res['decided_by']} → {chip(res['final_colour'])} see by {res['see_by']}")
        print_queue(await hub.call("triage-desk", "get_queue", {}, caller="agent"))


def print_queue(q: dict[str, Any]) -> None:
    print(f"\n{BOLD}Queue (confirmed by a nurse):{RESET}")
    for i, r in enumerate(q["queue"], 1):
        print(f"  {i}. {r['patient_id']:<9} {chip(r['confirmed_colour'])} see by {r['see_by']}  ({r['confirmed_by']})")
    if q["awaiting_nurse"]:
        print(f"{BOLD}Awaiting a nurse:{RESET}")
        for r in q["awaiting_nurse"]:
            print(f"  - {r['patient_id']} {r['kind']} proposal {r['proposal_id']} ({r['proposed_colour']})")


async def show_queue(args) -> None:
    print_queue(store.queue())


def main() -> None:
    ap = argparse.ArgumentParser(prog="triage-desk", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--patients", default="data/demo_patients.json")
    d.add_argument("--nurse", help="name recorded on every decision")
    r = sub.add_parser("recheck")
    r.add_argument("--file", default="data/demo_recheck.json")
    r.add_argument("--nurse")
    sub.add_parser("queue")
    e = sub.add_parser("eval")
    e.add_argument("--runs", type=int, default=3)
    e.add_argument("--cases", default="evals/cases.yaml")
    e.add_argument("--only", help="comma-separated case ids")
    e.add_argument("--out", default="evals/results.json")
    sub.add_parser("serve")
    args = ap.parse_args()
    if args.cmd == "serve":
        from .server import main as serve
        return serve()
    if args.cmd == "eval":
        from .evals import run_evals
        return asyncio.run(run_evals(args))
    asyncio.run({"demo": demo, "recheck": recheck, "queue": show_queue}[args.cmd](args))
