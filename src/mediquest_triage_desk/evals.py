"""Eval harness: runs every case in evals/cases.yaml N times through the real graph, real MCP servers and the real
local model, then writes evals/results.json. Nothing is mocked.

The gate is answered by a scripted reviewer named "EVAL HARNESS (not a clinician)" that always REJECTS, so an eval
run can never leave a confirmed priority behind and the audit log says plainly that no clinician was involved.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .agent import MODEL, TEMPERATURE


async def run_evals(args) -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.environ["TRIAGE_DATA_DIR"] = str(Path("evals/runs") / stamp)  # must be set before the MCP servers spawn
    from .agent import MCPHub, build_graph
    from .cli import run_patient

    cases = yaml.safe_load(Path(args.cases).read_text())
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    async def decide(payload):
        return {"nurse_name": "EVAL HARNESS (not a clinician)", "decision": "reject",
                "reason": "automated eval run - no clinical decision"}

    results = []
    async with MCPHub() as hub:
        graph = build_graph(hub)
        for case in cases:
            runs = []
            for i in range(args.runs):
                patient = {"patient_id": f"EVAL-{case['id']}-{i}", "age_years": case["age_years"],
                           "complaint": case["complaint"], "vitals": case["vitals"]}
                t0 = time.perf_counter()
                state = await run_patient(graph, patient, decide)
                seconds = round(time.perf_counter() - t0, 1)
                runs.append(judge(case, state, seconds))
                print(f"{case['id']:<26} run {i + 1}: {'PASS' if runs[-1]['pass'] else 'FAIL'}  "
                      f"colour={runs[-1]['colour']} accepted={runs[-1]['accepted']} unverified={runs[-1]['unverified']} {seconds}s "
                      f"{'; '.join(runs[-1]['failures'])}")
            results.append({"id": case["id"], "why": case["why"], "expect": case["expect"],
                            "pass": all(r["pass"] for r in runs),
                            "pass_rate": f"{sum(r['pass'] for r in runs)}/{len(runs)}",
                            "colours": sorted({str(r["colour"]) for r in runs}),
                            "median_seconds": statistics.median(r["seconds"] for r in runs), "runs": runs})

    out = {"model": MODEL, "temperature": TEMPERATURE, "runs_per_case": args.runs, "started": stamp,
           "passed": sum(r["pass"] for r in results), "total": len(results), "cases": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n{out['passed']}/{out['total']} cases passed on every run ({args.runs} runs each, model {MODEL}). "
          f"Wrote {args.out}; audit log in {os.environ['TRIAGE_DATA_DIR']}/audit.jsonl")


def judge(case: dict[str, Any], state: dict[str, Any], seconds: float) -> dict[str, Any]:
    exp = case["expect"]
    proposal = state.get("proposal") or {}
    outcome = state.get("outcome")
    got_outcome = "proposed" if outcome in ("rejected", "confirmed", "overridden") else outcome
    accepted = [c["id"] for c in state.get("candidates", [])]
    colour = proposal.get("colour")
    failures = []
    want_outcome = exp.get("outcome", "proposed")
    if got_outcome != want_outcome:
        failures.append(f"outcome {got_outcome} != {want_outcome}")
    if "colour" in exp and colour != exp["colour"]:
        failures.append(f"colour {colour} != {exp['colour']}")
    for d in exp.get("must_include", []):
        if d not in accepted:
            failures.append(f"missing {d}")
    for d in exp.get("must_not", []):
        if d in accepted:
            failures.append(f"wrongly matched {d}")
    if proposal.get("proposal_id") and not proposal.get("citations"):
        failures.append("proposal without citations")
    return {"pass": not failures, "failures": failures, "outcome": got_outcome, "colour": colour,
            "accepted": accepted, "unverified": [u["id"] for u in proposal.get("unverified", [])],
            "extract_turns": state.get("extract_turns"),
            "extract_note": state.get("extract_note"), "seconds": seconds}
