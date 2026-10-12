"""`triage-desk` MCP server — SATS adult scoring tools for an outpatient triage desk.

Tools that only compute: list_discriminators, compute_tews, match_discriminators, get_queue.
Tools that act:         propose_priority (writes a PENDING proposal), flag_recheck (raises a PENDING re-check flag),
                        commit_priority (records a named nurse's decision — the only tool that changes the queue).

The server trusts nothing from its caller: propose_priority re-scores the vitals and re-validates every
discriminator quote itself, so a model that skips or fakes a step cannot get an unsourced colour written down.
Every tool call is appended to the audit log with its inputs, output and timestamp.

Run standalone (stdio):  uv run python -m mediquest_triage_desk.server
"""

from __future__ import annotations

import functools
import json
from datetime import datetime, timedelta
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from . import sats, store

mcp = FastMCP("triage-desk", log_level="WARNING")


class Vitals(BaseModel):
    respiratory_rate: float | None = Field(None, description="breaths per minute")
    heart_rate: float | None = Field(None, description="beats per minute")
    systolic_bp: float | None = Field(None, description="mmHg")
    temperature_c: float | None = Field(None, description="degrees Celsius")
    avpu: Literal["alert", "voice", "pain", "unresponsive"] | None = None
    mobility: Literal["walking", "with_help", "stretcher"] | None = None
    trauma: bool | None = None


class Candidate(BaseModel):
    id: str = Field(description="discriminator id from list_discriminators, e.g. 'vu.chest_pain'")
    quote: str = Field(description="the exact words from the presenting complaint that support it")


def audited(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # logged, then surfaced to the agent as a tool error it can recover from
            store.audit("tool_call", server="triage-desk", tool=fn.__name__, inputs=_plain(kwargs), error=str(exc))
            raise
        store.audit("tool_call", server="triage-desk", tool=fn.__name__, inputs=_plain(kwargs), output=result)
        return result
    return wrapper


def _plain(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o)))


def _vitals(v: Vitals | dict[str, Any]) -> dict[str, Any]:
    return v.model_dump() if isinstance(v, Vitals) else dict(v)


@mcp.tool()
@audited
def list_discriminators() -> dict[str, Any]:
    """The closed list of SATS adult clinical discriminators (id, label, colour). Only these ids can be matched."""
    return {"source": sats.chart()["source"],
            "discriminators": [{"id": d["id"], "label": d["label"], "colour": d["colour"]}
                               for d in sats.chart()["discriminators"]]}


@mcp.tool()
@audited
def compute_tews(vitals: Vitals) -> dict[str, Any]:
    """Score the adult Triage Early Warning Score from vitals. Each point cites its chart row.
    Missing vitals are reported back, never assumed normal."""
    return sats.compute_tews(_vitals(vitals))


@mcp.tool()
@audited
def match_discriminators(complaint: str, candidates: list[Candidate]) -> dict[str, Any]:
    """Validate candidate SATS discriminators against the presenting complaint. A candidate is accepted only if its id
    is on the chart and its quote appears word-for-word in the complaint. Rejections say why, so you can retry."""
    return sats.check_discriminators(complaint, [c.model_dump() for c in candidates])


@mcp.tool()
@audited
def propose_priority(patient_id: str, age_years: int, complaint: str, vitals: Vitals,
                     candidates: list[Candidate], unverified: list[Candidate] | None = None) -> dict[str, Any]:
    """Write a PENDING priority proposal for a nurse to confirm. Re-scores TEWS and re-validates every discriminator
    server-side; the proposal carries the citation for each contributing vital band and complaint phrase.
    `unverified`: discriminators the agent raised but could not cite. They never affect the colour; they are shown to
    the nurse as "check with the patient". Does NOT change the queue."""
    if age_years < sats.chart()["min_age_years"]:
        return {"status": "refused", "reason": f"age {age_years}: SATS paediatric charts are not implemented — "
                                               "triage this child by hand"}
    v = _vitals(vitals)
    tews = sats.compute_tews(v)
    checked = sats.check_discriminators(complaint, [c.model_dump() for c in candidates])
    result = sats.combine(tews, checked["accepted"])
    result["rejected_candidates"] = checked["rejected"]
    index = sats.discriminator_index()
    result["unverified"] = [{"id": u.id, "label": index[u.id]["label"], "colour": index[u.id]["colour"],
                             "quote": u.quote} for u in (unverified or []) if u.id in index]
    store.register_patient(patient_id, age_years, complaint, v)
    if result["status"] == "needs_vitals":
        return {"patient_id": patient_id, "proposal_id": None, **result,
                "message": f"No colour proposed: measure {', '.join(tews['missing'] + tews['invalid'])} first."}
    proposal_id = store.create_proposal(patient_id, "priority", result["colour"], result)
    return {"patient_id": patient_id, "proposal_id": proposal_id, **result}


@mcp.tool()
@audited
def flag_recheck(patient_id: str, vitals: Vitals) -> dict[str, Any]:
    """Re-score a waiting patient's repeat vitals. If the TEWS colour is now worse than the confirmed priority,
    raise a PENDING re-check flag for a nurse's second look. Never moves the patient itself."""
    p = store.patient(patient_id)
    if p is None:
        raise ValueError(f"unknown patient {patient_id}")
    tews = sats.compute_tews(_vitals(vitals))
    current = p["confirmed_colour"]
    if not tews["complete"]:
        return {"flag": False, "reason": f"re-check incomplete: measure {', '.join(tews['missing'])}", "tews": tews}
    if not sats.worse(tews["colour"], current):
        return {"flag": False, "reason": f"TEWS {tews['total']} ({tews['colour']}) is not worse than confirmed "
                                         f"{current}", "tews": tews}
    payload = {"previous_colour": current, "tews_total": tews["total"], "colour": tews["colour"],
               "target_minutes": sats.chart()["target_minutes"][tews["colour"]],
               "citations": [{"kind": "vital", "cite": i["cite"], "detail": f'{i["label"]} ({i["value"]}) -> {i["score"]}'}
                             for i in tews["items"] if i["score"] > 0]}
    proposal_id = store.create_proposal(patient_id, "recheck", tews["colour"], payload)
    return {"flag": True, "proposal_id": proposal_id, **payload}


@mcp.tool()
@audited
def commit_priority(proposal_id: str, nurse_name: str, decision: Literal["confirm", "override", "reject"],
                    confirmed_at_local: str, override_colour: Literal["red", "orange", "yellow", "green"] | None = None,
                    reason: str | None = None) -> dict[str, Any]:
    """Record a NAMED nurse's decision on a pending proposal. This is the only tool that changes the queue.
    `confirmed_at_local` is the facility-local ISO time (from the time server); the see-by time is computed from it.
    Called only by the human gate — never on the model's own initiative."""
    if not nurse_name.strip():
        raise ValueError("a named nurse or clinician must make this decision")
    proposal = store.get_proposal(proposal_id)
    if proposal is None:
        raise ValueError(f"no proposal {proposal_id}")
    if decision == "override" and (not override_colour or not (reason or "").strip()):
        raise ValueError("an override needs the new colour and a reason")
    if decision == "reject" and not (reason or "").strip():
        raise ValueError("a rejection needs a reason")
    final = {"confirm": proposal["colour"], "override": override_colour, "reject": None}[decision]
    see_by = None
    if final:
        minutes = sats.chart()["target_minutes"][final]
        see_by = (datetime.fromisoformat(confirmed_at_local) + timedelta(minutes=minutes)).isoformat(timespec="minutes")
    status = {"confirm": "confirmed", "override": "overridden", "reject": "rejected"}[decision]
    decided = store.decide_proposal(proposal_id, nurse_name.strip(), status, final, reason, see_by)
    return {"proposal_id": proposal_id, "patient_id": decided["patient_id"], "status": status,
            "final_colour": final, "decided_by": nurse_name.strip(), "see_by": see_by}


@mcp.tool()
@audited
def get_queue() -> dict[str, Any]:
    """The waiting queue ordered by CONFIRMED priority, plus proposals still awaiting a nurse."""
    return store.queue()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
