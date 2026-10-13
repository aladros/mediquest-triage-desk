"""The safety rules of the triage-desk MCP server, tested without any model: the agent proposes, only a named nurse
decides, and nothing reaches the queue any other way."""

import json

import pytest

from mediquest_triage_desk import server, store

VITALS = {"respiratory_rate": 16, "heart_rate": 80, "systolic_bp": 124, "temperature_c": 36.8,
          "avpu": "alert", "mobility": "walking", "trauma": False}
NOW = "2026-10-13T09:00:00+03:00"


@pytest.fixture(autouse=True)
def fresh_desk(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIAGE_DATA_DIR", str(tmp_path))


def propose(pid="P1", age=40, complaint="Crushing chest pain since morning", vitals=VITALS, candidates=None):
    cands = candidates if candidates is not None else [{"id": "vu.chest_pain", "quote": "chest pain"}]
    return server.propose_priority(patient_id=pid, age_years=age, complaint=complaint, vitals=server.Vitals(**vitals),
                                   candidates=[server.Candidate(**c) for c in cands])


def test_proposal_is_pending_and_does_not_touch_the_queue():
    p = propose()
    assert p["colour"] == "orange" and p["proposal_id"]
    q = server.get_queue()
    assert q["queue"] == [] and [a["proposal_id"] for a in q["awaiting_nurse"]] == [p["proposal_id"]]


def test_only_a_named_nurse_can_commit():
    p = propose()
    with pytest.raises(ValueError, match="named nurse"):
        server.commit_priority(proposal_id=p["proposal_id"], nurse_name="  ", decision="confirm", confirmed_at_local=NOW)
    assert server.get_queue()["queue"] == []


def test_override_and_reject_need_a_reason():
    p = propose()
    with pytest.raises(ValueError, match="reason"):
        server.commit_priority(proposal_id=p["proposal_id"], nurse_name="Nurse A", decision="override",
                               override_colour="red", confirmed_at_local=NOW)
    with pytest.raises(ValueError, match="reason"):
        server.commit_priority(proposal_id=p["proposal_id"], nurse_name="Nurse A", decision="reject",
                               confirmed_at_local=NOW)


def test_confirm_sets_colour_and_see_by_from_the_facility_clock():
    p = propose()
    r = server.commit_priority(proposal_id=p["proposal_id"], nurse_name="Nurse A", decision="confirm",
                               confirmed_at_local=NOW)
    assert r["final_colour"] == "orange" and r["see_by"] == "2026-10-13T09:10+03:00"
    assert server.get_queue()["queue"][0]["confirmed_by"] == "Nurse A"


def test_a_proposal_cannot_be_decided_twice():
    p = propose()
    server.commit_priority(proposal_id=p["proposal_id"], nurse_name="Nurse A", decision="confirm",
                           confirmed_at_local=NOW)
    with pytest.raises(ValueError, match="already"):
        server.commit_priority(proposal_id=p["proposal_id"], nurse_name="Nurse B", decision="override",
                               override_colour="green", reason="second opinion", confirmed_at_local=NOW)


def test_server_revalidates_quotes_it_is_given():
    p = propose(candidates=[{"id": "em.cardiac_arrest", "quote": "cardiac arrest"}])  # not in the complaint
    assert p["colour"] == "green" and p["rejected_candidates"][0]["id"] == "em.cardiac_arrest"


def test_children_are_refused_and_missing_vitals_get_no_colour():
    assert propose(age=8)["status"] == "refused"
    no_bp = {k: v for k, v in VITALS.items() if k != "systolic_bp"}
    p = propose(complaint="closed fracture of the wrist", vitals=no_bp,
                candidates=[{"id": "ur.fracture_closed", "quote": "closed fracture"}])
    assert p["status"] == "needs_vitals" and p["proposal_id"] is None


def test_recheck_flags_but_never_moves_the_patient():
    p = propose(complaint="came for a refill", candidates=[])
    server.commit_priority(proposal_id=p["proposal_id"], nurse_name="Nurse A", decision="confirm",
                           confirmed_at_local=NOW)
    worse = {**VITALS, "respiratory_rate": 24, "heart_rate": 118, "systolic_bp": 92, "mobility": "with_help"}
    flag = server.flag_recheck(patient_id="P1", vitals=server.Vitals(**worse))
    assert flag["flag"] and flag["colour"] == "orange"
    assert server.get_queue()["queue"][0]["confirmed_colour"] == "green"  # unchanged until a nurse decides


def test_every_tool_call_is_audited():
    p = propose()
    server.commit_priority(proposal_id=p["proposal_id"], nurse_name="Nurse A", decision="confirm",
                           confirmed_at_local=NOW)
    lines = [json.loads(x) for x in (store.data_dir() / "audit.jsonl").read_text().splitlines()]
    tools = [x["tool"] for x in lines if x["event"] == "tool_call"]
    assert tools == ["propose_priority", "commit_priority"]
    assert lines[-1]["inputs"]["nurse_name"] == "Nurse A" and lines[-1]["ts"]
