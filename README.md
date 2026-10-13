# MediQuest Triage Desk

**An agent for the outpatient triage desk that turns vitals and a presenting complaint into a South African Triage
Scale (SATS) priority in seconds — every colour cited to the chart row or the patient's own words — and then stops.
A named nurse confirms, overrides or rejects it. Nothing moves in the queue without her.**

Built for the *African Agentic AI Design Challenge* — Healthcare track, "The Triage Desk",
theme **Symptom-based priority scoring**. Synthetic patients only. Not a medical device; not for clinical use.

**Demo video (2:25, unedited run on a Kaggle T4, narrated):** https://youtu.be/GpeM8qaBvJo

## Run it (one command)

Prerequisites: [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/download). Then:

```bash
./run.sh
```

That starts Ollama if needed, pulls the open-weights model (`qwen2.5:7b`, ~4.7 GB, first run only), and runs the demo:
six synthetic patients from a morning outpatient queue. **You are the nurse at the gate** — type your name, then
confirm / override / reject each proposal. Every MCP tool call prints on screen as it happens.

```bash
./run.sh recheck          # 45 min later: repeat vitals on a waiting patient -> deterioration flag -> your decision
./run.sh queue            # the queue, ordered only by nurse-confirmed priorities
./run.sh eval --runs 3    # the eval set (see EVALS.md)
uv run pytest             # 19 tests: SATS scorer + gate/safety rules (no model needed)
```

Fresh-clone check: [`kaggle/smoke_fresh_clone.py`](kaggle/smoke_fresh_clone.py) clones this public repo on a clean Kaggle
T4, installs only uv + Ollama, and runs `./run.sh demo`, `recheck` and `queue` end to end (last passed 2026-10-13).

Swap the model with `TRIAGE_MODEL=qwen2.5:3b ./run.sh` (any Ollama model with tool calling). A GPU makes it seconds per
patient; CPU-only works but is slow.

**No GPU? Run it on Kaggle.** [`kaggle/demo.ipynb`](kaggle/demo.ipynb) is the same agent as a notebook — open it in
Kaggle with GPU T4 + Internet on and *Run All*. The nurse's decision is its own cell, so the pause at the gate is
visible. `kaggle/push.sh` runs the full eval set on a Kaggle GPU and is how EVALS.md was produced.
Audit log: `var/audit.jsonl`. State: `var/triage.db`. Delete `var/` to start a fresh morning.

## Who it is for

- **Triage nurses** at the adult triage desk of busy public outpatient departments, starting in Kenya: the agent
  gives them a cited SATS priority in seconds and never acts without their decision.
- **Patients waiting in that queue**, who are the people at risk when an urgent case sits unflagged.
- **Facility leads** who need an audit trail of who set each priority, when and why.

## What it does

1. **Score vitals** — `compute_tews` turns RR, HR, systolic BP, temperature, AVPU, mobility and trauma into the SATS
   Triage Early Warning Score. Each point cites its chart row (`hr.111-129 -> 2`). Missing vitals are reported, never
   assumed normal.
2. **Read the complaint** — the local model calls `match_discriminators` with SATS discriminator ids and the exact
   words that support each. The server rejects anything off the chart or not word-for-word in the complaint, points to
   the closest real words, and the model retries. This is the only step a model does. Anything it raised but could not
   cite goes to the nurse as UNVERIFIED — never into the colour, never silently dropped.
3. **Propose** — `propose_priority` re-checks everything server-side and takes the higher of the TEWS colour and the
   discriminator colour (the SATS rule), with a target time (red now, orange 10 min, yellow 60 min, green 4 h).
   It writes a *pending* proposal.
4. **Nurse decides** — the graph pauses (`interrupt()`). The nurse confirms, overrides (colour + reason) or rejects
   (reason). Only then does `commit_priority` set the colour and a see-by time from the facility clock.
5. **Deterioration re-check** — repeat vitals on a waiting patient; if the colour is now worse, `flag_recheck` raises
   a pending flag for a second look. The agent never moves the patient.

It refuses to apply the adult chart to children under 13 and hands them to the nurse.

## Requirements map

| Requirement | Where |
|---|---|
| Own MCP server, ≥3 tools, ≥1 that acts | `src/mediquest_triage_desk/server.py` — 7 tools; `propose_priority`, `flag_recheck`, `commit_priority` act |
| One MCP server we didn't write | `mcp-server-time` (modelcontextprotocol/servers) — facility-local clock for see-by deadlines; tz correctness maintained upstream beats owning it in a clinical log |
| Open-source orchestration | LangGraph — `src/mediquest_triage_desk/agent.py` |
| Full task on an open-weights model | Qwen 2.5 7B via Ollama on the facility's own machine; no patient text leaves it. Our runs: Kaggle T4, synthetic data only |
| Logged tool calls + gate on irreversible actions | `var/audit.jsonl` (every call: caller, inputs, output, timestamp, latency); `interrupt()` gate with named nurse before any colour reaches the queue |
| ARCHITECTURE.md / EVALS.md | in this repo |

## Triage scale

SATS adult chart (EMSSA, 2012 training manual), encoded in
[`src/mediquest_triage_desk/sats_adult.yaml`](src/mediquest_triage_desk/sats_adult.yaml). Every band and
discriminator has an id; every proposed colour cites the ids it came from.

## What is unfinished — honestly

- **The chart was transcribed by hand** and has not been checked line-by-line against the official EMSSA poster. Any
  facility would need to verify every band in `sats_adult.yaml` first.
- **Paediatric SATS is not implemented** (children are handed to the nurse).
- **Kiswahili complaints are missed** by the 7B model (unfixed — see EVALS.md), and a verbatim quote can still be
  irrelevant to the discriminator it is attached to; the nurse reading the citation is the check.
- **Not yet reviewed by a triage nurse.** The workflow is modelled on the published SATS process, not on time spent at
  a desk.
- Single desk, single machine; no EMR integration, no auth on the CLI (the name typed is the name recorded).
- Discriminators that depend on a measurement (glucose, burn %) rely on the nurse writing the number in the complaint.

## Next steps

1. Review with practising triage nurses (not done yet) and verify `sats_adult.yaml` against the official EMSSA chart.
2. Kiswahili/Sheng complaints: translate-then-match while quoting the original words, or a multilingual open-weights
   model (e.g. Aya) for extraction only; add a nurse-written Kiswahili eval set.
3. A second, independent check that each quote actually *supports* its discriminator (not just that it exists).
4. Paediatric SATS charts; a simple tablet UI for the desk; EMR/FHIR hand-off of confirmed priorities.

## Licence

Apache-2.0. © MediQuest by AlgoCracked.
