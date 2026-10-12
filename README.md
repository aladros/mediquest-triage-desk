# MediQuest Triage Desk

**An agent for the outpatient triage desk that turns vitals and a presenting complaint into a South African Triage
Scale (SATS) priority in seconds — every colour cited to the chart row or the patient's own words — and then stops.
A named nurse confirms, overrides or rejects it. Nothing moves in the queue without her.**

Built for the *African Agentic AI Design Challenge* — Healthcare track, "The Triage Desk",
theme **Symptom-based priority scoring**. Synthetic patients only. Not a medical device; not for clinical use.

## Run it (one command)

Prerequisites: [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/download). Then:

```bash
./run.sh
```

That starts Ollama if needed, pulls the open-weights model (`qwen2.5:3b`, ~2 GB, first run only), and runs the demo:
six synthetic patients from a morning outpatient queue. **You are the nurse at the gate** — type your name, then
confirm / override / reject each proposal. Every MCP tool call prints on screen as it happens.

```bash
./run.sh recheck          # 45 min later: repeat vitals on a waiting patient -> deterioration flag -> your decision
./run.sh queue            # the queue, ordered only by nurse-confirmed priorities
./run.sh eval --runs 3    # the eval set (see EVALS.md)
uv run pytest             # deterministic scorer tests (no model needed)
```

Swap the model with `TRIAGE_MODEL=qwen2.5:7b ./run.sh` (any Ollama model with tool calling).
Audit log: `var/audit.jsonl`. State: `var/triage.db`. Delete `var/` to start a fresh morning.

## What it does

1. **Score vitals** — `compute_tews` turns RR, HR, systolic BP, temperature, AVPU, mobility and trauma into the SATS
   Triage Early Warning Score. Each point cites its chart row (`hr.111-129 -> 2`). Missing vitals are reported, never
   assumed normal.
2. **Read the complaint** — the local model calls `match_discriminators` with SATS discriminator ids and the exact
   words that support each. The server rejects anything off the chart or not word-for-word in the complaint, and the
   model retries. This is the only step a model does.
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
| Full task on an open-weights model | Qwen 2.5 via Ollama, on the device; no patient text leaves the machine |
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
- **Negation and injected text** can fool the quote check — see the unfixed failure in EVALS.md.
- **Not yet reviewed by a triage nurse.** The workflow is modelled on the published SATS process, not on time spent at
  a desk.
- Single desk, single machine; no EMR integration, no auth on the CLI (the name typed is the name recorded).
- Discriminators that depend on a measurement (glucose, burn %) rely on the nurse writing the number in the complaint.

## Licence

Apache-2.0. © MediQuest by AlgoCracked.
