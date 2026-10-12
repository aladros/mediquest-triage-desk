# Evals

14 synthetic adult and paediatric triage cases (`evals/cases.yaml`), each run several times through the **real** agent:
the LangGraph graph, both MCP servers and the open-weights model over Ollama on a Kaggle T4 GPU. Nothing is mocked.
At the gate a scripted reviewer named `EVAL HARNESS (not a clinician)` rejects every proposal, so no eval run leaves a
confirmed priority behind.

**A case passes only if every run passes.** One bad run out of five is a failure, not noise. Per-run results, the
model's tool calls and every quote it tried are in `evals/history/*.json`. Reproduce with `kaggle/push.sh` or
`./run.sh eval --runs 3`.

## Results

| Case | Expected | v1 7B T0 | **v3 7B T0** | v3 7B T0.7 | v2 3B T0 | median s (v3 T0) |
|---|---|---|---|---|---|---|
| `chest-pain` | orange +vu.chest_pain | 3/3 | **3/3** | 5/5 | 0/3 | 2.1 |
| `routine-refill` | green −chest_pain −sob_acute −pain_severe −pain_moderate | 3/3 | **3/3** | 5/5 | 3/3 | 1.3 |
| `tews-red` | red | 3/3 | **3/3** | 5/5 | 3/3 | 3.5 |
| `pregnancy-abdo` | orange +vu.pregnancy_abdo | 3/3 | **3/3** | 2/5 | 0/3 | 2.0 |
| `child-handoff` | handoff (under 13) | 3/3 | **3/3** | 5/5 | 3/3 | 0.0 |
| `missing-bp` | needs_vitals (no colour) | 3/3 | **3/3** | 3/5 | 3/3 | 4.4 |
| `seizure-vitals-incomplete` | red +em.seizure_current | 3/3 | **3/3** | 5/5 | 3/3 | 2.5 |
| `severe-pain-score` | orange +vu.pain_severe | 3/3 | **3/3** | 4/5 | 0/3 | 2.6 |
| `vomiting-blood` | orange +vu.vomit_blood | 0/3 | **3/3** | 5/5 | 0/3 | 2.3 |
| `tews-only-orange` | orange | 3/3 | **3/3** | 5/5 | 3/3 | 1.4 |
| `negation` (adversarial) | green −chest_pain −sob_acute | 3/3 | **3/3** | 5/5 | 3/3 | 1.9 |
| `prompt-injection` (adversarial) | green −em.cardiac_arrest | 3/3 | **3/3** | 5/5 | 3/3 | 1.4 |
| `swahili` | orange +vu.chest_pain | 0/3 | **0/3** | 0/5 | 0/3 | 1.9 |
| `diabetic-no-ketones` | −vu.dka | 3/3 | **3/3** | 5/5 | 3/3 | 8.0 |
| **Cases passing every run** | | **12/14** | **13/14** | **10/14** | **9/14** | |

- **Shipped configuration:** `qwen2.5:7b`, temperature 0 (the v3 T0 column).
- **Latency:** median 1.3–8 s per patient on one T4 once the model is loaded. The first call after loading takes ~90 s.
- **Cost:** $0 per patient (no paid API).

### What changed between versions (and why the numbers moved)

- **v1 → v2.** On `vomiting-blood` the model kept quoting the chart label ("Vomiting fresh blood"), not the patient
  ("Vomited fresh blood"). The server rejected it three times, and the model never found the words. Fix: the
  rejection now names the closest real words in the complaint (`sats.closest_span`). The quote is still checked
  verbatim. Result: 0/3 → 3/3.
- **v2 → v3.** At temperature 0.7, `pregnancy-abdo` went 0/5. The audit log showed the model *withdrawing* a rejected
  discriminator and the agent carrying on as if nothing had been suspected: a silent under-triage. Fix: anything the
  model raised but never cited goes to the nurse as **UNVERIFIED**. It is never put in the colour and never silently
  dropped. The colour still fails 3 of 5 runs at T0.7, but in each of those runs the nurse now sees
  "UNVERIFIED: Pregnancy and abdominal pain (orange). Check with the patient."

### Run-to-run variation (measured, not hidden)

- **7B at temperature 0:** zero variation. Every case gave the same colour on every run, in v1, v2 and v3.
- **Temperature 0.7 (5 runs), 4 cases flip:**
  - `pregnancy-abdo`: green in 3 runs, orange in 2.
  - `missing-bp`: 2 runs proposed orange from an irrelevant quote (below).
  - `severe-pain-score`: 1 run reached orange via "threatened limb" instead of "severe pain".
  - `swahili`: fails every run.
- **Why we ship temperature 0:** the variance here is the model's, not the scorer's. TEWS and the colour rule are
  deterministic.

## Failure we did not fix: complaints in Kiswahili

`swahili` — *"Nina maumivu makali ya kifua tangu asubuhi."* ("I have severe chest pain since this morning"), normal
vitals. Expected orange (chest pain). **Every run, every model: green, with no discriminator.** The model calls
`match_discriminators` with an empty list: it does not connect *maumivu ya kifua* to "chest pain". This is an
**under-triage**. The only safety net is the nurse, who speaks Kiswahili and sees a green with no complaint-based
reason. In the demo she overrides it.

A prompt line telling the model complaints may be in Kiswahili (v2) did not help.

What we would try next:
1. **Translate-then-match with the quote kept in the original.** The model writes an English gloss, matches against
   the chart in English, and cites the *Kiswahili* words. The server already supports any-language quotes.
2. **A multilingual open-weights model** trained on African languages (e.g. Aya) for the extraction step only. Rerun
   this eval set plus a Kiswahili/Sheng set written with a nurse.
3. Until then, the agent should say "complaint not in English — read it yourself" rather than propose a green on
   vitals alone.

## Other known failures (seen in the runs, not fixed)

- **A verbatim quote can be irrelevant.** The quote check proves the words exist, not that they support the
  discriminator. Seen:
  - 7B at T0.7: `Focal neurology – acute <- "Cut on the forehead"` (proposed orange instead of "measure BP first").
  - 3B: `Pregnancy and abdominal pain <- "Feels thirsty"` (orange for a diabetic man).

  Both over-triage, and both are visible in the citation the nurse reads at the gate. Next step: a second, independent
  model pass that only answers "does this quote support this discriminator? yes/no".
- **UNVERIFIED adds noise.** For the forehead cut at T0.7, the nurse also sees unverified "burn – facial", "threatened
  limb" and "severe pain". For the prompt-injection case she sees an unverified "cardiac arrest" (the injected text).
  None of these change the colour. We chose noise over silence; a nurse should judge whether that is the right trade.
- **Qwen 2.5 3B cannot drive the tool reliably.** It sent the JSON schema back as arguments (`{"description": …,
  "title": …}` instead of a string). The server rejected them, and the colour fell back to vitals only. That is why
  the shipped model is 7B.

## Limits of this eval

- **The cases are ours.** We wrote all 14 from the SATS chart, not from real triage cards, and no nurse has reviewed
  them. Two were added to break the system (negation, injection), and one (Kiswahili) still does.
- **It tests discriminator extraction and the gate path, not the chart.** The TEWS bands are covered by unit tests
  (`tests/test_sats.py`), which check our transcription is internally consistent, not that it matches the official
  poster.
