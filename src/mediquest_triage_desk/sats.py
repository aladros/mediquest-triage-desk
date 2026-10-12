"""Deterministic SATS (adult) scoring. No model is involved in anything this module returns.

Every number that contributes to a colour carries the `id` of the chart row it came from (see sats_adult.yaml),
so a proposed priority can always answer the audit question "why this colour?".
"""

from __future__ import annotations

import re
from functools import cache
from importlib import resources
from typing import Any

import yaml

COLOUR_RANK = {"green": 0, "yellow": 1, "orange": 2, "red": 3}
NUMERIC_VITALS = ("respiratory_rate", "heart_rate", "systolic_bp", "temperature_c")
CATEGORICAL_VITALS = ("avpu", "mobility", "trauma")
ALL_VITALS = NUMERIC_VITALS + CATEGORICAL_VITALS


@cache
def chart() -> dict[str, Any]:
    text = resources.files(__package__).joinpath("sats_adult.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def discriminator_index() -> dict[str, dict[str, Any]]:
    return {d["id"]: d for d in chart()["discriminators"]}


def _band(param: str, value: float) -> dict[str, Any]:
    for band in chart()["tews"][param]["bands"]:
        lo, hi = band["from"], band["below"]
        if (lo is None or value >= lo) and (hi is None or value < hi):
            return band
    raise ValueError(f"{param}={value} falls in no band of the chart")


def tews_colour(total: int) -> str:
    for row in chart()["tews_colour"]:
        if total >= row["min"] and (row["max"] is None or total <= row["max"]):
            return row["colour"]
    raise ValueError(f"TEWS {total} has no colour")


def compute_tews(vitals: dict[str, Any]) -> dict[str, Any]:
    """Score adult TEWS. Missing or invalid vitals are reported, never guessed."""
    items: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: list[str] = []
    for param in NUMERIC_VITALS:
        raw = vitals.get(param)
        if raw is None or raw == "":
            missing.append(param)
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            invalid.append(f"{param}={raw!r} is not a number")
            continue
        band = _band(param, value)
        items.append({"param": param, "value": value, "score": band["score"],
                      "cite": band["id"], "label": band["label"]})
    for param in CATEGORICAL_VITALS:
        raw = vitals.get(param)
        if raw is None or raw == "":
            missing.append(param)
            continue
        key = str(raw).strip().lower().replace(" ", "_")
        if isinstance(raw, bool):
            key = "yes" if raw else "no"
        options = chart()["tews"][param]["values"]
        if key not in options:
            invalid.append(f"{param}={raw!r} is not one of {sorted(options)}")
            continue
        row = options[key]
        items.append({"param": param, "value": key, "score": row["score"],
                      "cite": row["id"], "label": row["label"]})
    complete = not missing and not invalid
    total = sum(i["score"] for i in items)
    return {
        "complete": complete,
        "total": total if complete else None,
        "partial_total": total,
        "colour": tews_colour(total) if complete else None,
        "items": items,
        "missing": missing,
        "invalid": invalid,
        "source": chart()["source"],
    }


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def check_discriminators(complaint: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Accept a candidate only if its id is on the chart AND its quote appears verbatim in the complaint."""
    index = discriminator_index()
    haystack = _normalise(complaint)
    accepted, rejected = [], []
    seen: set[str] = set()
    for cand in candidates:
        did = str(cand.get("id", "")).strip()
        quote = str(cand.get("quote", "")).strip()
        if did not in index:
            rejected.append({"id": did, "quote": quote, "reason": "not a discriminator on the SATS adult chart"})
        elif not quote:
            rejected.append({"id": did, "quote": quote, "reason": "no supporting quote given"})
        elif _normalise(quote) not in haystack:
            rejected.append({"id": did, "quote": quote,
                             "reason": "quote does not appear verbatim in the presenting complaint"})
        elif did in seen:
            continue
        else:
            seen.add(did)
            d = index[did]
            accepted.append({"id": did, "quote": quote, "colour": d["colour"], "label": d["label"], "cite": did})
    return {"accepted": accepted, "rejected": rejected}


def combine(tews: dict[str, Any], accepted: list[dict[str, Any]]) -> dict[str, Any]:
    """Final colour = the higher of the TEWS colour and any discriminator colour.

    With incomplete vitals we only propose a colour when a discriminator already forces orange or red — vitals can
    raise a colour but never lower one, so that colour stands. Otherwise there is no colour: the nurse must measure.
    """
    disc_colour = max((a["colour"] for a in accepted), key=COLOUR_RANK.__getitem__, default=None)
    citations = [{"kind": "vital", "cite": i["cite"], "detail": f'{i["label"]} ({i["value"]}) -> {i["score"]}'}
                 for i in tews["items"] if i["score"] > 0]
    citations += [{"kind": "discriminator", "cite": a["cite"], "detail": f'{a["label"]} <- "{a["quote"]}"'}
                  for a in accepted]

    if tews["complete"]:
        colours = [tews["colour"]] + ([disc_colour] if disc_colour else [])
        colour = max(colours, key=COLOUR_RANK.__getitem__)
        if tews["total"] == 0 and not accepted:
            citations.append({"kind": "vital", "cite": "tews.total", "detail": "TEWS 0 – all vitals in normal bands"})
        driver = "tews" if colour == tews["colour"] and (not disc_colour or
                                                          COLOUR_RANK[tews["colour"]] >= COLOUR_RANK[disc_colour]) \
            else "discriminator"
        status = "proposed"
    elif disc_colour and COLOUR_RANK[disc_colour] >= COLOUR_RANK["orange"]:
        colour, driver, status = disc_colour, "discriminator", "proposed_vitals_incomplete"
    else:
        colour, driver, status = None, None, "needs_vitals"

    target = chart()["target_minutes"][colour] if colour else None
    return {
        "status": status,
        "colour": colour,
        "driver": driver,
        "tews_total": tews["total"],
        "tews_colour": tews["colour"],
        "discriminator_colour": disc_colour,
        "target_minutes": target,
        "missing_vitals": tews["missing"],
        "citations": citations,
    }


def worse(new_colour: str | None, old_colour: str | None) -> bool:
    if new_colour is None or old_colour is None:
        return False
    return COLOUR_RANK[new_colour] > COLOUR_RANK[old_colour]
