"""Queue state (SQLite) and the append-only audit log (JSONL).

The queue only ever orders patients by a CONFIRMED priority. A proposal the agent writes sits in `proposals` with
status `pending` and does not move anyone until a named nurse confirms or overrides it at the gate.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
  patient_id TEXT PRIMARY KEY,
  arrived_at TEXT NOT NULL,
  age_years INTEGER,
  complaint TEXT NOT NULL,
  vitals_json TEXT NOT NULL,
  confirmed_colour TEXT,
  confirmed_at TEXT,
  confirmed_by TEXT,
  see_by TEXT
);
CREATE TABLE IF NOT EXISTS proposals (
  proposal_id TEXT PRIMARY KEY,
  patient_id TEXT NOT NULL,
  kind TEXT NOT NULL,              -- 'priority' | 'recheck'
  colour TEXT,
  payload_json TEXT NOT NULL,      -- citations, TEWS, discriminators, target time
  status TEXT NOT NULL,            -- 'pending' | 'confirmed' | 'overridden' | 'rejected'
  created_at TEXT NOT NULL,
  decided_at TEXT,
  decided_by TEXT,
  final_colour TEXT,
  reason TEXT
);
"""

COLOUR_ORDER = "CASE confirmed_colour WHEN 'red' THEN 0 WHEN 'orange' THEN 1 WHEN 'yellow' THEN 2 WHEN 'green' THEN 3 END"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def data_dir() -> Path:
    d = Path(os.environ.get("TRIAGE_DATA_DIR", "var"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(data_dir() / "triage.db")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def audit(event: str, **fields: Any) -> dict[str, Any]:
    """Append one line to the audit log. Never rewritten, never truncated by the app."""
    record = {"ts": now_iso(), "event": event, **fields}
    with (data_dir() / "audit.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return record


def register_patient(patient_id: str, age_years: int | None, complaint: str, vitals: dict[str, Any]) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO patients (patient_id, arrived_at, age_years, complaint, vitals_json) VALUES (?,?,?,?,?) "
            "ON CONFLICT(patient_id) DO UPDATE SET complaint=excluded.complaint, vitals_json=excluded.vitals_json",
            (patient_id, now_iso(), age_years, complaint, json.dumps(vitals)),
        )


def create_proposal(patient_id: str, kind: str, colour: str | None, payload: dict[str, Any]) -> str:
    proposal_id = f"{kind[:2]}-{uuid.uuid4().hex[:8]}"
    with connect() as conn:
        conn.execute(
            "INSERT INTO proposals (proposal_id, patient_id, kind, colour, payload_json, status, created_at) "
            "VALUES (?,?,?,?,?,'pending',?)",
            (proposal_id, patient_id, kind, colour, json.dumps(payload), now_iso()),
        )
    return proposal_id


def get_proposal(proposal_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
    return dict(row) if row else None


def decide_proposal(proposal_id: str, nurse: str, decision: str, final_colour: str | None,
                    reason: str | None, see_by: str | None) -> dict[str, Any]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
        if row is None:
            raise ValueError(f"no proposal {proposal_id}")
        if row["status"] != "pending":
            raise ValueError(f"proposal {proposal_id} already {row['status']}")
        conn.execute(
            "UPDATE proposals SET status=?, decided_at=?, decided_by=?, final_colour=?, reason=? WHERE proposal_id=?",
            (decision, now_iso(), nurse, final_colour, reason, proposal_id),
        )
        if decision in ("confirmed", "overridden") and final_colour:
            conn.execute(
                "UPDATE patients SET confirmed_colour=?, confirmed_at=?, confirmed_by=?, see_by=? WHERE patient_id=?",
                (final_colour, now_iso(), nurse, see_by, row["patient_id"]),
            )
    return get_proposal(proposal_id) or {}


def patient(patient_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,)).fetchone()
    return dict(row) if row else None


def queue() -> dict[str, list[dict[str, Any]]]:
    with connect() as conn:
        confirmed = conn.execute(
            f"SELECT patient_id, confirmed_colour, confirmed_by, see_by, arrived_at FROM patients "
            f"WHERE confirmed_colour IS NOT NULL ORDER BY {COLOUR_ORDER}, arrived_at"
        ).fetchall()
        awaiting = conn.execute(
            "SELECT p.patient_id, p.arrived_at, pr.proposal_id, pr.kind, pr.colour AS proposed_colour "
            "FROM proposals pr JOIN patients p USING(patient_id) WHERE pr.status='pending' ORDER BY pr.created_at"
        ).fetchall()
    return {"queue": [dict(r) for r in confirmed], "awaiting_nurse": [dict(r) for r in awaiting]}
