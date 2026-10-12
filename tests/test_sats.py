from mediquest_triage_desk import sats

NORMAL = {"respiratory_rate": 12, "heart_rate": 80, "systolic_bp": 120, "temperature_c": 36.8,
          "avpu": "alert", "mobility": "walking", "trauma": False}


def test_normal_vitals_score_zero_green():
    t = sats.compute_tews(NORMAL)
    assert t["complete"] and t["total"] == 0 and t["colour"] == "green"


def test_every_point_cites_a_chart_row():
    t = sats.compute_tews({**NORMAL, "heart_rate": 135, "respiratory_rate": 31})
    assert t["total"] == 6 and t["colour"] == "orange"
    assert {i["cite"] for i in t["items"] if i["score"]} == {"hr.gt129", "rr.gt29"}


def test_band_edges_have_no_gaps():
    for temp, score in [(34.9, 2), (34.95, 2), (35.0, 0), (38.4, 0), (38.45, 0), (38.5, 2)]:
        t = sats.compute_tews({**NORMAL, "temperature_c": temp})
        assert t["total"] == score, temp
    for hr, score in [(40, 3), (41, 1), (50, 1), (51, 0), (100, 0), (101, 1), (129, 2), (130, 3)]:
        assert sats.compute_tews({**NORMAL, "heart_rate": hr})["total"] == score, hr


def test_tews_colour_thresholds():
    assert [sats.tews_colour(n) for n in (0, 2, 3, 4, 5, 6, 7, 15)] == \
        ["green", "green", "yellow", "yellow", "orange", "orange", "red", "red"]


def test_missing_vitals_are_reported_not_assumed():
    t = sats.compute_tews({k: v for k, v in NORMAL.items() if k != "systolic_bp"})
    assert not t["complete"] and t["colour"] is None and t["missing"] == ["systolic_bp"]


def test_quote_must_appear_verbatim():
    c = "Crushing chest pain since this morning"
    r = sats.check_discriminators(c, [{"id": "vu.chest_pain", "quote": "chest  PAIN"},
                                      {"id": "vu.sob_acute", "quote": "short of breath"},
                                      {"id": "made.up", "quote": "chest pain"}])
    assert [a["id"] for a in r["accepted"]] == ["vu.chest_pain"]
    assert {x["id"] for x in r["rejected"]} == {"vu.sob_acute", "made.up"}


def test_discriminator_raises_but_never_lowers():
    t = sats.compute_tews({**NORMAL, "heart_rate": 132, "respiratory_rate": 31, "systolic_bp": 70})  # 9 -> red
    acc = sats.check_discriminators("abdominal pain", [{"id": "ur.abdo_pain", "quote": "abdominal pain"}])["accepted"]
    out = sats.combine(t, acc)
    assert out["colour"] == "red" and out["driver"] == "tews"


def test_discriminator_drives_colour_over_low_tews():
    t = sats.compute_tews(NORMAL)
    acc = sats.check_discriminators("chest pain", [{"id": "vu.chest_pain", "quote": "chest pain"}])["accepted"]
    out = sats.combine(t, acc)
    assert out["colour"] == "orange" and out["driver"] == "discriminator" and out["target_minutes"] == 10


def test_incomplete_vitals_no_colour_unless_orange_or_red():
    partial = {k: v for k, v in NORMAL.items() if k != "systolic_bp"}
    t = sats.compute_tews(partial)
    yellow = sats.check_discriminators("closed fracture", [{"id": "ur.fracture_closed", "quote": "closed fracture"}])
    assert sats.combine(t, yellow["accepted"])["status"] == "needs_vitals"
    red = sats.check_discriminators("seizure now", [{"id": "em.seizure_current", "quote": "seizure now"}])
    out = sats.combine(t, red["accepted"])
    assert out["status"] == "proposed_vitals_incomplete" and out["colour"] == "red"
