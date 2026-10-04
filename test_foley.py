"""python3 test_foley.py — synthetic clip: square bounces at frame 20, hard cut at 40, flash at 50-52."""
import numpy as np
import foley

T, S = 60, 48
v = np.zeros((T, S, S, 3), np.uint8)
v[:40] = (20, 20, 40)
v[40:] = (90, 40, 20)
for i in range(T):
    y = 2 * i if i < 20 else 40 - 2 * (i - 20) if i < 40 else 10 + i % 5
    v[i, y:y + 8, 20:28] = 255
v[50:53] = 250

an = foley.analyze(v)
F = foley.FPS
hits = [round(h["t"] * F) for h in an["hits"]]
assert any(abs(h - 20) <= 3 for h in hits), hits
assert [round(c * F) for c in an["cuts"]] == [40], an["cuts"]
assert any(abs(round(f["t"] * F) - 50) <= 1 for f in an["flashes"]), an["flashes"]
assert len(an["scenes"]) == 2

plan = foley.guess_plan(an)
for pal in foley.PALETTES:  # every palette renders
    for s in an["scenes"]:
        s["palette"] = pal
    master, events, meters = foley.compose(an, plan, seed=1)
    assert master.shape == (2, int(T / F * foley.SR)) and np.isfinite(master).all() and np.abs(master).max() <= 1
    assert np.sqrt(np.mean(master.astype(np.float64) ** 2)) <= 10 ** (-16.6 / 20) + 1e-6
for vibe in foley.VIBES:  # every sound world renders
    plan["vibe"] = vibe
    master, events, meters = foley.compose(an, plan, seed=3)
    assert np.isfinite(master).all() and np.abs(master).max() <= 1, vibe
    assert np.sqrt(np.mean(master.astype(np.float64) ** 2)) <= 10 ** (-16.6 / 20) + 1e-6, vibe
assert all(c['type'] not in ('boom', 'swell') for c in foley.auto_score(an)[1])  # neither a contact nor a cut requires a cinematic cue
assert not any(e['track'] == 'BOOM' and e['label'].startswith('cut') for e in events)
assert any('midi' in e for e in events if e['track'] == 'VOICE')  # the visual score receives the synthesized pitch
assert foley.parse_json('log {x} {"key":"D","cues":[]} bye')["key"] == "D"

# agent score: refs resolve to frame-accurate times, near-misses snap, junk is dropped
ids, _ = foley.event_table(an)
hit = next(k for k in ids if k.startswith("h"))
plan2 = foley.guess_plan(an)
foley.apply_score({"key": "E", "mode": "lydian", "scenes": [{"palette": "glass", "what": "x"}],
                   "sections": [{"start": 0, "end": 1, "energy": 2}, {"start": 1, "end": 2, "energy": "?"}],
                   "cues": [{"t": 0, "type": "hit", "ref": hit, "dur": 0, "strength": .9, "sound": "ring", "pitch": "low", "note": "hit"},
                            {"t": ids[hit]["t"] + .1, "type": "boom", "ref": None, "dur": 0, "strength": 1, "sound": "thump", "pitch": "auto", "note": ""},
                            {"t": .3, "type": "glide", "ref": None, "dur": .8, "strength": .7, "sound": "auto", "pitch": "follow", "note": ""},
                            {"t": .6, "type": "chop", "ref": None, "dur": .6, "strength": .7, "sound": "auto", "pitch": "auto", "note": ""},
                            {"t": 1.2, "type": "stutter", "ref": None, "dur": .3, "strength": .7, "sound": "auto", "pitch": "auto", "note": ""},
                            {"t": 1.33, "type": "tapestop", "ref": None, "dur": .5, "strength": .7, "sound": "auto", "pitch": "auto", "note": ""},
                            {"t": 1.35, "type": "dropout", "ref": None, "dur": .1, "strength": .7, "sound": "auto", "pitch": "auto", "note": ""},
                            {"t": 1.4, "type": "swell", "ref": None, "dur": .6, "strength": .7, "sound": "auto", "pitch": "auto", "note": ""},
                            {"t": 1, "type": "explode", "ref": None}, {"t": 99, "type": "blip", "ref": None}]}, an, plan2, ids)
assert plan2["key"] == "E" and an["scenes"][0]["palette"] == "glass"
anchored = [c for c in plan2["cues"] if c["type"] in ("hit", "boom")]
assert [c["type"] for c in anchored] == ["hit", "boom"] and all(c["t"] == ids[hit]["t"] for c in anchored), plan2["cues"]
assert "explode" not in [c["type"] for c in plan2["cues"]] and all(c["t"] < T / F for c in plan2["cues"])
assert plan2["sections"][0]["energy"] == 1 and plan2["sections"][1]["energy"] == .5 and plan2["sections"][-1]["b"] == T / F
master, events, _ = foley.compose(an, plan2, seed=2)
assert np.isfinite(master).all() and np.abs(master).max() <= 1
assert {"HIT", "BOOM", "VOICE", "CHOP"} <= {e["track"] for e in events}, {e["track"] for e in events}
assert not hasattr(foley, "fit_grid")  # sound design: no tempo grid, no melody
print("ok", hits, an["cuts"])
