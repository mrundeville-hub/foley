#!/usr/bin/env python3
"""foley — procedural sound design for videos & gifs, with an animated terminal score.

    foley                      paste a local video path → sound design + animated player
    foley clip.mp4 [--agent codex|claude|gemini|none] [--model M] [--seed N] [-o out.mp4] [--no-ui]
"""
import argparse, colorsys, json, os, re, shlex, shutil, subprocess, sys, tempfile, time, wave
from pathlib import Path
from urllib.parse import unquote, urlsplit
import numpy as np

SR, FPS = 44100, 30
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".gif", ".webm", ".mkv", ".avi"}


def local_video(value):
    """Accept a local path, Finder drag/drop, or file:// URL; never fetch remote media."""
    value = str(value).strip()
    if value.startswith("file:"):
        url = urlsplit(value)
        if url.netloc not in ("", "localhost"):
            raise ValueError("use a file on this computer")
        value = unquote(url.path)
    elif "://" in value:
        raise ValueError("paste a local path or file:// URL")
    path = Path(value).expanduser()
    if not path.is_file():
        try:
            parts = shlex.split(value)
        except ValueError:
            parts = []
        if len(parts) == 1:
            path = Path(parts[0]).expanduser()
    if not path.is_file():
        raise ValueError(f"no such file: {value}")
    if path.suffix.lower() not in VIDEO_EXT:
        raise ValueError("choose a video or GIF (.mp4, .mov, .webm, .mkv, .avi, .m4v, .gif)")
    return str(path.resolve())


def output_path(video):
    """New runs and variations keep previous results."""
    base = os.path.splitext(video)[0] + ".foley"
    candidate, i = base + ".mp4", 2
    while os.path.exists(candidate) or os.path.exists(os.path.splitext(candidate)[0] + ".wav"):
        candidate = f"{base}-{i}.mp4"
        i += 1
    return candidate

NOTES = "C C# D D# E F F# G G# A A# B".split()
SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10], "dorian": [0, 2, 3, 5, 7, 9, 10],
          "lydian": [0, 2, 4, 6, 7, 9, 11], "phrygian": [0, 1, 3, 5, 7, 8, 10], "pentatonic": [0, 2, 4, 7, 9]}
# per scene: the material of its contacts → which gesture a hit makes
PALETTES = dict(glass="ping", metal="ring", electric="chirp", organic="bend", cute="blob", dark="thump", glitch="crunch")
TRACKS = ["HIT", "VOICE", "GRAIN", "CHOP", "SWEEP", "BOOM", "SUB", "BED"]
# Compatibility names remain available at the CLI; each now selects a material balance.
VIBES = {
    "formant": dict(desc="elastic pressure, rubber friction and deforming resonances", voice="elastic"),
    "static": dict(desc="rough ceramic contact, abrasive friction, electrical debris", voice="abrasive"),
    "pluck": dict(desc="close dry contacts, short object resonance, restrained friction", voice="contact"),
    "flicker": dict(desc="material fragments and friction interrupted by light changes", voice="fragments"),
    "resonant": dict(desc="inharmonic metal or glass objects with detailed attacks and natural decay", voice="modal"),
    "haze": dict(desc="soft diffuse textures and long object resonance, driven by motion", voice="diffuse"),
    "swarm": dict(desc="dense micro-contacts, granular debris and mechanical surface motion", voice="particles"),
    "abyss": dict(desc="heavy membranes, low object pressure and coarse friction", voice="mass"),
}


# ───────────────────────────── analysis ─────────────────────────────

def decode(path, width=96, extra=(), fps=FPS):
    """All frames at FPS as uint8 (T,H,W,3). PPM stream so ffmpeg handles rotation and we read the size from the header."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={fps},scale={width}:-2", *extra,
                        "-f", "image2pipe", "-c:v", "ppm", "-"], capture_output=True)
    m = re.match(rb"P6\s(\d+)\s(\d+)\s255\s", p.stdout)
    if p.returncode or not m:
        raise RuntimeError(f"ffmpeg could not decode {path}: {p.stderr.decode()[-300:]}")
    w, h, hl = int(m[1]), int(m[2]), m.end()
    fs = hl + w * h * 3
    n = len(p.stdout) // fs
    return np.frombuffer(p.stdout[:n * fs], np.uint8).reshape(n, fs)[:, hl:].reshape(n, h, w, 3)


def smooth(x, k):
    return np.convolve(x, np.ones(k) / k, "same")


def peaks(x, thr, gap):
    """Greedy peak pick: strongest first, at least `gap` frames apart."""
    out = []
    for i in np.argsort(-x):
        if x[i] <= thr:
            break
        if all(abs(i - j) >= gap for j in out):
            out.append(int(i))
    return sorted(out)


def runs(mask):
    e = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def analyze(frames):
    # ponytail: whole clip in RAM (~60MB per 30s at 96px); stream in chunks if people feed it minutes of video
    T, H, W, _ = frames.shape
    Y = frames @ np.float32([.299, .587, .114]) / 255
    lum = Y.mean((1, 2))
    Yc = Y - lum[:, None, None]  # mean-removed so light changes don't read as motion

    # camera motion: global shift between frames by phase correlation, estimated only when the frame's outer ring is
    # textured (a flat background would just track the subject). Object motion = what's left after undoing the camera.
    gy, gx = np.gradient(Y, axis=(1, 2))
    ring = np.ones((H, W), bool)
    ring[int(H * .2):int(H * .8), int(W * .2):int(W * .8)] = False
    textured = (((np.abs(gx) + np.abs(gy)) > .02) & ring).sum((1, 2)) / ring.sum() > .3
    Fq = np.fft.rfft2(Yc)
    cam = np.zeros((T, 2), int)
    for i in range(1, T):
        if textured[i] and textured[i - 1]:
            R = Fq[i] * np.conj(Fq[i - 1])
            c = np.fft.irfft2(R / (np.abs(R) + 1e-9), (H, W))
            iy, ix = np.unravel_index(np.argmax(c), c.shape)
            s = ((iy + H // 2) % H - H // 2, (ix + W // 2) % W - W // 2)
            if c.max() > .08 and max(map(abs, s)) <= 12:
                cam[i] = s
    prev = Yc[:-1].copy()
    for i in np.flatnonzero(cam[1:].any(1)):
        prev[i] = np.roll(Yc[i], tuple(cam[i + 1]), (0, 1))
    d = np.abs(Yc[1:] - prev)
    d[:, :3] = d[:, -3:] = 0
    d[:, :, :3] = d[:, :, -3:] = 0
    camv = smooth(np.hypot(*cam.T) / W, 5)
    pans = [dict(t=a / FPS, d=(b - a) / FPS, s=float(np.clip(camv[a:b].max() / .04, .3, 1)),
                 dir="left" if cam[a:b, 1].sum() < 0 else "right") for a, b in runs(camv > .012) if b - a >= FPS * .25]
    w = d.sum((1, 2)) + 1e-6
    m = np.r_[0, d.mean((1, 2))]
    cx = np.r_[.5, d.sum(1) @ np.linspace(0, 1, W) / w]
    cy = np.r_[.5, d.sum(2) @ np.linspace(0, 1, H) / w]
    dl = np.diff(lum, prepend=lum[0])

    # cuts: colour-histogram jump that doesn't come back (a flash comes back)
    q = (frames >> 6).astype(np.int32)
    hist = np.stack([np.bincount(x.ravel(), minlength=64) for x in q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]]) / (H * W)
    hd = np.r_[0, .5 * np.abs(np.diff(hist, axis=0)).sum(1)]
    k = FPS // 3
    dist = lambda i, j: .5 * np.abs(hist[max(0, min(T - 1, i))] - hist[max(0, min(T - 1, j))]).sum()
    cut_score = np.array([h if h > 3 * np.median(hd[max(0, i - 15):i + 15]) and dist(i - 1, i + k) > .3
                          and dist(i - k, i) > .3 else 0 for i, h in enumerate(hd)])
    # same-palette cuts (stairs → other stairs): a one-frame spike of structural change, unlike sustained camera shake
    near = lambda i: np.r_[m[max(0, i - 6):i], m[i + 1:i + 7]]
    cut_score = np.maximum(cut_score, [.5 if x > .1 and x > 4 * np.median(near(i)) else 0 for i, x in enumerate(m)])
    cuts = peaks(cut_score, .4, FPS // 2)

    flashes = peaks(np.where(np.isin(np.arange(T), cuts), 0, dl), .06, FPS // 4)

    # hits: peaks of the most active REGION's motion that stand out from their neighbourhood. A small subject in a big
    # frame (a deer's steps) vanishes in whole-frame averages, and continuous motion has no onsets — but it has peaks.
    # Scored against the reference artist's accents this beats the old onset detector (measured on the 4 refs).
    ms = smooth(m, 3)
    G = 8
    hs, ws = H // G, W // G
    Ec = np.abs(np.diff(Yc, axis=0))[:, :hs * G, :ws * G].reshape(T - 1, G, hs, G, ws).mean((2, 4)).reshape(T - 1, -1)
    Ec = np.maximum(0, Ec - np.median(Ec, 1, keepdims=True))  # what every cell shares is the camera, not the subject
    top = np.argsort(Ec, 1)[:, -3:]
    reg = np.convolve(np.r_[0, np.take_along_axis(Ec, top, 1).mean(1)], [.25, .5, .25], "same")
    rx = np.r_[.5, (top % G).mean(1) / (G - 1)]  # where the most active region is
    ry = np.r_[.5, (top // G).mean(1) / (G - 1)]
    prom = reg - np.array([reg[max(0, i - 4):i + 5].min() for i in range(T)])
    for c in cuts + flashes:
        prom[max(0, c - 2):c + 3] = 0
    ismax = (reg >= np.r_[reg[1:], 0]) & (reg >= np.r_[0, reg[:-1]])
    score = np.where(ismax & (prom > 3 * np.median(prom) + 1e-5) & (reg > .003), prom, 0)
    score /= np.percentile(score[score > 0], 90) + 1e-9 if score.any() else 1
    # ...plus sudden direction changes (a bounce at constant speed has no energy peak, only a turn)
    v = np.stack([np.gradient(smooth(cx, 3)), np.gradient(smooth(cy, 3))])
    jerk = np.linalg.norm(np.gradient(v, axis=1), axis=0) * ms
    for c in cuts + flashes:
        jerk[max(0, c - 2):c + 3] = 0
    bounds = [0, *cuts, T]
    for a, b in zip(bounds, bounds[1:]):  # per shot: a violent shot shouldn't mute a gentle one
        jerk[a:a + 6] = 0  # the motion centre only settles after the first frames of a shot (smoothing spreads it)
        j = jerk[a:b] / (jerk[a:b].max() + 1e-9)
        score[a:b] = np.maximum(score[a:b], np.where((j > .5) & (ms[a:b] > .003), .8 * j, 0))  # region peaks win ties
    hits = peaks(score, 0, 4)

    # flicker: dense small light wobbles (screens, neon, train lights)
    small = np.abs(dl) > .012
    cand = np.abs(dl) * (np.convolve(small, np.ones(FPS // 2), "same") >= 4) * (np.abs(dl) > m)  # light, not objects
    for c in cuts + flashes:
        cand[max(0, c - 2):c + 3] = 0
    flicker = peaks(cand, .012, 3)

    # whooshes: sustained fast motion
    thr = max(np.percentile(ms, 75), .008)
    p99 = np.percentile(ms, 99) + 1e-9
    whooshes = [dict(t=a / FPS, d=(b - a) / FPS, s=float(np.clip(ms[a:b].max() / p99, .3, 1)),
                     env=ms[a:b] / ms[a:b].max(), pan=cx[a:b])
                for a, b in runs(ms > thr) if b - a >= FPS * .3]

    scenes = []
    for a, b in zip(bounds, bounds[1:]):
        px = frames[a:b:3, ::4, ::4].reshape(-1, 3).astype(np.float32)
        hue = colorsys.rgb_to_hls(*(px.mean(0) / 255))[0]
        sat = float(((px.max(1) - px.min(1)) / (px.max(1) + 1)).mean())
        count = lambda xs: sum(a <= i < b for i in xs)
        scenes.append(dict(a=a / FPS, b=b / FPS, lum=float(lum[a:b].mean()), hue=hue, sat=sat,
                           hits=count(hits), flashes=count(flashes), flicker=count(flicker)))

    ev = lambda idx, s: [dict(t=i / FPS, s=float(s(i)), x=float(cx[i]), y=float(cy[i])) for i in idx]
    return dict(T=T, cuts=[c / FPS for c in cuts], scenes=scenes, whooshes=whooshes, pans=pans,
                motion=dict(ms=ms, cx=cx, cy=cy, lum=lum, cam=camv, reg=reg, rx=rx, ry=ry),
                hits=[dict(t=i / FPS, s=float(np.clip(score[i], .2, 1)), x=float(rx[i]), y=float(ry[i])) for i in hits],
                flashes=ev(flashes, lambda i: np.clip(dl[i] / .3, .2, 1)),
                flicker=ev(flicker, lambda i: .4))


def guess_plan(an):
    for s in an["scenes"]:
        L = max(s["b"] - s["a"], .1)
        s["palette"] = ("electric" if s["flashes"] / L > .3 else "glitch" if s["flicker"] >= 3 else
                        "dark" if s["lum"] < .22 else "cute" if s["sat"] > .35 and s["lum"] > .45 else
                        "glass" if .45 < s["hue"] < .75 else "metal" if s["hits"] / L > 1.5 else "organic")
    lum = np.mean([s["lum"] for s in an["scenes"]])
    hue = an["scenes"][0]["hue"]
    return dict(key=NOTES[int(hue * 12) % 12], mode="minor" if lum < .35 else "dorian", vibe=guess_vibe(an), summary="", via="heuristics")


def guess_vibe(an):
    """no agent: read the vibe off the picture — light, flicker, how busy the motion is."""
    dur, sc = an["T"] / FPS, an["scenes"]
    lum, sat = np.mean([s["lum"] for s in sc]), np.mean([s["sat"] for s in sc])
    hits, flashes, flick = len(an["hits"]) / dur, len(an["flashes"]) / dur, sum(s["flicker"] for s in sc) / dur
    ms = an["motion"]["ms"]
    busy = float(np.mean(ms > .004))  # fraction of frames with visible motion
    score = dict(static=(flashes > .3) * 2.5, flicker=(flick > .5) * 2.5, abyss=(lum < .18) * 2.5,
                 swarm=(hits > 3) + (busy > .9) * .8,
                 pluck=(sat > .35 and lum > .4) + (1 < hits < 3) + (busy > .6 and hits < 1) * 2.1,
                 haze=(hits < 1) * 1.5 + (lum > .45) * .5, resonant=(len(an["cuts"]) / dur > .3) + (hits > 1.5) * .5, formant=1.)
    return max(score, key=score.get)


# ───────────────────────────── agent: the sound director ─────────────────────────────
# The agent gets timestamped contact sheets + every detected event (with ids) and writes a cue sheet:
# what happens, the musical frame, energy sections, and cues anchored to the detected events.
# The engine then renders that score, snapping anchored cues to the exact detected frame.

AGENTS = {  # add another CLI here: (prompt, image filenames in cwd, json-schema path) -> argv
    "codex":  lambda p, imgs, schema: ["codex", "exec", "--skip-git-repo-check", "-s", "read-only", "--output-schema", schema,
                                       p, *[a for i in imgs for a in ("-i", i)]],
    "claude": lambda p, imgs, schema: ["claude", "-p", p, "--allowedTools", "Read"],
    "gemini": lambda p, imgs, schema: ["gemini", "-p", p + "\n" + " ".join("@" + i for i in imgs)],
}
CUE_TYPES = ["hit", "boom", "glide", "sweep", "riser", "chop", "stutter", "tapestop", "dropout", "swell"]
SOUNDS = ["auto", "bend", "ping", "ring", "chirp", "blob", "thump", "crunch"]
PITCHES = ["auto", "low", "mid", "high", "up", "down", "follow"]

_obj = lambda props: {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}
_enum = lambda xs: {"type": "string", "enum": xs}
SCHEMA = _obj(dict(
    summary={"type": "string"}, story={"type": "string"}, vibe=_enum(list(VIBES)), vibe_why={"type": "string"},
    tweaks=_obj(dict(brightness={"type": "number"}, space={"type": "number"}, grit={"type": "number"})),
    key=_enum(NOTES), mode=_enum(list(SCALES)),
    scenes={"type": "array", "items": _obj(dict(palette=_enum(list(PALETTES)), what={"type": "string"}))},
    sections={"type": "array", "items": _obj(dict(start={"type": "number"}, end={"type": "number"}, energy={"type": "number"}))},
    cues={"type": "array", "items": _obj(dict(t={"type": "number"}, type=_enum(CUE_TYPES), ref={"type": ["string", "null"]},
                                              dur={"type": "number"}, strength={"type": "number"}, sound=_enum(SOUNDS),
                                              pitch=_enum(PITCHES), note={"type": "string"}))},
))

PROMPT = """You are an experimental sound designer scoring a short SILENT video, the way sound artists post reels where they
"play" a clip on synthesizers: strange, textural, vibey sound design that is glued to the picture. NOT MUSIC.

Sound direction: designed physical materials with a surreal twist, not a keyboard or backing track.
- Contacts combine a dry micro-impact, broadband body, inharmonic object resonance and tiny debris.
- Continuous action excites friction, stick-slip textures and pressure deformations. It does not play a scale.
- A material stays recognizable across its scene; energy, position and interaction change its excitation.
- Use short tape stops, micro-repeats and deliberate silence when justified by the visible action.
- Preserve detailed attacks and natural tails; do not fill every moment with a drone or pad.
There is NO beat, NO tempo, NO melody, NO chords. Every prominent event must be grounded in the picture.

INPUT
- Contact sheets (images, in order): {sheets}
  Each tile is one moment; its timestamp in seconds is printed in the top-left corner. Tiles run left→right, top→bottom.
  Coloured squares in a tile's top-right corner mark detected events inside that tile's time window:
  orange = impact, yellow = light flash, pink = cut, blue = camera pan.
- Duration: {dur:.2f}s. Scenes (hard cuts at): {cuts}
- Motion energy per 0.5s (0 = still … 9 = violent): {energy}
- Detected events — computer vision found these, with ids. Times are frame-accurate; x,y = where on screen (0..1, y=0 top):
{events}

HOW TO WORK
1. Study the sheets: subjects, materials, actions, contacts, light, mood, the arc.
2. THE VIBE — the material and texture balance for the whole clip. Match the look and the feeling; different videos must sound different.
{vibes}
   vibe_why: one line. tweaks: brightness, space, grit 0..1 (0.5 = the vibe's default).
   key + mode: legacy metadata required by the schema; use C / minor. They do not tune the sound engine.
3. Scenes: one entry per scene in order (there are {nscenes}), a palette = the material of its contacts, and one line:
     glass → brittle resonant glass   metal → inharmonic metal   electric → rough ceramic   organic → elastic rubber
     cute → soft gel                  dark → heavy membrane     glitch → granular mechanism
4. Sections: 2-6 sections covering 0..{dur:.2f}s, energy 0..1 each. Energy controls excitation intensity; it does not automatically add sub-bass.
5. Cues — interactions alongside motion-driven friction. Types:
     hit       a contact / landing / step / bounce / grab. ref = the detected id. sound = gesture or auto (scene palette).
     boom      a clear crash or explosive event: layered heavy membrane impact. Ordinary steps, gestures and reveals
               remain hit/glide cues. Use a separate dropout cue if a pause before the impact is visible.
     glide     a short pitch slide (0.2-1 s) WITH a subject over t..t+dur. pitch: follow (its height on screen), up, down.
               Use it for falls, lifts, jumps, flights, things that grow. Not for the whole clip.
     sweep     a fast movement or camera move: filtered noise travelling with it; pitch up/down = direction.
     riser     tension building INTO a moment; t = the start, dur = 0.5-3s.
     chop      t..t+dur the surface texture is gated by changes in light.
     stutter   t..t+dur the sound at t repeats like a broken sampler (glitchy moments, impacts that echo, shaking).
     tapestop  the whole sound slows down and dies over dur (0.3-1s) ending at t — use right before a cut or a still.
     dropout   dead silence t..t+dur (0.05-0.6s): before a reveal, on a black frame, a held breath.
     swell     from t to t+dur a layered material texture swells and opens up: the reveal, the payoff, the ending.
   Fields: t (s), type, ref (detected id like "h12" or null), dur (s, 0 for instant), strength 0..1,
   sound (auto or: {sounds}), pitch (auto or: {pitches}), note (≤8 words: what it's for, e.g. "phone hits 3rd stair").
RULES
- Timing is everything: for anything that matches a detected event set ref to its id and copy its time. Only add a cue
  without ref when you clearly see an action the detector missed.
- False positives (camera shake, noise) → skip. Real contacts → hit, strength = how hard.
- Density like the reference reels: about 2-4 accents per second while something moves — every step, head turn,
  tail flick, blink, grab gets its own hit or a short glide, each a little different. Near-still moments stay sparse.
- Weird is good: tape stops, stutters, dropouts and short glides where the picture allows it — but keep them short;
  long held tones smear the physical interactions.
- summary: one line — what you see and the sound idea. story: 2-3 sentences on the arc of the sound.
Reply with ONLY the JSON object matching the schema, no prose."""


GLYPHS = {"0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
          "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001001001001",
          "8": "111101111101111", "9": "111101111001111", ".": "000000000000010"}
MARKS = dict(impact=(255, 122, 72), flash=(253, 224, 71), cut=(251, 113, 133), pan=(125, 211, 252))


def stamp(img, text, k=3):
    """burn a timestamp into the top-left corner with a 3x5 pixel font (this ffmpeg build has no drawtext)."""
    img[:5 * k + 4, :len(text) * 4 * k + 4] //= 4
    for j, ch in enumerate(text):
        g = np.array([int(b) for b in GLYPHS.get(ch, "0" * 15)], bool).reshape(5, 3)
        img[2:2 + 5 * k, 2 + j * 4 * k:2 + j * 4 * k + 3 * k][np.kron(g, np.ones((k, k), bool))] = 255


def contact_sheets(video, an, tmp, cols=4, rows=3, max_tiles=48):
    dur = an["T"] / FPS
    step = max(.25, dur / max_tiles)
    frames = decode(video, 320, fps=1 / step)
    ev = [(h["t"], "impact") for h in an["hits"] if h["s"] >= .5] + [(f["t"], "flash") for f in an["flashes"]] + \
         [(c, "cut") for c in an["cuts"]] + [(p["t"], "pan") for p in an["pans"]]
    tiles = []
    for i, f in enumerate(frames):
        f, t = f.copy(), i * step
        stamp(f, f"{t:.2f}")
        kinds = sorted({k for te, k in ev if t - step / 2 <= te < t + step / 2}, key=list(MARKS).index)
        for j, k in enumerate(kinds):
            f[4:16, f.shape[1] - 16 - 14 * j:f.shape[1] - 4 - 14 * j] = MARKS[k]
        f[-2:], f[:, -2:] = 0, 0  # thin separators
        tiles.append(f)
    names, per = [], cols * rows
    for s in range(0, len(tiles), per):
        chunk = tiles[s:s + per] + [np.zeros_like(tiles[0])] * (per - len(tiles[s:s + per]))
        sheet = np.concatenate([np.concatenate(chunk[r * cols:(r + 1) * cols], 1) for r in range(rows)], 0)
        fn = f"sheet{len(names) + 1}_{s * step:.1f}-{min(dur, (s + per) * step):.1f}s.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{sheet.shape[1]}x{sheet.shape[0]}",
                        "-i", "-", "-q:v", "3", os.path.join(tmp, fn)], input=sheet.tobytes(), check=True)
        names.append(fn)
    return names


def event_table(an):
    """ids the agent can reference → (time, x, y, strength). Also the text table for the prompt."""
    ids, lines = {}, []
    for i, h in enumerate(an["hits"], 1):
        ids[f"h{i}"] = h
        lines.append(f"  h{i:<3} {h['t']:6.2f}s  motion hit   strength {h['s']:.2f}  at x={h['x']:.2f} y={h['y']:.2f}")
    for i, f in enumerate(an["flashes"], 1):
        ids[f"f{i}"] = f
        lines.append(f"  f{i:<3} {f['t']:6.2f}s  light flash  strength {f['s']:.2f}")
    for i, c in enumerate(an["cuts"], 1):
        ids[f"c{i}"] = dict(t=c, s=1., x=.5, y=.5)
        lines.append(f"  c{i:<3} {c:6.2f}s  cut")
    for i, w in enumerate(an["whooshes"], 1):
        ids[f"m{i}"] = dict(t=w["t"], s=w["s"], x=float(np.mean(w["pan"])), y=.5, d=w["d"])
        lines.append(f"  m{i:<3} {w['t']:6.2f}s  sustained motion for {w['d']:.2f}s  strength {w['s']:.2f}")
    for i, p in enumerate(an["pans"], 1):
        ids[f"p{i}"] = dict(t=p["t"], s=p["s"], x=.5, y=.5, d=p["d"])
        lines.append(f"  p{i:<3} {p['t']:6.2f}s  camera moves, picture slides {p['dir']} for {p['d']:.2f}s")
    for i, g in enumerate(an["flicker"], 1):
        ids[f"l{i}"] = g
        lines.append(f"  l{i:<3} {g['t']:6.2f}s  light flicker")
    lines.sort(key=lambda s: float(s.split()[1][:-1]))
    return ids, "\n".join(lines[:220]) or "  (none)"


def parse_json(text):
    dec = json.JSONDecoder()
    for i in reversed([m.start() for m in re.finditer(r"\{", text)]):
        try:
            obj, _ = dec.raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(obj, dict) and "cues" in obj:
            return obj


def ask_agent(name, video, an, plan, model=None):
    tmp = tempfile.mkdtemp(prefix="foley-")
    try:
        sheets = contact_sheets(video, an, tmp)
        ids, table = event_table(an)
        ms = an["motion"]["ms"]
        e = [ms[i:i + FPS // 2].mean() for i in range(0, len(ms), FPS // 2)]
        energy = "".join(str(int(np.clip(v / (np.percentile(ms, 95) + 1e-9) * 7, 0, 9))) for v in e)
        prompt = PROMPT.format(vibes="\n".join(f"     {k:<9} {v['desc']}" for k, v in VIBES.items()),
                               sheets=", ".join(sheets), dur=an["T"] / FPS, cuts=", ".join(f"{c:.2f}s" for c in an["cuts"]) or "none",
                               energy=energy, events=table, nscenes=len(an["scenes"]), sounds=", ".join(SOUNDS[1:]),
                               pitches=", ".join(PITCHES[1:]))
        schema = os.path.join(tmp, "score.schema.json")
        json.dump(SCHEMA, open(schema, "w"))
        argv = AGENTS[name](prompt, sheets, schema) + (["--model", model] if model else [])
        p = subprocess.run(argv, cwd=tmp, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    r = parse_json(p.stdout)
    if not r:
        raise RuntimeError(f"no score from {name} (exit {p.returncode}): {(p.stderr or p.stdout).strip()[-200:]}")
    apply_score(r, an, plan, ids)
    plan["via"], plan["raw"] = name, r



def apply_score(r, an, plan, ids):
    """agent output is data: keep only known values, clamp numbers, resolve refs to frame-accurate times."""
    dur = an["T"] / FPS
    num = lambda v, lo, hi, d: float(np.clip(v, lo, hi)) if isinstance(v, (int, float)) and np.isfinite(v) else d
    if r.get("key") in NOTES:
        plan["key"] = r["key"]
    if r.get("mode") in SCALES:
        plan["mode"] = r["mode"]
    if r.get("vibe") in VIBES:
        plan["vibe"], plan["vibe_why"] = r["vibe"], str(r.get("vibe_why", ""))[:160]
    if isinstance(r.get("tweaks"), dict):
        plan["tweaks"] = {k: num(r["tweaks"].get(k), 0, 1, .5) for k in ("brightness", "space", "grit")}
    for s, x in zip(an["scenes"], r.get("scenes") or []):
        if isinstance(x, dict) and x.get("palette") in PALETTES:
            s["palette"], s["what"] = x["palette"], str(x.get("what", ""))[:120]
    secs = []
    for x in r.get("sections") or []:
        if isinstance(x, dict):
            a, b = num(x.get("start"), 0, dur, 0), num(x.get("end"), 0, dur, dur)
            if b - a > .2:
                secs.append(dict(a=a, b=b, energy=num(x.get("energy"), 0, 1, .5)))
    if secs:
        secs.sort(key=lambda s: s["a"])
        secs[0]["a"], secs[-1]["b"] = 0, dur
        for s0, s1 in zip(secs, secs[1:]):
            s1["a"] = s0["b"] = max(s0["a"] + .2, (s0["b"] + s1["a"]) / 2)
        plan["sections"] = secs
    anchors = [(h["t"], h) for h in ids.values()]
    cues = []
    for x in r.get("cues") or []:
        if not isinstance(x, dict) or x.get("type") not in CUE_TYPES:
            continue
        c = dict(type=x["type"], t=num(x.get("t"), -1, dur, None), d=num(x.get("dur"), 0, 8, 0), s=num(x.get("strength"), 0, 1, .6),
                 sound=x.get("sound") if x.get("sound") in SOUNDS else "auto", pitch=x.get("pitch") if x.get("pitch") in PITCHES else "auto",
                 note=str(x.get("note") or "")[:60], x=.5, y=.5)
        ref = ids.get(str(x.get("ref") or ""))
        if ref is None and c["t"] is not None and c["type"] in ("hit", "boom") and anchors:
            t0, h = min(anchors, key=lambda a: abs(a[0] - c["t"]))
            ref = h if abs(t0 - c["t"]) < .15 else None  # snap near-misses onto the detected frame
        if ref is not None:
            c.update(t=ref["t"], x=ref.get("x", .5), y=ref.get("y", .5))
            if c["type"] in ("sweep", "glide") and not c["d"]:
                c["d"] = ref.get("d", .5)
        if c["t"] is not None and 0 <= c["t"] < dur:
            cues.append(c)
    if cues:
        plan["cues"] = sorted(cues, key=lambda c: c["t"])
    plan["summary"] = str(r.get("summary", ""))[:160]
    plan["story"] = str(r.get("story", ""))[:400]


# ───────────────────────────── visual score ─────────────────────────────

def auto_score(an):
    """the no-agent director: turn detections into sections + cues (same format the agent returns)."""
    dur = an["T"] / FPS
    ms, cy = an["motion"]["ms"], smooth(an["motion"]["cy"], 9)
    act = np.clip(smooth(ms, 5) / (np.percentile(ms, 95) + 1e-9), 0, 1.5) * (smooth(ms, 5) > .004)
    secs = []
    for s in an["scenes"]:
        if secs and (s["b"] - s["a"] < .5 or s["a"] - secs[-1]["a"] < 2):
            secs[-1]["b"] = s["b"]
        else:
            secs.append(dict(a=s["a"], b=s["b"]))
    for s in secs:
        s.update(energy=float(np.clip(act[int(s["a"] * FPS):int(s["b"] * FPS) + 1].mean(), 0, 1)))
    cue = lambda typ, t, s=.6, d=0., x=.5, y=.5, pitch="auto", sound="auto", note="": dict(type=typ, t=t, s=s, d=d, x=x, y=y, sound=sound, pitch=pitch, note=note)
    cues = []
    strong = [h for h in an["hits"] if h["s"] >= .5]
    for h in strong:
        # Contacts articulate the synth. A clip's strongest contact alone is not evidence for a cinematic boom.
        cues.append(cue("hit", h["t"], h["s"], x=h["x"], y=h["y"]))
    lt = -1
    for h in an["hits"]:
        if h["s"] < .5 and h["t"] - lt >= .2:
            cues.append(cue("hit", h["t"], .25 + .5 * h["s"], x=h["x"], y=h["y"]))
            lt = h["t"]
    for f in an["flashes"]:
        cues.append(cue("hit", f["t"], f["s"], sound="crunch", note="flash"))
    ft = [g["t"] for g in an["flicker"]]  # dense flicker → the bed gets chopped by the light
    i = 0
    while i < len(ft):
        j = i
        while j + 1 < len(ft) and ft[j + 1] - ft[j] < .4:
            j += 1
        if j - i >= 2:
            cues.append(cue("chop", ft[i] - .05, .7, ft[j] - ft[i] + .2, note="light chops the sound"))
        i = j + 1
    for w in an["whooshes"] + an["pans"]:
        if w["s"] >= .45:
            cues.append(cue("sweep", w["t"], w["s"], w["d"], pitch="up" if w.get("dir") != "left" else "down"))
    cuts = an["cuts"]
    if cuts and cuts[-1] > 1.5:  # a short break at the cut; a cut alone does not justify a cinematic swell
        cues.append(cue("tapestop", cuts[-1], .8, min(.7, cuts[-1] - .5)))
    return secs, sorted(cues, key=lambda c: c["t"])


def compose(an, plan, seed):
    """Render the material engine while preserving the CLI/player result interface."""
    from foley_engine import render
    result = render(an, plan, seed, auto_score(an))
    return result.audio, result.events, result.meters


def write_wav(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x.T, -1, 1) * 32767).astype("<i2").tobytes())


def mux(video, wav, out):
    copy = os.path.splitext(video)[1].lower() in (".mp4", ".mov", ".m4v")
    v = ["-c:v", "copy"] if copy else ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-i", wav, "-map", "0:v:0", "-map", "1:a:0", *v,
                    "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", out], check=True)


# ───────────────────────────── pipeline ─────────────────────────────

STEPS = ["decode video", "analyze motion · light · cuts", "scene analysis", "synthesize sound", "render mp4 + wav"]


def pipeline(S):
    """decode → analyze → agent → compose → render, reporting progress into S["steps"] (read by the UI thread).
    Reuses S["frames"] / S["an"] when present, so a new variation only re-synthesizes."""
    st = [dict(name=n, state="wait", note="") for n in STEPS]
    if S.get("agent"):
        st[2]["name"] = f"{S['agent']} · {S.get('model') or 'default'} is watching the frames"
    S["steps"] = st

    def mark(i, state, note=""):
        if S.get("cancel") and S["cancel"].is_set():
            raise InterruptedError("generation stopped")
        st[i].update(state=state, note=note, **{"t0" if state == "run" else "t1": time.monotonic()})
        if state != "run" and S.get("log"):
            S["log"](st[i])

    if "frames" in S:
        mark(0, "ok", "cached")
    else:
        mark(0, "run")
        S["frames"] = f = decode(S["video"])
        if len(f) < FPS // 2:
            raise RuntimeError("video is too short (need at least 0.5s)")
        mark(0, "ok", f"{len(f) / FPS:.1f}s · {len(f)} frames")
    if "an" in S:
        mark(1, "ok", "cached")
        mark(2, "ok", "cached")
    else:
        mark(1, "run")
        an = analyze(S["frames"])
        S.update(an=an, plan=guess_plan(an))
        mark(1, "ok", f"{len(an['hits'])} hits · {len(an['flashes'])} flashes · {len(an['cuts'])} cuts · "
                      f"{len(an['whooshes'])} whooshes · {len(an['flicker'])} flicker")
        if S.get("agent"):
            mark(2, "run")
            try:
                ask_agent(S["agent"], S["video"], an, S["plan"], S.get("model"))
                mark(2, "ok", f"{len(S['plan'].get('cues', []))} cues · {S['plan']['summary']}")
            except Exception as e:  # the agent is optional
                mark(2, "fail", f"{str(e)[:160]} — using heuristics")
        else:
            mark(2, "skip", "heuristics only")
    p = S["plan"]
    if S.get("vibe") in VIBES:  # the user's pick wins over the director's
        p["vibe"] = S["vibe"]
    mark(3, "run")
    S["master"], S["events"], S["meters"] = compose(S["an"], p, S["seed"])
    mark(3, "ok", f"{p['vibe']} · {p.get('engine', 'material')} · {len(S['events'])} events · seed {S['seed']}")
    mark(4, "run")
    S["out"] = S.get("out") or output_path(S["video"])
    S["wav"] = os.path.splitext(S["out"])[0] + ".wav"
    source = os.path.realpath(S["video"])
    if source in (os.path.realpath(S["out"]), os.path.realpath(S["wav"])):
        raise RuntimeError("output must be different from the source file")
    write_wav(S["wav"], S["master"])
    mux(S["video"], S["wav"], S["out"])
    mark(4, "ok", S["out"].replace(os.path.expanduser("~"), "~"))



# ───────────────────────────── cli ─────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Procedural sound design for a video or gif. Run without arguments for the interactive app.")
    ap.add_argument("video", nargs="?")
    ap.add_argument("--agent", default="none", choices=["auto", "none", *AGENTS],
                    help="optional frame-analysis agent (default: none, fully local; auto: first installed CLI)")
    ap.add_argument("--model", help="model passed to the optional agent CLI")
    ap.add_argument("--vibe", choices=list(VIBES), help="force a sound world instead of letting the director pick")
    ap.add_argument("--seed", type=int, help="same seed = same sound design (default: random)")
    ap.add_argument("-o", "--out", help="output .mp4 (default: a new <video>.foley.mp4 next to the input)")
    ap.add_argument("--no-ui", action="store_true", help="just render, print progress")
    a = ap.parse_args()
    agent = next((n for n in AGENTS if shutil.which(n)), None) if a.agent == "auto" else None if a.agent == "none" else a.agent
    for command in ("ffmpeg", "ffprobe"):
        if not shutil.which(command):
            ap.error(f"{command} is required; install ffmpeg (brew install ffmpeg on macOS)")
    if a.video:
        try:
            a.video = local_video(a.video)
        except ValueError as e:
            ap.error(str(e))
    if a.out:
        a.out = os.path.abspath(os.path.expanduser(a.out))
        if Path(a.out).suffix.lower() != ".mp4":
            ap.error("output must end in .mp4")
        if a.video == os.path.realpath(a.out):
            ap.error("output must be different from the source file")
    if a.no_ui or not sys.stdout.isatty():
        if not a.video:
            sys.exit("pass a video path")
        S = dict(video=a.video, out=a.out, model=a.model, agent=agent, vibe=a.vibe,
                 seed=a.seed if a.seed is not None else int.from_bytes(os.urandom(3)),
                 log=lambda s: print(f"  {'✓' if s['state'] == 'ok' else '✗' if s['state'] == 'fail' else '–'} {s['name']}  {s['note']}", file=sys.stderr))
        try:
            pipeline(S)
        except (RuntimeError, OSError, subprocess.CalledProcessError) as e:
            sys.exit(str(e))
        print(S["out"])
        return
    from foley_tui import FoleyApp  # textual only needed for the interactive app
    app = FoleyApp(video=a.video, agent=agent, model=a.model, vibe=a.vibe, seed=a.seed, out=a.out)
    app.run()
    if not app.S.get("error") and app.S.get("out") and os.path.exists(app.S["out"]):
        print(f"  ◉ foley  {app.S['out']}")


if __name__ == "__main__":
    main()
