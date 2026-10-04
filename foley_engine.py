"""Material-based sound design. No note grid, scale quantization, or oscillator melody.

A scene owns one resonant object. Contacts excite its modes, motion excites friction,
and small particles produce debris. Independent per-event RNGs keep variations
reproducible without changing every later sound when one cue is inserted.
"""
from dataclasses import dataclass

import numpy as np
import pedalboard as pb
from scipy import signal
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

SR, FPS = 44100, 30
TRACKS = ("HIT", "VOICE", "GRAIN", "CHOP", "SWEEP", "BOOM", "SUB", "BED")
ENGINE_VERSION = "material-2"
# Inharmonic transients need codec headroom: AAC can overshoot the PCM sample peaks.
MASTER_PEAK = .78
# Contact / resonance / friction / debris balance. These shape objects, not a keyboard patch.
PROFILES = {
    "resonant": (1., 1.15, .045, .65), "formant": (.9, 1.05, .05, .5),
    "static": (.85, .65, .085, 1.), "pluck": (1., .88, .022, .55),
    "flicker": (.9, .75, .055, 1.1), "haze": (.65, 1., .045, .4),
    "swarm": (.7, .70, .09, 1.5), "abyss": (1.1, 1.05, .04, .45),
}


@dataclass(frozen=True)
class Material:
    name: str
    base: float
    ratios: tuple
    decay: float
    hardness: float
    roughness: float
    elasticity: float


MATERIALS = {
    "metal": Material("metal", 210, (1, 1.47, 2.09, 2.56, 3.18, 4.02, 4.81, 5.93, 7.12, 8.46, 10.27, 12.39), 1.1, .88, .54, .06),
    "glass": Material("glass", 390, (1, 1.61, 2.32, 3.08, 4.13, 5.43, 6.72, 8.91, 11.12, 13.61), .75, .95, .18, .02),
    "organic": Material("rubber", 245, (1, 1.28, 1.81, 2.24, 2.94, 3.83, 4.71, 5.97, 7.32), .23, .30, .65, .72),
    "cute": Material("gel", 320, (1, 1.36, 1.89, 2.72, 3.47, 4.41, 5.38, 6.83), .28, .20, .46, .91),
    "electric": Material("ceramic", 280, (1, 1.43, 2.14, 2.69, 3.71, 4.63, 6.17, 7.72, 9.43), .38, .77, .79, .14),
    "glitch": Material("mechanism", 185, (1, 1.19, 1.73, 2.41, 3.32, 4.74, 6.11, 8.53, 11.41), .30, .78, .92, .12),
    "dark": Material("membrane", 115, (1, 1.59, 2.14, 2.30, 2.65, 2.92, 3.49, 4.10, 5.28), .58, .38, .56, .26),
}


@dataclass
class Render:
    audio: np.ndarray
    events: list
    meters: dict
    stems: dict
    materials: list


def _unit(x):
    return x / max(float(np.max(np.abs(x), initial=0)), 1e-9)


def _filter(x, cutoff, kind="lowpass", order=2):
    return signal.sosfilt(signal.butter(order, cutoff, btype=kind, fs=SR, output="sos"), x, axis=-1)


def _noise(n, rng, low=80, high=6500):
    x = rng.standard_normal(n)
    x = _filter(_filter(x, min(high, SR * .44)), max(low, 20), "highpass")
    return x / max(float(np.std(x)), 1e-8)


def _curve(values, n):
    values = np.asarray(values, dtype=float)
    return np.interp(np.linspace(0, max(0, len(values) - 1), n), np.arange(len(values)), values)


def _fade(x, attack=.004, release=.035):
    n = x.shape[-1]
    env = np.ones(n)
    a, b = min(n, max(1, int(SR * attack))), min(n, max(1, int(SR * release)))
    env[:a] *= np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    env[-b:] *= np.sin(np.linspace(np.pi / 2, 0, b)) ** 2
    return x * env


def _saturate(x, drive=1.6):
    # Oversampling keeps the nonlinear surface layer from spraying aliases into the top end.
    up = signal.resample_poly(x, 2, 1, axis=-1)
    wet = np.tanh(drive * up) / drive
    return signal.resample_poly(wet, 1, 2, axis=-1)[..., :x.shape[-1]]


def _peak_control(audio):
    """Stereo-linked, oversampled peak control with short lookahead and smooth gain."""
    up = signal.resample_poly(audio, 4, 1, axis=-1)
    peak = np.max(np.abs(up), axis=0)
    ahead = maximum_filter1d(peak, size=round(.008 * SR * 4) | 1)
    target = np.minimum(1., MASTER_PEAK / np.maximum(ahead, 1e-9))
    radius = round(.020 * SR * 4)
    kernel = signal.windows.gaussian(2 * radius + 1, .005 * SR * 4)
    smoothed = signal.fftconvolve(np.pad(target, radius, mode="edge"), kernel / kernel.sum(), mode="valid")
    gain = np.minimum(target, smoothed)
    limited = signal.resample_poly(up * gain, 1, 4, axis=-1)[:, :audio.shape[-1]]
    return limited, gain[::4][:audio.shape[-1]]


class ObjectVoice:
    """A stable family of inharmonic modes, independently excited by each action."""

    def __init__(self, material, rng, weight=1.):
        self.material = material
        self.base = material.base * rng.uniform(.86, 1.16) / weight
        self.frequencies = self.base * np.asarray(material.ratios) * rng.uniform(.98, 1.02, len(material.ratios))
        self.decays = material.decay * rng.uniform(.65, 1.25, len(material.ratios)) / np.asarray(material.ratios) ** .43
        self.gains = rng.uniform(.6, 1.1, len(material.ratios)) / np.asarray(material.ratios) ** .72

    def resonate(self, excitation, shift=1., damping=1.):
        out = np.zeros(len(excitation))
        for f, decay, gain in zip(self.frequencies * shift, self.decays * damping, self.gains):
            if f >= SR * .43:
                continue
            radius = np.exp(-3 / (max(.015, decay) * SR))
            angle = 2 * np.pi * f / SR
            mode = signal.lfilter([1 - radius], [1, -2 * radius * np.cos(angle), radius ** 2], excitation)
            out += mode * gain * np.sin(angle)
        return _unit(out)

    def contact(self, strength, rng, grit=.5):
        m = self.material
        duration = float(np.clip(.18 + m.decay * (.45 + .5 * strength), .25, 1.7))
        n = max(32, int(duration * SR))
        t = np.arange(n) / SR
        # An imperfect contact has several microscopic collisions, not one oscillator envelope.
        excitation = _noise(n, rng, 130, 1700 + m.hardness * 10500)
        attack = .0015 + (1 - m.hardness) * .012
        excitation *= (1 - np.exp(-t / attack)) * np.exp(-t / (.006 + .017 * (1 - m.hardness)))
        for _ in range(rng.integers(3, 8)):
            at = int(rng.uniform(.003, .065) * SR)
            length = min(n - at, int(rng.uniform(.006, .025) * SR))
            if length > 1:
                excitation[at:at + length] += rng.uniform(.1, .4) * _noise(length, rng, 300, 4500) * np.exp(-np.arange(length) / (.003 * SR))
        resonance = self.resonate(excitation, shift=rng.uniform(.95, 1.06), damping=.55 + .55 * strength)
        if m.elasticity > .2:
            # Time deformation bends the entire object spectrum, including its noisy partials.
            speed = 1 + m.elasticity * rng.uniform(-.32, .55) * np.exp(-t / .075)
            index = np.cumsum(speed) - speed[0]
            resonance = np.interp(index, np.arange(n), resonance, right=0)
        body = _filter(_noise(n, rng, 45, 900), 180 + self.base * .75)
        body *= (1 - np.exp(-t / .003)) * np.exp(-t / (.035 + .055 * strength))
        debris = _noise(n, rng, 700, 5500) * np.exp(-t / .07)
        debris *= _curve(rng.uniform(0, 1, max(3, int(duration * 110))), n) ** 5
        dry = .48 * _unit(excitation) + .32 * _unit(body)
        ring = resonance * (.26 + .32 * m.hardness)
        chips = _unit(debris) * (.05 + .12 * m.roughness)
        # Parallel saturation glues the layers without erasing their leading edges.
        dry = .70 * dry + .30 * _saturate(dry, 1.3 + 3.2 * grit)
        return _fade(dry), _fade(ring, release=.06), _fade(chips)

    def friction(self, envelope, displacement, rng):
        n = len(envelope)
        m = self.material
        # Stick-slip asperities form a texture with a changing rate and pressure.
        density = 18 + 125 * envelope * (.4 + m.roughness)
        impulses = rng.random(n) < density / SR
        exciter = impulses * rng.uniform(-1, 1, n)
        kernel = np.exp(-np.arange(int(.007 * SR)) / (.0015 * SR))
        grit = signal.fftconvolve(exciter, kernel, mode="full")[:n]
        pressure = _noise(n, rng, 65, 1600)
        fibres = _noise(n, rng, 500, 6500)
        grain_env = _curve(rng.uniform(.1, 1, max(3, int(n / SR * 35))), n)
        raw = (.65 * grit + .055 * pressure + .025 * fibres * grain_env ** 3) * envelope
        modes = self.resonate(raw, damping=.35)
        warp = 1 + m.elasticity * .12 * (displacement - .5) + .015 * _curve(rng.normal(0, 1, max(3, int(n / SR * 9))), n)
        modes = np.interp(np.cumsum(warp) - warp[0], np.arange(n), modes, right=0)
        # No regular pulse train and no semitone staircase: resonance evolves with contact pressure.
        texture = (.65 * _unit(raw) + (.15 + .18 * m.elasticity) * modes) * envelope
        return _fade(texture, .025, .06)

    def gesture(self, envelope, displacement, rng, direction="follow"):
        """A few contacts excite a bending object; air is only a quiet trailing layer."""
        n = len(envelope)
        t = np.arange(n) / SR
        phase = np.linspace(0, 1, n)
        excitation = np.zeros(n)
        # One leading action plus a small number of imperfect rebounds.
        times = [0.] + sorted(rng.uniform(.025, min(.32, n / SR * .65), 3))
        for i, at in enumerate(times):
            start = int(at * SR)
            count = min(n - start, int(.035 * SR))
            if count > 1:
                burst = _noise(count, rng, 100, 2200 + 2800 * self.material.hardness)
                burst *= np.exp(-np.arange(count) / (SR * (.004 + .005 * self.material.elasticity)))
                excitation[start:start + count] += burst * (.38 ** i)
        modes = self.resonate(excitation, damping=1.2 + .9 * self.material.elasticity)
        path = gaussian_filter1d(displacement, max(1, .025 * SR))
        delta = path - path[0]
        delta /= max(float(np.max(np.abs(delta))), .08)
        bow = np.sin(np.pi * phase) * (.35 + .7 * self.material.elasticity)
        slope = -1 if direction == "down" else 1
        contour = .65 * delta + slope * bow - .35 * phase
        speed = 2 ** np.clip(contour, -1.2, 1.2)
        indexes = np.cumsum(speed) - speed[0]
        bent = np.interp(indexes, np.arange(n), modes, right=0)
        # A second, darker resonant body opens behind the attack, then closes.
        body = self.resonate(excitation, shift=.57, damping=.8)
        body = np.interp(indexes * .91, np.arange(n), body, right=0)
        articulation = (.45 + .55 * envelope) * np.exp(-t / max(.12, n / SR * .72))
        sound = (.76 * bent + .22 * body * np.sin(np.pi * phase)) * articulation
        air = _noise(n, rng, 500, 3400) * np.exp(-t / .045) * .008
        audio = _fade(.82 * sound + .18 * _saturate(sound, 2.2) + air, .003, .045)
        return audio, self.base * speed[np.linspace(0, n - 1, 48).astype(int)]


def _stereo(x, pan=0., width=.12):
    pan = np.clip(pan, -.9, .9)
    if np.ndim(pan):
        pan = _curve(pan, len(x))
    angle = (pan + 1) * np.pi / 4
    pair = np.stack([x * np.cos(angle), x * np.sin(angle)])
    # A very short reflected component adds size; the direct contact stays localized.
    delay = int(.0037 * SR)
    if len(x) > delay and width:
        pair[1, delay:] += width * x[:-delay]
    return pair


def _room(stem, rng, amount):
    n = stem.shape[-1]
    length = int(.48 * SR)
    t = np.arange(length) / SR
    for channel in range(2):
        ir = np.zeros(length)
        for delay, level in ((.009, .32), (.017, .22), (.029, .18), (.043, .10), (.067, .07)):
            ir[int((delay + rng.uniform(-.001, .001)) * SR)] = level * rng.choice([-1, 1])
        diffuse = _noise(length, rng, 250, 3900) * np.exp(-t / .073) * np.minimum(t / .023, 1)
        ir += diffuse * .0015
        stem[channel] += amount * signal.fftconvolve(stem[channel], ir, mode="full")[:n]
    return stem


def _regions(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def render(an, plan, seed, automatic_score):
    """Render audio, a material score and diagnostic stems (before master EQ/compression)."""
    n = max(1, round(an["T"] / FPS * SR))
    duration = n / SR
    root_rng = np.random.default_rng(seed)
    scenes = an["scenes"]
    vibe = plan.get("vibe", "resonant")
    contact_level, ring_level, friction_level, debris_level = PROFILES.get(vibe, PROFILES["resonant"])
    tweaks = plan.get("tweaks") or {}
    brightness = float(np.clip(tweaks.get("brightness", .5), 0, 1))
    space = float(np.clip(tweaks.get("space", .5), 0, 1))
    grit = float(np.clip(tweaks.get("grit", .5), 0, 1))
    stems = {track: np.zeros((2, n), np.float32) for track in TRACKS}
    events = []
    objects = []
    for scene in scenes:
        material = MATERIALS.get(scene.get("palette"), MATERIALS["organic"])
        if vibe == "abyss":
            material = MATERIALS["dark"]
        elif vibe == "formant":
            material = MATERIALS["organic"]
        elif vibe == "swarm":
            material = MATERIALS["glitch"]
        objects.append(ObjectVoice(material, root_rng, weight=1.3 if vibe == "abyss" else 1.))

    def scene_at(t):
        return min(len(scenes) - 1, max(0, sum(s["a"] <= t + 1e-6 for s in scenes) - 1))

    def rng_for(t, kind):
        code = sum((i + 1) * ord(c) for i, c in enumerate(kind))
        return np.random.default_rng(np.random.SeedSequence([int(seed) % 2 ** 32, max(0, round(t * SR)), code]))

    def put(track, x, t, gain=1., pan=0., width=.12):
        x = _stereo(x, pan, width) if x.ndim == 1 else x
        start = round(t * SR)
        if start < 0:
            x, start = x[:, -start:], 0
        count = min(x.shape[-1], max(0, n - start))
        if count:
            stems[track][:, start:start + count] += (gain * x[:, :count]).astype(np.float32)

    def event(track, t, d, strength, label, obj=None):
        if t >= duration or t + d <= 0:
            return
        item = dict(track=track, t=max(0., float(t)), d=min(float(d), duration - max(0., float(t))),
                    s=float(np.clip(strength, 0, 1)), label=label, token=label.split(" ")[0])
        if obj:
            item.update(material=obj.material.name, frequency_hz=obj.base,
                        midi=float(69 + 12 * np.log2(obj.base / 440)))
        events.append(item)
        return item

    mo = an["motion"]
    reg = np.asarray(mo["reg"])
    ms = np.asarray(mo["ms"])
    activity = np.zeros(len(ms))
    for scene in scenes:
        a, b = max(0, round(scene["a"] * FPS)), min(len(ms), round(scene["b"] * FPS))
        if b <= a:
            continue
        value = .7 * reg[a:b] + .3 * ms[a:b]
        floor = np.percentile(value, 15)
        activity[a:b] = np.clip((value - floor) / max(np.percentile(value, 92) - floor, .015), 0, 1)
    activity = gaussian_filter1d(activity, .7)
    activity *= np.clip(ms / .003, 0, 1)
    energy = np.ones(len(ms))
    for section in plan.get("sections") or []:
        a, b = max(0, round(section["a"] * FPS)), min(len(ms), round(section["b"] * FPS))
        energy[a:b] = .65 + .7 * np.clip(section.get("energy", .5), 0, 1)

    cues = [dict(c) for c in (plan.get("cues") if plan.get("cues") is not None else automatic_score[1])]
    contacts = sorted((c for c in cues if c["type"] in ("hit", "boom")), key=lambda c: c["t"])
    selected = []
    for cue in contacts:
        if selected and cue["t"] - selected[-1]["t"] < .09:
            if cue["s"] > selected[-1]["s"]:
                selected[-1] = cue
        else:
            selected.append(cue)
    # Gentle local maxima supply texture articulation when the visual detector sees ongoing movement.
    for index in signal.find_peaks(activity, prominence=.17, distance=6)[0]:
        t = index / FPS
        if all(abs(c["t"] - t) > .16 for c in selected):
            selected.append(dict(type="touch", t=t, s=.25 + .25 * activity[index], x=float(mo["rx"][index]), y=float(mo["ry"][index])))
    selected.sort(key=lambda c: c["t"])
    for cue in selected:
        t = float(cue["t"])
        obj = objects[scene_at(t)]
        strength = float(np.clip(cue["s"], .05, 1))
        rng = rng_for(t, "contact")
        chosen = {"bend": "organic", "blob": "cute", "ping": "glass", "ring": "metal", "chirp": "electric", "crunch": "glitch", "thump": "dark"}.get(cue.get("sound"))
        if chosen or cue.get("pitch") in ("low", "high"):
            weight = {"low": 1.5, "high": .7}.get(cue.get("pitch"), 1.)
            obj = ObjectVoice(MATERIALS[chosen] if chosen else obj.material, rng, weight)
        dry, ring, chips = obj.contact(strength, rng, grit)
        gain = (.16 + .63 * strength ** 1.6) * energy[min(len(energy) - 1, round(t * FPS))]
        if cue["type"] == "touch":
            gain *= .38
        pan = (float(cue.get("x", .5)) - .5) * 1.3
        put("HIT", dry, t, gain * contact_level, pan, .05)
        put("VOICE", ring, t, gain * ring_level, pan, .18)
        put("GRAIN", chips, t + .009, gain * 1.25 * debris_level, pan, .32)
        event("HIT", t, min(.16, len(dry) / SR), strength, obj.material.name + " contact", obj)
        event("VOICE", t, len(ring) / SR, strength, obj.material.name + " resonance", obj)
        if cue["type"] == "boom":
            mass = ObjectVoice(MATERIALS["dark"], rng, weight=1.8)
            body, tail, _ = mass.contact(1, rng)
            put("BOOM", body + tail, t, .9, pan, .12)
            event("BOOM", t, len(tail) / SR, strength, "mass impact", mass)

    # A motion gesture is a changing contact surface, not a sustained synthesizer note.
    for scene_index, scene in enumerate(scenes):
        a, b = max(0, round(scene["a"] * FPS)), min(len(ms), round(scene["b"] * FPS))
        obj = objects[scene_index]
        for lo, hi in _regions(activity[a:b] > .075):
            lo, hi = a + lo, a + hi
            if hi - lo < 4:
                continue
            # Bounded chunks cap working memory; overlaps preserve continuous friction.
            for start in range(lo, hi, 60):
                end = min(hi, start + 63)
                if end - start < 3:
                    continue
                t, size = start / FPS, round((end - start) / FPS * SR)
                env = _curve(activity[start:end] * energy[start:end], size) ** .8
                displacement = _curve(np.asarray(mo["cy"])[start:end], size)
                rng = rng_for(t, "friction")
                friction = obj.friction(env, displacement, rng)
                put("GRAIN", friction, t, friction_level, _curve(np.asarray(mo["cx"])[start:end] * 1.4 - .7, size), .25)
                event("GRAIN", t, size / SR, float(env.max()), obj.material.name + " friction", obj)
                # Elastic deformation adds an irregular, breath-like pressure layer to soft objects.
                if obj.material.elasticity > .2:
                    breath = _noise(size, rng, 90, 1100) * env ** 1.4
                    pressure = obj.resonate(breath, shift=.78, damping=.22)
                    put("BED", _fade(pressure * env, .035, .07), t, .045, .15, .25)
                    event("BED", t, size / SR, .3, "pressure", obj)

    for cue in cues:
        t, kind = float(cue["t"]), cue["type"]
        if kind in ("hit", "boom", "dropout", "tapestop", "stutter"):
            continue
        obj = objects[scene_at(t)]
        rng = rng_for(t, kind)
        length = float(np.clip(cue.get("d", .5), .12, min(4., max(.12, duration - t))))
        size = max(16, round(length * SR))
        strength = float(np.clip(cue.get("s", .6), 0, 1))
        a = min(len(ms) - 1, max(0, round(t * FPS)))
        b = min(len(ms), max(a + 1, round((t + length) * FPS)))
        env = np.sin(np.linspace(0, np.pi, size)) ** 1.3 * (.35 + .65 * _curve(activity[a:b], size))
        if kind in ("sweep", "riser", "swell", "glide"):
            moving = _curve(np.asarray(mo["cy"])[a:b], size)
            texture, frequencies = obj.gesture(env, moving, rng, cue.get("pitch", "follow"))
            track = "VOICE" if kind == "glide" else "BOOM" if kind == "swell" else "SWEEP"
            put(track, texture, t, .27 + .24 * strength,
                np.asarray(mo["cx"])[a:b] * 1.6 - .8, .22)
            item = event(track, t, length, strength, obj.material.name + " " + kind, obj)
            if item is not None:
                item["pitch_curve"] = (69 + 12 * np.log2(frequencies / 440)).tolist()
        elif kind == "chop":
            texture = obj.friction(np.maximum(env, .1), _curve(np.asarray(mo["cy"])[a:b], size), rng)
            light = np.asarray(mo["lum"])[a:b]
            gate = _curve(light > np.median(light), size)
            gate = gaussian_filter1d(gate, .0015 * SR)
            put("CHOP", texture * gate, t, .35, 0, .25)
            event("CHOP", t, length, strength, "light-gate", obj)

    # Short, object-sized acoustic space; the intrinsic material decay carries most of the tail.
    for track in ("HIT", "VOICE", "GRAIN", "SWEEP", "BOOM", "BED", "CHOP"):
        amount = (.11 if track in ("HIT", "GRAIN", "CHOP") else .23) * (.5 + space)
        stems[track] = _room(stems[track], rng_for(0, "room-" + track), amount).astype(np.float32)

    # Directional rests and tape gestures operate on every stem so graphics share their true envelope.
    gate = np.ones(n)
    if vibe == "flicker":
        lum = np.asarray(mo["lum"])
        local_dark = lum < gaussian_filter1d(lum, 6) - .015
        gate *= _curve(np.where(local_dark, .04, 1.), n)
    for cue in cues:
        t, kind = float(cue["t"]), cue["type"]
        if kind == "dropout":
            a, b = max(0, round(t * SR)), min(n, round((t + max(.03, cue.get("d", .1))) * SR))
            gate[a:b] = 0
            event("CHOP", t, (b - a) / SR, 1, "silence")
        elif kind == "tapestop":
            length = min(max(.08, cue.get("d", .3)), max(0, t))
            a, b = max(0, round((t - length) * SR)), min(n, round(t * SR))
            if b - a > 32:
                speed = np.linspace(1, .035, b - a) ** 1.3
                indexes = np.cumsum(speed) - speed[0]
                fade = np.linspace(1, 0, b - a) ** .7
                for stem in stems.values():
                    for channel in range(2):
                        stem[channel, a:b] = np.interp(indexes, np.arange(b - a), stem[channel, a:b]) * fade
                event("SWEEP", a / SR, (b - a) / SR, .8, "tape-stop")
        elif kind == "stutter":
            rng = rng_for(t, "stutter")
            a, b = max(0, round(t * SR)), min(n, round((t + max(.08, cue.get("d", .3))) * SR))
            grain = min(b - a, round(rng.uniform(.018, .055) * SR))
            if grain > 16:
                for stem in stems.values():
                    fragment = _fade(stem[:, a:a + grain].copy(), .002, .004)
                    for start in range(a, b, grain):
                        count = min(grain, b - start)
                        stem[:, start:start + count] = fragment[:, :count] * np.exp(-(start - a) / max(1, b - a))
                event("CHOP", t, (b - a) / SR, .8, "micro-repeat")
    gate = gaussian_filter1d(gate, .0015 * SR)
    for track in stems:
        stems[track] *= gate
        stems[track] = _fade(stems[track], .006, .025).astype(np.float32)

    mix = sum(stems.values())
    # Gentle finishing, not a blanket low-pass used to imitate a reference's spectral average.
    board = pb.Pedalboard([pb.HighpassFilter(30), pb.HighShelfFilter(cutoff_frequency_hz=5500, gain_db=-3 + brightness * 3),
                          pb.Compressor(threshold_db=-12, ratio=1.45, attack_ms=22, release_ms=95)])
    master = board(mix, SR)
    rms = float(np.sqrt(np.mean(master.astype(float) ** 2)))
    gain = min(10 ** (-16.6 / 20) / max(rms, 1e-9), 8.)
    before_limit = master * gain
    master, limiter_gain = _peak_control(before_limit)
    # A common envelope keeps exported stems and meters aligned with peak control.
    post_rms = float(np.sqrt(np.mean(master.astype(float) ** 2)))
    true_peak = float(np.abs(signal.resample_poly(master, 4, 1, axis=-1)).max())
    trim = min(1., 10 ** (-16.6 / 20) / max(post_rms, 1e-9), MASTER_PEAK / max(true_peak, 1e-9))
    master *= trim
    for track in stems:
        stems[track] *= gain * limiter_gain * trim
    hop = SR // FPS
    meters = {k: np.sqrt(np.mean(v[:, :n // hop * hop].reshape(2, -1, hop) ** 2, axis=(0, 2))) for k, v in stems.items()}
    top = max((float(v.max(initial=0)) for v in meters.values()), default=1)
    meters = {k: np.sqrt(v / max(top, 1e-9)) for k, v in meters.items()}
    plan.update(engine=ENGINE_VERSION, materials=[o.material.name for o in objects], vibe=vibe)
    return Render(master.astype(np.float32), sorted(events, key=lambda e: e["t"]), meters, stems, plan["materials"])
