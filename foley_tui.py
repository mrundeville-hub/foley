"""One terminal canvas: local video → weirdo sound design → animated score."""
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.theme import Theme
from textual.widget import Widget
from textual.widgets import Button, Input, Static

import foley as F
from foley_preview import VideoPlayer

BLUE, WHITE = "#0814b9", "#ffffff"
INK = f"{WHITE} on {BLUE}"
CURSOR = f"{BLUE} on {WHITE}"
MUTED = f"#6475df on {BLUE}"
SELECTED = "#ffffff on #070f89"
STAGE_NAMES = ("reading the picture", "following motion", "finding its voice", "making it weird", "writing the sound")


class Score(Widget):
    """Three inline punchcards, following the user's live-code reference."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.t = 0.
        self.live = False
        self.duration = 1.
        self.events = []
        self.meters = {}
        self.waveform = np.zeros(1)
        self._layout = None
        self._base = None
        self.plan = {}

    def load(self, state):
        self.duration = state["master"].shape[1] / F.SR
        self.events = state["events"]
        self.meters = state["meters"]
        self.plan = state["plan"]
        audio = state["master"]
        hop = max(1, F.SR // 100)
        n = audio.shape[1] // hop
        # Absolute peaks preserve transients and stereo phase differences.
        self.waveform = np.abs(audio[:, :n * hop]).reshape(2, n, hop).max(axis=(0, 2))
        self.live, self.t, self._layout = True, 0., None
        self.refresh()

    def idle(self):
        self.live, self._layout = False, None
        self.plan, self.events, self.meters = {}, [], {}
        self.duration = 1.

    def animate(self, t):
        self.t = t
        self.refresh()

    def build_score(self, width, height, section=0):
        grid = np.full((height, width), " ", dtype="<U1")
        tracks = (("VOICE", "SWEEP"),
                  ("HIT", "BOOM", "GRAIN"), ("CHOP",))[section]
        pitched = [e["midi"] for e in self.events if e["track"] in tracks and "midi" in e]
        pitched += [p for e in self.events if e["track"] in tracks for p in e.get("pitch_curve", [])]
        low, high = (min(pitched) - 1, max(pitched) + 1) if pitched else (36, 72)
        for event in self.events:
            track = event["track"]
            if track not in tracks:
                continue
            x = min(width - 1, max(0, int(event["t"] / self.duration * width)))
            end = min(width, max(x + 1, int((event["t"] + max(event["d"], .04)) / self.duration * width)))
            level = (event["midi"] - low) / (high - low) if "midi" in event else event.get("y", event["s"])
            y = min(height - 1, round((1 - np.clip(level, 0, 1)) * (height - 1)))
            if section == 0 and event.get("pitch_curve"):
                curve = event["pitch_curve"]
                levels = np.interp(np.linspace(0, len(curve) - 1, end - x), np.arange(len(curve)), curve)
                rows = np.rint((1 - np.clip((levels - low) / (high - low), 0, 1)) * (height - 1)).astype(int)
                grid[rows, np.arange(x, end)] = "▬"
            else:
                grid[y, x:end if section == 0 else x + 1] = "▬" if section == 0 else "▪"
        if section == 1:
            values = self.meters.get("GRAIN", np.zeros(1))
            v = values[np.linspace(0, len(values) - 1, width).astype(int)]
            for x in np.flatnonzero(v > .18):
                grid[x % height, x] = "▪"
        elif section == 2:
            # Full-height slices: boundaries come from chops and audio transients.
            peaks = np.array([float(c.max()) if len(c) else 0. for c in np.array_split(self.waveform, width)])
            values = self.meters.get("CHOP", np.zeros(1))
            v = values[np.linspace(0, len(values) - 1, width).astype(int)]
            gate = (v > .08) | (peaks > .12)
            for x in range(width):
                if gate[x]:
                    grid[:, x] = "▏" if x == 0 or not gate[x - 1] or abs(peaks[x] - peaks[x - 1]) > .12 else "█"
        return grid

    def build_idle(self, width, height, section=0):
        grid = np.full((height, width), " ", dtype="<U1")
        if section == 2:
            for x in range(width):
                phase = int(x + self.t * 3)
                if phase % 19 not in (0, 1, 11):
                    grid[:, x] = "▏" if phase % 5 == 0 else "█"
            return grid
        for lane in range(2 if section == 0 else 3):
            for x in range(width):
                phase = x * .055 + self.t * .3 + lane * 1.6
                y = min(height - 1, int((np.sin(phase) + 1) * .5 * (height - 1)))
                if section == 0 and (x + lane * 11) % 23 < 18:
                    grid[y, x] = "▬"
                elif section == 1 and x % 3 == 0:
                    grid[y, x] = "▪"
        return grid

    def heading(self, section, width, detail=2):
        line = Text(style=INK)
        line.append(f"{section * 4 + 1:02} ", style=MUTED)
        name = ("voice", "gestures", "chops")[section]
        tracks = (("VOICE", "SWEEP"), ("HIT", "BOOM", "GRAIN"), ("CHOP",))[section]
        events = [e for e in self.events if e["track"] in tracks]
        active_index = max((i for i, e in enumerate(events) if e["t"] <= self.t), default=0)
        start = max(0, active_index - 4)
        line.append(name + (': sound("<' if section < 2 else ': slice("<'))
        if self.live:
            for e in events[start:start + 9]:
                token = e.get("token") or (F.NOTES[int(e["midi"]) % 12].lower() + str(int(e["midi"]) // 12 - 1) if "midi" in e else e["label"].split()[0])
                values = self.meters.get(e["track"], np.zeros(1))
                meter = values[min(len(values) - 1, int(self.t / self.duration * len(values)))]
                active = e["t"] <= self.t < e["t"] + e["d"] and meter > .08
                line.append(f"[{token}]" if active else token, style="underline " + SELECTED if active else INK)
                line.append(" ")
            if not events:
                line.append("~")
        else:
            line.append("waiting for video" if section == 0 else "motion preview" if section == 1 else "amplitude preview")
        line.append('>")')
        line.truncate(width, overflow="ellipsis")
        lines = [line]
        if detail == 3:
            patch = F.VIBES.get(self.plan.get("vibe"), {})
            descriptions = (f'.sound("{patch.get("voice", "auto")}").vibe("{self.plan.get("vibe", "auto")}")',
                            '.follow("motion / contacts / flashes")', '.follow("cuts / flicker / amplitude")')
            meta = Text(f"{section * 4 + 2:02} ", style=MUTED)
            meta.append("  " + descriptions[section], style=INK)
            meta.truncate(width, overflow="ellipsis")
            lines.append(meta)
        if detail >= 2:
            chart_tracks = (("VOICE", "SWEEP"), ("HIT", "BOOM", "GRAIN"), ("CHOP",))[section]
            count = sum(e["track"] in chart_tracks for e in self.events)
            info = (f"{count} event{'s' if count != 1 else ''}" if section != 2 or count else "output amplitude") if self.live else "preview"
            sub = Text(f"{section * 4 + detail:02} ", style=MUTED)
            sub.append(f"  .punchcard()  /  {info}", style=INK)
            sub.truncate(width, overflow="ellipsis")
            lines.append(sub)
        return lines

    def render(self):
        width, height = self.size.width, self.size.height
        if width < 1 or height < 6:
            return Text("")
        header_height = 3 if height >= 22 else 2 if height >= 12 else 1
        gaps = 2 if height >= 11 else 0
        available = height - header_height * 3 - gaps
        sizes = [max(1, available * 2 // 5), max(1, available * 2 // 5)]
        sizes.append(max(1, available - sum(sizes)))
        chart_width = max(1, int((width - 5) * .80)) if width >= 70 else max(1, width - 5)
        layout = (chart_width, *sizes)
        if self.live and self._layout != layout:
            self._base = [self.build_score(chart_width, h, section) for section, h in enumerate(sizes)]
            self._layout = layout
        cursor = min(chart_width - 1, int(self.t / self.duration * chart_width)) if self.live else int(self.t * 9) % chart_width
        out = Text(style=INK)
        for section, h in enumerate(sizes):
            for line in self.heading(section, width, header_height):
                out.append_text(line)
                out.append("\n")
            grid = self._base[section] if self.live else self.build_idle(chart_width, h, section)
            for row in grid:
                out.append("     ")
                for x, char in enumerate(row):
                    if x == cursor:
                        out.append("│", style=MUTED)
                    elif section == 2 and char in ("█", "▏"):
                        out.append(" " if char == "█" else char, style=CURSOR)
                    else:
                        out.append(char)
                out.append("\n")
            if section < 2 and gaps:
                out.append("\n")
        return out[:-1]  # retain styled spaces in the final white slice row


WEIRDO = Theme(
    name="weirdo-blue", primary=WHITE, secondary=WHITE, accent=WHITE,
    foreground=WHITE, background=BLUE, surface=BLUE, panel=BLUE,
    boost=BLUE, success=WHITE, warning=WHITE, error=WHITE, dark=True,
    variables={"text-muted": WHITE, "input-cursor-background": WHITE,
               "input-cursor-foreground": BLUE, "input-selection-background": BLUE},
)


class FoleyApp(App):
    TITLE = "foley / weirdo sound design"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("ctrl+l", "new_file", "new file", show=False),
        Binding("n", "new_file", "new video", show=False),
        Binding("enter", "process_file", "process video", show=False),
        Binding("space", "pause", "pause", show=False),
        Binding("r", "replay", "replay", show=False),
        Binding("v", "variation", "variation", show=False),
        Binding("o", "open_result", "open video", show=False),
        Binding("f", "reveal_result", "Finder", show=False),
        Binding("escape", "quit", "quit", show=False),
        Binding("ctrl+c", "quit", "quit", show=False, priority=True),
    ]
    CSS = """
    Screen { background: #0814b9; color: #ffffff; padding: 1; }
    #brand { height: 1; color: #ffffff; text-style: bold; margin-bottom: 1; }
    Score { height: 1fr; color: #ffffff; margin-bottom: 1; }
    #status { height: 1; color: #ffffff; }
    #path { height: 3; background: #0814b9; color: #ffffff;
            border: solid #6475df; padding: 0 1; }
    #path:focus { border: solid #ffffff; background-tint: transparent; }
    #path > .input--placeholder { color: #ffffff; }
    #path > .input--cursor { background: #ffffff; color: #0814b9; }
    #actions { height: 1; margin-top: 1; }
    #actions Button { height: 1; min-width: 0; width: auto; padding: 0; line-pad: 1;
        border: none; background: #0814b9; color: #ffffff; margin-right: 0; }
    #actions Button:hover, #actions Button:focus { background: #ffffff; color: #0814b9; text-style: none; }
    #actions Button:disabled { color: #6475df; }
    #actions #new { width: 11; }
    #actions #make { width: 9; text-style: bold; }
    #actions #replay { width: 8; }
    #actions #vary { width: 6; }
    #actions #files { width: 7; }
    #keys { height: 1; color: #6475df; }
    """

    def __init__(self, video=None, agent=None, model=None, vibe=None, seed=None, out=None):
        super().__init__()
        self.S = {}
        self.first_video = video
        self.agent, self.model, self.vibe = agent, model, vibe
        self.seed, self.requested_out = seed, out
        self.busy = False
        self.proc = None
        self.playing = False
        self.finished = False
        self.selecting = not bool(video)
        self.input_error = False
        self.epoch = time.monotonic()

    def compose(self) -> ComposeResult:
        yield Static("foley / weirdo sound design", id="brand", markup=False)
        yield Score(id="score")
        yield Static("give the picture a strange voice", id="status", markup=False)
        yield Input(placeholder="paste / drop a video path → Enter to process", id="path")
        with Horizontal(id="actions"):
            yield Button("New video", id="new")
            yield Button("Process", id="make")
            yield Button("Replay", id="replay")
            yield Button("Vary", id="vary")
            yield Button("Files", id="files")
        yield Static("enter process   n new video   esc quit", id="keys", markup=False)

    def on_mount(self):
        self.register_theme(WEIRDO)
        self.theme = WEIRDO.name
        self.query_one(Input).focus()
        self.update_controls()
        self.set_interval(1 / 24, self.tick)
        if self.first_video:
            self.submit(self.first_video)

    def on_input_submitted(self, event):
        if not self.busy:
            self.submit(event.value)

    def submit(self, value):
        if self.busy:
            return
        try:
            video = F.local_video(value)
        except ValueError as e:
            self.input_error = True
            self.status(str(e))
            self.query_one(Input).focus()
            return
        self.input_error = False
        self.stop_playback()
        self.S = dict(video=video, agent=self.agent, model=self.model, vibe=self.vibe,
                      seed=self.seed if self.seed is not None else int.from_bytes(os.urandom(3)),
                      out=self.requested_out)
        self.generate()

    def update_controls(self):
        self.query_one(Input).disabled = self.busy
        for name in ("new", "make"):
            self.query_one("#" + name, Button).disabled = self.busy
        for name in ("replay", "vary", "files"):
            self.query_one("#" + name, Button).disabled = self.busy or self.selecting or "master" not in self.S or bool(self.S.get("error"))

    def on_button_pressed(self, event):
        actions = {"new": self.action_new_file, "make": self.action_process_file,
                   "replay": self.action_replay, "vary": self.action_variation,
                   "files": self.action_reveal_result}
        action = actions.get(event.button.id)
        if action:
            action()
            if event.button.id == "files":
                self.set_focus(None)

    def action_process_file(self):
        self.submit(self.query_one(Input).value)

    def status(self, text):
        self.query_one("#status", Static).update(text)

    def keys(self, text):
        self.query_one("#keys", Static).update(text)

    def generate(self):
        self.busy, self.finished = True, False
        self.selecting = False
        self.S.pop("error", None)
        self.S["cancel"] = threading.Event()
        self.query_one(Input).value = self.S["video"]
        self.update_controls()
        self.query_one(Score).idle()
        self.set_focus(None)
        self.status("reading the picture")
        self.keys("esc quit")
        self.run_worker(self.generate_sound, thread=True, exclusive=True)

    def generate_sound(self):
        state = self.S
        try:
            F.pipeline(state)
        except InterruptedError:
            return
        except Exception as e:
            state["error"] = f"{type(e).__name__}: {e}"
        if not state["cancel"].is_set():
            self.call_from_thread(self.generated)

    def generated(self):
        self.busy = False
        if self.S.get("error"):
            self.selecting = True
            self.update_controls()
            self.status("could not make sound · " + self.S["error"])
            inp = self.query_one(Input)
            inp.value = self.S["video"]
            inp.focus()
            self.keys("enter retry   ·   ctrl+l another file   ·   esc quit")
            return
        self.query_one(Score).load(self.S)
        self.requested_out = None  # -o applies to the first successful clip only.
        self.update_controls()
        self.set_focus(None)
        self.keys("space pause  r replay  v vary  n new video  f files  esc quit")
        self.start_playback()

    def start_playback(self):
        self.stop_playback()
        try:
            self.proc = VideoPlayer(self.S["out"])
        except OSError as e:
            self.finish(f"playback unavailable: {e}")
            return
        self.playing, self.finished = True, False
        self.query_one(Score).animate(0.)

    def stop_playback(self):
        if self.proc:
            self.proc.stop()
        self.proc, self.playing = None, False

    @property
    def paused(self):
        return self.now() if self.proc and self.proc.state["paused"] else None

    def now(self):
        return self.proc.state["position"] if self.proc else 0.

    def finish(self, note=None):
        self.stop_playback()
        self.finished = True
        if not note:
            self.query_one(Score).animate(self.query_one(Score).duration)
        self.status(note or f"saved / {Path(self.S['out']).name} / choose another video below")

    def tick(self):
        score = self.query_one(Score)
        if self.busy:
            steps = self.S.get("steps", [])
            active = next((i for i, s in enumerate(steps) if s["state"] == "run"), None)
            if active is not None:
                self.status(STAGE_NAMES[active] + " · " + "·" * (1 + int(time.monotonic() * 3) % 3))
            score.animate(time.monotonic() - self.epoch)
        elif self.playing:
            t = min(self.now(), score.duration)
            score.animate(t)
            state = "opening video" if not self.proc.state["ready"] else "paused" if self.paused is not None else "playing"
            if not self.input_error:
                self.status(f"{state}  {t:05.1f} / {score.duration:05.1f}  ·  {self.S['plan']['vibe']}  ·  {Path(self.S['video']).name}")
            code = self.proc.poll()
            if self.proc.state["error"]:
                self.finish(self.proc.state["error"])
            elif code is not None:
                self.finish("video player stopped; press r to reopen" if code else None)
            elif self.proc.state["eof"]:
                self.finish()
        elif not self.finished:
            score.animate(time.monotonic() - self.epoch)

    def action_pause(self):
        if not self.playing or self.proc.poll() is not None:
            return
        self.proc.toggle_pause()

    def action_replay(self):
        if not self.busy and not self.selecting and "master" in self.S and not self.S.get("error"):
            self.set_focus(None)
            self.start_playback()

    def action_variation(self):
        if self.busy or self.selecting or "master" not in self.S or self.S.get("error"):
            return
        self.stop_playback()
        self.S["seed"] = int.from_bytes(os.urandom(3))
        self.S["out"] = None
        self.generate()

    def action_new_file(self):
        if self.busy:
            return
        self.stop_playback()
        self.finished = False
        self.selecting = True
        self.input_error = False
        self.query_one(Score).idle()
        inp = self.query_one(Input)
        inp.display, inp.value = True, ""
        self.update_controls()
        inp.focus()
        self.status("new video / paste a path below, then Enter")
        self.keys("enter process   ·   esc quit")

    def open_output(self, reveal=False):
        if self.busy or not self.S.get("out") or not Path(self.S["out"]).is_file():
            return
        if shutil.which("open"):
            subprocess.Popen(["open", *(["-R"] if reveal else []), self.S["out"]])
        else:
            self.status(self.S["out"])

    def action_open_result(self):
        self.action_replay()

    def action_reveal_result(self):
        self.open_output(reveal=True)

    def action_quit(self):
        if self.S.get("cancel"):
            self.S["cancel"].set()
        self.stop_playback()
        self.exit()

    def on_unmount(self):
        if self.S.get("cancel"):
            self.S["cancel"].set()
        self.stop_playback()
