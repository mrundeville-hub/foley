"""Integration checks for file input, rendering, and the terminal player."""
import asyncio
import shlex
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
from textual.widgets import Button, Input, Static

import foley as F
from foley_tui import FoleyApp, Score
from foley_preview import VideoPlayer


class Player:
    """Native preview boundary used by the UI test without opening a window."""

    def __init__(self):
        self.code = None
        self.state = dict(position=0., paused=False, ready=True, eof=False, error=None)

    def poll(self):
        return self.code

    def toggle_pause(self):
        self.state["paused"] = not self.state["paused"]

    def stop(self):
        self.code = 0


class Workflow(unittest.TestCase):
    def test_score_shows_sweep_contour(self):
        score = Score()
        score.duration = 2
        score.events = [dict(track="SWEEP", t=0, d=2, s=.7, midi=48, pitch_curve=[48, 72, 48])]
        grid = score.build_score(21, 8)
        rows = [np.flatnonzero(grid[:, x] == "▬")[0] for x in (0, 10, 20)]
        self.assertGreater(rows[0], rows[1])
        self.assertEqual(rows[0], rows[2])

    def test_score_displays_pitch_independently_of_strength(self):
        score = Score()
        score.duration = 2
        score.events = [dict(track="VOICE", t=0, d=.5, s=.7, midi=48),
                        dict(track="VOICE", t=1, d=.5, s=.7, midi=72)]
        grid = score.build_score(20, 6)
        low = np.flatnonzero(grid[:, 0] == "▬")
        high = np.flatnonzero(grid[:, 10] == "▬")
        self.assertEqual(len(low), 1)
        self.assertEqual(len(high), 1)
        self.assertGreater(low[0], high[0])

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="foley-check-")
        cls.root = Path(cls.tmp.name).resolve()
        cls.video = cls.root / "strange [clip] name.mp4"
        frames = np.zeros((90, 64, 96, 3), np.uint8)
        for i in range(len(frames)):
            frames[i] = (20, 20, 55) if i < 45 else (65, 20, 10)
            x = int(36 + 25 * np.sin(i * .15))
            y = int(22 + 15 * np.sin(i * .2))
            frames[i, y:y + 12, x:x + 12] = 245
            if i in (30, 31):
                frames[i] = 230
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", "96x64", "-r", "30", "-i", "-", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(cls.video)], input=frames.tobytes(), check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_local_input(self):
        for value in (str(self.video), self.video.as_uri(), shlex.quote(str(self.video)),
                      str(self.video).replace(" ", "\\ ")):
            self.assertEqual(F.local_video(value), str(self.video))
        for value in ("https://example.org/clip.mp4", "file://server/clip.mp4", "no-file.mp4"):
            with self.assertRaises(ValueError):
                F.local_video(value)

    def test_cli_render_and_source_protection(self):
        original = self.video.read_bytes()
        out = self.root / "render.mp4"
        cmd = [sys.executable, str(Path(F.__file__)), self.video.as_uri(), "--no-ui",
               "--vibe", "resonant", "--seed", "42", "-o", str(out)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), str(out))
        self.assertIn("heuristics only", p.stderr)
        self.assertEqual(self.video.read_bytes(), original)
        with wave.open(str(out.with_suffix(".wav"))) as w:
            self.assertEqual((w.getnchannels(), w.getframerate(), w.getsampwidth()), (2, 44100, 2))
            audio = np.frombuffer(w.readframes(w.getnframes()), "<i2")
            self.assertGreater(np.max(np.abs(audio.astype(int))), 1000)
        info = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                               "-of", "csv=p=0", str(out)], capture_output=True, text=True, check=True)
        self.assertEqual(set(info.stdout.split()), {"video", "audio"})
        bad = subprocess.run(cmd[:-1] + [str(self.video)], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("different from the source", bad.stderr)
        self.assertEqual(self.video.read_bytes(), original)

    def test_existing_results_are_preserved(self):
        first = Path(F.output_path(str(self.video)))
        first.write_bytes(b"existing video")
        second = Path(F.output_path(str(self.video)))
        self.assertNotEqual(first, second)
        second.with_suffix(".wav").write_bytes(b"existing audio")
        third = Path(F.output_path(str(self.video)))
        self.assertNotIn(third, (first, second))
        self.assertEqual(first.read_bytes(), b"existing video")

    def test_terminal_workflow(self):
        players = []

        def player(path):
            self.assertTrue(Path(path).is_file())
            self.assertEqual(Path(path).suffix, ".mp4")
            p = Player()
            players.append(p)
            return p

        async def until(pilot, condition):
            for _ in range(200):
                if condition():
                    return
                await pilot.pause(.03)
            self.fail("generation timed out")

        async def check():
            app = FoleyApp(vibe="formant", seed=17, out=str(self.root / "ui.mp4"))
            with patch("foley_tui.VideoPlayer", side_effect=player):
                async with app.run_test(size=(80, 24)) as pilot:
                    await pilot.pause(.1)
                    score = app.query_one(Score)
                    a = score.render().plain
                    await pilot.pause(.15)
                    self.assertNotEqual(a, score.render().plain)
                    await pilot.press("r", "space", "v")
                    self.assertEqual(app.query_one(Input).value, "r v")
                    app.query_one(Input).value = self.video.as_uri()
                    await pilot.press("enter")
                    await until(pilot, lambda: not app.busy)
                    self.assertNotIn("error", app.S)
                    self.assertTrue(app.playing)
                    self.assertEqual((app.S["plan"]["vibe"], app.S["seed"]), ("formant", 17))
                    self.assertEqual(app.S["out"], str(self.root / "ui.mp4"))
                    old_out = app.S["out"]
                    await pilot.press("space")
                    pos = app.now()
                    await pilot.pause(.1)
                    self.assertEqual(pos, app.now())
                    self.assertTrue(players[-1].state["paused"])
                    await pilot.press("space")
                    self.assertIsNone(app.paused)
                    await pilot.press("r")
                    self.assertLess(app.now(), .3)
                    await pilot.press("v")
                    await until(pilot, lambda: not app.busy)
                    self.assertTrue(app.playing)
                    self.assertNotEqual(old_out, app.S["out"])
                    self.assertTrue(Path(old_out).is_file())
                    for size in ((50, 18), (120, 40)):
                        await pilot.resize_terminal(*size)
                        await pilot.pause(.05)
                        self.assertGreaterEqual(score.size.height, 6)
                        self.assertLessEqual(app.query_one("#keys").region.bottom, size[1])
                        self.assertLessEqual(app.query_one("#files").region.right, size[0])
                        for button in app.query(Button):
                            self.assertIn(str(button.label), button.render_line(0).text)
                    await pilot.press("ctrl+l")
                    self.assertFalse(app.playing)
                    self.assertTrue(app.query_one(Input).display)
                    app.query_one(Input).value = "/no-such-video.mp4"
                    await pilot.press("enter")
                    self.assertIn("no such file", str(app.query_one("#status", Static).content))
                    corrupt = self.root / "broken.mp4"
                    corrupt.write_bytes(b"not a video")
                    app.query_one(Input).value = str(corrupt)
                    await pilot.press("enter")
                    await until(pilot, lambda: not app.busy)
                    self.assertIn("error", app.S)
                    self.assertTrue(app.query_one(Input).display)
                    await pilot.press("escape")
                self.assertTrue(all(p.poll() is not None for p in players))

        asyncio.run(check())

    def test_real_player_clock_and_pause(self):
        # Real IPC and real decoding, using null output so CI does not need speakers or a display.
        player = VideoPlayer(str(self.video), extra_args=("--vo=null", "--ao=null"))

        def until(condition):
            import time
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                self.assertIsNone(player.state["error"], player.state["error"])
                if condition():
                    return
                time.sleep(.03)
            self.fail(f"player timed out: {player.state}")

        try:
            until(lambda: player.state["ready"] and player.state["position"] > .1)
            player.toggle_pause()
            until(lambda: player.state["paused"])
            import time
            time.sleep(.1)
            pos = player.state["position"]
            time.sleep(.25)
            self.assertAlmostEqual(player.state["position"], pos, delta=.001)
            player.toggle_pause()
            until(lambda: not player.state["paused"] and player.state["position"] > pos + .1)
        finally:
            address = Path(player.address)
            player.stop()
            self.assertIsNotNone(player.poll())
            self.assertFalse(address.exists())

    def test_process_another_video_in_same_session(self):
        second = self.root / "another clip.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(self.video),
                        "-t", "1.5", "-an", str(second)], check=True)
        first_out = self.root / "first-explicit.mp4"

        async def ready(pilot, app):
            for _ in range(200):
                if not app.busy and app.playing:
                    return
                await pilot.pause(.03)
            self.fail(str(app.S.get("error", "generation timed out")))

        async def check():
            app = FoleyApp(video=str(self.video), seed=31, out=str(first_out))
            with patch("foley_tui.VideoPlayer", side_effect=lambda path: Player()):
                async with app.run_test(size=(100, 36)) as pilot:
                    await ready(pilot, app)
                    frames = app.S["frames"]
                    old_bytes = first_out.read_bytes()
                    self.assertTrue(app.query_one(Input).display)
                    self.assertFalse(app.query_one(Input).disabled)
                    await pilot.click("#new")
                    self.assertFalse(app.playing)
                    self.assertIs(app.focused, app.query_one(Input))
                    self.assertTrue(app.query_one("#vary", Button).disabled)
                    app.action_variation()
                    self.assertFalse(app.busy)
                    app.query_one(Input).value = second.as_uri()
                    await pilot.click("#make")
                    await ready(pilot, app)
                    self.assertEqual(app.S["video"], str(second))
                    self.assertIsNot(app.S["frames"], frames)
                    self.assertAlmostEqual(app.query_one(Score).duration, 1.5, delta=.04)
                    self.assertNotEqual(app.S["out"], str(first_out))
                    self.assertEqual(first_out.read_bytes(), old_bytes)
                    self.assertTrue(Path(app.S["out"]).is_file())
                    await pilot.press("n")
                    self.assertFalse(app.playing)
                    self.assertEqual(app.query_one(Input).value, "")
                    await pilot.press("escape")
        asyncio.run(check())

    def test_video_argument_keeps_playback_keys_active(self):
        async def check():
            app = FoleyApp(video=str(self.video), seed=17, out=str(self.root / "argument.mp4"))
            with patch("foley_tui.VideoPlayer", side_effect=lambda path: Player()):
                async with app.run_test(size=(80, 24)) as pilot:
                    for _ in range(200):
                        if not app.busy and app.playing:
                            break
                        await pilot.pause(.03)
                    self.assertTrue(app.playing, app.S.get("error"))
                    self.assertIsNone(app.focused)
                    await pilot.press("space")
                    self.assertIsNotNone(app.paused)
                    await pilot.press("escape")
        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
