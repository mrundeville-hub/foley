"""One native mpv window plays the rendered video and sound on the same media clock."""
import json
import queue
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path


class VideoPlayer:
    def __init__(self, path, extra_args=()):
        executable = shutil.which("mpv")
        if executable is None:
            raise OSError("mpv is required for synchronized preview (brew install mpv)")
        self.state = dict(position=0., paused=False, ready=False, eof=False, error=None)
        self.commands = queue.Queue()
        self.stopped = threading.Event()
        self.tmp = tempfile.TemporaryDirectory(prefix="foley-preview-", dir="/tmp")
        self.address = str(Path(self.tmp.name) / "ipc")
        self.serial = 0
        command = [executable, "--no-config", "--no-terminal", "--force-window=yes", "--no-osc",
                   "--osd-level=0", "--autofit=960x720", "--keep-open=yes", "--pause=yes",
                   f"--input-ipc-server={self.address}", f"--title=foley / {Path(path).name}",
                   *extra_args, "--", str(path)]
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            self.tmp.cleanup()
            raise
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()

    def request(self, command):
        self.serial += 1
        self.connection.sendall((json.dumps(dict(command=command, request_id=self.serial)) + "\n").encode())
        while not self.stopped.is_set():
            line = self.reader.readline()
            if not line:
                raise OSError("video player disconnected")
            response = json.loads(line)
            # mpv sends property/file events on the same socket as command responses.
            if response.get("request_id") == self.serial:
                return response.get("data") if response.get("error") == "success" else None

    def monitor(self):
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.connection.settimeout(2)
        try:
            deadline = time.monotonic() + 12
            while not self.stopped.is_set():
                try:
                    self.connection.connect(self.address)
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if self.process.poll() is not None:
                        raise OSError("video player could not open the preview")
                    if time.monotonic() >= deadline:
                        raise OSError("video player did not start")
                    self.stopped.wait(.04)
            if self.stopped.is_set():
                return
            self.reader = self.connection.makefile("rb")
            started = False
            while not self.stopped.is_set() and self.process.poll() is None:
                while not self.commands.empty():
                    self.request(self.commands.get_nowait())
                position = self.request(["get_property", "time-pos"])
                paused = self.request(["get_property", "pause"])
                eof = self.request(["get_property", "eof-reached"])
                if position is not None and not started:
                    self.request(["set_property", "pause", False])
                    paused, started = False, True
                self.state = dict(position=float(position or 0), paused=bool(paused),
                                  ready=started, eof=bool(eof), error=None)
                if not started and time.monotonic() >= deadline:
                    raise OSError("video player could not load the clip")
                self.stopped.wait(1 / 30)
        except (OSError, ValueError) as e:
            if not self.stopped.is_set():
                self.state = {**self.state, "error": str(e)}
        finally:
            if hasattr(self, "reader"):
                self.reader.close()
            self.connection.close()

    def toggle_pause(self):
        self.commands.put(["cycle", "pause"])

    def poll(self):
        return self.process.poll()

    def stop(self):
        self.stopped.set()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.thread.join(timeout=2)
        self.tmp.cleanup()
