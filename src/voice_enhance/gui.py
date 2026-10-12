"""voice-enhance --gui: the same pipeline behind a page in your browser.

A small local web server (127.0.0.1 only) that keeps the models loaded and
reuses the slow steps (AI cleanup, rebuild) between runs, so moving a gate or
blend slider re-renders in seconds. The page plays the original and the
enhanced version in sync and switches between them at the same position.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from . import __version__
from . import cli
from . import progress as prog
from . import aidereverb, sidon, stupase
from .audio import ENCODERS, SR, load, save_wav
from .backends import clearvoice_available
from .guard import GuardSettings

PAGE = Path(__file__).with_name("gui.html")
DIALOGS = Path(__file__).with_name("dialogs.js")

# settings key (argparse dest) -> flag; only values that differ from the CLI defaults are passed
FLAGS = {"backend": "-b", "max_reduction": "--max-reduction", "protect": "--protect", "blend": "--blend",
         "gate": "--gate", "laughter_guard": "--laughter-guard", "ai_dereverb": "--ai-dereverb", "lufs": "--lufs",
         "format": "-f", "stupase_crossover": "--stupase-crossover"}
OFF_SWITCHES: dict[str, str] = {}  # flag given when False
ON_SWITCHES = {"declip": "--declip", "rebuild": "--rebuild"}  # flag given when True
STUDIO = {"sidon": "--studio", "stupase": "--stupase"}  # studio model -> flag (None = off)
DEFAULTS = {k: v for k, v in vars(cli.parser().parse_args([])).items()
            if k in FLAGS or k in OFF_SWITCHES or k in ON_SWITCHES or k == "studio"}


def to_argv(settings: dict) -> list[str]:
    argv: list[str] = []
    for key, flag in FLAGS.items():
        v = settings.get(key, DEFAULTS[key])
        if v is not None and v != DEFAULTS[key] and (key != "stupase_crossover" or settings.get("studio") == "stupase"):
            argv += [flag, f"{v:g}" if isinstance(v, float) else str(v)]
    argv += [f for k, f in OFF_SWITCHES.items() if not settings.get(k, True)]
    argv += [f for k, f in ON_SWITCHES.items() if settings.get(k, False)]
    if settings.get("studio") in STUDIO:
        argv.append(STUDIO[settings["studio"]])
    return argv


# short names for the settings in saved file names
LABELS = {"-b": "noise", "--max-reduction": "maxred", "--laughter-guard": "laugh", "--ai-dereverb": "aidereverb",
          "--stupase-crossover": "xover", "--studio": "sidon"}


def settings_label(settings: dict) -> str:
    """The settings that differ from the defaults, e.g. "noise-none xover4000 stupase", for file names."""
    argv = to_argv(settings)
    if "-f" in argv:  # the format is already the extension
        i = argv.index("-f")
        del argv[i:i + 2]
    words: list[str] = []
    for a in argv:
        if a.startswith("-") and not a.lstrip("-.").replace(".", "").isdigit():
            words.append(LABELS.get(a, a.lstrip("-")))
        else:
            words[-1] += a if a.lstrip("-.").replace(".", "").isdigit() else f"-{a}"
    return " ".join(words) or "defaults"


def job_key(job: tuple[int, dict]) -> tuple:
    """What a render depends on: the output format only matters when saving.
    No settings means just load the file."""
    fid, settings = job
    return fid, None if settings is None else sorted((k, v) for k, v in settings.items() if k != "format")


class NewFile(Exception):
    """A different file was opened while this one was processing."""


class Session:
    def __init__(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="voice-enhance-gui-"))
        self.cond = threading.Condition()
        self.wanted: tuple[int, dict | None] | None = None  # (file id, settings) to render next; None = just load it
        self.running: tuple | None = None  # job_key of the render in progress
        self.done: tuple | None = None  # job_key of the last finished render
        self.files: dict[int, tuple[Path, Path | None]] = {}  # id -> (source, folder to save next to, None if uploaded)
        self.next_id = 1
        self.loaded: int | None = None
        self.rendered: int | None = None  # file id the latest render is of
        self.rendered_settings: dict = {}  # and the settings it used
        self.audio = None
        self.ref_lufs = float("nan")
        self.cache: dict = {}
        self.renders: list[Path] = []
        self.dialogs = Dialogs()
        self.status = {"file": None, "file_id": None, "state": "empty", "stage": "", "pct": 0.0, "note": "",
                       "result": "", "error": "", "version": 0, "command": "", "duration": 0.0,
                       "queued": 0, "finished": 0}  # job numbers, so the page knows when its change is done
        threading.Thread(target=self._worker, daemon=True).start()

    # --- requests from the page ---------------------------------------------

    def open(self, path: Path, folder: Path | None, settings: dict | None = None) -> None:
        """Loads the file so the original can be played; it's processed when the page asks."""
        with self.cond:
            fid = self.next_id
            self.next_id += 1
            self.files[fid] = (path, folder)
            self.status.update(file=path.name, file_id=None, state="working", stage="reading", pct=0.0,
                               error="", note="", result="", queued=self.status["queued"] + 1,
                               command=self._command(fid, settings or dict(DEFAULTS)))
            self.wanted = (fid, None)
            self.cond.notify()

    def apply(self, settings: dict, process: bool = True) -> int | None:
        """Queues a render if the settings change the sound; returns its job number.
        Without `process`, only updates the command line shown on the page."""
        with self.cond:
            if self.wanted is None:
                return None
            self.status.update(command=self._command(self.wanted[0], settings))
            if not process:
                return None
            self.wanted = (self.wanted[0], settings)
            if job_key(self.wanted) == (self.running or self.done):
                return None
            self.status.update(state="working", stage="queued", pct=0.0, error="", queued=self.status["queued"] + 1)
            self.cond.notify()
            return self.status["queued"]

    def _command(self, fid: int, settings: dict) -> str:
        src = self.files[fid][0]
        name = src.name if self.files[fid][1] is None else str(src)
        return " ".join(["voice-enhance", shlex.quote(name), *map(shlex.quote, to_argv(settings))])

    # --- the worker ---------------------------------------------------------

    def _worker(self) -> None:
        while True:
            with self.cond:
                while self.wanted is None or job_key(self.wanted) == self.done:
                    self.cond.wait()
                job = (self.wanted[0], self.wanted[1] and dict(self.wanted[1]))
                n = self.status["queued"]
                self.running = job_key(job)
            try:
                if job[1] is None:
                    self._open(job[0])
                else:
                    self._run(*job)
            except NewFile:
                continue
            except BaseException as e:  # SystemExit from bad settings included
                if not isinstance(e, SystemExit):
                    traceback.print_exc()
                msg = str(e) if str(e) not in ("", "2") else "Invalid setting"
                self.status.update(state="error", error=msg)
            with self.cond:
                self.done, self.running = job_key(job), None
                self.status["finished"] = n

    def _progress(self, fid: int):
        def on_move(label: str, frac: float) -> None:
            if self.wanted and self.wanted[0] != fid:
                raise NewFile()
            self.status.update(stage=label, pct=frac)
        return on_move

    def _load(self, fid: int) -> None:
        prog.stage("decode")
        self.loaded, self.audio, self.cache = None, load(self.files[fid][0]), {}
        self.ref_lufs = cli.source_lufs(self.audio)
        save_wav(self.audio, self.tmp / f"original_{fid}.wav")
        self.loaded = fid
        self.status.update(file_id=fid, duration=self.audio.size / SR)

    def _open(self, fid: int) -> None:
        self.status.update(state="working", error="")
        prog.listener = self._progress(fid)
        try:
            with prog.progress(cli.short(self.files[fid][0].stem), ["decode"]):
                self._load(fid)
        finally:
            prog.listener = None
        self.status.update(state="loaded", stage="", pct=0.0, note="", result="")

    def _run(self, fid: int, settings: dict) -> None:
        argv = to_argv(settings)
        args = cli.parser().parse_args(argv)
        cli.configure(args)
        rebuild_hf = cli.rebuild_wanted(args) and clearvoice_available()
        guard = GuardSettings(strength=args.laughter_guard)
        src, _ = self.files[fid]
        new_file = self.loaded != fid
        self.status.update(state="working", error="", command=self._command(fid, settings))
        prog.listener = self._progress(fid)
        try:
            with prog.progress(cli.short(src.stem), cli.stages_for(args, args.backend, guard, rebuild_hf, decode=new_file),
                               backend=args.backend):
                if new_file:
                    self._load(fid)
                y, note = cli.process(self.audio, args.backend, mix=1 - args.blend, guard=guard, cache=self.cache,
                                      rebuild_hf=rebuild_hf, ai_dereverb=args.ai_dereverb, repair_clipping=args.declip,
                                      protect_db=args.protect, rebuild_above=args.rebuild_above, gate_db=args.gate)
                version = self.status["version"] + 1
                dst = self.tmp / f"enhanced_{version}.wav"
                result = cli.render(y, dst, args, self.ref_lufs)
        finally:
            prog.listener = None
        self.renders.append(dst)
        self.rendered, self.rendered_settings = fid, settings
        while len(self.renders) > 3:  # keep the one playing and the one being saved
            self.renders.pop(0).unlink(missing_ok=True)
        self.status.update(state="ready", stage="done", pct=1.0, note=note, version=version,
                           result=result.split("  ", 1)[-1])

    # --- saving -------------------------------------------------------------

    def save(self, fmt: str) -> dict:
        version, fid = self.status["version"], self.status["file_id"]
        if not version or fid is None or self.rendered != fid:
            return {"error": "Nothing processed yet"}
        ext = "." + fmt.lower().lstrip(".")
        if ext not in ENCODERS:
            return {"error": f"Unsupported format {fmt}"}
        src, folder = self.files[fid]
        where = folder or Path.home() / "Desktop"
        dst = self.dialogs.choose_save(f"{src.stem}_enhanced [{settings_label(self.rendered_settings)}]{ext}", where)
        if dst is None:
            return {"cancelled": True}
        if dst.suffix.lower() not in ENCODERS:
            dst = dst.with_name(dst.name + ext)
        rendered = self.tmp / f"enhanced_{version}.wav"
        if dst.suffix.lower() == ".wav":
            shutil.copyfile(rendered, dst)
        else:
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(rendered),
                            *ENCODERS[dst.suffix.lower()], str(dst)], check=True)
        return {"path": str(dst)}


# --- macOS dialogs ------------------------------------------------------------

class Dialogs:
    """Native Open/Save panels from a helper process started with the GUI.

    Running osascript per click took ~4 s before the panel appeared (it has to
    launch as a GUI app every time); the helper is already up, so it's ~0.1 s.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self._start()

    def _start(self) -> None:
        self.proc = subprocess.Popen(["osascript", "-l", "JavaScript", str(DIALOGS)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.proc.stdout.readline()  # {"ready": true}

    def ask(self, **request) -> Path | None:
        """None if cancelled."""
        with self.lock:
            for _ in range(2):  # restart the helper once if it died
                try:
                    if self.proc.poll() is not None:
                        self._start()
                    self.proc.stdin.write(json.dumps(request) + "\n")
                    self.proc.stdin.flush()
                    line = self.proc.stdout.readline()
                    if line:
                        path = json.loads(line).get("path")
                        return Path(path) if path else None
                except (OSError, ValueError):
                    pass
                self.proc.kill()
                self._start()
            return None

    def choose_open(self) -> Path | None:
        return self.ask(prompt="Choose a recording to clean up")

    def choose_save(self, name: str, folder: Path) -> Path | None:
        return self.ask(save=True, prompt="Save the enhanced audio", name=name, folder=str(folder))

    def close(self) -> None:
        if self.proc:
            self.proc.kill()


# --- HTTP -------------------------------------------------------------------

def handler(session: Session, port: int, initial: Path | None):
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _file(self, path: Path) -> None:
            if not path.exists():
                return self._json({"error": "not found"}, 404)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as f:
                shutil.copyfileobj(f, self.wfile)

        def _allowed(self, post: bool) -> bool:
            # Only this machine's browser: the Host check stops DNS rebinding, and the
            # custom header forces a CORS preflight (which we never answer) on other sites.
            if self.headers.get("Host") not in hosts or (post and self.headers.get("X-Voice-Enhance") != "1"):
                self._json({"error": "forbidden"}, 403)
                return False
            return True

        def do_GET(self) -> None:
            if not self._allowed(post=False):
                return
            path = self.path.split("?", 1)[0]
            if path == "/":
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/info":
                return self._json({"version": __version__, "defaults": DEFAULTS,
                                   "rebuild_available": clearvoice_available(), "ai_dereverb_available": aidereverb.available(),
                                   "studio_available": {"sidon": sidon.available(), "stupase": stupase.available()}, "formats": [e[1:] for e in ENCODERS],
                                   "initial": bool(initial)})
            if path == "/api/status":
                return self._json(session.status)
            if path.startswith("/audio/original_") or path.startswith("/audio/enhanced_"):
                name = Path(path).name
                return self._file(session.tmp / name)
            self._json({"error": "not found"}, 404)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_POST(self) -> None:
            if not self._allowed(post=True):
                return
            path = self.path.split("?", 1)[0]
            if path == "/api/start":
                if initial and session.wanted is None:
                    session.open(initial, initial.parent, self._body().get("settings"))
                return self._json({"ok": True})
            if path == "/api/choose":
                chosen = session.dialogs.choose_open()
                if chosen:
                    session.open(chosen, chosen.parent, self._body().get("settings"))
                return self._json({"opened": bool(chosen)})
            if path == "/api/upload":
                name = Path(unquote(self.headers.get("X-Filename", "upload.wav"))).name or "upload.wav"
                folder = session.tmp / f"upload_{time.time_ns()}"
                folder.mkdir()
                dst = folder / name
                left = int(self.headers.get("Content-Length") or 0)
                with dst.open("wb") as f:
                    while left > 0:
                        chunk = self.rfile.read(min(left, 1 << 20))
                        if not chunk:
                            break
                        f.write(chunk)
                        left -= len(chunk)
                settings = json.loads(unquote(self.headers.get("X-Settings", "")) or "null")
                session.open(dst, None, settings)
                return self._json({"ok": True})
            if path == "/api/apply":
                body = self._body()
                return self._json({"job": session.apply(body["settings"], body.get("process", True))})
            if path == "/api/save":
                return self._json(session.save(self._body().get("format", "wav")))
            if path == "/api/reveal":
                subprocess.run(["open", "-R", self._body()["path"]])
                return self._json({"ok": True})
            self._json({"error": "not found"}, 404)

    return Handler


def serve(initial: str | None = None) -> None:
    start = Path(initial).expanduser().resolve() if initial else None
    if start and not start.is_file():
        raise SystemExit(f"Not found: {start}")
    session = Session()
    server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = server.server_address[1]
    server.RequestHandlerClass = handler(session, port, start)
    url = f"http://127.0.0.1:{port}/"
    prog.write(f"voice-enhance {__version__} is running at {url}\nLeave this window open; press Ctrl+C to quit.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.dialogs.close()
        shutil.rmtree(session.tmp, ignore_errors=True)
