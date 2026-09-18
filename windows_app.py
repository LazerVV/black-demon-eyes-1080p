from __future__ import annotations

import json
import os
import platform
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:
    DND_FILES = None
    TkinterDnD = None

import cv2

from demon_eye import Config, process_video


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
VIDEO_TYPES = [
    ("Video files", "*.mp4 *.mov *.mkv *.avi *.webm"),
    ("All files", "*.*"),
]


def default_output(path: str) -> str:
    p = Path(path)
    return str(p.with_name(f"{p.stem}_demon.mp4"))


def settings_path() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home()))
    return base / "DemonEye" / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(data: dict) -> None:
    try:
        path = settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def write_error_log(error: BaseException, context: str = "") -> Path:
    base = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    path = base / "DemonEye-error.txt"
    try:
        header = [
            "Demon Eye error report",
            f"Platform: {platform.platform()}",
            f"Python: {sys.version}",
            f"Executable: {sys.executable}",
            f"Frozen: {bool(getattr(sys, 'frozen', False))}",
            f"CWD: {os.getcwd()}",
            f"Args: {sys.argv!r}",
        ]
        if context:
            header.append(f"Context: {context}")
        header.append("")
        path.write_text(
            "\n".join(header)
            + "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            encoding="utf-8",
        )
    except Exception:
        pass
    return path


def valid_video(path: str | Path) -> bool:
    p = Path(path)
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS


def is_1080p(path: str | Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width == 1920 and height == 1080
    finally:
        cap.release()


class DemonEyeApp:
    def __init__(self, root: tk.Tk, initial_paths: list[str] | None = None):
        self.root = root
        self.busy = False
        self.paths: list[str] = []
        self.last_output_dir: Path | None = None
        self.watch_folder: Path | None = None
        self.watch_seen: set[str] = set()
        self.watch_sizes: dict[str, tuple[int, int]] = {}

        root.title("Demon Eye")
        root.resizable(False, False)

        saved = load_settings()
        self.input_var = tk.StringVar(value="")
        self.output_var = tk.StringVar(value="")
        self.smoke_var = tk.DoubleVar(value=float(saved.get("strength", 0.82)))
        self.speed_var = tk.DoubleVar(value=float(saved.get("speed", 1.0)))
        self.size_var = tk.DoubleVar(value=float(saved.get("size", 1.0)))
        self.blur_var = tk.DoubleVar(value=float(saved.get("blur", 1.0)))
        self.transition_var = tk.DoubleVar(value=float(saved.get("transition", 0.0)))
        self.size_final_var = tk.StringVar(value=str(saved.get("size_final", "4.0")))
        self.speed_final_var = tk.StringVar(value=str(saved.get("speed_final", "")))
        self.status_var = tk.StringVar(value="Drop video clips here or choose them.")

        frame = ttk.Frame(root, padding=12)
        frame.grid()

        self.drop_box = ttk.Label(
            frame,
            text="DROP VIDEO CLIPS HERE",
            anchor="center",
            relief="groove",
            padding=(12, 20),
        )
        self.drop_box.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))

        ttk.Label(frame, text="Input").grid(row=1, column=0, sticky="w")
        self.input_entry = ttk.Entry(frame, textvariable=self.input_var, width=58)
        self.input_entry.grid(row=1, column=1, padx=6)
        ttk.Button(frame, text="Choose", command=self.choose_input).grid(row=1, column=2)

        ttk.Label(frame, text="Output").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.output_entry = ttk.Entry(frame, textvariable=self.output_var, width=58)
        self.output_entry.grid(row=2, column=1, padx=6, pady=(6, 0))
        self.save_button = ttk.Button(frame, text="Save", command=self.choose_output)
        self.save_button.grid(row=2, column=2, pady=(6, 0))

        controls = ttk.Frame(frame)
        controls.grid(row=3, column=0, columnspan=3, pady=12)

        self._number(controls, "Strength", self.smoke_var, 0)
        self._number(controls, "Speed", self.speed_var, 1)
        self._number(controls, "Size", self.size_var, 2)
        self._number(controls, "Blur", self.blur_var, 3)
        self._number(controls, "Transition s", self.transition_var, 0, row=1)
        self._number(controls, "Final size", self.size_final_var, 1, row=1)
        self._number(controls, "Final speed", self.speed_final_var, 2, row=1)

        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, columnspan=3, sticky="ew")

        self.process_button = ttk.Button(buttons, text="Process", command=self.start)
        self.process_button.pack(side="left", fill="x", expand=True)

        self.folder_button = ttk.Button(
            buttons, text="Open folder", command=self.open_output_folder
        )
        self.folder_button.pack(side="left", padx=(8, 0))
        self.folder_button.state(["disabled"])

        self.watch_button = ttk.Button(
            buttons, text="Watch folder", command=self.toggle_watch_folder
        )
        self.watch_button.pack(side="left", padx=(8, 0))

        self.help_button = ttk.Button(
            buttons, text="Help", command=self.show_help
        )
        self.help_button.pack(side="left", padx=(8, 0))

        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        ttk.Label(frame, textvariable=self.status_var).grid(
            row=6, column=0, columnspan=3, pady=(8, 0)
        )

        self._enable_drop(frame)

        if initial_paths:
            self.set_inputs(initial_paths)

        self.root.after(1500, self._poll_watch_folder)

    @staticmethod
    def _number(parent, label, variable, col, row=0):
        ttk.Label(parent, text=label).grid(
            row=row, column=col * 2, padx=(0, 3), pady=(4 if row else 0, 0)
        )
        ttk.Entry(parent, textvariable=variable, width=6).grid(
            row=row, column=col * 2 + 1, padx=(0, 10), pady=(4 if row else 0, 0)
        )

    def _enable_drop(self, frame):
        if DND_FILES is None:
            return
        for widget in (self.root, frame, self.drop_box):
            try:
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self.on_drop)
            except Exception:
                pass

    def on_drop(self, event):
        if self.busy:
            return

        try:
            raw_paths = list(self.root.tk.splitlist(event.data))
        except Exception:
            raw_paths = [str(event.data)]

        paths = []
        for raw in raw_paths:
            path = Path(raw.strip("{}")).expanduser()
            if valid_video(path):
                paths.append(str(path.resolve()))

        if not paths:
            messagebox.showerror("Demon Eye", "Drop video files.")
            return

        self.set_inputs(paths)
        self.root.after(80, self.start)

    def set_inputs(self, paths: list[str]):
        clean = [str(Path(p).resolve()) for p in paths if valid_video(p)]
        if not clean:
            return

        self.paths = clean
        if len(clean) == 1:
            src = clean[0]
            self.input_var.set(src)
            self.output_var.set(default_output(src))
            self.output_entry.state(["!disabled"])
            self.save_button.state(["!disabled"])
            self.status_var.set(Path(src).name)
        else:
            self.input_var.set(f"{len(clean)} clips queued")
            self.output_var.set("next to each source as *_demon.mp4")
            self.output_entry.state(["disabled"])
            self.save_button.state(["disabled"])
            self.status_var.set(f"{len(clean)} clips ready")

    def choose_input(self):
        paths = filedialog.askopenfilenames(filetypes=VIDEO_TYPES)
        if paths:
            self.set_inputs(list(paths))

    def choose_output(self):
        if len(self.paths) != 1:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4")],
            initialfile=Path(self.output_var.get()).name
            if self.output_var.get()
            else "demon.mp4",
        )
        if path:
            self.output_var.set(path)

    def _params(self):
        transition = max(0.0, float(self.transition_var.get()))
        size_final_text = self.size_final_var.get().strip()
        speed_final_text = self.speed_final_var.get().strip()

        if transition > 0.0 and not size_final_text:
            raise ValueError("Final size is required when Transition is greater than 0.")

        size_final = (
            max(0.25, float(size_final_text))
            if size_final_text
            else None
        )
        speed_final = (
            max(0.0, float(speed_final_text))
            if speed_final_text
            else None
        )

        return (
            max(0.0, min(1.0, float(self.smoke_var.get()))),
            max(0.0, float(self.speed_var.get())),
            max(0.25, float(self.size_var.get())),
            max(0.0, float(self.blur_var.get())),
            transition,
            size_final,
            speed_final,
        )

    def start(self):
        if self.busy:
            return

        if not self.paths:
            typed = self.input_var.get().strip()
            if valid_video(typed):
                self.paths = [str(Path(typed).resolve())]

        if not self.paths:
            messagebox.showerror("Demon Eye", "Choose at least one video.")
            return

        try:
            params = self._params()
        except (TypeError, ValueError) as exc:
            messagebox.showerror("Demon Eye", str(exc) or "Invalid effect settings.")
            return

        jobs = []
        if len(self.paths) == 1:
            src = self.paths[0]
            dst = self.output_var.get().strip() or default_output(src)
            jobs.append((src, dst))
        else:
            jobs = [(src, default_output(src)) for src in self.paths]

        for src, dst in jobs:
            if os.path.abspath(src) == os.path.abspath(dst):
                messagebox.showerror("Demon Eye", "Output cannot overwrite the input.")
                return
            if not is_1080p(src):
                messagebox.showerror(
                    "Demon Eye",
                    f"{Path(src).name} is not 1920x1080.\n\n"
                    "This build is intentionally tuned for 1080p clips only.",
                )
                return

        save_settings({
            "strength": params[0],
            "speed": params[1],
            "size": params[2],
            "blur": params[3],
            "transition": params[4],
            "size_final": "" if params[5] is None else params[5],
            "speed_final": "" if params[6] is None else params[6],
        })

        self.busy = True
        self.process_button.state(["disabled"])
        self.progress.start(12)
        self.status_var.set(
            "Processing..." if len(jobs) == 1 else f"Processing 1/{len(jobs)}..."
        )
        threading.Thread(
            target=self._run_jobs, args=(jobs, params), daemon=True
        ).start()

    def _run_jobs(self, jobs, params):
        try:
            (
                strength,
                speed,
                size,
                blur,
                transition,
                size_final,
                speed_final,
            ) = params
            cfg = Config(
                smoke=strength,
                shader_speed=speed,
                shader_size=size,
                shader_blur=blur,
                transition=transition,
                shader_size_final=size_final,
                shader_speed_final=speed_final,
            )

            outputs = []
            total = len(jobs)
            for index, (src, dst) in enumerate(jobs, start=1):
                if total > 1:
                    self.root.after(
                        0,
                        lambda i=index, n=total, name=Path(src).name:
                            self.status_var.set(f"Processing {i}/{n}: {name}")
                    )
                process_video(src, dst, cfg, "auto")
                outputs.append(dst)

        except Exception as exc:
            current = ""
            try:
                current = f"input={src!r} output={dst!r} params={params!r}"
            except Exception:
                pass
            log = write_error_log(exc, current)
            self.root.after(0, lambda: self._failed(str(exc), log))
            return

        self.root.after(0, lambda: self._done(outputs))

    def _stop_busy(self):
        self.busy = False
        self.progress.stop()
        self.process_button.state(["!disabled"])

    def _failed(self, error: str, log: Path):
        self._stop_busy()
        self.status_var.set("Failed.")
        messagebox.showerror("Demon Eye", f"{error}\n\nDetails: {log}")

    def _done(self, outputs: list[str]):
        self._stop_busy()
        if outputs:
            self.last_output_dir = Path(outputs[-1]).resolve().parent
            self.folder_button.state(["!disabled"])

        if len(outputs) == 1:
            self.status_var.set(f"Done: {Path(outputs[0]).name}")
            messagebox.showinfo("Demon Eye", f"Done.\n\n{outputs[0]}")
        else:
            self.status_var.set(f"Done: {len(outputs)} clips")
            messagebox.showinfo(
                "Demon Eye",
                f"Done. Processed {len(outputs)} clips next to their source files.",
            )

    def show_help(self):
        messagebox.showinfo(
            "Demon Eye - Controls",
            "Defaults match the Linux CLI.\n\n"
            "Strength 0.82\n"
            "Overall effect intensity. Same as Linux --smoke 0.82.\n\n"
            "Speed 1.0\n"
            "Shader animation speed. 2.0 = twice as fast.\n\n"
            "Size 1.0\n"
            "Effect scale/spread. Bigger values also increase black-root and plasma intensity.\n\n"
            "Blur 1.0\n"
            "Shader softness. 0 = none, 1 = default, 2 = older stronger blur.\n\n"
            "Transition 0\n"
            "Seconds used to grow from the initial Size/Speed to the final values. 0 disables it.\n\n"
            "Final size\n"
            "Required when Transition > 0. Example: Size 1 -> Final size 4 over 1.5 seconds.\n\n"
            "Final speed\n"
            "Optional. Leave blank to keep the initial Speed during the transition.\n\n"
            "All creator builds are tuned for 1920x1080 video."
        )

    def open_output_folder(self):
        if self.last_output_dir is None:
            return
        try:
            os.startfile(str(self.last_output_dir))
        except Exception:
            pass

    def toggle_watch_folder(self):
        if self.watch_folder is not None:
            self.watch_folder = None
            self.watch_seen.clear()
            self.watch_sizes.clear()
            self.watch_button.configure(text="Watch folder")
            self.status_var.set("Watch folder stopped.")
            return

        folder = filedialog.askdirectory(title="Watch export folder")
        if not folder:
            return

        self.watch_folder = Path(folder).resolve()
        self.watch_seen = {
            str(p.resolve())
            for p in self.watch_folder.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        }
        self.watch_sizes.clear()
        self.watch_button.configure(text="Stop watch")
        self.status_var.set(f"Watching: {self.watch_folder}")

    def _poll_watch_folder(self):
        try:
            folder = self.watch_folder
            if folder is not None and folder.is_dir():
                for path in folder.iterdir():
                    if (
                        not path.is_file()
                        or path.suffix.lower() not in VIDEO_EXTS
                        or path.stem.lower().endswith("_demon")
                    ):
                        continue

                    key = str(path.resolve())
                    if key in self.watch_seen:
                        continue

                    try:
                        size = path.stat().st_size
                    except OSError:
                        continue

                    old_size, stable = self.watch_sizes.get(key, (-1, 0))
                    stable = stable + 1 if size == old_size and size > 0 else 0
                    self.watch_sizes[key] = (size, stable)

                    # Two stable polls means Resolve/ffmpeg has almost certainly
                    # finished writing the clip. Process one at a time.
                    if stable >= 2 and not self.busy:
                        self.watch_seen.add(key)
                        self.watch_sizes.pop(key, None)
                        self.set_inputs([key])
                        self.status_var.set(f"Auto-processing: {path.name}")
                        self.root.after(50, self.start)
                        break
        except Exception:
            pass
        finally:
            self.root.after(1500, self._poll_watch_folder)


def make_root():
    if TkinterDnD is not None:
        try:
            return TkinterDnD.Tk()
        except Exception:
            pass
    return tk.Tk()


def command_line_videos() -> list[str]:
    paths = []
    for raw in sys.argv[1:]:
        if valid_video(raw):
            paths.append(str(Path(raw).resolve()))
    return paths


def main() -> int:
    initial_paths = command_line_videos()

    root = make_root()
    app = DemonEyeApp(root, initial_paths)

    # Dropping one or more clips onto DemonEye.exe/shortcut is zero-decision:
    # process all of them with remembered settings and write beside each source.
    if initial_paths:
        root.after(250, app.start)

    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
