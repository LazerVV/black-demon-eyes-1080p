# demon-eye

A fast 1080p black-eye + living-plasma effect for short creator clips.

The current look is deliberately tuned for **1920x1080** footage. Animation is time-based, not frame-based, so `--speed 1` has the same timing at normal constant frame rates such as 24/25/30/50/60 fps.

Built iteratively with OpenAI ChatGPT assisting the implementation.

## Install

```bash
pip install -r requirements.txt
```

## CLI

Input and output are mandatory:

```bash
python demon_eye.py input.mp4 output.mp4
```

Useful controls:

```text
--smoke 0.82        effect strength
--speed 1.0         animation speed
--size 1.0          spread around the eyes
--shader-blur 1.0   softness; 2.0 = the older stronger blur
--transition SEC     ramp to a final size/speed
--size-final N       required when transition > 0
--speed-final N      optional final animation speed
```

Example:

```bash
python demon_eye.py input.mp4 output.mp4 --size 1 --speed 1 --transition 1.5 --size-final 4 --speed-final 2 --shader-blur 1
```

The plasma renderer is OpenGL/EGL GPU rendering. Larger `--size` values also increase black-root density and plasma intensity instead of merely spreading the effect thinner. Transition speed is integrated over time, so changing toward `--speed-final` does not introduce an animation phase jump. NVENC is used automatically when ffmpeg can initialize it; otherwise encoding falls back to x264.

## Windows one-click build

`windows_app.py` is the creator-facing wrapper. Drop one or several videos onto `DemonEye.exe` / its shortcut, **or drop them directly into the open window**. Dropping starts processing immediately with the remembered settings. Batch clips are written beside each source as `*_demon.mp4`.

The GUI remembers Strength / Speed / Size / Blur settings, shows processing status, can open the output folder, and writes `DemonEye-error.txt` next to the EXE if processing fails. The Windows control defaults are the same as the CLI defaults: Strength `0.82` = `--smoke 0.82`, Speed `1.0`, Size `1.0`, Blur `1.0`, Transition `0`.

For Resolve-style use, **Watch folder** watches an export directory. Existing files are ignored; when a new video finishes writing, DemonEye automatically renders `*_demon.mp4` beside it.

The Windows creator app is intentionally strict about **1920x1080** input so the tuned effect scale cannot silently drift.

GitHub Actions builds two Windows variants: a single `DemonEye.exe` and a `DemonEye-Portable.zip` folder build. The portable build starts faster and is less likely to upset antivirus software; neither requires Python to be installed. The portable ZIP also contains a helper that creates Desktop and Windows **Send to** shortcuts.

## Private Docker / Gradio

Requires Docker + NVIDIA Container Toolkit on the Linux host.

The container does **not** install CUDA or CuPy. The active effect is OpenGL/EGL, so NVIDIA Container Toolkit injects the host driver libraries needed for `graphics` and `video`. This avoids CUDA toolkit / CuPy version mismatch entirely.

```bash
docker build -t demon-eye .
docker run --rm --gpus all -p 7860:7860 \
  -e DEMON_EYE_USER=creator \
  -e DEMON_EYE_PASSWORD='change-this' \
  demon-eye
```

Open port 7860 through your private network/reverse proxy. The Gradio page requires the configured username/password. If exposing it beyond a trusted network, put HTTPS in front of it.

## License

GPL-3.0-or-later. See `LICENSE`.
