from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr

from demon_eye import Config, process_video


def render(video, strength, speed, size, blur, transition, size_final, speed_final):
    if not video:
        raise gr.Error("Choose a video.")

    src = str(video)
    work = Path(tempfile.mkdtemp(prefix="demon-eye-web-"))
    out = work / f"{Path(src).stem}_demon.mp4"

    transition = max(0.0, float(transition))
    if transition > 0.0 and size_final is None:
        raise gr.Error("Final size is required when Transition is greater than 0.")

    cfg = Config(
        smoke=max(0.0, min(1.0, float(strength))),
        shader_speed=max(0.0, float(speed)),
        shader_size=max(0.25, float(size)),
        shader_blur=max(0.0, float(blur)),
        transition=transition,
        shader_size_final=(
            max(0.25, float(size_final))
            if size_final is not None
            else None
        ),
        shader_speed_final=(
            max(0.0, float(speed_final))
            if speed_final is not None
            else None
        ),
    )
    process_video(src, str(out), cfg, "auto")
    return str(out)


demo = gr.Interface(
    fn=render,
    inputs=[
        gr.Video(label="Clip"),
        gr.Slider(0.0, 1.0, value=0.82, step=0.01, label="Strength"),
        gr.Slider(0.0, 3.0, value=1.0, step=0.05, label="Speed"),
        gr.Slider(0.25, 3.0, value=1.0, step=0.05, label="Size"),
        gr.Slider(0.0, 3.0, value=1.0, step=0.05, label="Blur"),
        gr.Slider(0.0, 10.0, value=0.0, step=0.1, label="Transition seconds"),
        gr.Number(value=4.0, label="Final size"),
        gr.Number(value=None, label="Final speed (optional)"),
    ],
    outputs=gr.Video(label="Demon clip"),
    title="Demon Eye",
)


if __name__ == "__main__":
    user = os.environ.get("DEMON_EYE_USER", "creator")
    password = os.environ.get("DEMON_EYE_PASSWORD")
    if not password:
        raise RuntimeError("Set DEMON_EYE_PASSWORD before starting the web app.")

    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        auth=(user, password),
        show_error=True,
    )
