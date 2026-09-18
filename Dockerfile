FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,video,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg libgl1 libegl1 libglib2.0-0 libglvnd0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-web.txt ./
RUN python3 -m pip install --no-cache-dir \
    -r requirements.txt -r requirements-web.txt

COPY demon_eye.py glsl_smoke.py web_app.py ./

EXPOSE 7860
CMD ["python3", "web_app.py"]
