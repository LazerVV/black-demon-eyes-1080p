#!/usr/bin/env python3
"""
demon-eye v1

Fast single-face video effect:
- MediaPipe face/iris tracking
- highlight-preserving black sclera
- localized, temporally coherent "living smoke"
- skin-colour gate to reduce painting over fingers/eyelids
- NVENC auto-detection with x264 fallback
- audio remux after processing

This is intentionally optimized for short 1080p creator snippets.
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from glsl_smoke import GLSLSmokeRenderer

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None


# Shared near-black socket tone. Keeping the shader transition on this exact
# color avoids a visible pitch mismatch at the eyelid border.
SOCKET_BGR = np.array([5.0, 5.0, 5.0], dtype=np.float32)
PLASMA_BODY_BGR = np.array([6.0, 6.0, 18.0], dtype=np.float32)
PLASMA_HOT_BGR = np.array([10.0, 10.0, 52.0], dtype=np.float32)


# MediaPipe Face Mesh landmark rings.
# These follow the visible eyelid opening rather than a broad face region.
RIGHT_EYE = np.array(
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    dtype=np.int32,
)
LEFT_EYE = np.array(
    [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    dtype=np.int32,
)

# Ordered lid arcs. We spline these instead of drawing straight landmark-to-
# landmark polygon edges, which removes the "GIMP lasso" geometry.
RIGHT_UPPER = np.array([33, 246, 161, 160, 159, 158, 157, 173, 133], dtype=np.int32)
RIGHT_LOWER = np.array([133, 155, 154, 153, 145, 144, 163, 7, 33], dtype=np.int32)
LEFT_UPPER = np.array([263, 466, 388, 387, 386, 385, 384, 398, 362], dtype=np.int32)
LEFT_LOWER = np.array([362, 382, 381, 380, 374, 373, 390, 249, 263], dtype=np.int32)

# refine_landmarks=True gives 10 extra iris landmarks.
RIGHT_IRIS = np.array([469, 470, 471, 472], dtype=np.int32)
LEFT_IRIS = np.array([474, 475, 476, 477], dtype=np.int32)


@dataclass
class Config:
    blackness: float = 1.0
    smoke: float = 0.82
    shader_speed: float = 1.0
    shader_size: float = 1.0
    shader_blur: float = 1.0
    transition: float = 0.0
    shader_size_final: Optional[float] = None
    shader_speed_final: Optional[float] = None
    feather_px: float = 1.25
    iris_guard: float = 1.08
    hold_frames: int = 0
    skin_gate: bool = False
    tracking_alpha: float = 0.44
    track_width: int = 576
    track_fps: float = 0.0
    debug_mask: bool = False


def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def transition_values(t: float, cfg: Config):
    """Return current size, current speed and integrated shader time."""
    t = max(0.0, float(t))
    duration = max(0.0, float(cfg.transition))
    size0 = max(0.25, float(cfg.shader_size))
    size1 = (
        max(0.25, float(cfg.shader_size_final))
        if cfg.shader_size_final is not None
        else size0
    )
    speed0 = max(0.0, float(cfg.shader_speed))
    speed1 = (
        max(0.0, float(cfg.shader_speed_final))
        if cfg.shader_speed_final is not None
        else speed0
    )

    if duration <= 1e-6:
        return size0, speed0, t * speed0

    if t >= duration:
        # Integral of smoothstep(0..1) is 0.5.
        shader_time = (
            duration * (speed0 + 0.5 * (speed1 - speed0))
            + (t - duration) * speed1
        )
        return size1, speed1, shader_time

    u = t / duration
    eased = u * u * (3.0 - 2.0 * u)
    size = size0 + (size1 - size0) * eased
    speed = speed0 + (speed1 - speed0) * eased

    # Integral of smoothstep(u)=3u^2-2u^3 is u^3-0.5u^4.
    integrated_ease = u * u * u - 0.5 * u * u * u * u
    shader_time = (
        speed0 * t
        + (speed1 - speed0) * duration * integrated_ease
    )
    return size, speed, shader_time


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (x >= edge1).astype(np.float32)
    t = clamp01((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)

_ELLIPSE_KERNEL_CACHE = {}


def ellipse_kernel(size: int) -> np.ndarray:
    size = max(1, int(size)) | 1
    kernel = _ELLIPSE_KERNEL_CACHE.get(size)
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        _ELLIPSE_KERNEL_CACHE[size] = kernel
    return kernel


def ffmpeg_executable() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    if imageio_ffmpeg is not None:
        try:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and os.path.exists(exe):
                return exe
        except Exception:
            pass
    return None


def ffmpeg_exists() -> bool:
    return ffmpeg_executable() is not None


def nvenc_works() -> bool:
    if not ffmpeg_exists():
        return False
    cmd = [
        ffmpeg_executable() or "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=256x256:r=1",
        "-frames:v", "1", "-an",
        "-c:v", "h264_nvenc",
        "-f", "null", "-"
    ]
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except OSError:
        return False


def choose_encoder(requested: str) -> str:
    if requested == "x264":
        return "x264"
    if requested == "nvenc":
        if not nvenc_works():
            raise RuntimeError("NVENC was requested but ffmpeg could not initialize h264_nvenc.")
        return "nvenc"
    return "nvenc" if nvenc_works() else "x264"


def start_encoder(path: str, width: int, height: int, fps: float, encoder: str):
    common = [
        ffmpeg_executable() or "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s:v", f"{width}x{height}",
        "-r", f"{fps:.8f}",
        "-i", "-",
        "-an",
    ]

    if encoder == "nvenc":
        codec = [
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", "18",
            "-b:v", "0",
        ]
    else:
        codec = [
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
        ]

    # pad protects yuv420p encoders from odd source dimensions.
    tail = [
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        path,
    ]

    return subprocess.Popen(common + codec + tail, stdin=subprocess.PIPE)


def remux_audio(temp_video: str, source: str, output: str) -> None:
    cmd = [
        ffmpeg_executable() or "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", temp_video,
        "-i", source,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        output,
    ]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError("ffmpeg failed while remuxing audio.")


class LandmarkTracker:
    def __init__(self, cfg: Config):
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This mediapipe build does not expose mp.solutions. "
                "Install the pinned mediapipe==0.10.21 from requirements.txt."
            )

        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.60,
            min_tracking_confidence=0.65,
        )
        self.cfg = cfg
        self.prev: Optional[np.ndarray] = None
        self.misses = 0

    def close(self):
        self.mesh.close()

    @staticmethod
    def _eye_summary(points: np.ndarray):
        r = np.mean(points[RIGHT_EYE, :2], axis=0)
        l = np.mean(points[LEFT_EYE, :2], axis=0)
        mid = (r + l) * 0.5
        sep = float(np.linalg.norm(l - r))
        return r, l, mid, sep

    def _plausible(self, current: np.ndarray) -> bool:
        if current is None or current.shape[0] < 468 or not np.isfinite(current).all():
            return False

        _, _, eye_mid, eye_sep = self._eye_summary(current)
        if eye_sep < 8.0:
            return False

        # Basic facial vertical ordering. This rejects a lot of the "eyes moved
        # down onto the mouth" hallucinations when the real face is hidden.
        nose_y = float(current[1, 1])
        mouth_y = float((current[13, 1] + current[14, 1]) * 0.5)
        chin_y = float(current[152, 1])

        if not (eye_mid[1] < mouth_y and nose_y < chin_y):
            return False
        if (mouth_y - eye_mid[1]) < 0.18 * eye_sep:
            return False

        # For ASMR clips the head moves smoothly. A one-frame jump by a large
        # fraction of the inter-eye distance is almost always FaceMesh latching
        # onto a hand/mouth during occlusion, not real motion.
        if self.prev is not None:
            prev_r, prev_l, prev_mid, prev_sep = self._eye_summary(self.prev)
            denom = max(prev_sep, 12.0)
            delta = eye_mid - prev_mid
            jump = float(np.linalg.norm(delta)) / denom
            dx = float(delta[0]) / denom
            dy = float(delta[1]) / denom
            scale = eye_sep / denom

            # Occlusion failures tend to yank the inferred eyes downward or
            # rotate the eye pair. Real head motion is usually much more
            # coherent frame-to-frame. Bias heavily toward the last good pose.
            cur_r, cur_l, _, _ = self._eye_summary(current)
            prev_vec = prev_l - prev_r
            cur_vec = cur_l - cur_r
            prev_angle = math.atan2(float(prev_vec[1]), float(prev_vec[0]))
            cur_angle = math.atan2(float(cur_vec[1]), float(cur_vec[0]))
            angle_delta = abs((cur_angle - prev_angle + math.pi) % (2.0 * math.pi) - math.pi)

            if jump > 0.30:
                return False
            if dy > 0.12 and jump > 0.12:
                return False
            if abs(dx) > 0.24 and jump > 0.22:
                return False
            if angle_delta > math.radians(12.0):
                return False
            if scale < 0.68 or scale > 1.45:
                return False

        return True

    def _smooth(self, current: np.ndarray) -> np.ndarray:
        if self.prev is None or self.prev.shape != current.shape:
            self.prev = current.copy()
            return current

        dist = np.linalg.norm(current[:, :2] - self.prev[:, :2], axis=1)
        med = float(np.median(dist))

        # Low alpha kills high-frequency landmark jitter; movement raises alpha
        # automatically so head turns do not lag several frames behind.
        alpha = float(np.clip(self.cfg.tracking_alpha + med / 24.0, 0.42, 0.86))
        smoothed = alpha * current + (1.0 - alpha) * self.prev

        # Eye landmarks get a per-point deadband. Sub-pixel / ~1 px motion is
        # usually estimator noise and should barely move the mask; real blinks
        # and side glances move several pixels and pass through quickly.
        eye_idx = np.unique(np.concatenate([RIGHT_EYE, LEFT_EYE, RIGHT_IRIS, LEFT_IRIS]))
        eye_idx = eye_idx[eye_idx < current.shape[0]]
        if eye_idx.size:
            eye_delta = current[eye_idx, :2] - self.prev[eye_idx, :2]
            eye_motion = np.linalg.norm(eye_delta, axis=1)
            motion_t = smoothstep(2.15, 9.5, eye_motion)
            eye_alpha = 0.025 + 0.80 * motion_t
            eye_alpha = eye_alpha[:, None]
            smoothed[eye_idx] = (
                eye_alpha * current[eye_idx]
                + (1.0 - eye_alpha) * self.prev[eye_idx]
            )

        self.prev = smoothed
        return smoothed

    def detect(self, frame_bgr: np.ndarray, run_inference: bool = True) -> Tuple[Optional[np.ndarray], float]:
        h, w = frame_bgr.shape[:2]

        if not run_inference and self.prev is not None:
            return self.prev.copy(), 1.0

        detect_frame = frame_bgr
        if self.cfg.track_width > 0 and w > self.cfg.track_width:
            scale = self.cfg.track_width / float(w)
            detect_frame = cv2.resize(
                frame_bgr,
                (self.cfg.track_width, max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            points = np.array(
                [[p.x * w, p.y * h, p.z * w] for p in lm],
                dtype=np.float32,
            )

            if self._plausible(points):
                self.misses = 0
                return self._smooth(points), 1.0

            # A detected-but-implausible face is usually an occlusion glitch.
            # Keep the last good geometry stationary while the pixel-based eye
            # visibility logic decides how fast to fade the sockets away.
            self.misses += 1
            if self.prev is not None and self.misses <= 8:
                return self.prev.copy(), 0.25
        else:
            self.misses += 1

        # Do not paint stale eyes over a hand for seconds. A tiny hold can be
        # opted into from the CLI, but the default is zero.
        if self.prev is not None and self.misses <= self.cfg.hold_frames:
            confidence = 1.0 - (self.misses / (self.cfg.hold_frames + 1.0))
            return self.prev.copy(), float(confidence)

        # After a short sustained loss, throw away stale geometry. Otherwise a
        # face that reappears after a hand pass at a new position can be
        # rejected forever as an implausible jump from the pre-occlusion pose.
        if self.misses >= 12:
            self.prev = None

        return None, 0.0

def polygon_mask(shape_hw: Tuple[int, int], pts_xy: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.round(pts_xy).astype(np.int32)
    cv2.fillPoly(mask, [poly], 255, lineType=cv2.LINE_AA)
    return mask.astype(np.float32) / 255.0


def eye_geometry(points: np.ndarray, eye_idx: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    eye = points[eye_idx, :2]
    x0, y0 = np.min(eye, axis=0)
    x1, y1 = np.max(eye, axis=0)
    width = max(4.0, float(x1 - x0))
    center = (float((x0 + x1) * 0.5), float((y0 + y1) * 0.5))
    return eye, width, center


def catmull_rom_chain(ctrl: np.ndarray, samples_per_segment: int = 12) -> np.ndarray:
    """Dense smooth curve through all control points, including endpoints."""
    p = np.asarray(ctrl, dtype=np.float32)
    if len(p) < 2:
        return p.copy()
    ext = np.vstack([p[0], p, p[-1]])
    chunks = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False, dtype=np.float32)[:, None]
        t2 = t * t
        t3 = t2 * t
        q = 0.5 * (
            2.0 * p1
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        chunks.append(q)
    chunks.append(p[-1:])
    return np.vstack(chunks)


def smooth_eye_geometry(
    shape_hw: Tuple[int, int],
    upper: np.ndarray,
    lower: np.ndarray,
    medial: np.ndarray,
    outer: np.ndarray,
    eye_width: float,
    oversample: int = 4,
) -> np.ndarray:
    """
    Smooth eyelid-opening geometry with small anatomical corner carve-outs.

    Medial carve-out preserves the caruncle. The tiny outer carve prevents the
    socket from swallowing lateral lashes/canthus.
    """
    h, w = shape_hw
    s = max(2, int(oversample))
    hh, ww = h * s, w * s
    hi = np.zeros((hh, ww), dtype=np.uint8)

    up = catmull_rom_chain(upper, 14)
    lo = catmull_rom_chain(lower, 14)
    contour = np.vstack([up, lo[1:]])
    contour_hi = np.round(contour * s).astype(np.int32)
    cv2.fillPoly(hi, [contour_hi], 255, lineType=cv2.LINE_AA)

    all_pts = np.vstack([upper, lower])
    eye_height = max(3.0, float(np.max(all_pts[:, 1]) - np.min(all_pts[:, 1])))
    axis = outer - medial
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))

    # Preserve the pink caruncle at the nasal/medial corner.
    medial_center = medial + axis * 0.018
    cv2.ellipse(
        hi,
        tuple(np.round(medial_center * s).astype(int)),
        (
            max(1, int(round(eye_width * 0.075 * s))),
            max(1, int(round(eye_height * 0.30 * s))),
        ),
        angle,
        0,
        360,
        0,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # Smaller outer notch avoids cutting over the lash-heavy lateral corner.
    outer_center = outer - axis * 0.006
    cv2.ellipse(
        hi,
        tuple(np.round(outer_center * s).astype(int)),
        (
            max(1, int(round(eye_width * 0.022 * s))),
            max(1, int(round(eye_height * 0.20 * s))),
        ),
        angle,
        0,
        360,
        0,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    geom = cv2.resize(hi, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    geom = cv2.GaussianBlur(geom, (0, 0), sigmaX=0.62, sigmaY=0.62)
    return clamp01(geom)


def local_focus_sigma(
    frame_roi: np.ndarray,
    eye_width: float,
    gray: Optional[np.ndarray] = None,
) -> float:
    """Stable blur estimate with a soft baseline and limited frame-to-frame volatility."""
    if frame_roi.size == 0:
        return 1.25

    if gray is None:
        gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    # Avoid sudden 1-2 frame jumps to a razor-sharp synthetic edge.
    soft = float(np.clip((110.0 - sharpness) / 100.0, 0.0, 1.0))
    return 1.10 + soft * max(1.4, eye_width * 0.020)

def eye_visibility_factor(
    frame_roi: np.ndarray,
    eye_geom: np.ndarray,
    eye_width: float,
    lab: Optional[np.ndarray] = None,
    gray: Optional[np.ndarray] = None,
) -> float:
    """
    Visibility estimate independent of skin colour.

    A visible eye differs in colour/texture from the nearby eyelid; a closed
    lid or hand passing over it tends to become locally similar.
    """
    inside = eye_geom > 0.50
    if int(np.count_nonzero(inside)) < 12:
        return 0.0

    binary = inside.astype(np.uint8)
    k = max(3, int(round(max(3.0, eye_width * 0.11))) | 1)
    kernel = ellipse_kernel(k)
    ring = cv2.dilate(binary, kernel, iterations=1).astype(bool) & (~inside)
    if int(np.count_nonzero(ring)) < 12:
        return 1.0

    if lab is None:
        lab = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    if gray is None:
        gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    elif gray.dtype != np.float32:
        gray = gray.astype(np.float32)

    eye_lab = np.median(lab[inside], axis=0)
    ring_lab = np.median(lab[ring], axis=0)
    delta = float(np.linalg.norm(
        (eye_lab - ring_lab) / np.array([1.0, 1.35, 1.35], dtype=np.float32)
    ))
    texture = float(np.std(gray[inside]))

    color_score = float(np.clip((delta - 3.0) / 10.0, 0.0, 1.0))
    texture_score = float(np.clip((texture - 4.0) / 13.0, 0.0, 1.0))
    raw = max(color_score, texture_score)
    return float(smoothstep(0.08, 0.72, np.array(raw, dtype=np.float32)))


def base_socket_alpha(geom: np.ndarray, feather_px: float, focus_sigma: float) -> np.ndarray:
    """
    Interior stays opaque; only the physical edge is feathered/defocused.

    This is deliberately different from darkening the source eye: the source
    iris/sclera must not leak through the black socket interior.
    """
    binary = (geom > 0.16).astype(np.uint8)
    if int(binary.sum()) == 0:
        return np.zeros_like(geom, dtype=np.float32)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    edge = smoothstep(0.10, max(0.75, float(feather_px)), dist)
    alpha = clamp01(edge * np.maximum(geom, 0.0))

    if focus_sigma > 0.55:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=focus_sigma, sigmaY=focus_sigma)
        # Keep the blur local to the eye opening plus a tiny physically
        # plausible halo; do not let it spill across the face.
        halo = max(1, int(math.ceil(focus_sigma * 1.35)))
        k = halo * 2 + 1
        allowed = cv2.dilate(
            binary,
            ellipse_kernel(k),
            iterations=1,
        ).astype(np.float32)
        alpha *= allowed

    return clamp01(alpha)


def collapse_from_edges(mask: np.ndarray, visibility: float) -> np.ndarray:
    """
    Occlusion-only circular collapse.

    Keep the normal full eyeball mask for most of the fade, then visibly
    transition to an iris-sized circular core and finally a point.
    """
    v = float(np.clip(visibility, 0.0, 1.0))

    if v >= 0.72:
        return mask.copy()
    if v <= 0.002:
        return np.zeros_like(mask, dtype=np.float32)

    binary = mask > 0.05
    if int(np.count_nonzero(binary)) < 8:
        return np.zeros_like(mask, dtype=np.float32)

    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]].astype(np.float32)
    weights = np.maximum(mask, 0.0)
    total = float(weights.sum())
    if total <= 1e-6:
        return np.zeros_like(mask, dtype=np.float32)

    cx = float((xx * weights).sum() / total)
    cy = float((yy * weights).sum() / total)

    ys, xs = np.where(binary)
    rr = np.sqrt((xs.astype(np.float32) - cx) ** 2 + (ys.astype(np.float32) - cy) ** 2)
    rmax = max(2.0, float(rr.max()))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rmax

    # Stage 1: 0.72 -> 0.28 collapses from full eye to about iris-size.
    # Stage 2: 0.28 -> 0 collapses iris-size core to a point.
    if v >= 0.28:
        t = (v - 0.28) / (0.72 - 0.28)
        solid_radius = 0.30 + 0.82 * t
    else:
        t = v / 0.28
        solid_radius = 0.035 + 0.265 * t

    # Noticeably softer than the black-eye edge, but no longer so huge that
    # the circular progression becomes visually indistinct.
    feather = 0.46
    radial = 1.0 - smoothstep(solid_radius, solid_radius + feather, r)

    aperture_sigma = max(1.6, rmax * 0.12)
    radial = cv2.GaussianBlur(
        radial.astype(np.float32),
        (0, 0),
        sigmaX=aperture_sigma,
        sigmaY=aperture_sigma,
    )
    radial = clamp01(radial)

    # Preserve a black circular core almost to the end; only the last tiny
    # point fades in opacity.
    center_fade = float(smoothstep(0.006, 0.075, np.array(v, dtype=np.float32)))
    return clamp01(mask * radial * center_fade)


class SocketTemporalState:
    """Per-eye visibility smoothing plus graceful fade when tracking disappears."""

    def __init__(self):
        self.visibility = np.zeros(2, dtype=np.float32)
        self.last_bbox = None
        self.last_base_masks = None
        self.last_eye_info = None
        self.prev_render_mask = None
        self.prev_render_bbox = None

    def _advance_visibility(self, targets, fps: float):
        fps = max(1.0, float(fps))
        out = self.visibility.copy()
        for i in range(2):
            raw = float(np.clip(targets[i], 0.0, 1.0))

            # Hysteresis: ordinary tiny confidence changes around a visible eye
            # must not start/stop the occlusion fade. Only a meaningful drop
            # begins hiding; a clearly visible eye snaps the target back to 1.
            if raw >= 0.70:
                target = 1.0
            elif raw <= 0.34:
                target = 0.0
            else:
                target = float(out[i])

            tau = 0.070 if target < out[i] else 0.035
            a = 1.0 - math.exp(-1.0 / (fps * tau))
            out[i] += a * (target - out[i])
        self.visibility = np.clip(out, 0.0, 1.0)

    def detected(self, bbox, base_masks, eye_info, targets, fps: float):
        self._advance_visibility(targets, fps)

        # Under partial occlusion, prefer the last reliable socket geometry to
        # fresh MediaPipe landmarks that may drift onto a hand or nose.
        weak = min(float(targets[0]), float(targets[1])) < 0.70
        if (
            weak
            and self.last_bbox is not None
            and self.last_base_masks is not None
            and self.last_eye_info is not None
        ):
            bbox = self.last_bbox
            base_masks = self.last_base_masks
            eye_info = self.last_eye_info

        masks = [
            collapse_from_edges(base_masks[i], float(self.visibility[i]))
            for i in range(2)
        ]

        self.last_bbox = bbox
        self.last_base_masks = [m.copy() for m in base_masks]
        self.last_eye_info = eye_info

        info = [
            (*eye_info[i], float(self.visibility[i]))
            for i in range(2)
        ]
        combined = clamp01(masks[0] + masks[1])
        strong_stabilize = min(float(targets[0]), float(targets[1])) >= 0.78
        combined = self._stabilize_render_mask(bbox, combined, strong_stabilize)
        return bbox, combined, info

    def _stabilize_render_mask(self, bbox, mask, strong: bool):
        """
        Temporal alpha smoothing for the final eye mask.

        This attacks 1-3 px border chatter directly instead of trying to infer
        whether each tiny landmark move was real. On normal reliable tracking
        it damps small frame-to-frame contour noise; during occlusion we leave
        the circular fade untouched.
        """
        if bbox is None or mask is None:
            return mask

        if (
            strong
            and self.prev_render_mask is not None
            and self.prev_render_bbox == bbox
            and self.prev_render_mask.shape == mask.shape
        ):
            # Strong persistence on tiny contour changes, but larger genuine
            # shape changes still come through over a few frames.
            delta = float(np.mean(np.abs(mask - self.prev_render_mask)))
            alpha = 0.08 if delta < 0.030 else (0.20 if delta < 0.055 else 0.46)
            mask = alpha * mask + (1.0 - alpha) * self.prev_render_mask

        self.prev_render_bbox = bbox
        self.prev_render_mask = mask.copy()
        return clamp01(mask)

    def lost(self, fps: float):
        if self.last_bbox is None or self.last_base_masks is None:
            return None, None, []
        self._advance_visibility((0.0, 0.0), fps)
        masks = [
            collapse_from_edges(self.last_base_masks[i], float(self.visibility[i]))
            for i in range(2)
        ]
        if float(max(self.visibility)) < 0.01:
            return None, None, []
        info = [
            (*self.last_eye_info[i], float(self.visibility[i]))
            for i in range(2)
        ]
        combined = clamp01(masks[0] + masks[1])
        self.prev_render_bbox = self.last_bbox
        self.prev_render_mask = combined.copy()
        return self.last_bbox, combined, info


def build_masks(frame: np.ndarray, points: np.ndarray, cfg: Config):
    h, w = frame.shape[:2]

    right_eye, rw, rc = eye_geometry(points, RIGHT_EYE)
    left_eye, lw, lc = eye_geometry(points, LEFT_EYE)
    all_eye = np.vstack([right_eye, left_eye])
    max_eye_width = max(8.0, rw, lw)

    pad_x = int(max(14.0, max_eye_width * 0.25))
    pad_y = int(max(12.0, max_eye_width * 0.22))
    raw_x0 = max(0, int(math.floor(float(np.min(all_eye[:, 0])))) - pad_x)
    raw_x1 = min(w, int(math.ceil(float(np.max(all_eye[:, 0])))) + pad_x + 1)
    raw_y0 = max(0, int(math.floor(float(np.min(all_eye[:, 1])))) - pad_y)
    raw_y1 = min(h, int(math.ceil(float(np.max(all_eye[:, 1])))) + pad_y + 1)

    # Stable crop grid: tiny landmark noise should not change the ROI origin
    # every frame, because that defeats temporal edge smoothing.
    q = 4
    x0 = max(0, (raw_x0 // q) * q)
    y0 = max(0, (raw_y0 // q) * q)
    x1 = min(w, ((raw_x1 + q - 1) // q) * q)
    y1 = min(h, ((raw_y1 + q - 1) // q) * q)
    if x1 <= x0 or y1 <= y0:
        return None, None, [], (0.0, 0.0)

    roi_shape = (y1 - y0, x1 - x0)
    offset = np.array([x0, y0], dtype=np.float32)
    roi_frame = frame[y0:y1, x0:x1]

    ru = points[RIGHT_UPPER, :2] - offset
    rl = points[RIGHT_LOWER, :2] - offset
    lu = points[LEFT_UPPER, :2] - offset
    ll = points[LEFT_LOWER, :2] - offset

    right_geom = smooth_eye_geometry(
        roi_shape,
        ru, rl,
        points[133, :2] - offset,
        points[33, :2] - offset,
        rw,
    )
    left_geom = smooth_eye_geometry(
        roi_shape,
        lu, ll,
        points[362, :2] - offset,
        points[263, :2] - offset,
        lw,
    )

    roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    roi_lab = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2LAB).astype(np.float32)

    focus_sigma = local_focus_sigma(roi_frame, max(rw, lw), gray=roi_gray)
    right_base = base_socket_alpha(right_geom, cfg.feather_px, focus_sigma)
    left_base = base_socket_alpha(left_geom, cfg.feather_px, focus_sigma)

    right_vis = eye_visibility_factor(
        roi_frame, right_geom, rw, lab=roi_lab, gray=roi_gray
    )
    left_vis = eye_visibility_factor(
        roi_frame, left_geom, lw, lab=roi_lab, gray=roi_gray
    )

    eye_info = [
        (
            right_eye, rw, rc, 11.73,
            points[RIGHT_UPPER, :2].copy(),
            points[RIGHT_LOWER, :2].copy(),
            points[133, :2].copy(),
            points[33, :2].copy(),
        ),
        (
            left_eye, lw, lc, 29.31,
            points[LEFT_UPPER, :2].copy(),
            points[LEFT_LOWER, :2].copy(),
            points[362, :2].copy(),
            points[263, :2].copy(),
        ),
    ]
    return (x0, y0, x1, y1), [right_base, left_base], eye_info, (right_vis, left_vis)

def debug_sclera_overlay(frame: np.ndarray, bbox, mask: np.ndarray) -> np.ndarray:
    if bbox is None or mask is None or float(mask.max()) <= 0.001:
        return frame
    x0, y0, x1, y1 = bbox
    out = frame.copy()
    roi = out[y0:y1, x0:x1].astype(np.float32)
    a = clamp01(mask)[..., None] * 0.78
    green = np.zeros_like(roi)
    green[..., 1] = 255.0
    roi = roi * (1.0 - a) + green * a
    out[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
    return out


def blacken_sclera(
    frame: np.ndarray,
    bbox,
    mask: np.ndarray,
    blackness: float,
    confidence: float,
) -> np.ndarray:
    """Render an opaque black socket, not a contrast/darken filter."""
    if bbox is None or mask is None or blackness <= 0.0 or confidence <= 0.0:
        return frame
    if float(mask.max()) <= 0.001:
        return frame

    x0, y0, x1, y1 = bbox
    roi = frame[y0:y1, x0:x1].astype(np.float32)

    # Interior opacity is essentially 1.0. No iris, sclera, glare or original
    # eye brightness should show through. blackness is an effect-strength knob,
    # not a contrast multiplier.
    a = clamp01(mask * confidence)
    if blackness < 1.0:
        a *= float(np.clip(blackness, 0.0, 1.0))
    keep = 1.0 - a

    # Same alpha-over equation, channel-wise and in-place. This avoids several
    # ROI-sized RGB temporaries every frame.
    for channel in range(3):
        plane = roi[..., channel]
        plane *= keep
        plane += SOCKET_BGR[channel] * a

    frame[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
    return frame

class SpiritSmokeState:
    """Small inertial lag so the shader field trails head motion slightly."""

    def __init__(self):
        self.prev_centers = [None, None]
        self.offsets = [
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        ]
        self.motion = [
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        ]
        self._scratch = {}

    def scratch_buffers(self, height: int, width: int):
        key = (int(height), int(width))
        pair = self._scratch.get(key)
        if pair is None:
            # Crop dimensions are quantized, so this remains a tiny cache.
            if len(self._scratch) >= 12:
                self._scratch.clear()
            pair = (
                np.empty(key, dtype=np.float32),
                np.empty(key, dtype=np.uint8),
            )
            self._scratch[key] = pair

        mask, selector = pair
        mask.fill(0.0)
        selector.fill(0)
        return mask, selector

    def dynamics_for(self, eye_index: int, center, eye_width: float):
        cur = np.asarray(center, dtype=np.float32)
        prev = self.prev_centers[eye_index]
        off = self.offsets[eye_index]
        motion = self.motion[eye_index]

        if prev is not None:
            delta = cur - prev
            motion = motion * 0.74 + delta * 0.26
            off = off * 0.82 - motion * 0.18
            limit = max(1.0, eye_width * 0.10)
            mag = float(np.linalg.norm(off))
            if mag > limit:
                off *= limit / mag
        else:
            off *= 0.0
            motion *= 0.0

        self.prev_centers[eye_index] = cur
        self.offsets[eye_index] = off
        self.motion[eye_index] = motion
        return off, motion


def apply_smoke(
    frame: np.ndarray,
    eye_info,
    shader_time: float,
    strength: float,
    shader_size: float,
    shader_blur: float,
    confidence: float,
    socket_bbox,
    socket_mask: np.ndarray,
    spirit_state: Optional[SpiritSmokeState] = None,
    shader_renderer: Optional[GLSLSmokeRenderer] = None,
) -> np.ndarray:
    """Render fast GLSL plasma from the real upper/lower eyelid splines."""
    if (
        strength <= 0.0
        or confidence <= 0.0
        or not eye_info
        or socket_bbox is None
        or socket_mask is None
        or shader_renderer is None
    ):
        return frame

    if spirit_state is None:
        spirit_state = SpiritSmokeState()

    sx0, sy0, sx1, sy1 = socket_bbox
    fh, fw = frame.shape[:2]
    out = frame

    for eye_index, info in enumerate(eye_info):
        if len(info) < 9:
            continue

        (
            eye_poly,
            eye_width,
            center,
            seed,
            upper_pts,
            lower_pts,
            medial,
            outer_corner,
            visibility,
        ) = info

        v = float(np.clip(visibility, 0.0, 1.0))
        if v <= 0.002:
            continue

        size = max(0.25, float(shader_size))

        # 1x keeps the exact old ROI. Larger sizes used to multiply the WHOLE
        # crop, so size=4 rendered ~16x the area. Only the shader's reach grows
        # with size; grow the crop by that additional reach instead.
        extra_reach = max(0.0, size - 1.0) * 0.94
        pad_x = max(40, int(round(eye_width * (1.55 + extra_reach))))
        pad_up = max(48, int(round(eye_width * (1.70 + extra_reach))))
        pad_down = max(38, int(round(eye_width * (1.35 + extra_reach))))

        cx, cy = float(center[0]), float(center[1])
        x0 = max(0, int(math.floor(cx - pad_x)))
        x1 = min(fw, int(math.ceil(cx + pad_x)) + 1)
        y0 = max(0, int(math.floor(cy - pad_up)))
        y1 = min(fh, int(math.ceil(cy + pad_down)) + 1)

        # Stabilize invisible crop boundaries so tiny landmark jitter does not
        # create new CPU/GPU buffer sizes every frame.
        q = 8
        x0 = max(0, (x0 // q) * q)
        y0 = max(0, (y0 // q) * q)
        x1 = min(fw, ((x1 + q - 1) // q) * q)
        y1 = min(fh, ((y1 + q - 1) // q) * q)

        if x1 <= x0 or y1 <= y0:
            continue

        rh, rw = y1 - y0, x1 - x0
        local_mask, selector = spirit_state.scratch_buffers(rh, rw)

        ix0, iy0 = max(x0, sx0), max(y0, sy0)
        ix1, iy1 = min(x1, sx1), min(y1, sy1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue

        src_x0, src_y0 = ix0 - sx0, iy0 - sy0
        src_x1, src_y1 = ix1 - sx0, iy1 - sy0
        dst_x0, dst_y0 = ix0 - x0, iy0 - y0
        dst_x1, dst_y1 = ix1 - x0, iy1 - y0

        local_mask[dst_y0:dst_y1, dst_x0:dst_x1] = socket_mask[
            src_y0:src_y1, src_x0:src_x1
        ]

        # Isolate this eye from the combined two-eye socket mask without cutting
        # the shader's outer wisps.
        lcx = np.float32(cx - x0)
        lcy = np.float32(cy - y0)
        cv2.ellipse(
            selector,
            (int(round(float(lcx))), int(round(float(lcy)))),
            (
                max(4, int(round(eye_width * 0.72))),
                max(3, int(round(eye_width * 0.40))),
            ),
            0.0,
            0.0,
            360.0,
            255,
            thickness=-1,
            lineType=cv2.LINE_8,
        )
        cv2.multiply(
            local_mask,
            selector,
            dst=local_mask,
            scale=(1.0 / 255.0),
            dtype=cv2.CV_32F,
        )

        if float(local_mask.max()) < 0.005:
            continue

        offset = np.array([x0, y0], dtype=np.float32)
        local_upper = np.asarray(upper_pts, dtype=np.float32) - offset
        local_lower = np.asarray(lower_pts, dtype=np.float32) - offset
        local_medial = np.asarray(medial, dtype=np.float32) - offset

        # Keep the exact smooth eyelid geometry, but send only tiny point
        # arrays to GLSL. The GPU computes pixel-to-lid distance directly.
        upper_curve = catmull_rom_chain(local_upper, 2)
        lower_curve = catmull_rom_chain(local_lower, 2)

        motion_offset, motion = spirit_state.dynamics_for(
            eye_index, center, eye_width
        )

        fx = shader_renderer.render(
            local_mask,
            upper_curve,
            lower_curve,
            time_s=float(shader_time),
            seed=float(seed),
            visibility=v,
            strength=float(strength * confidence),
            center_xy=(
                float(lcx + motion_offset[0] * 0.24),
                float(lcy + motion_offset[1] * 0.24),
            ),
            motion_xy=(float(motion[0]), float(motion[1])),
            medial_xy=(
                float(local_medial[0]),
                float(local_medial[1]),
            ),
            eye_width=float(eye_width),
            size=float(shader_size),
            scale=0.64,
        )

        if fx.ndim != 3 or fx.shape[2] < 2:
            continue

        roi = out[y0:y1, x0:x1].astype(np.float32)

        # Plasma body stays dark at the root but is now visibly distinct from
        # the socket. A second highlight is derived from the SAME smooth alpha
        # field, so there is no extra contour / hard color boundary.
        # Bigger means nastier, not merely farther-reaching. Large sizes get
        # denser black root, denser plasma and stronger hot structure.
        size_over = max(0.0, size - 1.0)
        dark_boost = 1.0 + size_over * 0.32
        plasma_boost = 1.0 + size_over * 0.42
        hot_boost = 1.0 + size_over * 0.35

        dark_a = clamp01(fx[..., 0] * 0.96 * dark_boost)
        plasma_raw = clamp01(fx[..., 1] * plasma_boost)

        # User-facing shader blur:
        # 1x = half the previous blur, 2x = exactly the previous blur.
        blur_amount = max(0.0, float(shader_blur))
        if blur_amount > 0.001:
            blur_sigma = max(
                0.325,
                float(eye_width) * 0.00275,
            ) * blur_amount
            blur_mix = float(np.clip(0.125 * blur_amount, 0.0, 0.70))
            plasma_blur = cv2.GaussianBlur(
                plasma_raw.astype(np.float32, copy=False),
                (0, 0),
                sigmaX=blur_sigma,
                sigmaY=blur_sigma,
            )
            plasma_raw = clamp01(
                plasma_raw * (1.0 - blur_mix)
                + plasma_blur * blur_mix
            )

        body_a = clamp01(plasma_raw * 0.98)

        # Bright electric highlight fades in only as the near-black
        # inner bridge fades out. Body plasma is still strong at the lashes,
        # so the inner transition stays alive without a hard bright ring.
        root_dark = clamp01(fx[..., 0])
        hot_a = clamp01(
            smoothstep(0.30, 0.80, plasma_raw)
            * (1.0 - root_dark * 0.86)
            * 0.36
            * hot_boost
        )

        # Collapse the exact sequence
        #   socket alpha-over -> body alpha-over -> hot alpha-over
        # into equivalent scalar weights. Then operate on RGB planes in-place,
        # avoiding multiple full RGB temporary images.
        keep_hot = 1.0 - hot_a
        keep_body = 1.0 - body_a
        base_w = (1.0 - dark_a) * keep_body * keep_hot
        socket_w = dark_a * keep_body * keep_hot
        body_w = body_a * keep_hot

        for channel in range(3):
            plane = roi[..., channel]
            plane *= base_w
            plane += SOCKET_BGR[channel] * socket_w
            plane += PLASMA_BODY_BGR[channel] * body_w
            plane += PLASMA_HOT_BGR[channel] * hot_a

        out[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)

    return out


def process_video(input_path: str, output_path: str, cfg: Config, encoder_request: str) -> None:
    if not ffmpeg_exists():
        raise RuntimeError("ffmpeg is not available in PATH.")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        raise RuntimeError("Input video dimensions could not be read.")
    if not np.isfinite(fps) or fps <= 0.01:
        fps = 30.0

    encoder = choose_encoder(encoder_request)

    try:
        shader_renderer = GLSLSmokeRenderer()
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize the real EGL/OpenGL smoke shader: "
            f"{exc}"
        ) from exc

    transition_desc = ""
    if cfg.transition > 0.0 and cfg.shader_size_final is not None:
        transition_desc = (
            f" -> size={cfg.shader_size_final:.2f}x"
            + (
                f" speed={cfg.shader_speed_final:.2f}x"
                if cfg.shader_speed_final is not None
                else ""
            )
            + f" over {cfg.transition:.2f}s"
        )
    print(
        f"[demon-eye] {width}x{height} @ {fps:.3f} fps | "
        f"encoder={encoder} | "
        f"speed={cfg.shader_speed:.2f}x | size={cfg.shader_size:.2f}x"
        f"{transition_desc} | "
        f"blur={cfg.shader_blur:.2f}x | smoke={shader_renderer.backend_name}"
    )

    out_dir = str(Path(output_path).resolve().parent)
    os.makedirs(out_dir, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(prefix="demon-eye-", suffix=".mp4", dir=out_dir)
    os.close(fd)
    os.unlink(temp_path)

    proc = start_encoder(temp_path, width, height, fps, encoder)
    if proc.stdin is None:
        raise RuntimeError("Could not open ffmpeg stdin.")

    socket_state = SocketTemporalState()
    spirit_state = SpiritSmokeState()
    profile = os.environ.get("DEMON_EYE_PROFILE", "").strip() == "1"
    processed_frames = 0
    profile_ns = {
        "tracking": 0,
        "masks": 0,
        "socket": 0,
        "shader": 0,
        "encoder_write": 0,
    }
    if cfg.track_fps <= 0.0:
        track_every = 1
    else:
        track_every = max(1, int(round(fps / cfg.track_fps)))
    print(
        f"[demon-eye] tracking={cfg.track_width}px wide @ ~{fps / track_every:.1f} Hz "
        f"(inference every {track_every} frame{'s' if track_every != 1 else ''})"
    )

    # Pipeline CPU tracking one or two frames ahead while the main thread does
    # GLSL rendering/compositing/encoding. Tracking itself remains identical.
    tracked_frames: queue.Queue = queue.Queue(maxsize=2)
    tracking_stop = threading.Event()

    def queue_item(item) -> bool:
        while not tracking_stop.is_set():
            try:
                tracked_frames.put(item, timeout=0.10)
                return True
            except queue.Full:
                continue
        return False

    def tracking_worker():
        tracker = None
        try:
            tracker = LandmarkTracker(cfg)
            index = 0
            while not tracking_stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break

                run_inference = (
                    (index % track_every == 0)
                    or (tracker.prev is None)
                )
                t0 = time.perf_counter_ns() if profile else 0
                points, confidence = tracker.detect(
                    frame,
                    run_inference=run_inference,
                )
                if profile:
                    profile_ns["tracking"] += time.perf_counter_ns() - t0
                if not queue_item(
                    ("frame", index, frame, points, confidence)
                ):
                    return
                index += 1
        except BaseException as exc:
            queue_item(
                (
                    "error",
                    exc,
                    "".join(
                        traceback.format_exception(
                            type(exc), exc, exc.__traceback__
                        )
                    ),
                )
            )
        finally:
            if tracker is not None:
                try:
                    tracker.close()
                except Exception:
                    pass
            queue_item(("done",))

    tracking_thread = threading.Thread(
        target=tracking_worker,
        name="demon-eye-tracker",
        daemon=True,
    )
    tracking_thread.start()

    try:
        with tqdm(total=total if total > 0 else None, unit="frame", desc="demon-eye") as bar:
            while True:
                item = tracked_frames.get()
                kind = item[0]

                if kind == "done":
                    break
                if kind == "error":
                    _, exc, detail = item
                    raise RuntimeError(
                        "Face tracking worker failed:\n" + detail
                    ) from exc

                _, frame_index, frame, points, confidence = item
                out = frame

                if points is not None and confidence >= 0.95:
                    t0 = time.perf_counter_ns() if profile else 0
                    socket_bbox, base_masks, raw_eye_info, vis_targets = build_masks(
                        frame, points, cfg
                    )
                    if profile:
                        profile_ns["masks"] += time.perf_counter_ns() - t0
                    if socket_bbox is not None:
                        socket_bbox, socket_mask, eye_info = socket_state.detected(
                            socket_bbox, base_masks, raw_eye_info, vis_targets, fps
                        )
                    else:
                        socket_bbox, socket_mask, eye_info = socket_state.lost(fps)
                else:
                    # Never rebuild the eye from uncertain landmarks. Keep the
                    # last trusted position and let only the circular aperture
                    # fade it away.
                    socket_bbox, socket_mask, eye_info = socket_state.lost(fps)

                if socket_bbox is not None and socket_mask is not None:
                    t0 = time.perf_counter_ns() if profile else 0
                    if cfg.debug_mask:
                        out = debug_sclera_overlay(out, socket_bbox, socket_mask)
                    else:
                        out = blacken_sclera(
                            out, socket_bbox, socket_mask, cfg.blackness, 1.0
                        )
                    if profile:
                        profile_ns["socket"] += time.perf_counter_ns() - t0

                    smoke_strength = float(np.clip(cfg.smoke, 0.0, 1.0))
                    video_time = frame_index / fps
                    current_size, current_speed, shader_time = transition_values(
                        video_time, cfg
                    )
                    t0 = time.perf_counter_ns() if profile else 0
                    out = apply_smoke(
                        out,
                        eye_info,
                        shader_time,
                        smoke_strength,
                        current_size,
                        cfg.shader_blur,
                        1.0,
                        socket_bbox,
                        socket_mask,
                        spirit_state=spirit_state,
                        shader_renderer=shader_renderer,
                    )
                    if profile:
                        profile_ns["shader"] += time.perf_counter_ns() - t0

                try:
                    t0 = time.perf_counter_ns() if profile else 0
                    if not out.flags.c_contiguous:
                        out = np.ascontiguousarray(out)
                    proc.stdin.write(memoryview(out).cast("B"))
                    if profile:
                        profile_ns["encoder_write"] += time.perf_counter_ns() - t0
                except BrokenPipeError as exc:
                    raise RuntimeError("ffmpeg encoder terminated early.") from exc

                processed_frames += 1
                bar.update(1)

        tracking_thread.join()

        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg video encoder failed with exit code {rc}.")

        remux_audio(temp_path, input_path, output_path)
    finally:
        tracking_stop.set()
        try:
            tracking_thread.join(timeout=2.0)
        except Exception:
            pass
        cap.release()
        try:
            shader_renderer.release()
        except Exception:
            pass
        if proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except Exception:
                pass
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    if profile:
        processed = max(1, processed_frames)
        def ms_per_frame(name: str) -> float:
            return profile_ns[name] / processed / 1_000_000.0

        print(
            "[demon-eye] profile ms/frame | "
            f"tracking={ms_per_frame('tracking'):.2f} "
            f"masks={ms_per_frame('masks'):.2f} "
            f"socket={ms_per_frame('socket'):.2f} "
            f"shader+composite={ms_per_frame('shader'):.2f} "
            f"pipe={ms_per_frame('encoder_write'):.2f}"
        )

    print(f"[demon-eye] wrote: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(
        prog="demon_eye.py",
        usage="%(prog)s INPUT OUTPUT [--smoke N] [--speed N] [--size N] [--shader-blur N] [--transition SEC --size-final N [--speed-final N]]",
        description="1080p demon-eye video effect.",
    )
    p.add_argument("input", help="Input video")
    p.add_argument("output", help="Output MP4")
    p.add_argument(
        "--smoke", type=float, default=0.82,
        help="effect strength 0..1; Windows Strength uses the same value (default: 0.82)",
    )
    p.add_argument(
        "--speed", type=float, default=1.0,
        help="shader animation speed multiplier (default: 1.0x)",
    )
    p.add_argument(
        "--size", "--smoke-radius", dest="shader_size", type=float, default=1.0,
        help="effect scale/spread; values >1 also make the black root and plasma stronger (default: 1.0x)",
    )
    p.add_argument(
        "--shader-blur", type=float, default=1.0,
        help="shader softness; 0 disables post-blur, 1 is default, 2 matches the older stronger blur",
    )
    p.add_argument(
        "--transition", type=float, default=0.0,
        help="seconds to smoothly grow from initial size/speed to final values (default: off)",
    )
    p.add_argument(
        "--size-final", type=float, default=None,
        help="target size reached at the end of --transition; required when transition > 0",
    )
    p.add_argument(
        "--speed-final", type=float, default=None,
        help="optional target animation speed at the end of --transition; omitted = keep initial speed",
    )
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True, help=argparse.SUPPRESS)
    p.add_argument("--encoder", choices=["auto", "nvenc", "x264"], default="auto", help=argparse.SUPPRESS)

    # Kept only so old commands do not break.
    p.add_argument("--blackness", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--feather", type=float, default=1.25, help=argparse.SUPPRESS)
    p.add_argument("--iris-guard", type=float, default=1.08, help=argparse.SUPPRESS)
    p.add_argument("--hold-frames", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--skin-gate", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--debug-mask", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--track-width", type=int, default=576, help=argparse.SUPPRESS)
    p.add_argument("--track-fps", type=float, default=0.0, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.transition > 0.0 and args.size_final is None:
        p.error("--size-final is required when --transition > 0")
    return args


def main():
    args = parse_args()

    cfg = Config(
        blackness=float(np.clip(args.blackness, 0.0, 1.0)),
        smoke=float(np.clip(args.smoke, 0.0, 1.0)),
        shader_speed=max(0.0, float(args.speed)),
        shader_size=max(0.25, float(args.shader_size)),
        shader_blur=max(0.0, float(args.shader_blur)),
        transition=max(0.0, float(args.transition)),
        shader_size_final=(
            max(0.25, float(args.size_final))
            if args.size_final is not None
            else None
        ),
        shader_speed_final=(
            max(0.0, float(args.speed_final))
            if args.speed_final is not None
            else None
        ),
        feather_px=max(0.5, float(args.feather)),
        iris_guard=max(1.0, float(args.iris_guard)),
        hold_frames=max(0, int(args.hold_frames)),
        skin_gate=bool(args.skin_gate),
        track_width=max(256, int(args.track_width)),
        track_fps=max(0.0, float(args.track_fps)),
        debug_mask=bool(args.debug_mask),
    )

    output_path = args.output
    if os.path.abspath(output_path) == os.path.abspath(args.input):
        print("[demon-eye] ERROR: output would overwrite input.", file=sys.stderr)
        return 1

    try:
        process_video(args.input, output_path, cfg, args.encoder)
    except KeyboardInterrupt:
        print("\n[demon-eye] interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[demon-eye] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
