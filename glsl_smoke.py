from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

try:
    import moderngl
except Exception:
    moderngl = None


LID_SAMPLES = 13

VERTEX_SHADER = r"""
#version 330
in vec2 in_pos;
out vec2 v_uv;

void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


FRAGMENT_SHADER = r"""
#version 330

uniform sampler2D u_mask;
uniform sampler2D u_lid_points;

uniform float u_time;
uniform float u_seed;
uniform float u_visibility;
uniform float u_strength;
uniform vec2 u_center;
uniform vec2 u_motion;
uniform vec2 u_medial;
uniform vec4 u_lid_bounds;
uniform float u_aspect;
uniform float u_eye_width_metric;
uniform float u_size;

in vec2 v_uv;
out vec2 frag_out;

const int LID_SAMPLES = 13;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float noise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash12(i);
    float b = hash12(i + vec2(1.0, 0.0));
    float c = hash12(i + vec2(0.0, 1.0));
    float d = hash12(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm3(vec2 p) {
    float v = 0.0;
    float a = 0.57;
    mat2 rot = mat2(1.63, -0.57, 0.57, 1.63);
    for (int i = 0; i < 3; ++i) {
        v += a * noise2(p);
        p = rot * p + vec2(4.71, 7.93);
        a *= 0.48;
    }
    return v / 1.117;
}

vec2 metric(vec2 uv) {
    return vec2(uv.x * u_aspect, uv.y);
}

vec2 lid_point(int row, int index) {
    // Uploaded pre-transformed into the same metric space as metric(uv).
    return texelFetch(u_lid_points, ivec2(index, row), 0).rg;
}

float segment_distance(vec2 p, vec2 a, vec2 b) {
    vec2 ab = b - a;
    float h = clamp(
        dot(p - a, ab) / max(dot(ab, ab), 1e-8),
        0.0, 1.0
    );
    return length(p - (a + ab * h));
}

float lid_distance(vec2 uv, int row) {
    vec2 p = metric(uv);
    float d = 1e6;

    // Adjacent line segments share endpoints. Fetch each spline point once
    // instead of fetching both ends again for every segment.
    vec2 a = lid_point(row, 0);
    for (int i = 1; i < LID_SAMPLES; ++i) {
        vec2 b = lid_point(row, i);
        d = min(d, segment_distance(p, a, b));
        a = b;
    }

    return d / max(u_eye_width_metric, 1e-5);
}

float ring8(vec2 uv, float radius_eye_width) {
    float r = u_eye_width_metric * radius_eye_width;
    vec2 dx = vec2(r / u_aspect, 0.0);
    vec2 dy = vec2(0.0, r);
    vec2 dd = vec2((r * 0.70710678) / u_aspect, r * 0.70710678);

    float s = 0.0;
    s += texture(u_mask, uv + dx).r;
    s += texture(u_mask, uv - dx).r;
    s += texture(u_mask, uv + dy).r;
    s += texture(u_mask, uv - dy).r;
    s += texture(u_mask, uv + dd).r;
    s += texture(u_mask, uv - dd).r;
    s += texture(u_mask, uv + vec2(dd.x, -dd.y)).r;
    s += texture(u_mask, uv + vec2(-dd.x, dd.y)).r;
    return s * 0.125;
}

// Smooth support around the ACTUAL socket. This exists only to weld the plasma
// to the eyeslit and make the inner transition impossible to expose.
float socket_support(vec2 uv) {
    float c = texture(u_mask, uv).r;
    float root_scale =
        1.0 + max(u_size - 1.0, 0.0) * 0.18;
    float r1 = ring8(uv, 0.058 * root_scale);
    float r2 = ring8(uv, 0.142 * root_scale);
    float r3 = ring8(uv, 0.258 * root_scale);

    return clamp(
        c  * 0.39 +
        r1 * 0.35 +
        r2 * 0.24 +
        r3 * 0.14,
        0.0, 1.0
    );
}

float ridge01(float x) {
    return 1.0 - abs(x * 2.0 - 1.0);
}

// Returns x = continuous membrane coverage, y = electric energy.
// Energy NEVER defines topology; coverage remains connected underneath it.
vec2 plasma_membrane(
    vec2 p,
    float d,
    float t,
    float seed,
    float direction,
    float reach_mul
) {
    // fbm3() is bounded below 0.873 with the current octave weights, so the
    // reach expression can never exceed ~0.564 * reach_mul * u_size.
    // Reject farther pixels before any FBM work.
    if (d >= 0.565 * reach_mul * u_size) {
        return vec2(0.0);
    }

    float contour1 = fbm3(
        vec2(
            p.x * 3.8 + seed * 0.71,
            t * 0.48 * direction
        )
    );

    float contour2 = fbm3(
        vec2(
            p.x * 7.4 - seed * 0.43,
            t * 0.69 * direction + d * 2.1
        )
    );

    // Nonzero minimum reach keeps the membrane continuously attached.
    float reach =
        (
            0.18 +
            contour1 * 0.31 +
            contour2 * 0.13
        ) * reach_mul * u_size;

    // Broad outer feather: outer transition is an actual gradient, not a cut.
    float coverage =
        1.0 - smoothstep(
            reach * 0.36,
            reach,
            d
        );

    coverage *= exp(-d * 1.72);

    // Outside the membrane the caller would multiply energy by zero anyway.
    // Skip the expensive electric/noise stack entirely in those pixels.
    if (coverage <= 0.0) {
        return vec2(0.0);
    }

    // Domain-warped electric structure INSIDE the connected coverage.
    float warp = fbm3(
        vec2(
            p.x * 5.8 + seed * 0.91,
            d * 8.7 - t * 1.06 * direction
        )
    );

    float e1 = fbm3(
        vec2(
            p.x * 11.5 + (warp - 0.5) * 1.35 - seed * 0.37,
            d * 17.0 + t * 1.42 * direction
        )
    );

    float e2 = fbm3(
        vec2(
            p.x * 18.0 - (warp - 0.5) * 1.1 + seed * 0.19,
            d * 24.0 - t * 1.78 * direction
        )
    );

    // Electric cores plus a broader low-energy halo. The core still gives
    // readable plasma structure; the halo removes the synthetic hard-line look.
    float r1 = ridge01(e1);
    float r2 = ridge01(e2);

    float veins =
        smoothstep(0.54, 0.92, r1) * 0.58 +
        smoothstep(0.62, 0.95, r2) * 0.34;

    float vein_glow =
        smoothstep(0.28, 0.82, r1) * 0.15 +
        smoothstep(0.36, 0.86, r2) * 0.10;

    // More coherent rolling plasma underneath the fine veins.
    float rolling = fbm3(
        vec2(
            p.x * 7.8 + seed,
            d * 12.0 - t * 1.12 * direction
        )
    );

    float energy = clamp(
        0.11 +
        rolling * 0.24 +
        veins * 0.70 +
        vein_glow,
        0.0, 1.0
    );

    return vec2(coverage, energy);
}

float medial_corner_bias(vec2 uv) {
    // Keep the general caruncle shadow compact; do not solve the hard edge by
    // simply making a larger dark blob.
    vec2 d = metric(uv) - metric(u_medial);
    d.x *= 1.05;
    d.y *= 0.90;

    float ew = max(u_eye_width_metric, 1e-5);
    return exp(-dot(d, d) / (ew * ew * 0.0135));
}

// One-sided feather from the medial/carnucle point INTO the eye.
// This specifically smears the pointy black-mask cut in the horizontal-ish
// medial->eye-center direction without darkening a much larger surrounding area.
float medial_inner_feather(vec2 uv) {
    float ew = max(u_eye_width_metric, 1e-5);

    vec2 m = metric(u_medial);
    vec2 c = metric(u_center);
    vec2 axis = c - m;
    float axis_len = max(length(axis), 1e-6);
    axis /= axis_len;
    vec2 normal = vec2(-axis.y, axis.x);

    vec2 r = metric(uv) - m;
    float along = dot(r, axis) / ew;
    float across = dot(r, normal) / ew;

    // Keep almost all of this on the EYE side of the caruncle border.
    // Negative along = medial/nose side; positive along = into the eye.
    float along_in =
        smoothstep(-0.006, 0.040, along);
    float along_out =
        1.0 - smoothstep(0.13, 0.255, along);

    // Slightly tighter perpendicular feather, so darkness sits on the cut
    // instead of spreading through the whole medial corner.
    float cross_soft =
        exp(-pow(across / 0.082, 2.0));

    // Bias the strongest part a little inside the black/carnucle boundary.
    float border_weight =
        exp(-pow((along - 0.070) / 0.115, 2.0));

    return clamp(
        along_in * along_out * cross_soft * (0.72 + 0.28 * border_weight),
        0.0,
        1.0
    );
}

float circular_collapse_gate(vec2 uv, float vis) {
    if (vis >= 0.72) return 1.0;

    float solid_radius;
    if (vis >= 0.28) {
        float q = (vis - 0.28) / (0.72 - 0.28);
        solid_radius = 0.30 + 0.82 * q;
    } else {
        float q = vis / 0.28;
        solid_radius = 0.035 + 0.265 * q;
    }

    float radius = solid_radius * 0.56;
    float feather = 0.27;
    float r =
        length(metric(uv) - metric(u_center))
        / max(u_eye_width_metric, 1e-5);

    return 1.0 - smoothstep(radius, radius + feather, r);
}

void main() {
    vec2 uv = v_uv;

    // Cheap conservative reject before the expensive 24 segment checks + FBM.
    // The theoretical maximum upper reach is ~0.918 eye widths per size unit;
    // extra margin covers socket support, caruncle feather and interpolation.
    vec2 pm = metric(uv);
    vec2 bmin = u_lid_bounds.xy;
    vec2 bmax = u_lid_bounds.zw;
    vec2 outside_box = max(max(bmin - pm, pm - bmax), vec2(0.0));
    float box_distance =
        length(outside_box) / max(u_eye_width_metric, 1e-5);
    if (box_distance > (0.94 * u_size + 0.34)) {
        frag_out = vec2(0.0);
        return;
    }

    float maskv = texture(u_mask, uv).r;

    float du = lid_distance(uv, 0);
    float dl = lid_distance(uv, 1);
    float nearest = min(du, dl);

    float vis = clamp(u_visibility, 0.0, 1.0);
    float t = u_time;

    vec2 p = metric(uv - u_center);
    p -= metric(u_motion) * 0.018;

    float upper_side =
        smoothstep(-0.055, 0.100, u_center.y - uv.y);
    float lower_side =
        smoothstep(-0.045, 0.075, uv.y - u_center.y);

    vec2 up = vec2(0.0);
    vec2 lo = vec2(0.0);

    // smoothstep is exactly zero outside each lid's side. Avoid evaluating the
    // expensive FBM membrane that would only be multiplied by zero afterward.
    if (upper_side > 0.0) {
        up = plasma_membrane(
            p, du, t, u_seed, 1.0, 1.48
        ) * upper_side;
    }
    if (lower_side > 0.0) {
        lo = plasma_membrane(
            p, dl, t, u_seed + 8.71, -1.0, 0.90
        ) * lower_side;
    }

    float coverage =
        max(up.x, lo.x * 0.90);

    float energy =
        max(up.y * up.x, lo.y * lo.x * 0.90);

    float support = socket_support(uv);

    // INNER TRANSITION:
    // near the socket this stays essentially socket-black and continuous.
    // Visible plasma energy ramps in only after that black support has already
    // begun fading, so there is no black-eye -> colored-effect seam.
    float black_root =
        pow(support, 1.05);

    float root_spread =
        1.0 + max(u_size - 1.0, 0.0) * 0.12;
    float lid_root =
        exp(-nearest * (5.2 / root_spread));

    float medial_boost =
        medial_corner_bias(uv);

    float medial_feather =
        medial_inner_feather(uv);

    // Strong electric energy begins at the actual lash/root region.
    float inner_boost =
        1.0 + 4.2 * exp(-pow(nearest / 0.085, 2.0));

    // Continuous floor: the black socket crop must never be exposed directly.
    float membrane_floor =
        coverage * (0.19 + 0.20 * lid_root);

    float crop_hide_veil =
        black_root * exp(-nearest * 2.7) * 0.24;

    // Broad partial-black caruncle overlay. This is intentionally not tied to
    // the exact cutaway shape: it obscures the ~vertical/pointy mask boundary
    // with generous feathering while leaving anatomy discernible.
    float caruncle_shadow =
        medial_boost * (0.065 + 0.075 * exp(-nearest * 2.0));

    // Soften the precise socket-root shape a little in the compact caruncle
    // region, but do not enlarge the whole dark patch.
    float black_root_soft =
        black_root * (1.0 - medial_boost * 0.12);

    // Low-alpha smoky fill remains compact.
    float medial_fill =
        medial_boost * (0.04 + 0.045 * exp(-nearest * 1.8));

    // Directional partial-black feather over the actual cutaway border.
    // This is the main fix: it extends toward the eye center rather than
    // radially enlarging the caruncle shadow.
    float medial_cut_feather =
        medial_feather * (0.25 + 0.18 * exp(-nearest * 2.4));

    // Subtle second veil farther around the eye; no abrupt outer boundary.
    float outer_veil =
        coverage * (1.0 - exp(-nearest * 2.35)) * 0.15;

    float electric =
        coverage * energy * inner_boost;

    float electric_visible =
        1.0 - exp(-electric * 0.58);

    // Keep electric highlights from emphasizing the exact medial boundary.
    float medial_energy_gate =
        1.0 - medial_boost * 0.16 - medial_feather * 0.10;

    float plasma_alpha =
        black_root_soft * 0.060 +
        membrane_floor * 1.02 +
        crop_hide_veil +
        medial_fill +
        outer_veil +
        electric_visible * 0.74 * medial_energy_gate;

    float dark_alpha =
        black_root_soft * 0.97 +
        lid_root * 0.17 +
        caruncle_shadow +
        medial_cut_feather;

    // Don't repaint deep socket unnecessarily; taper continuously.
    dark_alpha *=
        1.0 - smoothstep(0.88, 1.0, maskv) * 0.40;

    plasma_alpha *=
        1.0 - smoothstep(0.91, 1.0, maskv) * 0.18;

    float collapse_gate =
        circular_collapse_gate(uv, vis);
    float spirit =
        pow(vis, 0.92) * u_strength;

    plasma_alpha *= collapse_gate * spirit;
    dark_alpha *= collapse_gate * spirit;

    // Fixed physical feather at the render-box edge. The previous 13% UV
    // feather forced oversized ROIs as --size increased.
    float edge_metric =
        min(
            min(uv.x * u_aspect, (1.0 - uv.x) * u_aspect),
            min(uv.y, 1.0 - uv.y)
        );
    float edge_eye =
        edge_metric / max(u_eye_width_metric, 1e-5);

    float roi_fade =
        smoothstep(0.0, 0.12, edge_eye);

    plasma_alpha *= roi_fade;
    dark_alpha *= roi_fade;

    frag_out = vec2(
        clamp(dark_alpha, 0.0, 1.0),
        clamp(plasma_alpha, 0.0, 1.0)
    );
}
"""


@dataclass
class _Target:
    mask_tex: object
    lid_tex: object
    out_tex: object
    fbo: object


class GLSLSmokeRenderer:
    """Headless EGL plasma renderer with GPU lid-distance evaluation."""

    def __init__(self):
        if moderngl is None:
            raise RuntimeError("moderngl is not installed")

        try:
            self.ctx = moderngl.create_context(
                standalone=True,
                backend="egl",
                require=330,
            )
        except Exception:
            self.ctx = moderngl.create_context(
                standalone=True,
                require=330,
            )

        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )

        vertices = np.array(
            [
                -1.0, -1.0,
                 1.0, -1.0,
                -1.0,  1.0,
                -1.0,  1.0,
                 1.0, -1.0,
                 1.0,  1.0,
            ],
            dtype="f4",
        )
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, "in_pos")

        self.program["u_mask"].value = 0
        self.program["u_lid_points"].value = 1

        self.targets: Dict[Tuple[int, int], _Target] = {}

    @property
    def backend_name(self) -> str:
        try:
            renderer = self.ctx.info.get("GL_RENDERER", "OpenGL")
            return f"glsl/egl-organic-transition ({renderer})"
        except Exception:
            return "glsl/egl-organic-transition"

    def _target(self, width: int, height: int) -> _Target:
        key = (width, height)
        target = self.targets.get(key)
        if target is not None:
            return target

        # Mask alpha needs smooth gradients but half float is already far beyond
        # 8-bit precision and halves transfer bandwidth vs float32.
        mask_tex = self.ctx.texture((width, height), 1, dtype="f2")
        mask_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        mask_tex.repeat_x = False
        mask_tex.repeat_y = False

        # Tiny 17x2 point texture: upper row + lower row.
        lid_tex = self.ctx.texture((LID_SAMPLES, 2), 2, dtype="f4")
        lid_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        lid_tex.repeat_x = False
        lid_tex.repeat_y = False

        # Only two values are ever returned to Python: bridge alpha + plasma alpha.
        out_tex = self.ctx.texture((width, height), 2, dtype="f2")
        out_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        out_tex.repeat_x = False
        out_tex.repeat_y = False

        fbo = self.ctx.framebuffer(color_attachments=[out_tex])
        target = _Target(mask_tex, lid_tex, out_tex, fbo)
        self.targets[key] = target
        return target

    @staticmethod
    def _resample_curve(points: np.ndarray, count: int = LID_SAMPLES) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(pts) == count:
            return pts
        if len(pts) < 2:
            return np.repeat(pts[:1], count, axis=0)

        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        if total <= 1e-6:
            return np.repeat(pts[:1], count, axis=0)

        q = np.linspace(0.0, total, count, dtype=np.float32)
        x = np.interp(q, s, pts[:, 0]).astype(np.float32)
        y = np.interp(q, s, pts[:, 1]).astype(np.float32)
        return np.column_stack([x, y]).astype(np.float32)

    def render(
        self,
        mask: np.ndarray,
        upper_points: np.ndarray,
        lower_points: np.ndarray,
        *,
        time_s: float,
        seed: float,
        visibility: float,
        strength: float,
        center_xy: Tuple[float, float],
        motion_xy: Tuple[float, float],
        medial_xy: Tuple[float, float],
        eye_width: float,
        size: float = 1.0,
        scale: float = 0.64,
    ) -> np.ndarray:
        h, w = mask.shape
        # Caller already rejects empty masks; avoid scanning this full ROI twice.
        if h <= 1 or w <= 1:
            return np.zeros((h, w, 2), dtype=np.float32)

        scale = float(np.clip(scale, 0.40, 1.0))
        sw = max(40, int(math.ceil((w * scale) / 8.0) * 8))
        sh = max(28, int(math.ceil((h * scale) / 8.0) * 8))

        # One half-float image upload instead of five float32 uploads.
        mask_small = cv2.resize(
            mask.astype(np.float32, copy=False),
            (sw, sh),
            interpolation=cv2.INTER_AREA,
        )
        mask_f16 = np.ascontiguousarray(mask_small.astype(np.float16))

        upper = self._resample_curve(upper_points)
        lower = self._resample_curve(lower_points)

        lids = np.stack([upper, lower], axis=0).astype(np.float32)

        # Upload lid points directly in shader metric space:
        # metric(uv) == (uv.x * render_aspect, uv.y).
        render_aspect = float(sw) / max(float(sh), 1.0)
        lids[..., 0] *= np.float32(
            render_aspect / max(float(w), 1.0)
        )
        lids[..., 1] /= np.float32(max(float(h), 1.0))
        lids = np.ascontiguousarray(lids)

        lid_min = np.min(lids.reshape(-1, 2), axis=0)
        lid_max = np.max(lids.reshape(-1, 2), axis=0)

        target = self._target(sw, sh)
        target.mask_tex.write(mask_f16.tobytes(), alignment=1)
        target.lid_tex.write(lids.tobytes(), alignment=1)

        target.mask_tex.use(location=0)
        target.lid_tex.use(location=1)
        target.fbo.use()

        self.ctx.viewport = (0, 0, sw, sh)
        target.fbo.clear(0.0, 0.0, 0.0, 0.0)

        cx = float(center_xy[0]) / max(float(w), 1.0)
        cy = float(center_xy[1]) / max(float(h), 1.0)
        mx = float(motion_xy[0]) / max(float(w), 1.0)
        my = float(motion_xy[1]) / max(float(h), 1.0)
        medx = float(medial_xy[0]) / max(float(w), 1.0)
        medy = float(medial_xy[1]) / max(float(h), 1.0)

        self.program["u_time"].value = float(time_s)
        self.program["u_seed"].value = float(seed)
        self.program["u_visibility"].value = float(np.clip(visibility, 0.0, 1.0))
        self.program["u_strength"].value = float(np.clip(strength, 0.0, 1.0))
        self.program["u_center"].value = (cx, cy)
        self.program["u_motion"].value = (mx, my)
        self.program["u_medial"].value = (medx, medy)
        self.program["u_lid_bounds"].value = (
            float(lid_min[0]),
            float(lid_min[1]),
            float(lid_max[0]),
            float(lid_max[1]),
        )
        self.program["u_aspect"].value = render_aspect
        self.program["u_eye_width_metric"].value = (
            float(eye_width) / max(float(h), 1.0)
        )
        self.program["u_size"].value = float(np.clip(size, 0.25, 4.0))

        self.vao.render(mode=moderngl.TRIANGLES)

        # Two half-float channels instead of RGBA float32: 1/4 the readback bytes.
        data = target.fbo.read(
            components=2,
            alignment=1,
            dtype="f2",
        )
        fx_small = np.frombuffer(data, dtype=np.float16).reshape(sh, sw, 2)
        fx_small = fx_small.astype(np.float32)

        fx = cv2.resize(
            fx_small,
            (w, h),
            interpolation=cv2.INTER_CUBIC,
        )
        return np.clip(fx, 0.0, 1.0).astype(np.float32, copy=False)

    def release(self):
        for target in self.targets.values():
            for obj in (
                target.fbo,
                target.mask_tex,
                target.lid_tex,
                target.out_tex,
            ):
                try:
                    obj.release()
                except Exception:
                    pass
        self.targets.clear()

        for obj in (self.vao, self.vbo, self.program, self.ctx):
            try:
                obj.release()
            except Exception:
                pass
