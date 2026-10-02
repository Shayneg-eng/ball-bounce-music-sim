"""
render.py

Renders the ball-bounce choreography (from simulate.py) to an MP4 by
drawing each frame with Pillow and piping raw RGB frames into ffmpeg. A
vertical camera follows the ball's long-term descent (panels scroll by),
while being heavily smoothed over time so it does NOT cancel out the
ball's short-term parabolic motion -- otherwise gravity would be
mathematically present but invisible, since a camera that tracks the ball
exactly, frame by frame, pins it to a fixed pixel position forever. Each
panel dims once it has been hit, since it's never used again.
"""

import math
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from simulate import position_at, BALL_RADIUS


def _lerp(a, b, t):
    return a + (b - a) * t


def _draw_background(draw, width, height, t):
    top = (18, 18, 30)
    bottom = (8, 8, 16)
    for y in range(0, height, 4):
        f = y / height
        r = int(_lerp(top[0], bottom[0], f))
        g = int(_lerp(top[1], bottom[1], f))
        b = int(_lerp(top[2], bottom[2], f))
        draw.rectangle([0, y, width, y + 4], fill=(r, g, b))


def _compute_camera_track(ball_y, fps, height, camera_frac, smooth_seconds):
    """Zero-phase smoothing of the ball's vertical position, so the camera
    follows the overall downward trend without cancelling the visible
    parabolic motion of any individual bounce. Since rendering is offline
    we can smooth using the whole trajectory (including "future" samples),
    which avoids the lag a live/causal filter would introduce."""
    window = max(1, int(round(smooth_seconds * fps)))
    if window % 2 == 0:
        window += 1
    if window <= 1 or len(ball_y) <= window:
        smoothed = ball_y.copy()
    else:
        kernel = np.ones(window) / window
        pad = window // 2
        padded = np.pad(ball_y, (pad, pad), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
    cam = smoothed - height * camera_frac
    # keep the camera from peeking above the very start of the world
    cam = np.maximum(cam, -height * 0.25)
    return cam


def render_video(segments, panels, gravity, duration, width, height, fps,
                  out_path, camera_frac=0.42, ball_radius=BALL_RADIUS,
                  camera_smooth_seconds=1.1, fmt_progress=True):
    total_frames = int(math.ceil(duration * fps))

    # --- pass 1: precompute the ball's raw trajectory, then derive a
    # smoothed camera track from it ---
    raw_by = np.empty(total_frames)
    raw_bx = np.empty(total_frames)
    for frame_idx in range(total_frames):
        t = frame_idx / fps
        bx, by, _, _ = position_at(segments, t, gravity)
        raw_bx[frame_idx] = bx
        raw_by[frame_idx] = by
    cam_track = _compute_camera_track(raw_by, fps, height, camera_frac, camera_smooth_seconds)

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "medium", "-crf", "20",
            out_path,
        ],
        stdin=subprocess.PIPE,
    )

    trail = []  # recent (x, y) screen-space positions for a motion trail

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        bx, by = raw_bx[frame_idx], raw_by[frame_idx]
        cam_y = cam_track[frame_idx]

        img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(img, "RGBA")
        _draw_background(draw, width, height, t)

        screen_x = bx
        screen_y = by - cam_y

        # --- panels ---
        for p in panels:
            py = p.y - cam_y
            if py < -150 or py > height + 150:
                continue
            hit = t >= p.hit_time
            appear_lead = 0.6
            spawn_t = p.hit_time - appear_lead
            if t < spawn_t:
                continue
            appear_f = min(1.0, (t - spawn_t) / appear_lead) if not hit else 1.0
            scale = 0.4 + 0.6 * appear_f

            half = (p.length * scale) / 2.0
            dx = math.cos(p.angle) * half
            dy = math.sin(p.angle) * half
            cx, cy = p.x, py

            color = p.color
            if hit:
                fade = min(1.0, (t - p.hit_time) / 2.5)
                dim = 0.55 - 0.25 * fade
                color = tuple(int(c * dim + 30 * (1 - dim)) for c in color)
                alpha = max(70, int(255 * (1 - 0.5 * fade)))
            else:
                alpha = 255

            width_px = 10 if not hit else 7
            draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)],
                      fill=(*color, alpha), width=width_px)

            if hit and (t - p.hit_time) < 0.12:
                flash_f = 1.0 - (t - p.hit_time) / 0.12
                r = 40 * flash_f
                draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                             fill=(255, 255, 255, int(160 * flash_f)))

        # --- motion trail ---
        trail.append((screen_x, screen_y))
        if len(trail) > 14:
            trail.pop(0)
        for i, (tx, ty) in enumerate(trail[:-1]):
            f = (i + 1) / len(trail)
            r = ball_radius * 0.5 * f
            draw.ellipse([tx - r, ty - r, tx + r, ty + r],
                         fill=(255, 255, 255, int(60 * f)))

        # --- ball ---
        r = ball_radius
        draw.ellipse([screen_x - r, screen_y - r, screen_x + r, screen_y + r],
                     fill=(255, 255, 255, 255), outline=(255, 210, 90, 255), width=3)

        ffmpeg.stdin.write(img.tobytes())

        if fmt_progress and frame_idx % max(1, fps) == 0:
            sys.stderr.write(f"\rrendering frame {frame_idx}/{total_frames} (t={t:5.1f}s)")
            sys.stderr.flush()

    if fmt_progress:
        sys.stderr.write("\n")
    ffmpeg.stdin.close()
    ffmpeg.wait()
    if ffmpeg.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {ffmpeg.returncode}")
