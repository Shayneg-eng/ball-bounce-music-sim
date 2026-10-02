#!/usr/bin/env python3
"""
BEAT DROP - a rhythm-synchronized ball drop simulation in an infinite 2D world.

How it works
------------
1. Pick a WAV file (Tkinter dialog, or pass a path on the command line).
2. Librosa extracts BPM, beat timestamps and onsets. Beats are quantized to the
   tempo grid and lightly humanized; onsets that collide with beats are dropped.
   If anything fails, a synthetic 120 BPM rhythm (with a generated click track)
   is used instead.
3. BUILDING PHASE: a physics lookahead predicts where the ball will be at every
   upcoming beat and places an angled platform there. The ONLY thing chosen is
   each platform's angle - the ball's motion is never adjusted. Bad placements
   (miss, out of bounds, stall, overlap) count as failures and rewind the
   builder to the previous successful hit, which is retried with slightly
   different angles (i.e. slightly different outgoing velocities). Too many
   failures without progress regenerate the whole course from scratch.
4. SIMULATION PHASE: the music plays and the ball falls through the course
   under plain gravity, air drag and one collision law shared by every surface.
   Physics is stepped against the playback clock, so each contact lands on its
   beat - it looks like happenstance because nothing is ever nudged.

Controls
--------
    SPACE  pause / resume          R  regenerate the course
    [ / ]  shift audio sync -/+5ms ESC quit

Usage
-----
    python beat_drop.py                 # opens a file dialog
    python beat_drop.py song.wav        # skip the dialog
    python beat_drop.py --synthetic     # skip audio, use the default rhythm
    python beat_drop.py song.wav --latency 60   # audio output delay in ms (default 30)

Requirements: pygame, librosa, numpy (tkinter ships with most Python installs).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import tempfile
import time
import wave
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pygame

try:
    import librosa  # noqa: F401
    HAVE_LIBROSA = True
    LIBROSA_ERROR = ""
except Exception as _exc:  # pragma: no cover - depends on the environment
    # Common on brand-new Python versions (numba lags behind). The game still
    # works: a built-in numpy beat tracker takes over.
    HAVE_LIBROSA = False
    LIBROSA_ERROR = f"{type(_exc).__name__}: {_exc}"[:120]


# =============================================================================
# Configuration
# =============================================================================
SCREEN_W, SCREEN_H = 1280, 720
TARGET_FPS = 120
DEFAULT_LATENCY_MS = 30.0  # typical mixer buffer + device output delay

# Physics (world units are pixels, time in seconds, +y points down)
PHYS_DT = 1.0 / 240.0
GRAVITY = 1500.0           # re-tuned per song by configure_physics()
ARC_SCALE = 375.0          # GRAVITY * beat_period^2: keeps arcs the same size on screen
AIR_DRAG = 0.12            # linear drag coefficient per second
MAX_SPEED = 5000.0         # safety cap only
BALL_R = 14.0
RESTITUTION = 0.60         # one collision law for every surface, beats included
FRICTION = 0.03            # tangential loss on contact

# World (vast, walled at the sides, floor very far below)
WORLD_HALF_W = 50000.0     # effectively endless; the camera follows the ball
WORLD_TOP = -20000.0
WORLD_FLOOR = 2000000.0

# Platforms
PLAT_LEN = 120.0
PLAT_THICK = 12.0
MIN_PLAT_SPACING = 150.0
PATH_CLEARANCE = BALL_R + PLAT_THICK / 2 + 6.0
MAX_TILT = math.radians(50)
MAX_PLATFORMS = 600
PALETTE = [
    (255, 94, 98), (255, 167, 38), (255, 221, 87), (102, 230, 140),
    (64, 196, 255), (124, 120, 255), (214, 102, 255), (255, 112, 190),
]

# Rhythm
MIN_TARGET_GAP = 0.30      # seconds between two consecutive target hits
MAX_TARGET_GAP = 1.20      # longer gaps get filled with grid beats
FIRST_DROP_BEATS = 1.0     # the ball is released one beat before the first hit
QUANTIZE_STRENGTH = 0.5    # 0 = detected attacks, 1 = hard local tempo grid
HUMANIZE_SIGMA = 0.004     # seconds
HUMANIZE_CLIP = 0.010

# Rhythmic collision
TIMING_TOL = 0.012         # bounce fires within this window of the beat
MISS_WINDOW = 0.15         # seconds past the beat that count as a miss

# Builder (aesthetic preferences only - they never alter the physics)
PLAN_DRAWS = 90            # platform angles tried per beat
GOOD_ENOUGH = 6            # stop after this many valid angles, keep the best
DESCENT_PER_SEC = 220.0    # preferred average drop rate of the course
CRUISE_SPEED = 750.0       # preferred ball speed at impact
FAIL_THRESHOLD = 45        # failures without new progress before regenerating
BUILD_BUDGET_S = 0.008     # build time per frame (keeps the window responsive)
SIM_FAIL_THRESHOLD = 8     # live misses before the course is rebuilt

BG_TOP = (12, 12, 24)
BG_BOTTOM = (22, 16, 40)
GRID_COLOR = (32, 30, 56)
TEXT = (230, 232, 245)
DIM_TEXT = (150, 152, 180)


# =============================================================================
# Rhythm analysis
# =============================================================================
@dataclass
class RhythmData:
    bpm: float
    beats: np.ndarray          # processed primary beats (seconds)
    onsets: np.ndarray         # secondary onsets that survived filtering
    targets: np.ndarray        # merged, gap-limited hit times for platforms
    duration: float
    audio_path: str | None
    synthetic: bool
    source_name: str
    temp_audio: bool = False   # audio_path is a temp file we created
    analyzer: str = "demo"     # how the beats were found


def quantize_and_humanize(raw_beats: np.ndarray, bpm: float, rng: np.random.Generator) -> np.ndarray:
    """Pull each beat toward a LOCAL tempo grid (a line fitted through its
    neighbours, so songs that speed up or slow down stay aligned), then add a
    few ms of human jitter."""
    if len(raw_beats) == 0:
        return raw_beats
    b = np.sort(np.asarray(raw_beats, float))
    period = 60.0 / bpm
    if len(b) >= 5:
        k = np.arange(len(b), dtype=float)
        pred = np.empty_like(b)
        for i in range(len(b)):
            lo, hi = max(0, i - 4), min(len(b), i + 5)
            slope, icpt = np.polyfit(k[lo:hi], b[lo:hi], 1)
            pred[i] = icpt + slope * i
        q = b + (pred - b) * QUANTIZE_STRENGTH
    else:
        q = b.copy()
    jitter = np.clip(rng.normal(0.0, HUMANIZE_SIGMA, len(q)), -HUMANIZE_CLIP, HUMANIZE_CLIP)
    q = np.sort(np.maximum(q + jitter, 0.0))
    # Collapse anything that landed on the same slot.
    keep = [q[0]]
    for t in q[1:]:
        if t - keep[-1] > period * 0.5:
            keep.append(t)
    return np.array(keep)


def filter_onsets(onsets: np.ndarray, strengths: np.ndarray, beats: np.ndarray, period: float) -> np.ndarray:
    """Keep strong onsets that do not conflict with a primary beat."""
    if len(onsets) == 0 or len(beats) == 0:
        return np.array([])
    conflict = max(0.09, period * 0.2)
    idx = np.searchsorted(beats, onsets)
    left = beats[np.clip(idx - 1, 0, len(beats) - 1)]
    right = beats[np.clip(idx, 0, len(beats) - 1)]
    nearest = np.minimum(np.abs(onsets - left), np.abs(onsets - right))
    strong = strengths >= np.percentile(strengths, 75) if len(strengths) else np.ones_like(onsets, bool)
    return onsets[(nearest > conflict) & strong]


def build_targets(beats: np.ndarray, onsets: np.ndarray, period: float, duration: float) -> np.ndarray:
    """Merge beats and onsets into a hit list with sane gaps."""
    events = sorted([(float(b), 0) for b in beats] + [(float(o), 1) for o in onsets])
    out: list[float] = []
    for t, _kind in events:
        if t < 0.35 or t > duration - 0.05:
            continue
        if out and t - out[-1] < MIN_TARGET_GAP:
            continue
        # Fill long silences with grid beats so the ball never free-falls forever.
        while out and t - out[-1] > MAX_TARGET_GAP:
            out.append(out[-1] + max(period, MIN_TARGET_GAP))
        out.append(t)
    return np.array(out)


# ---------------------------------------------------------------------------
# Decoding. The user's song ALWAYS plays; only the rhythm analysis can fall back.
# ---------------------------------------------------------------------------
def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Dependency-free WAV reader: 8/16/24/32-bit int PCM, 32/64-bit float,
    WAVE_FORMAT_EXTENSIBLE. Returns float32 audio shaped (channels, samples)."""
    import struct
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] not in (b"RIFF", b"RF64") or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
            if fmt[0] == 0xFFFE and len(body) >= 26:          # extensible: real tag in the GUID
                fmt = (struct.unpack("<H", body[24:26])[0],) + fmt[1:]
        elif cid == b"data":
            if size == 0xFFFFFFFF or pos + 8 + size > len(data):  # RF64 / truncated
                body = data[pos + 8:]
            raw = body
        pos += 8 + size + (size & 1)
    if fmt is None or raw is None:
        raise ValueError("missing fmt or data chunk")
    tag, channels, sr, _, _, bits = fmt
    width = bits // 8
    raw = raw[: len(raw) - len(raw) % (width * channels)]
    if tag == 3 and bits in (32, 64):
        y = np.frombuffer(raw, dtype="<f4" if bits == 32 else "<f8").astype(np.float32)
    elif tag == 1 and bits == 8:
        y = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif tag == 1 and bits == 16:
        y = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bits == 24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        y = (np.where(v >= 1 << 23, v - (1 << 24), v)).astype(np.float32) / float(1 << 23)
    elif tag == 1 and bits == 32:
        y = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"unsupported WAV encoding (format {tag}, {bits}-bit)")
    y = y.reshape(-1, channels).T
    return np.nan_to_num(y), int(sr)


def decode_audio(path: str) -> tuple[np.ndarray, int, str]:
    """Decode any supported file to float32 (channels, samples)."""
    errors = []
    try:
        y, sr = read_wav(path)
        return y, sr, "built-in WAV reader"
    except Exception as exc:
        errors.append(f"wav reader: {exc}")
    try:
        import soundfile as sf
        y, sr = sf.read(path, dtype="float32", always_2d=True)
        return y.T, int(sr), "soundfile"
    except Exception as exc:
        errors.append(f"soundfile: {exc}")
    if HAVE_LIBROSA:
        try:
            y, sr = librosa.load(path, sr=None, mono=False)
            return np.atleast_2d(y).astype(np.float32), int(sr), "librosa"
        except Exception as exc:
            errors.append(f"librosa: {exc}")
    raise RuntimeError("could not decode audio - " + "; ".join(errors))


def write_pcm16_wav(y: np.ndarray, sr: int) -> str:
    """Write float audio shaped (channels, samples) to a temp 16-bit WAV that
    SDL_mixer is guaranteed to play."""
    y = np.atleast_2d(y)[:2]
    if y.shape[0] == 1:
        y = np.vstack([y, y])
    peak = float(np.abs(y).max()) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    pcm = (np.clip(y.T, -1.0, 1.0) * 32767).astype("<i2")
    fd, out = tempfile.mkstemp(prefix="beatdrop_", suffix=".wav")
    os.close(fd)
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return out


# ---------------------------------------------------------------------------
# Built-in beat tracker (numpy only) - used when librosa is unavailable/fails.
# ---------------------------------------------------------------------------
def numpy_onset_envelope(y: np.ndarray, sr: int, hop: int = 512, n_fft: int = 2048):
    """Log-spectral-flux onset strength. Returns (envelope, frame_rate)."""
    if sr > 24000:  # light decimation keeps it fast; beats live well below 11 kHz
        k = int(sr // 22050) or 1
        y = y[: len(y) - len(y) % k].reshape(-1, k).mean(axis=1)
        sr = sr / k
    hop = max(128, int(round(hop * sr / 22050)))
    n_fft = max(512, int(round(n_fft * sr / 22050)))
    if len(y) < n_fft * 2:
        raise RuntimeError("audio too short to analyse")
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop]
    win = np.hanning(n_fft).astype(np.float32)
    env = np.empty(len(frames), dtype=np.float32)
    prev = None
    for i0 in range(0, len(frames), 1024):          # chunked to bound memory
        mag = np.log1p(40.0 * np.abs(np.fft.rfft(frames[i0:i0 + 1024] * win, axis=1)))
        if prev is not None:
            mag = np.vstack([prev, mag])
        flux = np.maximum(0.0, np.diff(mag, axis=0)).mean(axis=1)
        if prev is None:
            flux = np.concatenate([[0.0], flux])
        env[i0:i0 + len(flux)] = flux
        prev = mag[-1:]
    fr = sr / hop
    # Remove slow loudness changes so only attacks remain.
    w = max(3, int(fr * 0.5))
    env = np.maximum(0.0, env - np.convolve(env, np.ones(w) / w, mode="same"))
    return env / (env.max() or 1.0), fr


def numpy_beat_track(env: np.ndarray, fr: float):
    """Tempo by autocorrelation, then beat positions by dynamic programming."""
    n = len(env)
    ac = np.correlate(env, env, mode="full")[n - 1:]
    bpms = np.arange(60.0, 200.0, 0.25)
    lags = 60.0 * fr / bpms
    score = np.interp(lags, np.arange(n), ac)
    score += 0.5 * np.interp(lags * 2, np.arange(n), ac)        # reward metrical consistency
    score *= np.exp(-0.5 * (np.log2(bpms / 120.0) / 0.9) ** 2)  # mild prior toward ~120
    bpm = float(bpms[int(np.argmax(score))])
    period = 60.0 * fr / bpm
    # Ellis-style DP: each beat prefers to sit one period after the previous one.
    cum = env.astype(np.float64).copy()
    back = np.full(n, -1)
    lo, hi = int(round(period * 0.5)), int(round(period * 2.0))
    for i in range(hi, n):
        prev = np.arange(i - hi, i - lo + 1)
        pen = -100.0 * (np.log((i - prev) / period)) ** 2 * 0.01
        cand = cum[prev] + pen * env.max() * 4
        j = int(np.argmax(cand))
        cum[i] = env[i] + cand[j]
        back[i] = prev[j]
    tail = cum[max(0, n - int(period)):]
    i = n - int(period) + int(np.argmax(tail)) if n > period else int(np.argmax(cum))
    beats = []
    while i >= 0:
        beats.append(i)
        i = back[i]
    beats = np.array(beats[::-1], dtype=float) / fr
    return bpm, beats


def numpy_onsets(env: np.ndarray, fr: float):
    """Peak-pick the onset envelope. Returns (times, strengths)."""
    k = max(1, int(fr * 0.05))
    padded = np.pad(env, k, mode="edge")
    local_max = np.lib.stride_tricks.sliding_window_view(padded, 2 * k + 1).max(axis=1)
    thr = np.mean(env) + 0.8 * np.std(env)
    idx = np.where((env >= local_max) & (env > thr))[0]
    keep = []
    for i in idx:
        if not keep or (i - keep[-1]) / fr > 0.1:
            keep.append(i)
    keep = np.array(keep, dtype=int)
    return keep / fr, env[keep] if len(keep) else np.array([])


def attack_envelopes(y: np.ndarray, sr: int, hop_s: float = 0.005):
    """Fine (5 ms) attack envelopes: broadband and low band (kick/bass).
    Each is the positive slope of log energy, which peaks right at a hit's attack."""
    hop = max(1, int(sr * hop_s))
    n = len(y) // hop
    frames = y[: n * hop].reshape(n, hop)
    fr = sr / hop

    def slope(energy):
        e = np.log(energy + 1e-9)
        d = np.maximum(0.0, np.diff(e, prepend=e[0]))
        d = np.convolve(d, np.array([0.25, 0.5, 0.25]), mode="same")
        return d / (d.max() or 1.0)

    broad = slope(np.sqrt((frames ** 2).mean(axis=1)))
    # Low band: block-average to ~2 kHz, then FFT low-pass at 160 Hz.
    k = max(1, sr // 2000)
    y2 = y[: len(y) // k * k].reshape(-1, k).mean(axis=1)
    sr2 = sr / k
    spec = np.fft.rfft(y2)
    spec[int(160.0 / (sr2 / 2) * (len(spec) - 1)):] = 0.0
    low = np.fft.irfft(spec, n=len(y2))
    hop2 = max(1, int(round(sr2 * hop_s)))
    n2 = min(n, len(low) // hop2)
    low_e = np.sqrt((low[: n2 * hop2].reshape(n2, hop2) ** 2).mean(axis=1))
    low_env = np.zeros(n)
    low_env[:n2] = slope(low_e)
    return broad, low_env, fr


def refine_beats(y: np.ndarray, sr: int, beats: np.ndarray, period: float) -> tuple[np.ndarray, str]:
    """Fix the two classic tracker mistakes: locking onto the off-beat, and a
    constant timing lag. Returns (beats, note)."""
    if len(beats) < 4:
        return beats, ""
    broad, low, fr = attack_envelopes(y, sr)
    combo = 0.6 * low + 0.4 * broad
    n = len(combo)
    duration = len(y) / sr

    def strength(times, env, win=0.03):
        w = int(win * fr)
        out = []
        for t in times:
            i = int(t * fr)
            if w <= i < n - w:
                out.append(env[i - w:i + w + 1].max())
        return float(np.mean(out)) if out else 0.0

    # Phase search: slide the whole beat grid across one period and keep the
    # offset where the kick/bass attacks are strongest (beats live on the low end).
    note = ""
    base = strength(beats, low, 0.015)
    best_phi, best = 0.0, base
    for phi in np.linspace(-period / 2, period / 2, 25):
        sc = strength(beats + phi, low, 0.015)
        if sc > best:
            best_phi, best = phi, sc
    if best > 1.15 * base + 1e-6 and abs(best_phi) > 0.02:
        beats = beats + best_phi
        beats = beats[(beats > 0.0) & (beats < duration - 0.05)]
        note = f"phase {best_phi * 1000:+.0f} ms"
    return snap_to_attacks(beats, combo, fr), note


def snap_to_attacks(times: np.ndarray, combo: np.ndarray, fr: float, win: float = 0.06) -> np.ndarray:
    """Move each time to the clear attack within +-win seconds, if there is one."""
    n = len(combo)
    w = int(win * fr)
    out = np.asarray(times, float).copy()
    for j, t in enumerate(out):
        i = int(round(t * fr))
        lo, hi = max(0, i - w), min(n, i + w + 1)
        if hi - lo < 3:
            continue
        seg = combo[lo:hi]
        k = int(np.argmax(seg))
        if seg[k] > 0.15:
            out[j] = (lo + k) / fr
    return np.sort(out)


def analyze_rhythm(y_mono: np.ndarray, sr: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, str]:
    """Return (bpm, raw_beats, onset_times, onset_strengths, method)."""
    problems = []
    if HAVE_LIBROSA:
        try:
            y = librosa.resample(y_mono, orig_sr=sr, target_sr=22050) if sr != 22050 else y_mono
            oenv = librosa.onset.onset_strength(y=y, sr=22050)
            tempo, beat_frames = librosa.beat.beat_track(onset_envelope=oenv, sr=22050)
            bpm = float(np.atleast_1d(tempo)[0])
            beats = librosa.frames_to_time(beat_frames, sr=22050)
            if not np.isfinite(bpm) or bpm <= 0 or len(beats) < 4:
                raise RuntimeError("librosa found no steady beat")
            of = librosa.onset.onset_detect(onset_envelope=oenv, sr=22050, units="frames")
            ons = librosa.frames_to_time(of, sr=22050)
            strengths = oenv[np.clip(of, 0, len(oenv) - 1)] if len(of) else np.array([])
            return bpm, np.asarray(beats, float), np.asarray(ons, float), np.asarray(strengths, float), "librosa"
        except Exception as exc:
            problems.append(f"librosa failed: {exc}")
            print(f"[rhythm] librosa analysis failed: {exc!r} - using the built-in beat tracker")
    else:
        problems.append(f"librosa unavailable ({LIBROSA_ERROR})")
    env, fr = numpy_onset_envelope(y_mono, sr)
    bpm, beats = numpy_beat_track(env, fr)
    ons, strengths = numpy_onsets(env, fr)
    if len(beats) < 4:
        raise RuntimeError("no steady beat found")
    return bpm, beats, ons, strengths, "built-in tracker" + (f" ({problems[0]})" if problems else "")


def load_song(path: str, seed: int) -> RhythmData:
    """Decode the user's file, analyse its rhythm, prepare it for playback.
    Raises only if the file cannot be decoded at all."""
    y, sr, decoder = decode_audio(path)
    if y.shape[-1] < sr:
        raise RuntimeError("audio file is shorter than one second")
    duration = float(y.shape[-1] / sr)
    playback = write_pcm16_wav(y, sr)
    mono = y.mean(axis=0).astype(np.float32)
    rng = np.random.default_rng(seed)
    try:
        bpm, raw_beats, ons, strengths, method = analyze_rhythm(mono, sr)
        raw_beats, fix = refine_beats(mono, sr, np.asarray(raw_beats, float), 60.0 / bpm)
        if len(ons):
            broad, low, fr_a = attack_envelopes(mono, sr)
            order = np.argsort(ons)
            ons = snap_to_attacks(np.asarray(ons, float)[order], 0.6 * low + 0.4 * broad, fr_a, 0.04)
            strengths = np.asarray(strengths, float)[order]
        if len(raw_beats) >= 4:
            ibi = float(np.median(np.diff(raw_beats)))
            if ibi > 0:
                bpm = 60.0 / ibi
        if fix:
            method += f", {fix}"
    except Exception as exc:
        # Last resort: keep the music, use a steady 120 BPM grid for the course.
        print(f"[rhythm] analysis failed entirely ({exc!r}); using a steady grid")
        bpm, raw_beats = 120.0, np.arange(0.5, duration - 0.1, 0.5)
        ons, strengths, method = np.array([]), np.array([]), f"steady grid (analysis failed: {exc})"
    period = 60.0 / bpm
    beats = quantize_and_humanize(raw_beats, bpm, rng)
    onsets = filter_onsets(ons, strengths, beats, period)
    targets = build_targets(beats, onsets, period, duration)
    if len(targets) < 4:
        bpm, period = 120.0, 0.5
        beats = quantize_and_humanize(np.arange(0.5, duration - 0.1, 0.5), bpm, rng)
        onsets = np.array([])
        targets = build_targets(beats, onsets, period, duration)
        method += " + grid fallback"
    print(f"[audio] decoded with {decoder}: {sr} Hz, {y.shape[0]} ch, {duration:.1f}s")
    return RhythmData(bpm, beats, onsets, targets, duration, playback, False, os.path.basename(path),
                      temp_audio=True, analyzer=method)


def synthesize_click_track(beats: np.ndarray, offbeats: np.ndarray, duration: float, sr: int = 44100) -> str:
    """Render a simple kick/snare/hat track to a temp WAV and return its path."""
    n = int(duration * sr)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(7)

    def add(sample: np.ndarray, t: float):
        i = int(t * sr)
        if 0 <= i < n:
            j = min(n, i + len(sample))
            out[i:j] += sample[: j - i]

    kt = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * np.cumsum(50 + 110 * np.exp(-kt * 30)) / sr) * np.exp(-kt * 14)
    st = np.arange(int(0.18 * sr)) / sr
    snare = (rng.uniform(-1, 1, len(st)) * 0.6 + np.sin(2 * np.pi * 190 * st) * 0.4) * np.exp(-st * 22)
    ht = np.arange(int(0.05 * sr)) / sr
    hat = rng.uniform(-1, 1, len(ht)) * np.exp(-ht * 90) * 0.35
    for i, b in enumerate(beats):
        add(kick * 0.9, b)
        if i % 2 == 1:
            add(snare * 0.5, b)
    for o in offbeats:
        add(hat, o)
    out = np.clip(out / max(1.0, np.abs(out).max()) * 0.9, -1, 1)
    pcm = (out * 32767).astype(np.int16)
    fd, path = tempfile.mkstemp(prefix="beatdrop_", suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def make_synthetic_rhythm(seed: int, bpm: float = 120.0, duration: float = 90.0) -> RhythmData:
    rng = np.random.default_rng(seed)
    period = 60.0 / bpm
    raw = np.arange(0.5, duration - 0.5, period)
    beats = quantize_and_humanize(raw, bpm, rng)
    offbeats = beats[:-1] + period / 2
    # Only a sparse subset of off-beats become extra platform targets.
    extra = offbeats[rng.random(len(offbeats)) < 0.15]
    onsets = filter_onsets(extra, np.ones(len(extra)), beats, period)
    targets = build_targets(beats, onsets, period, duration)
    path = None
    try:
        path = synthesize_click_track(beats, offbeats, duration)
    except Exception as exc:
        print(f"[audio] could not write synthetic click track: {exc}")
    return RhythmData(bpm, beats, onsets, targets, duration, path, True, f"DEMO metronome {bpm:.0f} BPM",
                      temp_audio=path is not None, analyzer="demo metronome (no song loaded)")


def choose_audio_file() -> str | None:
    """Show a Tkinter open-file dialog. Returns None if unavailable or cancelled."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        print("[ui] tkinter is not available - falling back to the synthetic rhythm")
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Choose a WAV file for Beat Drop",
            filetypes=[("WAV audio", "*.wav *.WAV"), ("All files", "*.*")],
        )
        root.destroy()
        return path or None
    except Exception as exc:
        print(f"[ui] file dialog failed: {exc}")
        return None


# =============================================================================
# Audio clock
# =============================================================================
class AudioClock:
    """Tracks precise playback time.

    Starts the instant pygame.mixer.music.play() is called and runs on
    perf_counter (smooth, monotonic, pause-aware). `latency` shifts the visuals
    later to match the audio output delay; tweak it live with [ and ].
    (mixer.music.get_pos() is too coarse and platform-dependent to steer by.)
    In manual mode (self-test) time advances only through advance().
    """

    def __init__(self, has_audio: bool, latency: float = 0.0, manual: bool = False):
        self.has_audio = has_audio
        self.latency = latency
        self.manual = manual
        self.reset()

    def reset(self):
        self.running = False
        self.paused = False
        self._start = 0.0
        self._pause_began = 0.0
        self._manual_t = 0.0

    def start(self):
        self.reset()
        if self.has_audio:
            try:
                pygame.mixer.music.play()
            except pygame.error as exc:
                print(f"[audio] playback failed: {exc}")
                self.has_audio = False
        self._start = time.perf_counter()
        self.running = True

    def stop(self):
        self.running = False
        if self.has_audio:
            try:
                pygame.mixer.music.stop()
            except pygame.error:
                pass

    def set_paused(self, paused: bool):
        if paused == self.paused or not self.running:
            return
        self.paused = paused
        now = time.perf_counter()
        if paused:
            self._pause_began = now
            if self.has_audio:
                pygame.mixer.music.pause()
        else:
            self._start += now - self._pause_began
            if self.has_audio:
                pygame.mixer.music.unpause()

    def advance(self, dt: float):
        if self.manual and self.running and not self.paused:
            self._manual_t += dt

    def time(self) -> float:
        if not self.running:
            return 0.0
        if self.manual:
            return self._manual_t
        ref = self._pause_began if self.paused else time.perf_counter()
        return ref - self._start - self.latency


# =============================================================================
# Geometry + physics
# =============================================================================
@dataclass
class Platform:
    pid: int
    cx: float
    cy: float
    angle: float
    target_time: float
    contact: tuple[float, float]
    v_in: tuple[float, float]
    v_out: tuple[float, float]
    color: tuple[int, int, int]
    n_in: int = 0              # physics steps of the flight that ends here
    hit_flash: float = 0.0
    hit: bool = False
    nx: float = field(init=False)
    ny: float = field(init=False)
    ax: float = field(init=False)
    ay: float = field(init=False)
    bx: float = field(init=False)
    by: float = field(init=False)

    def __post_init__(self):
        self.nx, self.ny = platform_normal(self.angle)
        tx, ty = math.cos(self.angle), math.sin(self.angle)
        h = PLAT_LEN / 2
        self.ax, self.ay = self.cx - tx * h, self.cy - ty * h
        self.bx, self.by = self.cx + tx * h, self.cy + ty * h

    def corners(self):
        tx, ty = math.cos(self.angle), math.sin(self.angle)
        h, t = PLAT_LEN / 2, PLAT_THICK / 2
        return [
            (self.cx - tx * h + self.nx * t, self.cy - ty * h + self.ny * t),
            (self.cx + tx * h + self.nx * t, self.cy + ty * h + self.ny * t),
            (self.cx + tx * h - self.nx * t, self.cy + ty * h - self.ny * t),
            (self.cx - tx * h - self.nx * t, self.cy - ty * h - self.ny * t),
        ]


def platform_normal(angle: float) -> tuple[float, float]:
    """Upward-facing surface normal of a platform tilted by `angle` radians."""
    return math.sin(angle), -math.cos(angle)


def seg_dist(P: np.ndarray, ax: float, ay: float, bx: float, by: float) -> np.ndarray:
    """Distance from each row of P (n,2) to segment AB."""
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    t = np.clip(((P[:, 0] - ax) * abx + (P[:, 1] - ay) * aby) / denom, 0.0, 1.0)
    dx = P[:, 0] - (ax + t * abx)
    dy = P[:, 1] - (ay + t * aby)
    return np.hypot(dx, dy)


def configure_physics(bpm: float) -> None:
    """Pick one gravity for the whole song so a beat-long arc has a pleasant
    size: slow songs get floaty arcs, fast songs snappy ones. It stays constant
    for the entire track, so the motion is ordinary projectile physics."""
    global GRAVITY
    period = 60.0 / max(40.0, min(240.0, bpm))
    GRAVITY = max(450.0, min(3400.0, ARC_SCALE / (period * period)))


def integrate(s: list[float], dt: float) -> str | None:
    """Advance ball state s=[x,y,vx,vy] one step with gravity, drag and walls.
    Returns 'wall' or 'floor' when a boundary was touched."""
    s[3] += GRAVITY * dt
    drag = 1.0 - AIR_DRAG * dt
    s[2] *= drag
    s[3] *= drag
    sp = math.hypot(s[2], s[3])
    if sp > MAX_SPEED:  # safety net only - never reached on a built course
        k = MAX_SPEED / sp
        s[2] *= k
        s[3] *= k
    s[0] += s[2] * dt
    s[1] += s[3] * dt
    event = None
    if s[0] - BALL_R < -WORLD_HALF_W:
        s[0] = -WORLD_HALF_W + BALL_R
        s[2], s[3] = surface_bounce(s[2], s[3], 1.0, 0.0)
        event = "wall"
    elif s[0] + BALL_R > WORLD_HALF_W:
        s[0] = WORLD_HALF_W - BALL_R
        s[2], s[3] = surface_bounce(s[2], s[3], -1.0, 0.0)
        event = "wall"
    if s[1] + BALL_R > WORLD_FLOOR:
        s[1] = WORLD_FLOOR - BALL_R
        s[2], s[3] = surface_bounce(s[2], s[3], 0.0, -1.0)
        event = "floor"
    return event


def surface_bounce(vx: float, vy: float, nx: float, ny: float) -> tuple[float, float]:
    """The single collision law used for EVERY contact (beat hits included):
    reflect the normal component with RESTITUTION, keep the tangential
    component minus a little FRICTION. No extra impulses, no speed shaping."""
    vn = vx * nx + vy * ny
    if vn >= 0:
        return vx, vy
    tx, ty = vx - vn * nx, vy - vn * ny
    return (tx * (1 - FRICTION) - vn * RESTITUTION * nx,
            ty * (1 - FRICTION) - vn * RESTITUTION * ny)


def collide_platform(s: list[float], p: Platform) -> bool:
    """Resolve circle-vs-platform contact with the shared collision law."""
    abx, aby = p.bx - p.ax, p.by - p.ay
    t = max(0.0, min(1.0, ((s[0] - p.ax) * abx + (s[1] - p.ay) * aby) / (abx * abx + aby * aby)))
    qx, qy = p.ax + abx * t, p.ay + aby * t
    dx, dy = s[0] - qx, s[1] - qy
    dist = math.hypot(dx, dy)
    reach = BALL_R + PLAT_THICK / 2
    if dist >= reach or dist < 1e-6:
        return False
    nx, ny = dx / dist, dy / dist
    s[0] += nx * (reach - dist)
    s[1] += ny * (reach - dist)
    s[2], s[3] = surface_bounce(s[2], s[3], nx, ny)
    return True


# =============================================================================
# Course builder (Building Phase)
# =============================================================================
@dataclass
class Arrival:
    """Predicted ball state at the moment it reaches a target beat."""
    t: float
    x: float
    y: float
    vx: float
    vy: float
    target_idx: int
    path: np.ndarray       # incoming flight samples
    steps: int             # physics steps of the incoming flight
    hint: float | None = None  # platform angle to perturb after a rewind


class CourseBuilder:
    """Places platforms so that plain, unmodified physics lands the ball on
    every beat. The only free choice at each hit is the platform's angle; the
    ball's speed always follows from gravity, drag and the collision law."""

    def __init__(self, targets: np.ndarray, seed: int, bpm: float):
        self.targets = targets
        self.period = 60.0 / bpm
        self.seed = seed
        self.rebuilds = 0
        self.total_failures = 0
        self.rewinds = 0
        self._restart(seed)

    # ---- lifecycle ---------------------------------------------------------
    def _restart(self, seed: int):
        self.rng = random.Random(seed)
        self.failures = 0
        self.best_depth = 0
        self.platforms: list[Platform] = []
        self.done = False
        self.spawn = (self.rng.uniform(-250, 250), 0.0)
        # First hit: the first target at least one beat in, so the opening drop
        # is a full beat of free fall from rest.
        drop = self.period * FIRST_DROP_BEATS
        i0 = int(np.searchsorted(self.targets, drop - 1e-6))
        i0 = min(i0, len(self.targets) - 1)
        first = float(self.targets[i0])
        self.release_time = max(0.0, first - drop)
        path, st, n = self.lookahead(self.spawn[0], self.spawn[1], 0.0, 0.0, first - self.release_time)
        self.stack = [Arrival(first, st[0], st[1], st[2], st[3], i0, path, n)]
        self._cache_dirty = True
        self.last_total = min(len(self.targets) - i0, MAX_PLATFORMS)

    def regenerate(self):
        self.rebuilds += 1
        self.seed = self.rng.randrange(1 << 30)
        self._restart(self.seed)

    @property
    def progress(self) -> float:
        return len(self.platforms) / max(1, self.last_total)

    @property
    def head(self) -> tuple[float, float]:
        a = self.stack[-1]
        return a.x, a.y

    # ---- physics lookahead ---------------------------------------------------
    @staticmethod
    def lookahead(x, y, vx, vy, duration):
        """Simulate free flight for `duration` seconds at the physics rate.
        Returns (samples, final_state, steps); final_state[4] flags a boundary hit."""
        steps = max(1, int(round(duration / PHYS_DT)))
        s = [x, y, vx, vy]
        pts = np.empty((steps + 1, 2))
        pts[0] = (x, y)
        boundary = False
        for i in range(1, steps + 1):
            if integrate(s, PHYS_DT):
                boundary = True
            pts[i] = (s[0], s[1])
        return pts, (s[0], s[1], s[2], s[3], boundary), steps

    # ---- validity checks ---------------------------------------------------
    def _past_samples(self) -> np.ndarray:
        if self._cache_dirty:
            parts = [a.path for a in self.stack[:-1]]
            self._samples = np.concatenate(parts) if parts else np.empty((0, 2))
            self._cache_dirty = False
        return self._samples

    def _platform_ok(self, p: Platform, incoming: np.ndarray) -> bool:
        if abs(p.cx) > WORLD_HALF_W - PLAT_LEN or p.cy > WORLD_FLOOR - 300 or p.cy < WORLD_TOP:
            return False
        if self.platforms:
            c = np.array([(q.cx, q.cy) for q in self.platforms[-80:]])
            if np.min(np.hypot(c[:, 0] - p.cx, c[:, 1] - p.cy)) < MIN_PLAT_SPACING:
                return False
        # The new platform must not cut through any earlier flight path.
        tail = max(3, len(incoming) // 10)
        for S in (self._past_samples(), incoming[:-tail]):
            if len(S) == 0:
                continue
            m = (np.abs(S[:, 0] - p.cx) < PLAT_LEN) & (np.abs(S[:, 1] - p.cy) < PLAT_LEN)
            if m.any() and seg_dist(S[m], p.ax, p.ay, p.bx, p.by).min() < PATH_CLEARANCE:
                return False
        return True

    def _path_ok(self, path: np.ndarray, new_plat: Platform) -> bool:
        lo, hi = path.min(axis=0) - PLAT_LEN, path.max(axis=0) + PLAT_LEN
        for q in self.platforms + [new_plat]:
            if not (lo[0] < q.cx < hi[0] and lo[1] < q.cy < hi[1]):
                continue
            S = path[len(path) // 5:] if q is new_plat else path
            if len(S) and seg_dist(S, q.ax, q.ay, q.bx, q.by).min() < PATH_CLEARANCE:
                return False
        return True

    def _landing_ok(self, x: float, y: float, new_plat: Platform) -> bool:
        """The next landing spot must leave room for its own platform."""
        if abs(x) > WORLD_HALF_W - 200 or y > WORLD_FLOOR - 400 or y < WORLD_TOP:
            return False
        need = MIN_PLAT_SPACING + BALL_R + PLAT_THICK
        for q in self.platforms[-40:] + [new_plat]:
            if math.hypot(q.cx - x, q.cy - y) < need:
                return False
        return True

    # ---- one build step ------------------------------------------------------
    def _candidate_angle(self, a: Arrival, draw: int) -> float:
        # After a rewind, first try small tweaks of the previous angle: a slight,
        # physically consistent change to the outgoing velocity.
        if a.hint is not None and draw < 6:
            return max(-MAX_TILT, min(MAX_TILT, a.hint + self.rng.gauss(0.0, math.radians(4))))
        return self.rng.uniform(-MAX_TILT, MAX_TILT)

    def _score(self, a: Arrival, st, gap: float) -> float:
        """Prefer courses that keep drifting down, stay central and zig-zag."""
        want_dy = DESCENT_PER_SEC * gap
        s = -abs(st[1] - a.y - want_dy) / 300.0
        s -= (abs(st[0]) / (WORLD_HALF_W * 0.4)) ** 2 * 2.0
        if st[0] * st[2] > 0:                      # heading further from the centre
            s -= (abs(st[0]) / 6000.0) ** 2
        s -= abs(math.hypot(st[2], st[3]) - CRUISE_SPEED) / 900.0
        return s + self.rng.uniform(0.0, 0.5)

    def step(self) -> None:
        if self.done:
            return
        a = self.stack[-1]
        idx = a.target_idx
        at_cap = len(self.platforms) + 1 >= MAX_PLATFORMS
        # Next hit: normally the next target. A short off-beat onset may be
        # skipped (the ball simply flies through it) if it cannot be reached.
        options = [idx + 1]
        if idx + 2 < len(self.targets) and self.targets[idx + 1] - self.targets[idx] < 0.75 * self.period:
            options.append(idx + 2)
        best = None
        for nxt in options:
            is_last = nxt >= len(self.targets) or at_cap
            gap = 0.8 if is_last else float(self.targets[nxt] - self.targets[idx])
            best = self._search(a, gap, is_last)
            if best is not None:
                break
        if best is None:
            self._register_failure()
            return
        _score, plat, path, st, n = best
        self.platforms.append(plat)
        if len(self.platforms) > self.best_depth:  # new progress
            self.best_depth = len(self.platforms)
            self.failures = 0
        if is_last:
            self.done = True
        else:
            self.stack.append(Arrival(float(self.targets[nxt]), st[0], st[1], st[2], st[3], nxt, path, n))
            self._cache_dirty = True

    def _search(self, a: Arrival, gap: float, is_last: bool):
        """Try platform angles for the hit at `a`; return the best valid one."""
        speed_in = math.hypot(a.vx, a.vy)
        if speed_in < 120.0 or abs(a.x) >= WORLD_HALF_W - 150:
            return None
        off = BALL_R + PLAT_THICK / 2
        best = None
        valid = 0
        for draw in range(PLAN_DRAWS):
            if valid >= GOOD_ENOUGH:
                break
            ang = self._candidate_angle(a, draw)
            nx, ny = platform_normal(ang)
            if a.vx * nx + a.vy * ny > -0.25 * speed_in:
                continue            # ball would not strike the top face
            v_out = surface_bounce(a.vx, a.vy, nx, ny)
            plat = Platform(
                pid=len(self.platforms) + 1, cx=a.x - nx * off, cy=a.y - ny * off, angle=ang,
                target_time=a.t, contact=(a.x, a.y), v_in=(a.vx, a.vy), v_out=v_out,
                color=PALETTE[len(self.platforms) % len(PALETTE)], n_in=a.steps,
            )
            if not self._platform_ok(plat, a.path):
                continue
            path, st, n = self.lookahead(a.x, a.y, v_out[0], v_out[1], gap)
            if st[4] or not self._path_ok(path, plat):
                continue
            if not is_last:
                if st[3] < 60.0:                        # must be coming down onto the next one
                    continue
                if math.hypot(st[2], st[3]) < 150.0:    # ball would stall
                    continue
                if not self._landing_ok(st[0], st[1], plat):
                    continue
            valid += 1
            score = self._score(a, st, gap)
            if best is None or score > best[0]:
                best = (score, plat, path, st, n)
        return best

    def _register_failure(self):
        self.failures += 1
        self.total_failures += 1
        if self.failures > FAIL_THRESHOLD:
            self.regenerate()
            return
        # Rewind to the previous successful hit. Its platform is re-chosen,
        # starting with slight angle tweaks - i.e. slightly different outgoing
        # velocities that still obey the same collision law.
        if len(self.stack) > 1:
            self.stack.pop()
            removed = self.platforms.pop()
            self.stack[-1].hint = removed.angle
            self._cache_dirty = True
            self.rewinds += 1


# =============================================================================
# Camera + effects
# =============================================================================
class Camera:
    def __init__(self):
        self.x, self.y = 0.0, 0.0
        self.svx = self.svy = 0.0

    def snap(self, x, y):
        self.x, self.y = x, y
        self.svx = self.svy = 0.0

    def follow(self, x, y, vx, vy, dt, stiffness=3.0):
        # Lead with a heavily smoothed velocity so bounces never jerk the view
        # (a jerking camera makes the ball itself look like it changes speed).
        a = 1.0 - math.exp(-dt / 0.9)
        self.svx += (vx - self.svx) * a
        self.svy += (vy - self.svy) * a
        lead_x = max(-200.0, min(200.0, self.svx * 0.25))
        lead_y = 90.0 + max(-100.0, min(160.0, self.svy * 0.25))
        k = 1.0 - math.exp(-stiffness * dt)
        self.x += (x + lead_x - self.x) * k
        self.y += (y + lead_y - self.y) * k

    def to_screen(self, x, y):
        return x - self.x + SCREEN_W / 2, y - self.y + SCREEN_H / 2

    def visible(self, x, y, margin=PLAT_LEN):
        return (abs(x - self.x) < SCREEN_W / 2 + margin) and (abs(y - self.y) < SCREEN_H / 2 + margin)


def lerp_color(a, b, t):
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t), int(a[2] + (b[2] - a[2]) * t))


# =============================================================================
# Game
# =============================================================================
class Game:
    def __init__(self, screen: pygame.Surface, rhythm: RhythmData, seed: int, has_audio: bool,
                 latency: float, manual_clock: bool, audio_note: str = ""):
        self.screen = screen
        self.audio_note = audio_note
        self.rhythm = rhythm
        self.seed = seed
        self.font = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 16)
        self.font_small = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 13)
        self.font_big = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 30, bold=True)
        configure_physics(rhythm.bpm)
        self.clock = AudioClock(has_audio, latency, manual_clock)
        self.camera = Camera()
        self.trail: deque = deque(maxlen=70)
        self.particles: list[list] = []
        self.label_cache: dict[int, pygame.Surface] = {}
        self.bg = self._make_background()
        self.hud_panel = pygame.Surface((420, 256), pygame.SRCALPHA)
        self.hud_panel.fill((8, 8, 18, 170))
        self.paused = False
        self.sim_misses = 0
        self.builder: CourseBuilder | None = None
        self.start_build(seed)

    # ---- phase control -----------------------------------------------------
    def start_build(self, seed: int):
        self.clock.stop()
        self.phase = "BUILDING"
        prev = self.builder
        self.builder = CourseBuilder(self.rhythm.targets, seed, self.rhythm.bpm)
        if prev is not None:  # keep lifetime tallies across resets
            self.builder.rebuilds = prev.rebuilds
            self.builder.total_failures = prev.total_failures
            self.builder.rewinds = prev.rewinds
        self.camera.snap(*self.builder.spawn)
        self.trail.clear()
        self.particles.clear()
        self.label_cache.clear()
        self.sim_misses = 0
        self.complete_timer = 0.0

    def start_simulation(self):
        b = self.builder
        self.phase = "SIMULATING"
        self.platforms = b.platforms
        self.release_time = b.release_time
        self.ball = [b.spawn[0], b.spawn[1], 0.0, 0.0]
        self.mode = "HOLD"
        self.target_i = 0
        self.hits = 0
        self.phys_time = 0.0
        self.seg_steps = 0
        self.last_offset_ms = 0.0
        self.camera.snap(b.spawn[0], b.spawn[1] + 110)
        self.trail.clear()
        self.clock.start()

    def reset_course(self):
        self.seed = random.randrange(1 << 30)
        self.paused = False
        self.start_build(self.seed)

    # ---- update --------------------------------------------------------------
    def update(self, dt: float):
        if self.paused:
            return
        self.clock.advance(dt)
        if self.phase == "BUILDING":
            self._update_build(dt)
        elif self.phase in ("SIMULATING", "COMPLETE"):
            self._update_sim(dt)
        for p in self.particles:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += GRAVITY * 0.4 * dt
            p[4] -= dt
        self.particles = [p for p in self.particles if p[4] > 0]

    def _update_build(self, dt: float):
        b = self.builder
        deadline = time.perf_counter() + BUILD_BUDGET_S
        while not b.done:
            b.step()
            if time.perf_counter() >= deadline:
                break
        hx, hy = b.head
        self.camera.follow(hx, hy, 0.0, 0.0, dt, stiffness=8.0)
        if b.done:
            self.start_simulation()

    def _current_target(self) -> Platform | None:
        return self.platforms[self.target_i] if self.target_i < len(self.platforms) else None

    def _update_sim(self, dt: float):
        t = self.clock.time()
        s = self.ball

        if self.mode == "HOLD" and t >= self.release_time:
            self.mode = "FLIGHT"
            self.phys_time = self.release_time
            self.seg_steps = 0

        # Pure, deterministic physics driven by the playback clock. The ball is
        # never slowed, pulled or boosted: the course was built so that this
        # exact integration reaches each platform on its beat.
        guard = 0
        while self.mode in ("FLIGHT", "FREE", "ARMED") and guard < 20000:
            guard += 1
            target = self._current_target()
            if self.mode == "ARMED":
                # The ball is touching its platform (at most half a physics step
                # early). The contact resolves when the playback clock reaches
                # the beat, within TIMING_TOL.
                if t >= target.target_time - TIMING_TOL:
                    self._bounce(target)
                    continue
                break
            if self.phys_time + PHYS_DT > t:
                break
            if self.mode == "FLIGHT" and self.seg_steps >= target.n_in:
                self.mode = "ARMED"
                continue
            event = integrate(s, PHYS_DT)
            for p in self._nearby_platforms(s[0], s[1]):
                if p is not target:
                    collide_platform(s, p)
            self.phys_time += PHYS_DT
            self.seg_steps += 1
            if event == "floor" and self.mode == "FLIGHT":
                self._register_miss()
                break
        if self.phase == "BUILDING":   # a miss triggered a full rebuild
            return
        target = self._current_target()
        if self.mode == "FLIGHT" and target is not None and self.phys_time > target.target_time + MISS_WINDOW:
            self._register_miss()

        self.trail.append((s[0], s[1]))
        self.camera.follow(s[0], s[1], s[2], s[3], dt)
        for p in self.platforms:
            if p.hit_flash > 0:
                p.hit_flash = max(0.0, p.hit_flash - dt * 2.5)

        # End of track -> regenerate a fresh course and play again.
        if self.phase == "SIMULATING" and self.mode == "FREE" and t > self.rhythm.duration + 0.5:
            self.phase = "COMPLETE"
        if self.phase == "COMPLETE":
            self.complete_timer += dt
            if self.complete_timer > 3.0:
                self.reset_course()

    def _nearby_platforms(self, x, y):
        # Upcoming and recent platforms are the only ones the ball can reach.
        lo, hi = max(0, self.target_i - 6), min(len(self.platforms), self.target_i + 8)
        for p in self.platforms[lo:hi]:
            if abs(p.cx - x) < PLAT_LEN and abs(p.cy - y) < PLAT_LEN:
                yield p

    def _bounce(self, p: Platform):
        """Ordinary collision with the platform - the same law as every other
        contact - resolved exactly on the beat."""
        s = self.ball
        s[2], s[3] = surface_bounce(s[2], s[3], p.nx, p.ny)
        # Physical contact time vs the beat (step quantisation only, <= ~2 ms).
        self.last_offset_ms = (self.phys_time - p.target_time) * 1000.0
        self.phys_time = p.target_time
        self.seg_steps = 0
        p.hit = True
        p.hit_flash = 1.0
        self.hits += 1
        for _ in range(10):
            a = random.uniform(0, math.tau)
            sp = random.uniform(60, 220)
            self.particles.append([s[0], s[1], math.cos(a) * sp, math.sin(a) * sp,
                                   random.uniform(0.2, 0.4), p.color])
        self.target_i += 1
        self.mode = "FLIGHT" if self.target_i < len(self.platforms) else "FREE"

    def _register_miss(self):
        """Safety net (never triggered on a validated course): rewind to the last
        successful hit and replay the deterministic physics up to 'now'."""
        self.sim_misses += 1
        if self.sim_misses > SIM_FAIL_THRESHOLD:
            self.reset_course()
            return
        j = self.target_i
        if j == 0:
            b = self.builder
            self.ball[:] = [b.spawn[0], b.spawn[1], 0.0, 0.0]
            self.phys_time = self.release_time
        else:
            prev = self.platforms[j - 1]
            self.ball[:] = [prev.contact[0], prev.contact[1], prev.v_out[0], prev.v_out[1]]
            self.phys_time = prev.target_time
        self.seg_steps = 0
        self.mode = "FLIGHT"

    # ---- drawing ---------------------------------------------------------------
    def _make_background(self) -> pygame.Surface:
        surf = pygame.Surface((SCREEN_W, SCREEN_H))
        for y in range(SCREEN_H):
            pygame.draw.line(surf, lerp_color(BG_TOP, BG_BOTTOM, y / SCREEN_H), (0, y), (SCREEN_W, y))
        return surf

    def draw(self):
        scr = self.screen
        scr.blit(self.bg, (0, 0))
        self._draw_grid()
        self._draw_walls()
        platforms = self.builder.platforms if self.phase == "BUILDING" else self.platforms
        t = self.clock.time()
        for p in platforms:
            if self.camera.visible(p.cx, p.cy):          # off-screen culling
                self._draw_platform(p)
        if self.phase == "BUILDING":
            self._draw_build_head()
        else:
            self._draw_trail()
            self._draw_particles()
            self._draw_ball(t)
        self._draw_beat_indicator(t)
        self._draw_hud(t)
        if self.paused:
            self._center_text("PAUSED", "SPACE to resume")
        elif self.phase == "COMPLETE":
            self._center_text("COURSE COMPLETE", "Generating a new course...")

    def _draw_grid(self):
        step = 200
        cx, cy = self.camera.x, self.camera.y
        x0 = math.floor((cx - SCREEN_W / 2) / step) * step
        y0 = math.floor((cy - SCREEN_H / 2) / step) * step
        x = x0
        while x < cx + SCREEN_W / 2 + step:
            sx, _ = self.camera.to_screen(x, 0)
            pygame.draw.line(self.screen, GRID_COLOR, (sx, 0), (sx, SCREEN_H))
            x += step
        y = y0
        while y < cy + SCREEN_H / 2 + step:
            _, sy = self.camera.to_screen(0, y)
            pygame.draw.line(self.screen, GRID_COLOR, (0, sy), (SCREEN_W, sy))
            y += step

    def _draw_walls(self):
        for wx in (-WORLD_HALF_W, WORLD_HALF_W):
            if abs(wx - self.camera.x) < SCREEN_W / 2 + 10:
                sx, _ = self.camera.to_screen(wx, 0)
                pygame.draw.line(self.screen, (120, 60, 90), (sx, 0), (sx, SCREEN_H), 4)

    def _draw_platform(self, p: Platform):
        pts = [self.camera.to_screen(x, y) for x, y in p.corners()]
        base = p.color
        fill = lerp_color(base, (20, 20, 35), 0.72 if p.hit else 0.55)
        if p.hit_flash > 0:
            fill = lerp_color(fill, (255, 255, 255), p.hit_flash * 0.8)
        pygame.draw.polygon(self.screen, fill, pts)
        outline = base if not p.hit else lerp_color(base, (90, 90, 110), 0.5)
        pygame.draw.polygon(self.screen, outline, pts, 2)
        label = self.label_cache.get(p.pid)
        if label is None:
            label = self.font_small.render(str(p.pid), True, base)
            self.label_cache[p.pid] = label
        lx, ly = self.camera.to_screen(p.cx - p.nx * 20, p.cy - p.ny * 20)
        self.screen.blit(label, label.get_rect(center=(lx, ly)))

    def _draw_trail(self):
        n = len(self.trail)
        if n < 2:
            return
        pts = [self.camera.to_screen(x, y) for x, y in self.trail]
        for i in range(1, n):
            age = i / n
            col = lerp_color(BG_BOTTOM, (120, 220, 255), age)
            pygame.draw.line(self.screen, col, pts[i - 1], pts[i], max(1, int(BALL_R * 0.9 * age)))

    def _draw_particles(self):
        for x, y, _vx, _vy, life, col in self.particles:
            sx, sy = self.camera.to_screen(x, y)
            pygame.draw.circle(self.screen, lerp_color(BG_BOTTOM, col, min(1.0, life * 2)), (sx, sy), max(1, int(life * 7)))

    def _draw_ball(self, t: float):
        s = self.ball
        x, y = s[0], s[1]
        if self.mode in ("FLIGHT", "FREE"):
            # Sub-step extrapolation so motion is smooth at any frame rate.
            frac = max(0.0, min(PHYS_DT, t - self.phys_time))
            x, y = x + s[2] * frac, y + s[3] * frac
        sx, sy = self.camera.to_screen(x, y)
        pygame.draw.circle(self.screen, (235, 245, 255), (sx, sy), int(BALL_R))
        pygame.draw.circle(self.screen, (120, 220, 255), (sx, sy), int(BALL_R), 3)

    def _draw_build_head(self):
        hx, hy = self.builder.head
        sx, sy = self.camera.to_screen(hx, hy)
        pygame.draw.circle(self.screen, (120, 220, 255), (sx, sy), int(BALL_R), 2)

    def _draw_beat_indicator(self, t: float):
        cx, cy = SCREEN_W - 70, 70
        beats = self.rhythm.beats
        active = self.phase != "BUILDING"
        i = int(np.searchsorted(beats, t)) if active else 0
        if active and 0 < i < len(beats):
            prev, nxt = beats[i - 1], beats[i]
            phase = (t - prev) / max(1e-3, nxt - prev)
        else:
            phase = 0.0
        ring = 14 + 34 * (1.0 - phase)          # ring closes in as the beat approaches
        flash = max(0.0, 1.0 - phase / 0.15) if active else 0.0
        pygame.draw.circle(self.screen, (40, 40, 70), (cx, cy), 50, 1)
        pygame.draw.circle(self.screen, lerp_color((70, 80, 130), (255, 255, 255), phase ** 3), (cx, cy), int(ring), 2)
        core = lerp_color((60, 70, 120), (255, 221, 87), flash)
        pygame.draw.circle(self.screen, core, (cx, cy), int(12 + 6 * flash))
        lbl = self.font_small.render(f"{self.rhythm.bpm:.1f} BPM", True, DIM_TEXT)
        self.screen.blit(lbl, lbl.get_rect(center=(cx, cy + 64)))

    def _draw_hud(self, t: float):
        self.screen.blit(self.hud_panel, (12, 12))
        b = self.builder
        if self.phase == "BUILDING":
            status = f"BUILDING  {len(b.platforms)}/{b.last_total}  ({b.progress * 100:4.0f}%)"
        elif self.paused:
            status = "PAUSED"
        elif self.phase == "COMPLETE":
            status = "COMPLETE"
        else:
            status = {"HOLD": "SIMULATING - drop on next beat", "FREE": "SIMULATING - free fall"}.get(
                self.mode, "SIMULATING")
        n_plat = len(b.platforms)
        hits = getattr(self, "hits", 0) if self.phase != "BUILDING" else 0
        nxt = self._current_target() if self.phase != "BUILDING" else None
        dur = self.rhythm.duration
        lines = [
            (f"BEAT DROP  -  {self.rhythm.source_name[:26]}", TEXT),
            (f"Phase     {status}", (120, 220, 255)),
            (f"Platforms {n_plat}", TEXT),
            (f"Hits      {hits}/{n_plat}" + (f"   next #{nxt.pid} @ {nxt.target_time:6.2f}s" if nxt else ""), TEXT),
            (f"Time      {fmt_time(t)} / {fmt_time(dur)}", TEXT),
            (self.audio_status, (120, 230, 150) if self.clock.has_audio else (255, 110, 110)),
            (f"Beats     {self.rhythm.analyzer}"[:52],
             (255, 200, 90) if self.rhythm.synthetic or "built-in" in self.rhythm.analyzer else TEXT),
            (f"Sync      audio offset {self.clock.latency * 1000:+.0f} ms   contact "
             f"{getattr(self, 'last_offset_ms', 0.0):+.1f} ms", TEXT),
            (f"Failures  build {b.total_failures}  rewinds {b.rewinds}", TEXT),
            (f"          rebuilds {b.rebuilds}  live misses {self.sim_misses}", TEXT),
        ]
        y = 22
        for text, col in lines:
            self.screen.blit(self.font.render(text, True, col), (24, y))
            y += 22
        if self.phase == "BUILDING":
            pygame.draw.rect(self.screen, (50, 50, 80), (24, y + 2, 370, 5))
            pygame.draw.rect(self.screen, (120, 220, 255), (24, y + 2, int(370 * b.progress), 5))
        ctrl = self.font_small.render("[SPACE] pause   [R] reset course   [ / ] audio sync -/+ 5 ms   "
                                      "drop a WAV on the window to switch songs   [ESC] quit",
                                      True, DIM_TEXT)
        self.screen.blit(ctrl, (16, SCREEN_H - 26))

    def _center_text(self, title: str, sub: str):
        a = self.font_big.render(title, True, TEXT)
        b = self.font.render(sub, True, DIM_TEXT)
        self.screen.blit(a, a.get_rect(center=(SCREEN_W / 2, SCREEN_H / 2 - 16)))
        self.screen.blit(b, b.get_rect(center=(SCREEN_W / 2, SCREEN_H / 2 + 20)))

    @property
    def audio_status(self) -> str:
        if self.clock.has_audio:
            return "Audio     PLAYING" if self.phase != "BUILDING" else "Audio     ready"
        return f"Audio     OFF - {self.audio_note or 'no audio device'}"[:52]

    # ---- input -----------------------------------------------------------------
    def handle_event(self, ev) -> bool:
        if ev.type == pygame.QUIT:
            return False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                return False
            if ev.key == pygame.K_SPACE and self.phase != "BUILDING":
                self.paused = not self.paused
                self.clock.set_paused(self.paused)
            if ev.key == pygame.K_r:
                self.reset_course()
            if ev.key == pygame.K_LEFTBRACKET:
                self.clock.latency -= 0.005
            if ev.key == pygame.K_RIGHTBRACKET:
                self.clock.latency += 0.005
        return True


def fmt_time(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60)}:{t % 60:06.3f}"


def draw_loading(screen, font, msg, sub=""):
    screen.fill(BG_TOP)
    s = font.render(msg, True, TEXT)
    screen.blit(s, s.get_rect(center=(SCREEN_W / 2, SCREEN_H / 2)))
    if sub:
        t = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 15).render(sub, True, DIM_TEXT)
        screen.blit(t, t.get_rect(center=(SCREEN_W / 2, SCREEN_H / 2 + 34)))
    pygame.display.flip()
    pygame.event.pump()


def start_screen(screen, message: str) -> str | None:
    """Shown when no song was chosen. Returns a file path, "DEMO", or None to quit.
    Accepts a file dragged onto the window, so it works even without tkinter."""
    big = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 34, bold=True)
    font = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 18)
    small = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 14)
    clock = pygame.time.Clock()
    t = 0.0
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return None
            if ev.type == pygame.DROPFILE:
                return ev.file
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_o:
                p = choose_audio_file()
                if p:
                    return p
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_d:
                return "DEMO"
        t += clock.tick(60) / 1000.0
        screen.fill(BG_TOP)
        pulse = 0.5 + 0.5 * math.sin(t * math.tau * 2)
        pygame.draw.rect(screen, lerp_color((50, 50, 90), (120, 220, 255), pulse),
                         (SCREEN_W / 2 - 330, SCREEN_H / 2 - 120, 660, 240), 3, border_radius=18)
        lines = [(big, "BEAT DROP", TEXT, -70), (font, "Drag a WAV file onto this window", TEXT, -10),
                 (font, "[O] open file browser     [D] demo metronome     [ESC] quit", DIM_TEXT, 30)]
        for f, text, col, dy in lines:
            s = f.render(text, True, col)
            screen.blit(s, s.get_rect(center=(SCREEN_W / 2, SCREEN_H / 2 + dy)))
        if message:
            s = small.render(message[:120], True, (255, 150, 120))
            screen.blit(s, s.get_rect(center=(SCREEN_W / 2, SCREEN_H / 2 + 160)))
        pygame.display.flip()


def prepare_song(screen, font, path: str | None, seed: int) -> tuple[RhythmData | None, str]:
    """Load the user's song (never silently replaced by the demo). Returns
    (rhythm, error_message). path=None/'DEMO' loads the demo metronome."""
    if not path or path == "DEMO":
        return make_synthetic_rhythm(seed), ""
    draw_loading(screen, font, f"Analysing {os.path.basename(path)} ...",
                 "librosa" if HAVE_LIBROSA else "built-in beat tracker (librosa not available)")
    try:
        rhythm = load_song(path, seed)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return None, f"Could not open {os.path.basename(path)}: {exc}"
    print(f"[rhythm] {rhythm.source_name}: {rhythm.bpm:.1f} BPM via {rhythm.analyzer}; "
          f"{len(rhythm.beats)} beats, {len(rhythm.onsets)} extra onsets, {len(rhythm.targets)} targets")
    return rhythm, ""


def load_into_mixer(rhythm: RhythmData, mixer_note: str) -> tuple[bool, str]:
    if not rhythm.audio_path:
        return False, mixer_note or "no audio file"
    if not pygame.mixer.get_init():
        return False, mixer_note or "no audio device"
    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.load(rhythm.audio_path)
        pygame.mixer.music.set_volume(1.0)
        print(f"[audio] ready: {rhythm.source_name} (mixer {pygame.mixer.get_init()})")
        return True, ""
    except pygame.error as exc:
        print(f"[audio] mixer could not load the file: {exc}")
        return False, f"load failed: {exc}"


def cleanup(rhythm: RhythmData | None):
    if rhythm and rhythm.temp_audio and rhythm.audio_path:
        try:
            pygame.mixer.music.unload()
        except Exception:
            pass
        try:
            os.remove(rhythm.audio_path)
        except OSError:
            pass


# =============================================================================
# Entry point
# =============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rhythm-synchronized ball drop simulation")
    ap.add_argument("wav", nargs="?", help="WAV file (skips the file dialog)")
    ap.add_argument("--synthetic", action="store_true", help="skip audio selection, use the demo metronome")
    ap.add_argument("--seed", type=int, default=None, help="course generation seed")
    ap.add_argument("--latency", type=float, default=DEFAULT_LATENCY_MS,
                    help=f"audio output latency compensation in ms (default {DEFAULT_LATENCY_MS:.0f}; "
                         "adjust live with [ and ])")
    ap.add_argument("--selftest", type=float, default=0.0, metavar="SECONDS",
                    help="run headless on a simulated clock for N seconds and print stats")
    args = ap.parse_args(argv)

    if args.selftest:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    random.seed(seed)
    if not HAVE_LIBROSA:
        print(f"[rhythm] librosa not available ({LIBROSA_ERROR}); the built-in beat tracker will be used")

    # 1. Pick the audio file before the pygame window exists (avoids Tk/SDL clashes).
    path = args.wav
    if not path and not args.synthetic and not args.selftest:
        path = choose_audio_file()
    if args.synthetic or (args.selftest and not path):
        path = "DEMO"

    pygame.mixer.pre_init(44100, -16, 2, 1024)
    pygame.init()
    mixer_note = ""
    if not pygame.mixer.get_init():
        try:
            pygame.mixer.init(44100, -16, 2, 1024)
        except pygame.error as exc:
            mixer_note = f"mixer init failed: {exc}"
            print(f"[audio] {mixer_note}")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Beat Drop")
    font = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 20)

    # 2. Load the song. If none was chosen (or it failed), show the start screen
    #    instead of silently substituting the demo metronome.
    rhythm, message = (None, "") if not path else prepare_song(screen, font, path, seed)
    while rhythm is None:
        if args.selftest:
            print(f"[selftest] {message}")
            return 1
        choice = start_screen(screen, message or "No song selected.")
        if choice is None:
            pygame.quit()
            return 0
        rhythm, message = prepare_song(screen, font, choice, seed)

    has_audio, note = load_into_mixer(rhythm, mixer_note)
    game = Game(screen, rhythm, seed, has_audio, args.latency / 1000.0,
                manual_clock=bool(args.selftest), audio_note=note)
    frame_clock = pygame.time.Clock()
    running = True
    sim_elapsed = 0.0
    while running:
        if args.selftest:
            dt = 1.0 / TARGET_FPS
            sim_elapsed += dt
            if sim_elapsed >= args.selftest:
                break
        else:
            dt = min(frame_clock.tick(TARGET_FPS) / 1000.0, 0.05)
        for ev in pygame.event.get():
            if ev.type == pygame.DROPFILE:          # drop a new song any time
                was_paused = game.paused
                game.clock.set_paused(True)
                new, msg = prepare_song(screen, font, ev.file, seed)
                if new is None:                     # keep playing the current song
                    print(f"[audio] {msg}")
                    game.clock.set_paused(was_paused)
                    continue
                game.clock.stop()
                cleanup(rhythm)
                rhythm = new
                has_audio, note = load_into_mixer(rhythm, mixer_note)
                game = Game(screen, rhythm, random.randrange(1 << 30), has_audio, game.clock.latency,
                            manual_clock=bool(args.selftest), audio_note=note)
                continue
            running = game.handle_event(ev) and running
        game.update(dt)
        game.draw()
        pygame.display.set_caption(f"Beat Drop  -  {rhythm.source_name}  -  {frame_clock.get_fps():.0f} fps")
        pygame.display.flip()

    if args.selftest:
        b = game.builder
        print(f"[selftest] song={rhythm.source_name} analyzer={rhythm.analyzer!r} bpm={rhythm.bpm:.1f} "
              f"audio={'on' if has_audio else 'off'}")
        print(f"[selftest] phase={game.phase} platforms={len(b.platforms)} hits={getattr(game, 'hits', 0)} "
              f"build_failures={b.total_failures} rewinds={b.rewinds} rebuilds={b.rebuilds} "
              f"live_misses={game.sim_misses} audio_t={game.clock.time():.2f}")
    game.clock.stop()
    cleanup(rhythm)
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())