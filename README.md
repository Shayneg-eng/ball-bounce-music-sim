# Ball Bounce Music Sim

Turns any music file into a video of a ball dropping and bouncing off
procedurally generated panels in time with the beat.

## How it works

1. **Beat detection** (`beat_detect.js`, Node): decodes the audio with the
   Web Audio API (via `node-web-audio-api`, a Node implementation of the
   standard `AudioContext`/`OfflineAudioContext`), computes a spectral-flux
   onset envelope frame by frame, and picks peaks as beat timestamps.
2. **Choreography** (`simulate.py`): the ball is dropped at t=0 and free-falls
   under gravity until it lands on a panel placed exactly where physics puts
   it at the first beat. For every following beat, a new panel is generated
   at a fresh on-screen location, and the exact launch velocity needed to
   travel from the previous panel to the new one in the time between beats
   is solved from the real projectile-motion equations. Every bounce is
   therefore an actual parabolic arc under constant gravity, and lands
   precisely on its beat by construction. Each panel is used exactly once,
   and every candidate flight path is checked against the real geometry of
   every nearby panel (not just its center point) before being accepted, so
   the ball's path doesn't cut through a panel it already passed. For very
   densely packed beats (many onsets within a fraction of a second) panels
   are also sized relative to the space actually available, and consecutive
   beats closer than ~0.16s apart are merged so bounces stay physically
   readable.
3. **Rendering** (`render.py`): draws every frame with Pillow (ball, motion
   trail, panels that dim once used) and pipes raw frames into `ffmpeg` to
   encode an MP4. The camera follows the ball's *smoothed* long-term descent
   rather than locking to its exact position every frame -- a rigid 1:1
   camera lock would cancel out the very motion gravity produces, leaving
   the ball looking like it hovers in place while panels slide past it.
4. **Mux** (`ball_bounce.py`): stitches the original audio back onto the
   rendered video.

## Setup

Requirements: Node.js 18+, Python 3.9+, `ffmpeg` on your PATH.

```bash
npm install
pip install pillow numpy
```

## Usage

```bash
python3 ball_bounce.py path/to/song.mp3
```

This writes `song_ball_bounce.mp4` next to the input file. Useful options:

```bash
python3 ball_bounce.py song.mp3 out.mp4 \
  --width 1080 --height 1920 --fps 30 \
  --gravity 2200 --seed 42 \
  --min-bpm 80 --max-bpm 160
```

- `--width/--height` — video resolution (default 1080x1920, portrait).
- `--gravity` — strength of gravity in px/s^2 (higher = snappier, more
  aggressive bounces).
- `--seed` — fix the RNG seed to get a reproducible panel layout.
- `--min-bpm/--max-bpm` — tempo search range for beat detection, narrow it
  if the detector locks onto double/half tempo.
- `--beats-json path.json` — skip re-running beat detection and reuse a
  previously generated `beats.json` (handy for iterating on visuals only).
- `--keep-temp` — keep the intermediate `beats.json`/silent video for
  debugging.

You can also run beat detection standalone:

```bash
node beat_detect.js song.mp3 --out beats.json
```

## Files

- `beat_detect.js` — Web Audio API beat/onset detector (Node).
- `simulate.py` — physics choreography (panel placement + projectile math).
- `render.py` — Pillow-based frame renderer + ffmpeg video encoding.
- `ball_bounce.py` — CLI that wires the above together end to end.
