#!/usr/bin/env python3
"""
ball_bounce.py

End-to-end pipeline:

  1. Runs beat_detect.js (Node, using the Web Audio API via
     node-web-audio-api) to extract beat timestamps from the input audio.
  2. Feeds those timestamps into simulate.py, which choreographs a ball
     drop + a sequence of procedurally generated panel bounces timed
     exactly to each beat, under real projectile-motion physics.
  3. Renders the choreography to video with render.py.
  4. Muxes the original audio back onto the rendered (silent) video.

Usage:
    python3 ball_bounce.py <audio-file> [output.mp4] [options]

Options:
    --width N            video width in px (default 1080)
    --height N            video height in px (default 1920)
    --fps N               frames per second (default 30)
    --gravity N            gravity in px/s^2 (default 2200)
    --seed N               RNG seed for panel layout (default random)
    --min-bpm / --max-bpm  tempo search range for beat detection
    --beats-json PATH       reuse a previously generated beats.json instead
                            of re-running beat detection
    --keep-temp             don't delete intermediate files
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def run_beat_detection(audio_path, work_dir, min_bpm, max_bpm):
    beats_path = os.path.join(work_dir, "beats.json")
    cmd = [
        "node", os.path.join(HERE, "beat_detect.js"), audio_path,
        "--out", beats_path, "--min-bpm", str(min_bpm), "--max-bpm", str(max_bpm),
    ]
    print(f"[1/4] Detecting beats: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)
    with open(beats_path) as f:
        data = json.load(f)
    print(f"      -> {data['beatCount']} beats, ~{data['estimatedBpm']:.1f} BPM, "
          f"duration {data['duration']:.1f}s", file=sys.stderr)
    return data


def mux_audio(video_path, audio_path, out_path):
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", out_path,
    ]
    print(f"[4/4] Muxing audio: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description="Beat-synced ball bounce video generator")
    parser.add_argument("audio", help="path to input music file (mp3/wav/etc)")
    parser.add_argument("output", nargs="?", default=None, help="output mp4 path")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gravity", type=float, default=2200.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-bpm", type=float, default=60)
    parser.add_argument("--max-bpm", type=float, default=200)
    parser.add_argument("--beats-json", default=None)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    audio_path = os.path.abspath(args.audio)
    if not os.path.isfile(audio_path):
        sys.exit(f"Input audio file not found: {audio_path}")

    out_path = args.output or (os.path.splitext(audio_path)[0] + "_ball_bounce.mp4")
    out_path = os.path.abspath(out_path)

    work_dir = tempfile.mkdtemp(prefix="ball_bounce_")
    try:
        if args.beats_json:
            with open(args.beats_json) as f:
                beat_data = json.load(f)
            print(f"[1/4] Reusing beats from {args.beats_json}", file=sys.stderr)
        else:
            beat_data = run_beat_detection(audio_path, work_dir, args.min_bpm, args.max_bpm)

        beats = beat_data["beats"]
        duration = beat_data["duration"]
        if not beats:
            sys.exit("No beats detected in this audio file.")

        print("[2/4] Choreographing bounce physics", file=sys.stderr)
        sys.path.insert(0, HERE)
        from simulate import build_simulation
        segments, panels, gravity = build_simulation(
            beats, args.width, args.height, gravity=args.gravity, seed=args.seed,
        )
        print(f"      -> {len(panels)} panels, {len(segments)} flight segments", file=sys.stderr)

        # total render duration: whichever is longer, the song or the
        # choreography's own trailing fly-off segment, capped to the song
        render_duration = min(duration, segments[-1]["t_end"]) if duration else segments[-1]["t_end"]
        render_duration = max(render_duration, segments[-1]["t_end"] if not duration else render_duration)
        render_duration = duration if duration else segments[-1]["t_end"]

        print(f"[3/4] Rendering {args.width}x{args.height} @ {args.fps}fps, "
              f"{render_duration:.1f}s", file=sys.stderr)
        from render import render_video
        silent_path = os.path.join(work_dir, "silent.mp4")
        render_video(segments, panels, gravity, render_duration,
                     args.width, args.height, args.fps, silent_path)

        mux_audio(silent_path, audio_path, out_path)
        print(f"\nDone: {out_path}", file=sys.stderr)
    finally:
        if not args.keep_temp:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"(intermediate files kept in {work_dir})", file=sys.stderr)


if __name__ == "__main__":
    main()
