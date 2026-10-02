#!/usr/bin/env node
/**
 * beat_detect.js
 *
 * Analyzes an audio file using the Web Audio API (via node-web-audio-api,
 * a Node implementation of the standard browser AudioContext/OfflineAudioContext
 * APIs) and extracts beat timestamps using spectral-flux onset detection +
 * autocorrelation tempo estimation.
 *
 * Usage: node beat_detect.js <audio-file> [--out beats.json]
 */

const fs = require('fs');
const path = require('path');
const { OfflineAudioContext } = require('node-web-audio-api');

function parseArgs(argv) {
  const args = { out: null, minBpm: 60, maxBpm: 200 };
  const positional = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--out') args.out = argv[++i];
    else if (a === '--min-bpm') args.minBpm = parseFloat(argv[++i]);
    else if (a === '--max-bpm') args.maxBpm = parseFloat(argv[++i]);
    else positional.push(a);
  }
  args.input = positional[0];
  return args;
}

async function decodeAudio(filePath) {
  const buf = fs.readFileSync(filePath);
  // Use a throwaway OfflineAudioContext purely for its decodeAudioData(),
  // exactly as you'd do in a browser with a real AudioContext.
  const probeCtx = new OfflineAudioContext({ length: 1, sampleRate: 44100, numberOfChannels: 1 });
  const audioBuffer = await probeCtx.decodeAudioData(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  return audioBuffer;
}

// Downmix all channels to mono Float32Array
function toMono(audioBuffer) {
  const ch = audioBuffer.numberOfChannels;
  const len = audioBuffer.length;
  const mono = new Float32Array(len);
  for (let c = 0; c < ch; c++) {
    const data = audioBuffer.getChannelData(c);
    for (let i = 0; i < len; i++) mono[i] += data[i] / ch;
  }
  return mono;
}

// Simple radix-2 FFT (in-place, iterative) returning magnitude spectrum
function fftMagnitude(frame) {
  const n = frame.length;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  re.set(frame);

  // bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curWr = 1, curWi = 0;
      for (let j = 0; j < len / 2; j++) {
        const ur = re[i + j], ui = im[i + j];
        const vr = re[i + j + len / 2] * curWr - im[i + j + len / 2] * curWi;
        const vi = re[i + j + len / 2] * curWi + im[i + j + len / 2] * curWr;
        re[i + j] = ur + vr;
        im[i + j] = ui + vi;
        re[i + j + len / 2] = ur - vr;
        im[i + j + len / 2] = ui - vi;
        const nwr = curWr * wr - curWi * wi;
        const nwi = curWr * wi + curWi * wr;
        curWr = nwr; curWi = nwi;
      }
    }
  }

  const mag = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) mag[i] = Math.hypot(re[i], im[i]);
  return mag;
}

function hannWindow(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}

/**
 * Spectral flux onset detection function, computed frame by frame like an
 * AnalyserNode would feed getByteFrequencyData() in a real-time browser app.
 */
function computeOnsetEnvelope(mono, sampleRate, frameSize = 1024, hopSize = 512) {
  const window = hannWindow(frameSize);
  const numFrames = Math.floor((mono.length - frameSize) / hopSize);
  const onset = new Float64Array(numFrames);
  let prevMag = new Float64Array(frameSize / 2);

  const frame = new Float64Array(frameSize);
  for (let f = 0; f < numFrames; f++) {
    const start = f * hopSize;
    for (let i = 0; i < frameSize; i++) frame[i] = mono[start + i] * window[i];
    const mag = fftMagnitude(frame);

    let flux = 0;
    for (let i = 0; i < mag.length; i++) {
      const diff = mag[i] - prevMag[i];
      if (diff > 0) flux += diff;
    }
    onset[f] = flux;
    prevMag = mag;
  }
  return { onset, hopSize, frameSize };
}

// Adaptive peak-picking over the onset envelope
function pickPeaks(onset, sampleRate, hopSize, minGapSec = 0.12) {
  const n = onset.length;
  // normalize
  let max = 0;
  for (let i = 0; i < n; i++) max = Math.max(max, onset[i]);
  if (max <= 0) return [];
  const norm = Float64Array.from(onset, v => v / max);

  // local moving average threshold
  const windowFrames = Math.max(1, Math.round((1.5 * sampleRate) / hopSize)); // ~1.5s
  const peaks = [];
  const minGapFrames = Math.max(1, Math.round((minGapSec * sampleRate) / hopSize));
  let lastPeak = -Infinity;

  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - windowFrames);
    const hi = Math.min(n, i + windowFrames);
    let sum = 0;
    for (let j = lo; j < hi; j++) sum += norm[j];
    const mean = sum / (hi - lo);
    const threshold = mean * 1.5 + 0.05;

    const isLocalMax = norm[i] > threshold &&
      (i === 0 || norm[i] >= norm[i - 1]) &&
      (i === n - 1 || norm[i] >= norm[i + 1]);

    if (isLocalMax && (i - lastPeak) >= minGapFrames) {
      peaks.push(i);
      lastPeak = i;
    }
  }

  return peaks.map(i => (i * hopSize) / sampleRate);
}

/**
 * Estimate a global tempo via autocorrelation of the onset envelope. Used
 * for reporting purposes (and could be used to snap peaks to a grid); the
 * actual bounce timestamps come from the detected onset peaks themselves.
 */
function estimateTempoBpm(onset, sampleRate, hopSize, minBpm, maxBpm) {
  const framesPerSec = sampleRate / hopSize;
  const minLag = Math.floor(framesPerSec * (60 / maxBpm));
  const maxLag = Math.ceil(framesPerSec * (60 / minBpm));

  let bestLag = -1, bestScore = -Infinity;
  for (let lag = minLag; lag <= maxLag && lag < onset.length; lag++) {
    let score = 0;
    for (let i = 0; i + lag < onset.length; i++) score += onset[i] * onset[i + lag];
    if (score > bestScore) { bestScore = score; bestLag = lag; }
  }
  if (bestLag <= 0) return null;
  return 60 / (bestLag / framesPerSec);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.input) {
    console.error('Usage: node beat_detect.js <audio-file> [--out beats.json] [--min-bpm N] [--max-bpm N]');
    process.exit(1);
  }

  const audioBuffer = await decodeAudio(args.input);
  const sampleRate = audioBuffer.sampleRate;
  const mono = toMono(audioBuffer);

  const { onset, hopSize } = computeOnsetEnvelope(mono, sampleRate);
  const bpm = estimateTempoBpm(onset, sampleRate, hopSize, args.minBpm, args.maxBpm);
  const beats = pickPeaks(onset, sampleRate, hopSize);

  const result = {
    file: path.basename(args.input),
    duration: audioBuffer.duration,
    sampleRate,
    estimatedBpm: bpm,
    beatCount: beats.length,
    beats,
  };

  const json = JSON.stringify(result, null, 2);
  if (args.out) {
    fs.writeFileSync(args.out, json);
    console.error(`Wrote ${beats.length} beats (~${bpm ? bpm.toFixed(1) : '?'} BPM) to ${args.out}`);
  } else {
    process.stdout.write(json);
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
