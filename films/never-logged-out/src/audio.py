"""The film's sound, synthesised from the cue sheet (no samples, no voice: his lines are left for a real voice-over).

Buses:
  ambience  beds per place (day, interior, night, night_in, tunnel, cavern, chamber, wrong, room, pc), crossfaded by
            the amb / amb_fade / amb_duck / silence cues; ambient events (birds, drips, creaks, cave sounds) go
            through the place's reverb; whispers sit under the underground and the changed world, too quiet to be
            sure of
  music     a quiet piano piece for the perfect world (its melody comes back reversed and slowed under the finale),
            the menu pad, the title drone
  drones    low beating drones and stings, never a screamer
  effects   footsteps per surface, the game's foley and interface sounds, each through the reverb of where it is
Silence is a cue too: silence_all empties the ambience, music and drones for a moment; cut_silence empties
everything. Under his subtitles the ambience and music dip a little, so a voice-over sits on top.

    python audio.py            # -> output/film_audio.wav (48 kHz stereo, -16 LUFS, -1 dBFS peak)
"""
import json
import os
import sys
import time
import wave

import numpy as np
from scipy import signal

SR = 48000
CTL = 100                                    # control rate of the gain curves (Hz)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'output')


# ---------------------------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------------------------
def _sos(kind, f, order=2):
    return signal.butter(order, f, btype=kind, fs=SR, output='sos')


def bp(x, lo, hi, order=2):
    return signal.sosfilt(_sos('bandpass', [max(lo, 5.0), min(hi, SR * 0.45)], order), x)


def lp(x, f, order=2):
    return signal.sosfilt(_sos('lowpass', min(f, SR * 0.45), order), x)


def hp(x, f, order=2):
    return signal.sosfilt(_sos('highpass', f, order), x)


def tt(n):
    return np.arange(n) / SR


def expenv(n, tau, attack=0.0):
    t = tt(n)
    e = np.exp(-t / tau)
    if attack > 0:
        e *= np.clip(t / attack, 0, 1)
    return e


def fade(x, a=0.005, r=0.02):
    n = len(x)
    t = tt(n)
    return x * np.clip(t / max(a, 1e-4), 0, 1) * np.clip((n / SR - t) / max(r, 1e-4), 0, 1)


def norm(x, peak=1.0):
    m = np.abs(x).max() if len(x) else 0.0
    return x * (peak / m) if m > 0 else x


def resample(x, ratio):
    """Pitch/speed change by ratio (>1 = higher and shorter)."""
    n = int(len(x) / ratio)
    if n < 2:
        return x
    return np.interp(np.arange(n) * ratio, np.arange(len(x)), x)


def modal(dur, f0, ratios, amps, taus, rng, detune=0.004):
    n = int(dur * SR)
    t = tt(n)
    x = np.zeros(n)
    for r, a, tau in zip(ratios, amps, taus):
        f = f0 * r * (1 + rng.uniform(-detune, detune))
        x += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t / tau)
    return x


def sweep(dur, f0, f1, tau_f):
    n = int(dur * SR)
    t = tt(n)
    f = f1 + (f0 - f1) * np.exp(-t / tau_f)
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def shaped_noise(dur, center, width_oct, env, rng, nfft=1024):
    """Noise with a time-varying log-gaussian band (center/width/env: callables of t)."""
    n = int(dur * SR)
    x = rng.standard_normal(n + nfft)
    f, t, Z = signal.stft(x, fs=SR, nperseg=nfft, noverlap=nfft * 3 // 4)
    c = np.maximum(center(t), 30.0)
    w = np.maximum(width_oct(t), 0.1)
    lf = np.log2(np.maximum(f, 1.0))[:, None]
    g = np.exp(-0.5 * ((lf - np.log2(c)[None, :]) / w[None, :]) ** 2)
    Z = Z * g * env(t)[None, :]
    _, y = signal.istft(Z, fs=SR, nperseg=nfft, noverlap=nfft * 3 // 4)
    return y[:n]


def smooth_curve(n, rate_hz, rng, lo=0.0, hi=1.0):
    """A slow random curve (n samples at SR) wandering between lo and hi at about rate_hz."""
    m = max(4, int(n / SR * CTL) + 4)
    k = rng.standard_normal(m)
    k = signal.sosfiltfilt(signal.butter(2, rate_hz, fs=CTL, output='sos'), k)
    k = (k - k.min()) / (k.max() - k.min() + 1e-9)
    return lo + (hi - lo) * np.interp(np.arange(n) / SR * CTL, np.arange(m), k)


def stereo_noise_bed(n, rng, lo, hi, order=2):
    L = bp(rng.standard_normal(n), lo, hi, order)
    R = bp(rng.standard_normal(n), lo, hi, order)
    return L, R


def formant(x, freqs, bws, gains):
    """A source through a bank of resonators (vowels)."""
    y = np.zeros_like(x)
    for f, b, g in zip(freqs, bws, gains):
        y += g * bp(x, max(f - b / 2, 20), f + b / 2, 2)
    return y


def glottal(dur, f0_fn, rng, jitter=0.01):
    """A pulse-train voice source with a pitch contour (f0_fn: callable of t)."""
    n = int(dur * SR)
    t = tt(n)
    f = f0_fn(t) * (1 + jitter * smooth_curve(n, 8.0, rng, -1, 1))
    ph = np.cumsum(f) / SR
    saw = 2 * (ph % 1.0) - 1
    return lp(saw, 3500, 2) + 0.03 * rng.standard_normal(n)


# ---------------------------------------------------------------------------------------------
# mixing
# ---------------------------------------------------------------------------------------------
class Bus:
    def __init__(self, n):
        self.L = np.zeros(n, np.float32)
        self.R = np.zeros(n, np.float32)
        self.n = n

    def add(self, x, t, gain=1.0, pan=0.0, xr=None):
        """Mono x (or stereo x, xr) at time t (s), equal-power panned."""
        s = int(round(t * SR))
        if s >= self.n or len(x) == 0:
            return
        if s < 0:
            x = x[-s:]
            xr = xr[-s:] if xr is not None else None
            s = 0
        e = min(self.n, s + len(x))
        pan = float(np.clip(pan, -1, 1))
        gl = np.cos((pan + 1) * np.pi / 4) * np.sqrt(2)
        gr = np.sin((pan + 1) * np.pi / 4) * np.sqrt(2)
        if xr is None:
            self.L[s:e] += (x[:e - s] * gain * gl).astype(np.float32)
            self.R[s:e] += (x[:e - s] * gain * gr).astype(np.float32)
        else:
            self.L[s:e] += (x[:e - s] * gain).astype(np.float32)
            self.R[s:e] += (xr[:e - s] * gain).astype(np.float32)

    def mul(self, g):
        self.L *= g
        self.R *= g


def ramp_curve(n_ctl, events, init=0.0):
    """events: (t, target, ramp_seconds) -> control-rate curve moving linearly to each target in turn."""
    out = np.empty(n_ctl, np.float32)
    ev = sorted(events, key=lambda e: e[0])
    cur = init
    k = 0
    i = 0
    while i < n_ctl:
        t = i / CTL
        if k < len(ev) and ev[k][0] <= t:
            t0, target, rdur = ev[k]
            k += 1
            nr = max(1, int(rdur * CTL))
            end = min(n_ctl, i + nr)
            # stop the ramp early if the next event starts during it
            if k < len(ev):
                end = min(end, max(i + 1, int(ev[k][0] * CTL)))
            seg = np.linspace(cur, target, nr + 1)[1:1 + end - i]
            out[i:end] = seg
            cur = float(seg[-1]) if len(seg) else cur
            if end - i < nr and len(seg):
                # interrupted: keep the value reached
                pass
            else:
                cur = target
            i = end
            continue
        nxt = int(ev[k][0] * CTL) if k < len(ev) else n_ctl
        nxt = max(nxt, i + 1)
        out[i:min(nxt, n_ctl)] = cur
        i = nxt
    return out


def to_audio(curve, n):
    return np.interp(np.arange(n) / SR * CTL, np.arange(len(curve)), curve).astype(np.float32)


def reverb_ir(decay, seed, bright=5200.0, pre=0.012, early=()):
    rng = np.random.default_rng(seed)
    n = int(max(decay * 1.5, 0.3) * SR)
    t = tt(n)
    irs = []
    for ch in range(2):
        ir = rng.standard_normal(n) * np.exp(-t / (decay / 6.9))
        ir = lp(ir, bright, 2) * 0.75 + lp(ir, bright / 4, 2) * 0.25
        ir[:int(pre * SR)] = 0.0
        for (dt, a) in early:                  # discrete early reflections (a tunnel's slap)
            k = int(dt * SR * (1.0 + 0.03 * ch))
            if k < n:
                ir[k] += a * 6.0
        ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
        irs.append(ir.astype(np.float32))
    return irs


ROOMS = {                 # reverb per place: (decay s, wet, brightness, pre-delay, early reflections)
    'dry': None,
    'hook': (1.6, 0.35, 4000.0, 0.02, ()),
    'out': (0.9, 0.12, 6000.0, 0.01, ()),
    'room': (0.45, 0.22, 5000.0, 0.004, ((0.011, 0.3), (0.017, 0.2))),
    'tunnel': (1.6, 0.4, 3800.0, 0.008, ((0.021, 0.5), (0.043, 0.35), (0.066, 0.2))),
    'cave': (4.2, 0.5, 3200.0, 0.03, ((0.07, 0.25),)),
    'hall': (7.5, 0.62, 2600.0, 0.05, ((0.12, 0.2),)),
    'wrong': (1.4, 0.22, 4200.0, 0.02, ()),
}
PLACE_ROOM = {'day': 'out', 'night': 'out', 'interior': 'room', 'night_in': 'room', 'tunnel': 'tunnel',
              'cavern': 'cave', 'chamber': 'hall', 'wrong': 'wrong', 'room': 'room', None: 'hook'}


def apply_reverb(L, R, room):
    from scipy.signal import oaconvolve
    spec = ROOMS[room]
    if spec is None:
        return L, R
    decay, wet, bright, pre, early = spec
    irl, irr = reverb_ir(decay, hash(room) % 1000, bright, pre, early)
    yl = oaconvolve(L, irl)[:len(L)].astype(np.float32)
    yr = oaconvolve(R, irr)[:len(R)].astype(np.float32)
    return L * (1 - wet) + yl * wet, R * (1 - wet) + yr * wet


# ---------------------------------------------------------------------------------------------
# sound effects (each returns mono float arrays around 0.5-0.9 peak)
# ---------------------------------------------------------------------------------------------
def grains(n, rng, count, lo, hi, dur=(0.002, 0.008), spread=0.06):
    x = np.zeros(n)
    for _ in range(count):
        p = int(rng.uniform(0, spread) * SR)
        L = max(8, int(rng.uniform(*dur) * SR))
        if p >= n:
            continue
        g = rng.standard_normal(L) * np.hanning(L)
        e = min(n, p + L)
        x[p:e] += g[:e - p] * rng.uniform(0.3, 1.0)
    return bp(x, lo, hi, 2)


def sfx_step(rng, surface='grass', fast=False):
    """A footstep on a surface."""
    if surface in ('grass', 'path'):
        n = int(0.16 * SR)
        x = grains(n, rng, int(rng.integers(10, 18)), 500 if surface == 'grass' else 900,
                   4500 if surface == 'grass' else 7000, spread=0.05 if fast else 0.07)
        x += 0.5 * lp(rng.standard_normal(n), 260, 2) * expenv(n, 0.02)
        return norm(x * expenv(n, 0.06, 0.003), 0.55)
    if surface == 'wood':
        n = int(0.22 * SR)
        x = modal(0.22, rng.uniform(150, 210), (1.0, 2.27, 3.9, 5.6), (1.0, 0.5, 0.25, 0.12),
                  (0.05, 0.03, 0.02, 0.012), rng, 0.02)
        x += 0.4 * bp(rng.standard_normal(n), 600, 3500, 2) * expenv(n, 0.008)
        return norm(x * expenv(n, 0.08, 0.002), 0.6)
    if surface in ('stone', 'deepslate'):
        n = int(0.14 * SR)
        f = 1.0 if surface == 'stone' else 0.72
        x = 0.8 * bp(rng.standard_normal(n), 900 * f, 6500 * f, 2) * expenv(n, 0.012)
        x += modal(0.14, rng.uniform(600, 800) * f, (1.0, 1.9, 3.1), (1.0, 0.4, 0.2), (0.02, 0.012, 0.008), rng, 0.03)
        x += 0.3 * lp(rng.standard_normal(n), 300, 2) * expenv(n, 0.02)
        return norm(x * expenv(n, 0.04, 0.001), 0.55)
    if surface == 'carpet':
        n = int(0.16 * SR)
        x = lp(rng.standard_normal(n), 380, 2) * expenv(n, 0.035, 0.004)
        x += 0.15 * bp(rng.standard_normal(n), 1500, 4000, 2) * expenv(n, 0.02)
        return norm(x, 0.5)
    return sfx_step(rng, 'grass')


def sfx_creak(rng, dur=0.6, f0=(70, 130), bright=2400.0, level=1.0):
    """Wood under stress: stick-slip pulses at a wandering rate through wooden resonances."""
    n = int(dur * SR)
    t = tt(n)
    f = f0[0] + (f0[1] - f0[0]) * smooth_curve(n, 3.0, rng)
    ph = np.cumsum(f) / SR
    pulses = np.maximum(0, np.sin(2 * np.pi * ph)) ** 8
    x = pulses * (0.6 + 0.4 * smooth_curve(n, 6.0, rng))
    x = bp(x, 300, bright, 2) + 0.5 * bp(x, 900, 1400, 2)
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.6
    return norm(x * env, 0.5 * level)


def sfx_door(rng, reverse=False):
    n = int(0.9 * SR)
    x = np.zeros(n)
    latch = modal(0.08, 1800, (1.0, 2.3), (1.0, 0.5), (0.01, 0.006), rng) + 0.4 * hp(rng.standard_normal(int(0.08 * SR)),
                                                                               2000, 2) * expenv(int(0.08 * SR), 0.004)
    x[:len(latch)] += latch
    cr = sfx_creak(rng, 0.55, (55, 110))
    s = int(0.06 * SR)
    x[s:s + len(cr)] += 0.9 * cr[:n - s]
    th = sfx_step(rng, 'wood')
    s2 = int(0.62 * SR)
    x[s2:s2 + len(th)] += 0.6 * th[:n - s2]
    x = norm(x, 0.6)
    return x[::-1].copy() if reverse else x


def sfx_chest(rng, close=False):
    n = int(0.6 * SR)
    x = np.zeros(n)
    if not close:
        cr = sfx_creak(rng, 0.45, (90, 170), 3000)
        x[:len(cr)] += cr
        th = sfx_step(rng, 'wood')
        x[int(0.4 * SR):int(0.4 * SR) + len(th)] += 0.4 * th[:n - int(0.4 * SR)]
    else:
        th = sfx_step(rng, 'wood')
        x[:len(th)] += th
        x[:int(0.03 * SR)] += 0.3 * hp(rng.standard_normal(int(0.03 * SR)), 1500, 2)
    return norm(x, 0.6)


def sfx_trapdoor(rng):
    n = int(0.5 * SR)
    x = np.zeros(n)
    cr = sfx_creak(rng, 0.25, (80, 140), 2800)
    x[:len(cr)] += 0.8 * cr
    k = sfx_step(rng, 'wood')
    x[int(0.22 * SR):int(0.22 * SR) + len(k)] += k[:n - int(0.22 * SR)] * 1.2
    return norm(x, 0.65)


def sfx_ladder(rng):
    n = int(0.25 * SR)
    x = 0.8 * sfx_step(rng, 'wood')[:n]
    x = np.pad(x, (0, n - len(x)))
    x += 0.2 * bp(rng.standard_normal(n), 1500, 6000, 2) * expenv(n, 0.05, 0.01)
    return norm(x, 0.45)


def sfx_page(rng, dur=0.32):
    n = int(dur * SR)
    t = tt(n)
    sw = shaped_noise(dur, lambda q: 1500 + 5000 * np.clip(q / dur, 0, 1), lambda q: 1.1 + 0 * q,
                      lambda q: np.sin(np.pi * np.clip(q / dur, 0, 1)) ** 1.5, rng)
    x = norm(sw) + 0.35 * grains(n, rng, 14, 2500, 9000, (0.001, 0.004), dur * 0.8)
    return norm(fade(x) * (0.6 + 0.4 * np.sin(np.pi * t / dur)), 0.45)


def sfx_book_open(rng):
    n = int(0.45 * SR)
    x = np.zeros(n)
    th = lp(rng.standard_normal(int(0.08 * SR)), 500, 2) * expenv(int(0.08 * SR), 0.02)
    x[:len(th)] += norm(th, 0.5)
    pg = sfx_page(rng, 0.3)
    x[int(0.08 * SR):int(0.08 * SR) + len(pg)] += pg
    return norm(x, 0.5)


def sfx_paper(rng):
    n = int(0.7 * SR)
    pg = norm(sfx_page(rng, 0.5))
    x = grains(n, rng, 60, 1800, 9000, (0.002, 0.012), 0.6)
    x[:len(pg)] += 0.3 * pg[:n]
    return norm(fade(x, 0.01, 0.1), 0.45)


def sfx_pop(rng, f0=(500, 1100), dur=0.09):
    n = int(dur * SR)
    t = tt(n)
    f = f0[0] + (f0[1] - f0[0]) * (1 - np.exp(-t / 0.02))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * expenv(n, 0.025, 0.001)
    return norm(x, 0.5)


def sfx_click(rng, bright=1.0, double=True):
    n = int(0.09 * SR)
    x = np.zeros(n)
    for k, s in enumerate((0.0, 0.055) if double else (0.0,)):
        m = int(0.012 * SR)
        c = hp(rng.standard_normal(m), 2500 * bright, 2) * expenv(m, 0.0015)
        c += 0.5 * modal(0.012, 3200 * bright, (1.0, 1.7), (1.0, 0.4), (0.002, 0.0015), rng)
        i = int(s * SR)
        x[i:i + m] += c * (1.0 if k == 0 else 0.6)
    return norm(x, 0.4)


def sfx_mc_button(rng):
    """The game's button: a hard wooden click."""
    n = int(0.12 * SR)
    x = modal(0.12, 1250, (1.0, 1.52, 2.4, 3.3), (1.0, 0.8, 0.5, 0.3), (0.018, 0.012, 0.008, 0.006), rng)
    x += 0.6 * hp(rng.standard_normal(n), 1800, 2) * expenv(n, 0.003)
    return norm(x * expenv(n, 0.03, 0.0005), 0.6)


def sfx_key(rng):
    n = int(0.08 * SR)
    x = 0.7 * hp(rng.standard_normal(n), 2000, 2) * expenv(n, 0.004) + lp(rng.standard_normal(n), 700, 2) * expenv(n, 0.012)
    return norm(x, 0.45)


def sfx_whoosh(rng, dur=0.45, lo=300, hi=3500, up=True):
    c = (lambda q: lo + (hi - lo) * np.clip(q / dur, 0, 1)) if up else (lambda q: hi - (hi - lo) * np.clip(q / dur, 0, 1))
    x = shaped_noise(dur, c, lambda q: 0.8 + 0 * q, lambda q: np.sin(np.pi * np.clip(q / dur, 0, 1)) ** 2, rng)
    return norm(x, 0.5)


def bell(rng, m, dur=1.6, bright=1.0):
    f0 = 440.0 * 2 ** ((m - 69) / 12.0)
    x = modal(dur, f0, (1.0, 2.0, 3.0, 4.2, 5.4), (1.0, 0.45 * bright, 0.25 * bright, 0.12, 0.06),
              (0.9, 0.5, 0.3, 0.18, 0.1), rng, 0.001)
    return norm(x * np.clip(tt(len(x)) / 0.002, 0, 1), 0.7)


def sfx_join(rng, fast=False, leave=False):
    """Our login sound: two soft glassy notes, up for joined, down for left."""
    notes = (76, 83) if not leave else (83, 76)
    gap = 0.07 if fast else 0.12
    dur = 0.8 if fast else 1.6
    n = int((dur + gap) * SR)
    x = np.zeros(n)
    for k, m in enumerate(notes):
        b = bell(rng, m, dur, 0.6 if leave else 1.0)
        s = int(k * gap * SR)
        x[s:s + len(b)] += b[:n - s] * (0.8 if k == 0 else 1.0)
    x = lp(x, 5000 if not leave else 3000, 2)
    return norm(x, 0.5)


def sfx_tick(rng):
    n = int(0.03 * SR)
    x = hp(rng.standard_normal(n), 3000, 2) * expenv(n, 0.002)
    return norm(x, 0.25)


def sfx_glitch(rng, dur=0.18):
    """Digital damage: a stuttered, bit-crushed burst."""
    n = int(dur * SR)
    x = rng.standard_normal(n) * 0.5 + np.sign(np.sin(2 * np.pi * rng.uniform(80, 400) * tt(n)))
    hold = int(rng.integers(20, 90))
    x = np.repeat(x[::hold], hold)[:n]
    x = np.round(x * 3) / 3
    grain = int(0.02 * SR)
    for k in range(0, n - grain, grain * 2):
        if rng.random() < 0.5:
            x[k + grain:k + 2 * grain] = x[k:k + grain]
    return norm(fade(bp(x, 150, 9000, 2), 0.001, 0.01), 0.5)


def sfx_freeze(rng, dur):
    """The game hangs: a few milliseconds of sound stuck in the buffer, repeating."""
    g = int(0.058 * SR)
    src = 0.7 * np.sin(2 * np.pi * 110 * tt(g)) + 0.5 * bp(rng.standard_normal(g), 300, 3000, 2)
    src = norm(src)
    reps = int(dur * SR / g) + 1
    x = np.tile(src, reps)[:int(dur * SR)]
    x *= 1.0 - 0.3 * np.clip(tt(len(x)) / dur, 0, 1)
    return norm(lp(x, 6000, 2), 0.35)


def sfx_unfreeze(rng):
    n = int(0.5 * SR)
    x = sfx_glitch(rng, 0.25)
    y = sweep(0.5, 60, 900, 0.3) * np.linspace(0, 1, n) ** 2
    x = np.pad(x, (0, n - len(x))) * 0.6 + 0.4 * y
    return norm(x[::-1].copy(), 0.45)


def sfx_dig(rng, k=0):
    n = int(0.18 * SR)
    x = modal(0.18, rng.uniform(420, 560) * (1 + 0.02 * k), (1.0, 2.1, 3.3), (1.0, 0.5, 0.25), (0.03, 0.02, 0.012), rng,
              0.03)
    x += 0.6 * bp(rng.standard_normal(n), 800, 5000, 2) * expenv(n, 0.01)
    return norm(x * expenv(n, 0.05, 0.001), 0.5)


def sfx_break(rng):
    n = int(0.6 * SR)
    x = np.zeros(n)
    for k in range(9):
        d = sfx_dig(rng, k)
        s = int(rng.uniform(0, 0.25) * SR)
        x[s:s + len(d)] += d[:n - s] * rng.uniform(0.3, 1.0)
    x += 0.4 * grains(n, rng, 40, 600, 5000, (0.004, 0.02), 0.4)
    return norm(x, 0.7)


def sfx_sign_land(rng):
    n = int(0.9 * SR)
    x = np.zeros(n)
    for k, (s, a) in enumerate(((0.0, 1.0), (0.16, 0.5), (0.27, 0.3), (0.34, 0.15))):
        d = sfx_step(rng, 'wood')
        i = int(s * SR)
        x[i:i + len(d)] += d[:n - i] * a
    return norm(x, 0.7)


def sfx_punch(rng):
    n = int(0.3 * SR)
    x = np.zeros(n)
    sw = sfx_whoosh(rng, 0.16, 600, 2500)
    x[:len(sw)] += 0.5 * sw
    th = lp(rng.standard_normal(int(0.1 * SR)), 700, 2) * expenv(int(0.1 * SR), 0.02)
    s = int(0.12 * SR)
    x[s:s + len(th)] += norm(th, 0.8)[:n - s]
    return norm(x, 0.6)


def sfx_cloth(rng, dur=0.3):
    n = int(dur * SR)
    x = lp(grains(n, rng, 30, 300, 3000, (0.004, 0.02), dur * 0.8), 2500, 2)
    return norm(fade(x, 0.005, 0.05), 0.4)


def sfx_hum_voice(rng, kind='hmm', dur=0.5):
    """A villager: a nasal 'hmm' (or the disapproving 'hurr')."""
    if kind == 'hmm':
        f0 = lambda t: 125 + 18 * np.sin(np.pi * np.clip(t / dur, 0, 1)) - 10 * (t / dur)
        src = glottal(dur, f0, rng)
        x = formant(src, (260, 1050, 2400), (80, 200, 300), (1.0, 0.25, 0.08))
    else:
        f0 = lambda t: 118 - 30 * np.clip(t / dur, 0, 1)
        src = glottal(dur, f0, rng)
        x = formant(src, (520, 1000, 2300), (120, 180, 300), (1.0, 0.45, 0.1))
    env = np.clip(tt(len(x)) / 0.04, 0, 1) * np.clip((dur - tt(len(x))) / 0.12, 0, 1)
    return norm(x * env, 0.5)


def sfx_moo(rng):
    dur = 1.3
    f0 = lambda t: 105 + 20 * np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.5
    src = glottal(dur, f0, rng, 0.02)
    n = len(src)
    t = tt(n)
    k = np.clip(t / dur, 0, 1)
    x = (1 - k) * formant(src, (300, 900, 2300), (100, 200, 300), (1.0, 0.3, 0.05)) + k * formant(
        src, (380, 800, 2200), (120, 160, 300), (1.0, 0.5, 0.08))
    return norm(x * np.clip(t / 0.1, 0, 1) * np.clip((dur - t) / 0.25, 0, 1), 0.5)


def sfx_hurt(rng):
    """A player hurt, somewhere: a short pained grunt."""
    dur = 0.24
    src = glottal(dur, lambda t: 175 - 70 * np.clip(t / dur, 0, 1), rng, 0.03)
    x = formant(src, (560, 1100, 2500), (160, 220, 300), (1.0, 0.6, 0.12))
    x += 0.2 * bp(rng.standard_normal(len(x)), 800, 3000, 2)
    t = tt(len(x))
    return norm(x * np.clip(t / 0.01, 0, 1) * np.clip((dur - t) / 0.08, 0, 1), 0.5)


def sfx_chirp(rng):
    x = []
    for _ in range(int(rng.integers(2, 6))):
        d = rng.uniform(0.04, 0.11)
        n = int(d * SR)
        t = tt(n)
        f = rng.uniform(2600, 4600) * (1 + rng.uniform(-0.35, 0.35) * t / d)
        f += 300 * np.sin(2 * np.pi * rng.uniform(25, 60) * t)
        x.append(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / d) ** 2)
        x.append(np.zeros(int(rng.uniform(0.03, 0.1) * SR)))
    return norm(np.concatenate(x), 0.4)


def sfx_drip(rng):
    n = int(0.25 * SR)
    t = tt(n)
    f = rng.uniform(900, 2600) * (1 + 0.8 * np.exp(-t / 0.01))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * expenv(n, 0.03, 0.0005)
    return norm(x, 0.4)


def sfx_cave_sound(rng, dur=7.0):
    """The deep places' ambience: a slow, breathy, slightly tonal swell that comes from nowhere."""
    n = int(dur * SR)
    t = tt(n)
    f = rng.uniform(80, 160)
    k = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2
    tone = np.sin(2 * np.pi * f * t + 2.0 * np.sin(2 * np.pi * 0.3 * t)) + 0.5 * np.sin(2 * np.pi * f * 1.49 * t)
    air = shaped_noise(dur, lambda q: 300 + 500 * np.sin(np.pi * np.clip(q / dur, 0, 1)), lambda q: 0.6 + 0 * q,
                       lambda q: 1 + 0 * q, rng)
    x = (0.5 * norm(lp(tone, 600, 2)) + norm(air)) * k
    return norm(x, 0.5)


def sfx_whisper(rng, dur=2.5):
    """Breath shaped into something like syllables (never words)."""
    n = int(dur * SR)
    t = tt(n)
    x = rng.standard_normal(n)
    f1 = 400 + 400 * smooth_curve(n, 4.0, rng)
    f2 = 1100 + 1200 * smooth_curve(n, 5.0, rng)
    y = np.zeros(n)
    blk = 2048
    for s in range(0, n, blk):
        e = min(n, s + blk)
        m = (s + e) // 2
        y[s:e] = bp(x[max(0, s - 512):e], f1[m] * 0.8, f1[m] * 1.25, 2)[-(e - s):] + \
            0.7 * bp(x[max(0, s - 512):e], f2[m] * 0.85, f2[m] * 1.15, 2)[-(e - s):]
    syl = np.maximum(0, np.sin(2 * np.pi * np.cumsum(3.5 + 1.5 * smooth_curve(n, 2.0, rng)) / SR)) ** 1.5
    y = hp(y, 700, 2) * syl * np.sin(np.pi * np.clip(t / dur, 0, 1))
    return norm(y, 0.35)


def sfx_fan(rng, dur, spin_up=1.5):
    """A computer fan and power supply: whir, a hum and a whisper of coil whine."""
    n = int(dur * SR)
    t = tt(n)
    k = np.clip(t / spin_up, 0, 1)
    whir = bp(rng.standard_normal(n), 180, 1800, 2) * (0.3 + 0.7 * k)
    hum = 0.4 * np.sin(2 * np.pi * 120 * t) + 0.2 * np.sin(2 * np.pi * 240 * t)
    whine = 0.03 * np.sin(2 * np.pi * 7800 * t)
    blade = 0.25 * np.sin(2 * np.pi * (35 * k) * t)
    return 0.6 * norm(whir) + 0.12 * hum * k + whine * k + 0.05 * blade * whir


def sfx_boom(rng, dur=3.0, f=(55, 32)):
    n = int(dur * SR)
    x = sweep(dur, f[0], f[1], 0.4) * expenv(n, dur / 3.5, 0.02)
    x += 0.4 * lp(rng.standard_normal(n), 180, 2) * expenv(n, dur / 5, 0.01)
    return norm(np.tanh(1.4 * x), 0.8)


def sfx_sting_low(rng):
    """A low hit with a short dissonant bloom: felt more than heard."""
    n = int(4.0 * SR)
    t = tt(n)
    x = 0.8 * sfx_boom(rng, 4.0, (48, 30))
    cl = np.zeros(n)
    for f in (98.0, 103.8, 146.8, 155.6):
        cl += np.sin(2 * np.pi * f * t + rng.uniform(0, 6))
    cl = lp(cl, 900, 2) * np.clip(t / 0.25, 0, 1) * np.exp(-t / 1.3)
    return norm(x + 0.35 * norm(cl), 0.8)


def sfx_sting_note(rng):
    """One high piano note and its ghost a half step above."""
    a = piano(90, 0.8, 0.55, 3)
    b = piano(91, 0.8, 0.25, 5)
    n = max(len(a), len(b))
    x = np.zeros(n)
    x[:len(a)] += a
    x[int(0.04 * SR):int(0.04 * SR) + len(b)] += b[:n - int(0.04 * SR)]
    return norm(x, 0.6)


def sfx_clock(rng, dur):
    n = int(dur * SR)
    x = np.zeros(n)
    for k in range(int(dur * 2)):
        s = int(k * 0.5 * SR)
        m = int(0.03 * SR)
        c = modal(0.03, 2600 if k % 2 == 0 else 2250, (1.0, 2.7), (1.0, 0.3), (0.006, 0.004), rng) + \
            0.4 * hp(rng.standard_normal(m), 3000, 2) * expenv(m, 0.002)
        x[s:s + m] += c[:n - s]
    return norm(x, 0.35)


def sfx_draft(rng, dur):
    x = shaped_noise(dur, lambda q: 260 + 140 * np.sin(2 * np.pi * 0.35 * q) + 60 * np.sin(2 * np.pi * 1.3 * q),
                     lambda q: 0.35 + 0 * q, lambda q: np.clip(q / 1.0, 0, 1) * np.clip((dur - q) / 1.0, 0, 1), rng)
    return norm(x, 0.45)


def sfx_fire(rng, dur):
    n = int(dur * SR)
    x = 0.25 * lp(rng.standard_normal(n), 900, 2) * (0.6 + 0.4 * smooth_curve(n, 1.5, rng))
    for _ in range(int(dur * 9)):
        s = int(rng.uniform(0, dur - 0.02) * SR)
        m = int(rng.uniform(0.002, 0.008) * SR)
        x[s:s + m] += hp(rng.standard_normal(m), 1500, 2) * rng.uniform(0.2, 1.0)
    return norm(fade(x, 0.4, 0.8), 0.4)


# ---------------------------------------------------------------------------------------------
# instruments
# ---------------------------------------------------------------------------------------------
def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


_NOTE = {}


def piano(m, dur, vel, seed=0):
    """A soft felt piano note."""
    key = (m, round(dur, 2), round(vel, 2), seed % 4)
    if key in _NOTE:
        return _NOTE[key]
    rng = np.random.default_rng(1000 * m + seed % 4)
    f0 = midi_hz(m)
    ring = 2.6
    n = int((dur + ring) * SR)
    t = tt(n)
    B = 2.5e-4 * (f0 / 261.6) ** 0.6
    tau0 = float(np.clip(3.4 * (261.6 / f0) ** 0.5, 0.9, 6.0))
    x = np.zeros(n)
    for k in range(1, 12):
        fk = k * f0 * np.sqrt(1 + B * k * k)
        if fk > 9000:
            break
        amp = (1.0 / k) ** (1.6 - 0.55 * vel)
        tau = tau0 / (1 + 0.45 * (k - 1))
        env = 0.62 * np.exp(-t / (tau * 0.22)) + 0.38 * np.exp(-t / tau)
        for d in (-1.0, 1.0):
            x += 0.5 * amp * np.sin(2 * np.pi * fk * (1 + d * 0.0005) * t + rng.uniform(0, 6.28)) * env
    x *= np.clip(t / 0.005, 0, 1)
    thump = lp(rng.standard_normal(n), 900, 2) * np.exp(-t / 0.018) * 0.05 * vel
    x = lp(x + thump, 1800 + 3000 * vel, 2)
    rel = np.where(t < dur, 1.0, 0.25 + 0.75 * np.exp(-np.clip(t - dur, 0, None) / 0.18))
    x = x * rel * np.clip((n / SR - t) / 0.3, 0, 1)
    out = (x / (np.abs(x).max() + 1e-9) * vel).astype(np.float32)
    if len(_NOTE) > 400:
        _NOTE.clear()
    _NOTE[key] = out
    return out


def pad(freqs, dur, rng, attack=1.2, release=1.5, bright=1600.0):
    n = int((dur + release) * SR)
    t = tt(n)
    x = np.zeros(n)
    for f in freqs:
        for d in (-0.006, 0.0, 0.007):
            vib = 1 + 0.003 * np.sin(2 * np.pi * rng.uniform(4.2, 5.4) * t + rng.uniform(0, 6.28))
            ph = np.cumsum(f * (1 + d) * vib) / SR + rng.uniform()
            x += 2 * np.mod(ph, 1.0) - 1
    x = lp(lp(x, bright, 2), bright * 1.5, 2)
    env = np.clip(t / attack, 0, 1) ** 1.5 * np.where(t < dur, 1.0, np.clip(1 - (t - dur) / release, 0, 1))
    return x * env / (3 * len(freqs))


def drone(rng, dur, level=1.0, base=36.7):
    """Something is wrong: a low beating drone (a minor second), air, and a slow swell."""
    n = int(dur * SR)
    t = tt(n)
    x = np.zeros(n)
    for f, a in ((base, 1.0), (base * 1.059, 0.8), (base * 2, 0.45), (base * 2.12, 0.35)):
        x += a * np.tanh(1.4 * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)))
    x = lp(x, 380, 2)
    air = bp(rng.standard_normal(n), 1500, 5000, 2) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.23 * t + rng.uniform(0, 6)))
    x = norm(x) + 0.12 * norm(air)
    env = np.clip(t / max(0.3 * dur, 0.5), 0, 1) ** 1.6 * np.clip((dur - t) / 1.5, 0, 1)
    return x * env * level


# ---------------------------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------------------------
BPM = 66.0
BEAT = 60.0 / BPM
CHORDS = {'F': (41, 48, 52, 57), 'C/E': (40, 47, 52, 55), 'Dm7': (38, 45, 48, 53), 'Bb': (34, 41, 45, 50),
          'Gm7': (31, 38, 41, 46), 'Csus': (36, 43, 48, 53), 'Dm9': (38, 45, 52, 53), 'Am7': (33, 40, 43, 48),
          'F/A': (33, 41, 48, 52), 'C7sus': (36, 43, 46, 53)}
F4, G4, A4, Bb4, C5, D5, E5, F5, G5, A5 = 65, 67, 69, 70, 72, 74, 76, 77, 79, 81
# the rules lullaby: (chord, [(beat, note, beats)])
SONG = [('F', [(0, A4, 1), (1, C5, 1), (2, F5, 2)]), ('C/E', [(0, E5, 1.5), (1.5, D5, 0.5), (2, C5, 2)]),
        ('Dm7', [(0, D5, 1), (1, F5, 1), (2, A5, 1.5), (3.5, G5, 0.5)]), ('Bb', [(0, F5, 3)]),
        ('F', [(0, A4, 1), (1, C5, 1), (2, F5, 1), (3, G5, 1)]), ('C/E', [(0, E5, 2), (2, C5, 2)]),
        ('Gm7', [(0, Bb4, 1), (1, D5, 1), (2, F5, 1), (3, E5, 1)]), ('Csus', [(0, D5, 2), (2, C5, 2)]),
        ('Dm9', [(0, A5, 2), (2, G5, 1), (3, F5, 1)]), ('Am7', [(0, E5, 2), (2, C5, 2)]),
        ('Bb', [(0, D5, 1), (1, F5, 1), (2, A5, 1), (3, G5, 1)]), ('F/A', [(0, F5, 3)]),
        ('Gm7', [(0, Bb4, 1), (1, D5, 1), (2, G5, 2)]), ('C7sus', [(0, F5, 1), (1, E5, 1), (2, D5, 1), (3, C5, 1)]),
        ('F', [(0, A4, 2), (2, C5, 2)]), ('F', [(0, F5, 4)])]


def render_song(bars, rng, melody=True, accomp=True, octave=0):
    """Mono render of the given bars of SONG."""
    dur = len(bars) * 4 * BEAT + 5.0
    x = np.zeros(int(dur * SR))

    def put(sig, t, g):
        s = int(t * SR)
        e = min(len(x), s + len(sig))
        x[s:e] += sig[:e - s] * g

    for bi, (chord, mel) in enumerate(bars):
        t0 = bi * 4 * BEAT
        if accomp:
            c = CHORDS[chord]
            put(piano(c[0], 4 * BEAT, 0.42, bi), t0, 0.9)
            for k, (m, b) in enumerate(zip(c[1:], (0.5, 1.0, 2.0))):
                put(piano(m, 3.0 * BEAT - b * BEAT, 0.3, bi + k), t0 + b * BEAT + rng.uniform(0, 0.015), 0.7)
            put(piano(c[2] + 12, 1.5 * BEAT, 0.22, bi), t0 + 3.0 * BEAT, 0.55)
        if melody:
            for (b, m, ln) in mel:
                put(piano(m + 12 * octave, ln * BEAT * 0.95, 0.5, bi), t0 + b * BEAT + rng.uniform(0, 0.012), 1.0)
    return x


def music_piece(name, rng):
    """(mono signal) of a named piece from its start."""
    if name == 'perfect':
        return render_song(SONG + SONG[:8], rng)
    if name == 'perfect2':
        return render_song(SONG[8:], rng)
    raise KeyError(name)


def reversed_lullaby(rng, dur):
    mel = render_song(SONG[:8], rng, accomp=False)
    L, R = apply_reverb(mel.astype(np.float32), mel.astype(np.float32), 'hall')
    y = resample(((L + R) * 0.5)[::-1].copy(), 0.72)
    y = lp(y, 1600, 2)
    n = int(dur * SR)
    y = np.pad(y, (0, max(0, n - len(y))))[:n]
    t = tt(n)
    return norm(y, 1.0) * np.clip(t / 3.0, 0, 1) * np.clip((dur - t) / 3.0, 0, 1)


# ---------------------------------------------------------------------------------------------
# ambience beds (stereo, n samples)
# ---------------------------------------------------------------------------------------------
def bed(kind, n, rng):
    t = tt(n)
    if kind == 'day':
        gust = smooth_curve(n, 0.15, rng, 0.35, 1.0)
        L, R = stereo_noise_bed(n, rng, 120, 800)
        lv, rv = stereo_noise_bed(n, rng, 1800, 7000)
        leaves = smooth_curve(n, 0.3, rng, 0.1, 1.0)
        return (0.5 * L * gust + 0.12 * lv * leaves * gust, 0.5 * R * gust + 0.12 * rv * leaves * gust)
    if kind == 'interior':
        L, R = bed('day', n, rng)
        tone = lp(rng.standard_normal(n), 120, 2)
        return (0.5 * lp(L, 600, 2) + 0.15 * tone, 0.5 * lp(R, 600, 2) + 0.15 * tone)
    if kind == 'night':
        gust = smooth_curve(n, 0.1, rng, 0.2, 0.7)
        L, R = stereo_noise_bed(n, rng, 90, 500)
        cr = np.zeros(n)
        crr = np.zeros(n)
        for k in range(5):                       # crickets
            f = rng.uniform(4200, 5200)
            rate = rng.uniform(1.4, 2.4)
            ph = rng.uniform(0, 1)
            gate = (((t * rate + ph) % 1.0) < 0.22).astype(float)
            puls = (np.sin(2 * np.pi * 32 * t) > 0.2).astype(float)
            c = np.sin(2 * np.pi * f * t) * gate * puls
            c = lp(c, 7000, 1) * rng.uniform(0.2, 0.5)
            p = rng.uniform(-0.8, 0.8)
            cr += c * np.cos((p + 1) * np.pi / 4)
            crr += c * np.sin((p + 1) * np.pi / 4)
        return (0.45 * L * gust + 0.035 * cr, 0.45 * R * gust + 0.035 * crr)
    if kind == 'night_in':
        L, R = bed('night', n, rng)
        tone = lp(rng.standard_normal(n), 110, 2)
        return (0.55 * lp(L, 700, 2) + 0.12 * tone, 0.55 * lp(R, 700, 2) + 0.12 * tone)
    if kind == 'tunnel':
        L, R = stereo_noise_bed(n, rng, 40, 220)
        air = smooth_curve(n, 0.2, rng, 0.4, 1.0)
        return (0.7 * L * air, 0.7 * R * air)
    if kind == 'cavern':
        L, R = stereo_noise_bed(n, rng, 30, 160)
        roar = smooth_curve(n, 0.05, rng, 0.5, 1.0)
        a, b = stereo_noise_bed(n, rng, 300, 1200)
        return (0.9 * L * roar + 0.05 * a, 0.9 * R * roar + 0.05 * b)
    if kind == 'chamber':
        sub = np.sin(2 * np.pi * 29.0 * t) * 0.3 + np.sin(2 * np.pi * 30.7 * t) * 0.25
        L, R = stereo_noise_bed(n, rng, 25, 90)
        breath = smooth_curve(n, 0.08, rng, 0.2, 1.0)
        return (0.25 * sub + 0.6 * L * breath, 0.25 * sub + 0.6 * R * breath)
    if kind == 'wrong':
        gust = smooth_curve(n, 0.12, rng, 0.3, 1.0)
        L, R = stereo_noise_bed(n, rng, 150, 900)
        whistle = (np.sin(2 * np.pi * 523 * t) + np.sin(2 * np.pi * 529.5 * t)) * 0.5
        wl = lp(whistle * smooth_curve(n, 0.1, rng, 0.0, 1.0), 800, 1)
        low = np.sin(2 * np.pi * 41.2 * t) * 0.15
        return (0.5 * L * gust + 0.03 * wl + low, 0.5 * R * gust + 0.03 * wl + low)
    if kind == 'room':
        tone = lp(rng.standard_normal(n), 140, 2)
        L, R = bed('wrong', n, rng)
        fan = sfx_fan(rng, n / SR, 0.1)
        return (0.3 * lp(L, 500, 2) + 0.1 * tone + 0.25 * fan, 0.3 * lp(R, 500, 2) + 0.1 * tone + 0.25 * fan)
    if kind == 'pc':
        fan = sfx_fan(rng, n / SR)
        tone = lp(rng.standard_normal(n), 150, 2)
        return (0.5 * fan + 0.1 * tone, 0.5 * fan + 0.1 * tone)
    raise KeyError(kind)


BED_LEVEL = {'day': 0.12, 'interior': 0.16, 'night': 0.24, 'night_in': 0.3, 'tunnel': 0.18, 'cavern': 0.2,
             'chamber': 0.22, 'wrong': 0.12, 'room': 0.14, 'pc': 0.06}


def ambient_events(kind, t0, t1, rng):
    """(time, sound, gain, pan, room) scattered through a stretch of a place."""
    out = []
    dur = t1 - t0
    if kind == 'day':
        for _ in range(int(dur * 0.35)):
            out.append((rng.uniform(t0, t1), sfx_chirp(rng), rng.uniform(0.04, 0.12), rng.uniform(-0.9, 0.9), 'out'))
        for _ in range(int(dur / 25)):
            out.append((rng.uniform(t0, t1), sfx_hum_voice(rng, 'hmm', rng.uniform(0.35, 0.6)), 0.03,
                        rng.uniform(-0.8, 0.8), 'out'))
    if kind in ('night_in', 'interior'):
        for _ in range(int(dur / 11)):
            out.append((rng.uniform(t0, t1), sfx_creak(rng, rng.uniform(0.3, 0.8), (50, 110)), 0.05,
                        rng.uniform(-0.6, 0.6), 'room'))
    if kind in ('tunnel', 'cavern'):
        for _ in range(int(dur * (0.5 if kind == 'tunnel' else 0.8))):
            out.append((rng.uniform(t0, t1), sfx_drip(rng), rng.uniform(0.03, 0.09), rng.uniform(-0.9, 0.9),
                        'tunnel' if kind == 'tunnel' else 'cave'))
    if kind in ('cavern', 'chamber'):
        for _ in range(max(1, int(dur / 22))):
            out.append((rng.uniform(t0, max(t0, t1 - 6)), sfx_cave_sound(rng, rng.uniform(5, 8)), 0.09,
                        rng.uniform(-0.7, 0.7), 'cave' if kind == 'cavern' else 'hall'))
    if kind in ('cavern', 'chamber', 'wrong', 'night_in'):
        for _ in range(max(1, int(dur / 14))):           # whispers, under everything
            out.append((rng.uniform(t0, max(t0, t1 - 3)), sfx_whisper(rng, rng.uniform(1.8, 3.2)), 0.012,
                        rng.uniform(-1, 1), 'cave' if kind != 'night_in' else 'room'))
    if kind == 'wrong':
        for _ in range(int(dur * 0.12)):                 # birds... backwards
            out.append((rng.uniform(t0, t1), sfx_chirp(rng)[::-1].copy(), 0.05, rng.uniform(-0.9, 0.9), 'wrong'))
    return out


# ---------------------------------------------------------------------------------------------
# building the mix
# ---------------------------------------------------------------------------------------------
UI = {'click', 'click_play', 'click_soft', 'window_open', 'key_click', 'toast', 'folder_pop', 'mc_button', 'loading',
      'log_tick', 'chat_msg', 'glitch', 'glitch_tick', 'gui_open', 'cut_in'}


def build(film_path=None, out_path=None, seed=4, report=True, probe=None):
    """probe: list of seconds; prints each bus's loudness there (after the master gain) instead of guessing."""
    t_start = time.time()
    d = json.load(open(film_path or os.path.join(OUT, 'cues.json')))
    dur = float(d['duration'])
    cues = [(float(c[0]), c[1], c[2]) for c in d['cues']]
    subs = d['subs']
    n = int((dur + 2.0) * SR)
    nc = int((dur + 2.0) * CTL) + 2
    rng = np.random.default_rng(seed)

    # --- ambience state: gain curves per bed ----------------------------------------------------------------------
    kinds = list(BED_LEVEL)
    ev = {k: [] for k in kinds}
    cur = None
    place_at = []                       # (t, place) for choosing reverbs
    for (t, name, kw) in sorted(cues, key=lambda c: c[0]):
        if name == 'amb':
            k = kw['kind']
            for o in kinds:
                if o != k and o != 'pc':
                    ev[o].append((t, 0.0, 0.4))
            ev['pc'].append((t, 0.0, 0.4))
            ev[k].append((t, 1.0, 0.4 if cur is not None else 0.8))
            cur = k
            place_at.append((t, k))
        elif name == 'amb_fade':
            k = kw['to']
            fd = float(kw.get('dur', 2.0))
            if cur is not None:
                ev[cur].append((t, 0.0, fd))
            ev[k].append((t, 1.0, fd))
            cur = k
            place_at.append((t + fd * 0.5, k))
        elif name == 'amb_duck' and cur is not None:
            ev[cur].append((t, float(kw.get('to', 0.3)), 1.0))
        elif name == 'silence':
            for o in kinds:
                ev[o].append((t, 0.0, 0.02))
            cur = None
            place_at.append((t, None))
        elif name == 'pc_hum_start':
            ev['pc'].append((t, 1.0, 1.0))
        elif name == 'cut_in':
            for o in kinds:
                ev[o].append((t, 0.0, 0.01))
            ev['pc'].append((t, 1.0, 0.05))
            cur = None
            place_at.append((t, None))
    gains = {k: ramp_curve(nc, ev[k]) for k in kinds}

    # masks: silence_all (ambience, music, drones) and cut_silence (everything)
    sil_ev, cut_ev = [], []
    for (t, name, kw) in cues:
        if name == 'silence_all':
            dd = float(kw.get('dur', 2.0))
            sil_ev += [(t, 0.0, 0.04), (t + dd, 1.0, 1.0)]
        if name == 'cut_silence':
            cut_ev += [(t, 0.0, 0.005), (t + 1.2, 1.0, 0.01)]
    sil = ramp_curve(nc, sil_ev, 1.0)
    cut = ramp_curve(nc, cut_ev, 1.0)
    # under his lines the beds and music dip a little (room for the voice-over)
    duck = np.ones(nc, np.float32)
    for (a, b, text) in subs:
        i0, i1 = int(a * CTL), int(b * CTL)
        duck[max(0, i0 - 15):i1 + 30] = 0.7
    duck = signal.sosfiltfilt(signal.butter(1, 2.0, fs=CTL, output='sos'), duck).astype(np.float32)

    def place(t):
        p = None
        for (tp, k) in place_at:
            if tp <= t:
                p = k
            else:
                break
        return p

    # --- ambience ---------------------------------------------------------------------------------------------------------
    amb = Bus(n)
    rooms = {r: Bus(n) for r in ROOMS}            # silenced by silence_all (ambient events, and tails into a silence)
    thru = {r: Bus(n) for r in ROOMS}             # sounds placed inside a silence: they are the point of it
    windows = [(t, t + float(kw.get('dur', 2.0))) for (t, name, kw) in cues if name == 'silence_all']

    def bus_for(room, t):
        return thru[room] if any(a - 0.02 <= t < b for (a, b) in windows) else rooms[room]
    for k in kinds:
        g = gains[k]
        on = np.nonzero(g > 1e-4)[0]
        if not len(on):
            continue
        # contiguous stretches
        breaks = np.nonzero(np.diff(on) > CTL)[0]
        starts = np.r_[on[0], on[breaks + 1]]
        ends = np.r_[on[breaks], on[-1]]
        for (s, e) in zip(starts, ends):
            t0, t1 = s / CTL, (e + 1) / CTL
            ns = int((t1 - t0) * SR) + 1
            L, R = bed(k, ns, rng)
            gl = to_audio(g[s:e + 2], ns) * BED_LEVEL[k]
            amb.add(L * gl, t0, 1.0, xr=R * gl)
            for (te, snd, gain, pan, room) in ambient_events(k, t0, t1, rng):
                gg = float(g[min(int(te * CTL), len(g) - 1)])
                if gg > 0.05:
                    rooms[room].add(snd, te, gain * gg, pan)
        print(f'[audio] bed {k}: {len(starts)} stretch(es)', flush=True)

    # --- music, drones --------------------------------------------------------------------------------------------------------
    mus = Bus(n)
    dro = Bus(n)
    stops = sorted(t for (t, name, kw) in cues if name == 'music_stop')
    for (t, name, kw) in cues:
        if name == 'music':
            x = music_piece(kw['piece'], rng)
            t_end = next((s for s in stops if s > t), t + len(x) / SR)
            m = int((t_end - t) * SR)
            x = x[:m + int(0.4 * SR)].copy()
            x[m:] *= np.linspace(1, 0, len(x) - m)
            mus.add(x, t, 0.08)
        elif name == 'mc_menu':
            x = pad([midi_hz(m) for m in (48, 55, 59, 62, 64)], 7.5, rng, 1.5, 2.0)
            for (dt, m) in ((1.2, 79), (3.4, 76), (5.3, 74)):
                p = piano(m, 1.5, 0.35, 1)
                x[int(dt * SR):int(dt * SR) + len(p)] += 0.5 * p[:len(x) - int(dt * SR)]
            mus.add(norm(x, 0.8), t, 0.1)
        elif name == 'title_drone':
            x = drone(rng, 4.6, 1.0, 32.7) + 0.4 * sfx_boom(rng, 4.6, (65, 30))
            mus.add(norm(x, 0.9), t, 0.32)
        elif name == 'lullaby_rev':
            mus.add(reversed_lullaby(rng, float(kw.get('dur', 25.0))), t, 0.06)
        elif name == 'drone':
            x = drone(rng, float(kw.get('dur', 6.0)), 1.0)
            dro.add(x, t, 0.11 * float(kw.get('level', 0.4)) / 0.4, rng.uniform(-0.2, 0.2))
        elif name == 'sting_low':
            dro.add(sfx_sting_low(rng), t, 0.12)
        elif name == 'sting_note':
            bus_for('hall', t).add(sfx_sting_note(rng), t, 0.04)
        elif name == 'sub_hit':
            dro.add(sfx_boom(rng, 2.2, (50, 28)), t, 0.35 * float(kw.get('gain', 1.0)))
        elif name == 'cave_reveal':
            x = sfx_boom(rng, 6.0, (42, 26)) * 0.6 + sfx_cave_sound(rng, 6.0)
            bus_for('cave', t).add(norm(x, 0.9), t, 0.07)

    # --- effects -------------------------------------------------------------------------------------------------------------------
    fire_on = []
    for (t, name, kw) in cues:
        pan = float(kw.get('pan', 0.0))
        dist = float(kw.get('dist', 1.0))
        g = float(kw.get('gain', 1.0)) / max(1.0, dist / 3.0) ** 0.8
        pl = place(t)
        room = PLACE_ROOM.get(pl, 'hook')
        room = 'dry' if name in UI else room
        s = None
        if name == 'step':
            surf = kw.get('surface', 'grass')
            srng = np.random.default_rng(7) if kw.get('even') else rng      # the finale's steps: all the same
            s = sfx_step(srng, surf, bool(kw.get('fast')))
            g *= 0.22 * (1.25 if kw.get('fast') else 1.0)
            if kw.get('echo'):
                room = 'cave' if pl == 'cavern' else 'tunnel'
        elif name == 'step_out':
            s = lp(sfx_step(rng, 'grass'), 900, 2)
            g *= 0.16
        elif name == 'step_in':
            s = resample(sfx_step(rng, 'wood'), 0.85)
            pre = s[::-1] * 0.25
            s = np.concatenate([pre[-int(0.08 * SR):], s])
            g *= 0.26
        elif name == 'step_above':
            w_ = lp(sfx_step(rng, 'wood'), 380, 2)                           # through the ceiling
            dust = grains(int(0.5 * SR), rng, 18, 2500, 9000, (0.001, 0.004), 0.4)
            s = np.zeros(int(0.5 * SR))
            s[:len(w_)] += norm(w_, 0.6)
            s += 0.05 * dust
            g *= 0.4
        elif name == 'step_behind':
            base = sfx_step(rng, 'carpet' if pl == 'room' else 'wood')
            s = resample(base, 0.7)
            s = np.concatenate([s, s[::-1] * 0.3])
            g = 0.4 / max(0.5, float(kw.get('dist', 1.0)))
            room = 'room'
        elif name == 'door_open':
            s = sfx_door(rng, reverse=bool(kw.get('reverse')))           # the finale's door plays backwards
            g *= 0.3
        elif name == 'chest_open':
            s = sfx_chest(rng)
            g *= 0.14
        elif name == 'chest_close':
            s = sfx_chest(rng, close=True)
            g *= 0.14
        elif name == 'trapdoor_open':
            s = sfx_trapdoor(rng)
            g *= 0.34
        elif name == 'ladder':
            s = sfx_ladder(rng)
            g *= 0.2
        elif name in ('page', 'page_far'):
            s = sfx_page(rng)
            g *= 0.2 if name == 'page' else 0.07
            if name == 'page_far':
                room = 'cave'
        elif name == 'book_open':
            s = sfx_book_open(rng)
            g *= 0.24
        elif name == 'paper':
            s = sfx_paper(rng)
            g *= 0.2
        elif name == 'item_pickup':
            s = sfx_pop(rng)
            g *= 0.22
        elif name == 'item_frame_take':
            s = sfx_pop(rng, (400, 800)) + 0.4 * np.pad(sfx_step(rng, 'wood'), (0, 0))[:int(0.09 * SR)]
            g *= 0.22
        elif name == 'gui_open':
            s = sfx_click(rng, 0.8, False)
            g *= 0.12
        elif name == 'dig':
            s = sfx_dig(rng, int(kw.get('k', 0)))
            g *= 0.3
        elif name == 'block_break':
            s = sfx_break(rng)
            g *= 0.4
        elif name == 'sign_land':
            s = sfx_sign_land(rng)
            g *= 0.36
        elif name == 'punch':
            s = sfx_punch(rng)
            g *= 0.3
        elif name == 'wool_break':
            s = sfx_cloth(rng, 0.25)
            g *= 0.3
        elif name == 'fire':
            fire_on.append((t, t + float(kw.get('dur', 10.0))))
            s = sfx_fire(rng, float(kw.get('dur', 10.0)))
            g *= 0.14
            pan = 0.3
        elif name == 'fire_off':
            s = lp(rng.standard_normal(int(0.6 * SR)), 2500, 2) * expenv(int(0.6 * SR), 0.15)
            g *= 0.02
        elif name == 'clock_tick':
            s = sfx_clock(rng, float(kw.get('dur', 3.0)))
            g *= 0.25
        elif name == 'draft':
            s = sfx_draft(rng, float(kw.get('dur', 5.0)))
            g *= 0.3 * float(kw.get('level', 1.0))
        elif name == 'bed_click':
            s = sfx_click(rng, 0.7, False) + 0.5 * np.pad(sfx_cloth(rng, 0.2), (0, 0))[:int(0.09 * SR)]
            g *= 0.2
        elif name in ('chair_creak', 'turn_creak'):
            dd = float(kw.get('dur', 1.4)) if name == 'turn_creak' else 1.4
            s = sfx_creak(rng, dd, (18, 34) if name == 'turn_creak' else (60, 120), 1400)
            g *= 0.035 if name == 'turn_creak' else 0.12
        elif name == 'turn_whoosh':
            s = sfx_whoosh(rng, 0.5, 400, 3000)
            g *= 0.4
        elif name == 'key_click':
            s = sfx_key(rng)
            g *= 0.22
        elif name in ('click', 'click_play', 'click_soft'):
            s = sfx_click(rng, 1.0 if name != 'click_soft' else 0.7)
            g *= 0.16 if name != 'click_soft' else 0.08
        elif name == 'window_open':
            s = sfx_whoosh(rng, 0.2, 1500, 6000)
            g *= 0.05
        elif name == 'toast':
            s = sfx_join(rng, True)
            g *= 0.15
        elif name == 'folder_pop':
            s = sfx_pop(rng, (300, 700), 0.06)
            g *= 0.1
        elif name == 'mc_button':
            s = sfx_mc_button(rng)
            g *= 0.3
        elif name == 'loading':
            nl = int(2.0 * SR)
            s = norm(sweep(2.0, 50, 90, 1.5) * np.clip(tt(nl) / 1.5, 0, 1), 0.5)
            g *= 0.12
        elif name in ('join', 'leave'):
            fast = any(abs(t - t2) < 0.9 and t2 != t for (t2, n2, _) in cues if n2 in ('join', 'leave'))
            s = sfx_join(rng, fast, leave=(name == 'leave'))
            g *= 0.09
            room = 'hook' if pl is None else room
        elif name == 'log_tick':
            s = sfx_tick(rng)
            g *= 0.3
        elif name == 'chat_msg':
            s = sfx_tick(rng)
            g *= 0.15
        elif name == 'glitch':
            s = sfx_glitch(rng, 0.14)
            g *= 0.28
        elif name == 'glitch_tick':
            s = sfx_glitch(rng, 0.05)
            g *= 0.2
        elif name == 'freeze_buzz':
            s = sfx_freeze(rng, float(kw.get('dur', 4.0)))
            g *= 0.2
            room = 'dry'
        elif name == 'unfreeze':
            s = sfx_unfreeze(rng)
            g *= 0.35
            room = 'dry'
        elif name == 'villager':
            s = sfx_hum_voice(rng, 'hmm', rng.uniform(0.4, 0.65))
            g *= 0.14
        elif name == 'villager_no':
            s = sfx_hum_voice(rng, 'no', 0.55)
            g *= 0.18
        elif name == 'cow':
            s = sfx_moo(rng)
            g *= 0.3
        elif name == 'hurt_far':
            s = lp(sfx_hurt(rng), 1600, 2)
            g = 0.09
            room = 'cave' if pl == 'cavern' else 'hall'
        elif name == 'wind_crater':
            s = shaped_noise(5.0, lambda q: 300 + 250 * np.sin(np.pi * np.clip(q / 5, 0, 1)), lambda q: 0.8 + 0 * q,
                             lambda q: np.sin(np.pi * np.clip(q / 5, 0, 1)) ** 2, rng)
            s = norm(s, 0.6)
            g *= 0.14
        elif name == 'pc_hum_start':
            nn = int(1.2 * SR)
            s = norm(sweep(1.2, 20, 60, 0.6) * np.clip(tt(nn) / 0.8, 0, 1) * np.clip((1.2 - tt(nn)) / 0.4, 0, 1), 0.3)
            g *= 0.1
        elif name in ('amb', 'amb_fade', 'amb_duck', 'silence', 'silence_all', 'music', 'music_stop', 'drone',
                      'sting_low', 'sting_note', 'sub_hit', 'cave_reveal', 'mc_menu', 'title_drone', 'lullaby_rev',
                      'chat', 'cut_in', 'cut_silence'):
            continue
        else:
            print('[audio] no sound for cue', name, flush=True)
            continue
        if dist > 3.0:
            s = lp(s, max(900.0, 9000.0 / (dist / 3.0)), 1)
        bus_for(room, t).add(s, t, g, pan)

    # --- reverbs, buses, masks, master --------------------------------------------------------------------------------------------
    print(f'[audio] events placed ({time.time() - t_start:.0f}s); reverbs...', flush=True)
    fx = Bus(n)
    fxt = Bus(n)
    for dst, src in ((fx, rooms), (fxt, thru)):
        for r, b in src.items():
            if not np.any(b.L) and not np.any(b.R):
                continue
            L, R = apply_reverb(b.L, b.R, r)
            dst.L += L
            dst.R += R
    silA = to_audio(sil, n)
    cutA = to_audio(cut, n)
    duckA = to_audio(duck, n)
    L = (amb.L * silA * duckA + mus.L * silA * duckA + dro.L * silA + fx.L * silA + fxt.L) * cutA
    R = (amb.R * silA * duckA + mus.R * silA * duckA + dro.R * silA + fx.R * silA + fxt.R) * cutA
    # the ambience's own rooms: ambient events already went through the rooms above
    L, R = compress(L, R)
    lufs = integrated_lufs(L, R)
    g = 10 ** ((-16.0 - lufs) / 20.0)
    if probe:
        parts = {'amb': (amb.L * silA * duckA * cutA, amb.R * silA * duckA * cutA),
                 'music': (mus.L * silA * duckA * cutA, mus.R * silA * duckA * cutA),
                 'drones': (dro.L * silA * cutA, dro.R * silA * cutA),
                 'fx': (fx.L * silA * cutA, fx.R * silA * cutA), 'fx_thru': (fxt.L * cutA, fxt.R * cutA)}
        for sec in probe:
            s0, s1 = int(sec * SR), int((sec + 1) * SR)
            row = []
            for k, (a, b) in parts.items():
                ka, kb = k_weight(a[s0:s1].astype(np.float64) * g), k_weight(b[s0:s1].astype(np.float64) * g)
                row.append(f'{k} {-0.691 + 10 * np.log10(np.mean(ka ** 2) + np.mean(kb ** 2) + 1e-12):6.1f}')
            print(f'[probe] {sec:6.1f}s  ' + '  '.join(row), flush=True)
    L, R = limiter(L * g, R * g, ceiling=0.89)
    out = out_path or os.path.join(OUT, 'film_audio.wav')
    write_wav(out, L[:int(dur * SR)], R[:int(dur * SR)])
    if report:
        print(f'[audio] {out}: {dur:.1f}s, integrated {integrated_lufs(L, R):.1f} LUFS (was {lufs:.1f}), '
              f'peak {max(np.abs(L).max(), np.abs(R).max()):.2f}, {time.time() - t_start:.0f}s', flush=True)
    return out


def compress(L, R, thresh_db=-24.0, ratio=2.5, attack=0.008, release=0.25):
    hop = SR // 1000
    m = 0.5 * (L.astype(np.float64) ** 2 + R.astype(np.float64) ** 2)
    win = int(0.03 * SR)
    c = np.cumsum(np.r_[0.0, m])
    rms = np.sqrt((c[win:] - c[:-win]) / win + 1e-12)
    rms = np.r_[rms, np.full(win - 1, rms[-1])][::hop]
    lev = 20 * np.log10(rms)
    target = -np.maximum(lev - thresh_db, 0.0) * (1.0 - 1.0 / ratio)
    aa = np.exp(-1.0 / (attack * 1000))
    ar = np.exp(-1.0 / (release * 1000))
    g = np.zeros_like(target)
    cur = 0.0
    for i, tv in enumerate(target):
        cur = aa * cur + (1 - aa) * tv if tv < cur else ar * cur + (1 - ar) * tv
        g[i] = cur
    gl = 10 ** (np.interp(np.arange(len(L)), np.arange(len(g)) * hop, g) / 20.0)
    return (L * gl).astype(np.float32), (R * gl).astype(np.float32)


def k_weight(x):
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    return signal.lfilter(b2, a2, signal.lfilter(b1, a1, x))


def integrated_lufs(L, R):
    kl, kr = k_weight(L.astype(np.float64)), k_weight(R.astype(np.float64))
    blk = int(0.4 * SR)
    hop = int(0.1 * SR)
    cl = np.cumsum(np.r_[0.0, kl ** 2])
    cr = np.cumsum(np.r_[0.0, kr ** 2])
    s = np.arange(0, len(kl) - blk, hop)
    ms = (cl[s + blk] - cl[s]) / blk + (cr[s + blk] - cr[s]) / blk
    lk = -0.691 + 10 * np.log10(ms + 1e-12)
    g = ms[lk > -70]
    if len(g) == 0:
        return -70.0
    rel = -0.691 + 10 * np.log10(g.mean()) - 10
    g2 = g[(-0.691 + 10 * np.log10(g + 1e-12)) > rel]
    return float(-0.691 + 10 * np.log10(g2.mean()))


def limiter(L, R, ceiling=0.89, hold=0.02):
    """Look-ahead peak limiter: the gain each sample needs, held for a moment and smoothed (vectorised)."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    peak = np.maximum(np.abs(L), np.abs(R)).astype(np.float64)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    need = minimum_filter1d(need, size=2 * int(0.002 * SR) + 1)
    red = maximum_filter1d(1.0 - need, size=int(hold * SR))
    red = signal.sosfiltfilt(signal.butter(1, 25.0, fs=SR, output='sos'), red)
    g = np.minimum(1.0 - np.clip(red, 0, 1), need)
    return (L * g).astype(np.float32), (R * g).astype(np.float32)


def write_wav(path, L, R):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = np.clip(np.stack([L, R], -1) * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'probe':
        build(probe=[float(x) for x in sys.argv[2].split(',')])
    else:
        build(*(sys.argv[1:3] if len(sys.argv) > 1 else ()))
