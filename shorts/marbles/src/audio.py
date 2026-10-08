"""Music and sound design for the marble machine, driven by the cue sheet of the render (cues.json: per video frame,
the story time and what happened in it).

Everything is synthesised here (no samples; the toolkit is sound.py). The effects come from the physics: every
marble that comes out of the hopper rattles down the end rods (glassy ticks off metal) and clacks onto the pile, so
the flow is a shimmering rain of glass that thins out and stops with the jam and comes back with a rush. The golden
ball rumbles as it rolls in and plugs the hole with a heavy clunk. The goat trots in, bleats at you, screams as it
charges and hits the stone with a thud and a clang of the whole machine. The last marble's bounces each ring a bell,
climbing, and its landing resolves on a chord and the level-up chime. Then the creeper's hiss, the blast, the glass,
and ten thousand marbles raining onto the meadow.

The music: a bouncy, glassy tune in G (a music box over plucked piano chords, a soft kick and shaker) that stops
dead, tape-stop style, when the hole jams; a sneaky pizzicato for the goat and a snare roll into its charge; the
full tune back at double strength when the goat frees it, building with strings; held strings and a heartbeat for
the last marble; a big G major chord at 10,000. Loudness normalised to -14 LUFS, true peak under -1 dBFS.
"""
import json

import numpy as np

from sound import (SR, Mix, bell, bp, compress, expenv, hp, integrated_lufs, limiter, lp, make_crash,
                   make_explosion, make_heartbeat, make_hiss, make_levelup, make_step, make_swell, make_whoosh,
                   midi_hz, modal, norm, piano, resample, reverb, shaped_noise, sine_sweep, strings, _write)

BPM = 112.0
BEAT = 60.0 / BPM
E8 = BEAT / 2
# G - Em - C - D
CHORDS = [(43, (55, 59, 62)), (40, (55, 59, 64)), (36, (55, 60, 64)), (38, (54, 57, 62))]
ARP = [(74, 71, 67, 71, 74, 71, 67, 71), (76, 71, 67, 71, 76, 71, 67, 71), (76, 72, 67, 72, 76, 72, 67, 72),
       (78, 74, 69, 74, 78, 74, 69, 74)]
MEL = [(79, 0, 2), (78, 2, 1), (76, 3, 1), (74, 4, 2), (71, 6, 2)]       # a hook on top (note, 8th, length)


# ---------------------------------------------------------------------------------------------
# sounds
# ---------------------------------------------------------------------------------------------
def _put(x, c, p, g=1.0):
    """Add c into x at sample p (clipped at the end)."""
    if p >= len(x):
        return
    m = min(len(c), len(x) - p)
    x[p:p + m] += c[:m] * g


def make_tick(rng):
    """A glass marble off an end rod: a bright tink, and the thin rod ringing for a moment."""
    dur = 0.09
    n = int(dur * SR)
    x = modal(dur, rng.uniform(3600, 5600), (1.0, 1.53, 2.21), (1.0, 0.5, 0.3), (0.006, 0.004, 0.003), rng)
    x += 0.45 * modal(dur, rng.uniform(1900, 2600), (1.0, 2.76, 5.4), (1.0, 0.4, 0.2), (0.03, 0.015, 0.01), rng)
    x += 0.3 * hp(rng.standard_normal(n), 5000, 2) * expenv(n, 0.0008)
    return norm(x * np.clip(np.arange(n) / SR / 0.0003, 0, 1), 0.8)


def make_clack(rng, dull=0.0):
    """Marble on marble (or, dull, on grass): a hard glassy click with a little body."""
    dur = 0.08
    n = int(dur * SR)
    f = rng.uniform(2400, 3600) * (1 - 0.55 * dull)
    x = modal(dur, f, (1.0, 1.7, 2.6), (1.0, 0.45, 0.2), (0.008 * (1 + dull * 2), 0.005, 0.003), rng)
    x += 0.5 * modal(dur, rng.uniform(900, 1400) * (1 - 0.5 * dull), (1.0, 2.2), (1.0, 0.3), (0.012, 0.006), rng)
    x += 0.4 * hp(rng.standard_normal(n), 3000, 2) * expenv(n, 0.001)
    if dull:
        x = lp(x, 2500, 2)
    return norm(x * np.clip(np.arange(n) / SR / 0.0003, 0, 1), 0.8)


def make_rattle_bed(dur, density, rng):
    """The dense part of the rain: a shimmer of glass noise, its loudness following density (per sample)."""
    n = int(dur * SR)
    x = bp(rng.standard_normal(n), 2600, 9000, 2)
    x += 0.5 * bp(rng.standard_normal(n), 1200, 2600, 2)
    # grainy: chop it into tiny bursts
    g = (rng.random(n) < 0.03).astype(float)
    g = np.convolve(g, np.hanning(96), 'same')
    return x * (0.15 + 0.85 * np.clip(g, 0, 1)) * density


def make_roll(rng, dur):
    """The golden ball rolling on marbles: a low rumble that goes round, and a crunch of marbles under it."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    rot = 0.5 + 0.5 * np.sin(2 * np.pi * (1.5 * t + 1.2 * t * t / dur))
    x = lp(rng.standard_normal(n), 260, 2) * (0.6 + 0.4 * rot)
    cr = np.zeros(n)
    for _ in range(int(60 * dur)):
        p = int(rng.uniform(0, n - 3000))
        c = make_clack(rng, 0.3)
        _put(cr, c, p, rng.uniform(0.2, 0.6))
    x = norm(x) + 0.35 * norm(cr)
    return x * np.clip(t / 0.3, 0, 1) * np.clip((dur - t) / 0.2, 0, 1)


def make_clunk(rng):
    """The ball dropping into the hole: a heavy dull metal 'dong' and the marbles around it settling."""
    dur = 2.4
    n = int(dur * SR)
    x = modal(dur, 92.0, (1.0, 2.33, 3.87, 5.4), (1.0, 0.6, 0.35, 0.2), (0.7, 0.4, 0.25, 0.15), rng)
    x += 0.8 * sine_sweep(dur, 120, 50, 0.06) * expenv(n, 0.12, 0.002)
    x += 0.5 * bp(rng.standard_normal(n), 300, 2500, 2) * expenv(n, 0.03)
    for _ in range(14):
        c = make_clack(rng, 0.2)
        p = int(rng.uniform(0.02, 0.35) * SR)
        _put(x, c, p, rng.uniform(0.1, 0.3))
    return norm(np.tanh(1.4 * x), 0.95)


def _formant(src, formants):
    out = np.zeros_like(src)
    for (f, bw, g) in formants:
        out += g * bp(src, max(f - bw / 2, 40), f + bw / 2, 2)
    return out


def make_bleat(rng):
    """The goat's 'meh-eh-eh': a buzzy voice, trembling."""
    dur = 0.75
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0 = 330 * (1 + 0.04 * np.sin(2 * np.pi * 5 * t)) * (1 - 0.12 * t / dur)
    ph = np.cumsum(f0) / SR
    src = 2 * np.mod(ph, 1.0) - 1 + 0.15 * rng.standard_normal(n)
    v = _formant(src, ((560, 160, 1.0), (1800, 260, 0.7), (2500, 300, 0.4)))
    trem = 0.55 + 0.45 * np.sin(2 * np.pi * 9.0 * t) ** 2
    env = np.clip(t / 0.04, 0, 1) * np.clip((dur - t) / 0.15, 0, 1)
    return norm(v * trem * env, 0.8)


def make_scream(rng):
    """The screaming goat: a ragged, almost human 'AAAAH'."""
    dur = 0.95
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0 = (520 + 120 * np.clip(t / 0.25, 0, 1) - 60 * np.clip((t - 0.6) / 0.35, 0, 1))
    f0 = f0 * (1 + 0.02 * np.sin(2 * np.pi * 7 * t) + 0.01 * rng.standard_normal(n).cumsum() / np.sqrt(n))
    ph = np.cumsum(f0) / SR
    src = 2 * np.mod(ph, 1.0) - 1
    src = np.tanh(2.5 * src) + 0.35 * rng.standard_normal(n)
    v = _formant(src, ((850, 220, 1.0), (1250, 260, 0.8), (2600, 400, 0.5), (3400, 500, 0.25)))
    env = np.clip(t / 0.03, 0, 1) * np.clip((dur - t) / 0.25, 0, 1)
    return norm(np.tanh(1.6 * v * env), 0.9)


def make_hoof(rng):
    """A hoof on grass: a soft dull knock with a crunch."""
    x = make_step(rng)
    k = modal(0.08, rng.uniform(380, 520), (1.0, 2.3), (1.0, 0.3), (0.02, 0.01), rng)
    y = np.zeros(max(len(x), len(k)))
    y[:len(x)] += 0.6 * x
    y[:len(k)] += 0.7 * k
    return norm(y, 0.6)


def make_ram(rng):
    """Head meets stone: a thud you feel, a crack, the whole machine ringing like a struck girder, marbles jumping."""
    dur = 3.0
    n = int(dur * SR)
    x = 1.0 * norm(sine_sweep(dur, 140, 42, 0.05) * expenv(n, 0.16, 0.001))
    x += 0.7 * norm(bp(rng.standard_normal(n), 400, 5000, 2) * expenv(n, 0.02, 0.0005))
    clang = modal(dur, 63.0, (1.0, 2.76, 5.40, 8.93, 13.3), (1.0, 0.7, 0.5, 0.3, 0.2), (1.6, 1.0, 0.7, 0.45, 0.3),
                  rng, 0.01)
    x += 0.55 * norm(clang)
    for _ in range(90):
        c = make_clack(rng)
        p = int(rng.uniform(0.03, 0.5) * SR)
        _put(x, c, p, rng.uniform(0.05, 0.25))
    return norm(np.tanh(1.5 * x), 0.98)


def make_pop(rng):
    """The golden ball popping out of the hole like a cork."""
    dur = 0.6
    n = int(dur * SR)
    x = sine_sweep(dur, 900, 260, 0.03) * expenv(n, 0.035, 0.0005)
    x += 0.6 * bp(rng.standard_normal(n), 600, 4000, 2) * expenv(n, 0.01)
    return norm(x, 0.8)


def make_tapestop(seg, dur=0.55):
    """The end of a stem played as the tape slows to a stop."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    speed = (1 - t / dur) ** 1.6
    pos = np.cumsum(speed)
    pos = np.clip(pos, 0, len(seg) - 2)
    i = pos.astype(int)
    f = pos - i
    return (seg[i] * (1 - f) + seg[i + 1] * f) * np.clip((dur - t) / 0.08, 0, 1)


def make_pluck(m, dur, rng, bright=1.0):
    """A plucked bass string (pizzicato)."""
    f0 = midi_hz(m)
    n = int((dur + 0.3) * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k in range(1, 9):
        x += (1.0 / k ** 1.3) * np.sin(2 * np.pi * f0 * k * t) * np.exp(-t / (0.35 / (1 + 0.6 * (k - 1)) * bright))
    x *= np.clip(t / 0.003, 0, 1)
    return norm(lp(x, 2500, 2), 0.8)


def make_kick(rng):
    dur = 0.4
    n = int(dur * SR)
    x = sine_sweep(dur, 160, 48, 0.035) * expenv(n, 0.14, 0.001)
    x += 0.2 * lp(rng.standard_normal(n), 2500, 2) * expenv(n, 0.004)
    return norm(x, 0.9)


def make_snare(rng):
    dur = 0.3
    n = int(dur * SR)
    x = bp(rng.standard_normal(n), 1200, 8000, 2) * expenv(n, 0.07, 0.001)
    x += 0.6 * sine_sweep(dur, 240, 170, 0.03) * expenv(n, 0.05, 0.001)
    return norm(x, 0.8)


def make_shaker(rng):
    dur = 0.08
    n = int(dur * SR)
    x = hp(rng.standard_normal(n), 6500, 2) * np.sin(np.pi * np.arange(n) / n) ** 2
    return norm(x, 0.5)


def make_wind(rng, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = lp(rng.standard_normal(n), 500, 2) + 0.3 * bp(rng.standard_normal(n), 600, 1800, 2)
    m = 0.6 + 0.4 * np.sin(2 * np.pi * 0.11 * t + 1.0) * np.sin(2 * np.pi * 0.047 * t)
    return norm(x * m, 0.5)


def make_chirp(rng):
    x = []
    for _ in range(int(rng.integers(2, 5))):
        d = rng.uniform(0.05, 0.11)
        m = int(d * SR)
        t = np.arange(m) / SR
        f = rng.uniform(3200, 4600) + rng.uniform(-900, 900) * t / d
        x.append(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / d) ** 2)
        x.append(np.zeros(int(rng.uniform(0.03, 0.08) * SR)))
    return norm(np.concatenate(x), 0.5)


# ---------------------------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------------------------
def play_bars(mus, rng, t0, t1, big=False, vel=1.0):
    """The tune from t0 for as many bars as fit before t1 (notes are cut off at t1)."""
    bar = 4 * BEAT
    k = 0
    t = t0
    kick, snare = make_kick(rng), make_snare(rng)
    shakers = [make_shaker(rng) for _ in range(4)]
    while t < t1 - 1e-6:
        bass, chord = CHORDS[k % 4]
        arp = ARP[k % 4]

        def add(x, at, g, pan=0.0):
            if at < t1:
                cut = int((t1 - at) * SR)
                mus.add(x[:cut], at, g, pan)
        # left hand: the bass note, chord stabs on 2 and 4 (held longer once it's big)
        add(piano(bass, 1.6 if big else 0.6, 0.55 * vel), t, 0.55)
        for b in (1, 3):
            for m in chord:
                add(piano(m, 0.9 if big else 0.18, 0.42 * vel), t + b * BEAT, 0.28, -0.2)
        # the music box: an arpeggio in eighths
        for j, m in enumerate(arp):
            add(bell(m + 12, rng, 0.9, 0.8), t + j * E8, 0.16 * vel, 0.25 if j % 2 else -0.1)
        if big and k % 2 == 1:
            for (m, s8, ln) in MEL:
                add(bell(m + 12, rng, 1.4, 1.0), t + s8 * E8, 0.20 * vel, 0.05)
        # beat: a soft kick on 1 and 3, shaker on the eighths; snare on 2 and 4 when it's big
        for b in (0, 2):
            add(kick, t + b * BEAT, (0.55 if big else 0.32) * vel)
        for j in range(8):
            add(shakers[j % 4], t + j * E8, (0.10 if j % 2 else 0.06) * vel, 0.4)
        if big:
            for b in (1, 3):
                add(snare, t + b * BEAT, 0.28 * vel)
            add(np.asarray(strings([midi_hz(m) for m in chord] + [midi_hz(bass + 12)], bar, rng, 0.3, 0.6)),
                t, 0.30 * vel, 0.0)
            add(make_pluck(bass, 0.4, rng), t, 0.35 * vel)
            add(make_pluck(bass, 0.3, rng), t + 2.5 * BEAT, 0.25 * vel)
        k += 1
        t += bar
    return t


def sneak(mus, rng, t0, t1):
    """The goat trots in: a sneaky pizzicato walk."""
    notes = [43, None, 43, None, 46, None, 47, None, 43, None, 43, 50, 49, None, 48, None]
    t = t0
    j = 0
    while t < t1:
        m = notes[j % len(notes)]
        if m is not None:
            mus.add(make_pluck(m, 0.12, rng, 0.6), t, 0.55, -0.1)
            mus.add(make_pluck(m + 12, 0.08, rng, 0.5), t, 0.15, 0.2)
        t += E8
        j += 1


def roll(mus, rng, t0, t1):
    """A snare roll building into the charge."""
    sn = [make_snare(rng) for _ in range(4)]
    t = t0
    j = 0
    while t < t1:
        u = (t - t0) / max(t1 - t0, 1e-6)
        mus.add(sn[j % 4] * 0.7, t, 0.08 + 0.35 * u ** 1.5, rng.uniform(-0.2, 0.2))
        t += 0.05
        j += 1


# ---------------------------------------------------------------------------------------------
# assembling
# ---------------------------------------------------------------------------------------------
def build(meta, out_path, seed=11, stems=False):
    rng = np.random.default_rng(seed)
    fps = meta['fps']
    frames = meta['frames']
    starts = dict(zip(meta['shots'], meta['starts']))
    n_fr = len(frames)
    dur = n_fr / fps
    vt = np.array([f['v'] for f in frames])
    ev = {}
    for f in frames:
        for e in f['events']:
            ev.setdefault(e, f['v'])
    fx = Mix(dur + 3.0)
    mus = Mix(dur + 3.0)
    amb = Mix(dur + 3.0)

    # -- the marbles: the rain of glass, from the physics ------------------------------------------
    ticks = [make_tick(rng) for _ in range(36)]
    clacks = [make_clack(rng) for _ in range(30)]
    dulls = [make_clack(rng, 1.0) for _ in range(20)]
    out_c = np.cumsum([f['out'] for f in frames])
    in_c = np.cumsum([f['in'] for f in frames])
    flight = np.maximum(out_c - in_c, 0)
    close = {'pegs': 1.6, 'fill_b': 1.4, 'last': 1.6, 'count': 1.2, 'stuck': 1.2, 'burst': 1.25, 'rise': 1.1,
             'open': 0.8, 'gold': 0.45, 'goat_a': 0.4, 'goat_b': 0.4, 'pop': 0.6, 'fill_a': 0.9, 'fill_c': 0.7,
             'done': 0.8, 'fuse': 0.7, 'boom': 1.0, 'after': 0.9}
    bed = np.zeros(int((dur + 3.0) * SR))
    for i, f in enumerate(frames):
        dt = f['dt']
        if dt <= 0:
            continue
        g = close.get(f['shot'], 1.0)
        n_tick = flight[i] * 2.0 * dt
        n_clack = f['in'] * 1.6
        if f['shot'] == 'last':
            n_tick, n_clack = 0.0, 0.0
        t0 = f['v'] - 0.5 / fps
        for (num, bank, gain, cap) in ((n_tick, ticks, 0.11, 7), (n_clack, clacks, 0.10, 5)):
            k = int(min(num, cap)) + (1 if rng.random() < (min(num, cap) % 1) else 0)
            for _ in range(k):
                s = bank[rng.integers(len(bank))]
                fx.add(s, t0 + rng.uniform(0, 1.0 / fps), gain * g * rng.uniform(0.4, 1.0), rng.uniform(-0.7, 0.7))
        dens = np.sqrt(max(n_tick + n_clack, 0.0) / max(dt * 60.0, 1e-6)) / 20.0
        a = int(t0 * SR)
        bed[a:a + int(SR / fps) + 1] = min(dens, 1.2) * g
    bed = lp(np.convolve(bed, np.ones(1200) / 1200, 'same'), 30, 1)
    fx.add(make_rattle_bed(len(bed) / SR, np.clip(bed, 0, 1.5), rng), 0.0, 0.055)

    # -- the hopper, the golden ball and the jam ----------------------------------------------------
    if 'gold_roll' in ev and 'jam' in ev:
        fx.add(make_roll(rng, ev['jam'] - ev['gold_roll'] + 0.1), ev['gold_roll'], 0.45, -0.1)
    if 'jam' in ev:
        fx.add(make_clunk(rng), ev['jam'], 0.55)
    # -- the goat ---------------------------------------------------------------------------------------
    if 'step_in' in ev:
        a = max(ev['step_in'], starts['goat_a'] - 0.2)
        b = ev.get('goat_stop', a + 1.0)
        t = a
        while t < b:
            fx.add(make_hoof(rng), t, 0.32, 0.25)
            t += 0.165
    if 'back' in ev:
        for k in range(3):
            fx.add(make_hoof(rng), ev['back'] + 0.12 * k, 0.25, 0.1)
    if 'bleat' in ev:
        fx.add(make_bleat(rng), ev['bleat'], 0.75, 0.1)
    if 'charge' in ev and 'ram' in ev:
        fx.add(make_scream(rng), ev['charge'] - 0.05, 0.8, 0.0)
        t = ev['charge']
        while t < ev['ram'] - 0.05:
            fx.add(make_hoof(rng), t, 0.45, 0.0)
            t += 0.11
        fx.add(make_ram(rng), ev['ram'], 1.0)
    if 'pop' in ev:
        fx.add(make_pop(rng), ev['pop'] + 0.05, 0.6, 0.1)
        fx.add(make_whoosh(rng, 0.7), ev['pop'] + 0.1, 0.35, 0.3)
    if 'gold_thud' in ev:
        th = make_explosion(rng, 0.3, distant=True)
        fx.add(lp(th, 400, 2), ev['gold_thud'], 0.25)
    if 'burst' in ev:
        fx.add(make_whoosh(rng, 1.0), ev['burst'] - 0.1, 0.4, -0.2)
        fx.add(make_swell(rng, 0.6), ev['burst'] - 0.6, 0.25)
    # -- the last marble: each bounce rings a note, climbing --------------------------------------------
    climb = [74, 76, 78, 79, 81, 83, 85, 86, 88, 90]
    j = 0
    for f in frames:
        for _ in range(f.get('hero_hits', 0)):
            m = climb[min(j, len(climb) - 1)]
            fx.add(make_tick(rng), f['v'], 0.5, 0.0)
            mus.add(bell(m, rng, 2.0, 1.1), f['v'], 0.42, 0.0)
            j += 1
    if 'land' in ev:
        fx.add(make_clack(rng), ev['land'], 0.6)
        fx.add(make_levelup(rng), ev['land'] + 0.05, 0.55)
    # -- the creeper ----------------------------------------------------------------------------------------
    if 'fuse' in ev:
        # the hiss runs right up to the blast (the fuse is in slow motion on screen), swelling
        hs = make_hiss(rng, ev.get('boom', ev['fuse'] + 1.5) - ev['fuse'] + 0.05)
        fx.add(hs * np.linspace(0.55, 1.25, len(hs)), ev['fuse'], 0.8)
    if 'boom' in ev:
        fx.add(make_explosion(rng, 2.4), ev['boom'], 1.15)
        # the blast is in slow motion on screen: under it, the same roar slowed down an octave
        slow = resample(make_explosion(rng, 2.4), 0.5)
        fx.add(lp(slow, 2500, 2), ev['boom'] + 0.02, 0.75)
        fx.add(make_crash(rng, 2.6), ev['boom'] + 0.02, 0.6, -0.4)
        fx.add(make_crash(rng, 2.2), ev['boom'] + 0.05, 0.6, 0.4)
        # glass tinkling down
        for _ in range(60):
            s = ticks[rng.integers(len(ticks))]
            fx.add(s, ev['boom'] + rng.exponential(0.6), 0.12 * rng.uniform(0.3, 1.0), rng.uniform(-0.8, 0.8))
        # ten thousand marbles coming down on the meadow
        for f in frames:
            h = f.get('blast_hits', 0)
            if h:
                for _ in range(int(min(h * 0.08, 10))):
                    s = dulls[rng.integers(len(dulls))]
                    fx.add(s, f['v'] + rng.uniform(0, 1 / fps), 0.12 * rng.uniform(0.4, 1.0), rng.uniform(-0.8, 0.8))

    # -- music ------------------------------------------------------------------------------------------------
    t_jam = ev.get('jam', dur)
    end1 = play_bars(mus, rng, 0.0, t_jam, big=False, vel=0.9)
    # the tape stops at the jam
    a = int(t_jam * SR)
    segL = mus.L[a:a + int(1.2 * SR)].copy()
    segR = mus.R[a:a + int(1.2 * SR)].copy()
    mus.L[a:] = 0.0
    mus.R[a:] = 0.0
    if len(segL) > 10:
        mus.add(make_tapestop(segL), t_jam, 1.0, -1.0)
        mus.add(make_tapestop(segR), t_jam, 1.0, 1.0)
    # the goat
    if 'step_in' in ev and 'goat_stop' in ev:
        sneak(mus, rng, max(ev['step_in'], starts['goat_a']), ev['goat_stop'] + 0.05)
    if 'back' in ev and 'charge' in ev:
        roll(mus, rng, ev['back'], ev['charge'] + 0.05)
    # back, big, from the burst until the last marble drops
    t_big = ev.get('burst', dur) - 2 * E8
    t_last = starts.get('count', dur)
    play_bars(mus, rng, t_big, t_last + 0.15, big=True, vel=1.0)
    # the last marble: held strings on D, a heartbeat
    if 'land' in ev:
        sus = strings([midi_hz(m) for m in (50, 57, 62, 66)], ev['land'] - t_last + 0.1, rng, 0.4, 0.4, 1800)
        mus.add(np.asarray(sus), t_last, 0.45)
        hb = make_heartbeat(rng)
        t = t_last + 0.2
        while t < ev['land'] - 0.2:
            fx.add(hb, t, 0.5)
            t += 0.62
        # 10,000: a big G major chord, held through 'it's a creeper'
        ch = [43, 55, 59, 62, 67, 71, 74, 79]
        hold = ev.get('fuse', ev['land'] + 2.5) - ev['land']
        for m in ch:
            mus.add(piano(m, hold, 0.7), ev['land'], 0.32)
        mus.add(np.asarray(strings([midi_hz(m) for m in ch[1:6]], hold, rng, 0.08, 0.3, 2600)), ev['land'], 0.5)
        mus.add(bell(91, rng, 2.4), ev['land'] + 0.1, 0.25)
    if 'fuse' in ev:
        a = int(ev['fuse'] * SR)
        fade = np.clip(1 - (np.arange(len(mus.L) - a) / SR) / 0.25, 0, 1)
        mus.L[a:] *= fade
        mus.R[a:] *= fade

    # -- the meadow: wind and the odd bird, quieter under the music -----------------------------------------
    amb.add(make_wind(rng, dur + 2.0), 0.0, 0.18)
    for t in np.arange(1.0, dur, 2.7):
        amb.add(make_chirp(rng), t + rng.uniform(0, 1.5), 0.05, rng.uniform(-0.8, 0.8))

    # -- mix ---------------------------------------------------------------------------------------------------
    mL, mR = reverb(mus.L, mus.R, seed=4, decay=1.8, wet=0.22)
    fL, fR = reverb(fx.L, fx.R, seed=5, decay=1.2, wet=0.12)
    L = 0.85 * mL + 1.0 * fL + amb.L
    R = 0.85 * mR + 1.0 * fR + amb.R
    n = int(dur * SR)
    L, R = L[:n], R[:n]
    L, R = hp(L, 30, 2), hp(R, 30, 2)
    L, R = compress(L, R, thresh_db=-20.0, ratio=2.5)
    lufs = integrated_lufs(L, R)
    g = 10 ** ((-14.0 - lufs) / 20.0)
    L, R = L * g, R * g
    L, R = limiter(L, R, ceiling=0.84)
    L, R = true_peak(L, R, -1.5)
    fade = np.clip((dur - np.arange(n) / SR) / 0.25, 0, 1)
    L, R = L * fade, R * fade
    _write(out_path, L, R)
    print(f'[audio] {dur:.2f}s, {integrated_lufs(L, R):.1f} LUFS', flush=True)


def true_peak(L, R, ceiling_db=-1.5):
    """Keep the inter-sample peaks (4x oversampled) under the ceiling: where they'd go over, a gain dip that opens
    a few milliseconds early and closes smoothly."""
    from scipy.ndimage import minimum_filter1d
    from scipy.signal import resample_poly
    c = 10 ** (ceiling_db / 20)
    peak = np.maximum(np.abs(resample_poly(L, 4, 1)), np.abs(resample_poly(R, 4, 1)))
    peak = peak[:len(L) * 4].reshape(-1, 4).max(1)
    need = np.minimum(1.0, c / np.maximum(peak, 1e-9))
    w = int(0.004 * SR)
    g = minimum_filter1d(need, size=2 * w + 1)
    g = np.convolve(g, np.ones(w) / w, 'same')
    return L * g, R * g


if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as fh:
        meta = json.load(fh)
    build(meta, sys.argv[2])
