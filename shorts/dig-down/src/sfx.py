"""The soundtrack, synthesised from the story's events (no samples).

Every swing of the pickaxe is heard, in the sound of what it hits (grass and dirt crunch, stone clicks, deepslate
thuds, amethyst rings), every block that breaks crumbles and its item pops into his pocket; falls whoosh and hurt;
the zombie groans, the cave spiders hiss, the sculk sensors click, the shrieker screams, the Warden's heart beats
and he roars; each diamond chimes. The music: a soft piano in the game's manner at the top, strings joining as he
goes deeper, a dark drone by the Ancient City, a bright chord for the diamonds, then nothing but a heartbeat and the
last slow swings, the lava, and the death sound. Loudness -14 LUFS, true peak under -2 dBFS.
"""
import numpy as np

import art as A
import audio as AU

SR = AU.SR
B = A.B


class Mix:
    def __init__(self, dur):
        self.n = int(dur * SR) + SR
        self.L = np.zeros(self.n)
        self.R = np.zeros(self.n)

    def add(self, x, t, gain=1.0, pan=0.0):
        s = int(t * SR)
        if s >= self.n or s + len(x) <= 0:
            return
        if s < 0:
            x = x[-s:]
            s = 0
        e = min(self.n, s + len(x))
        gl = np.cos((pan + 1) * np.pi / 4) * np.sqrt(2)
        gr = np.sin((pan + 1) * np.pi / 4) * np.sqrt(2)
        self.L[s:e] += x[:e - s] * gain * gl
        self.R[s:e] += x[:e - s] * gain * gr


def env(n, a, d):
    t = np.arange(n) / SR
    return np.clip(t / max(a, 1e-4), 0, 1) * np.exp(-t / d)


def noise_hit(rng, dur, lo, hi, decay, thump=0.0, f_thump=120):
    n = int(dur * SR)
    x = AU.bp(rng.standard_normal(n), lo, hi, 2) * env(n, 0.001, decay)
    if thump:
        t = np.arange(n) / SR
        x += thump * np.sin(2 * np.pi * f_thump * t * (1 - 0.3 * t / dur)) * env(n, 0.002, decay * 0.8)
    return AU.norm(x, 0.9)


MATERIAL = {}
for name in ('grass', 'dirt'):
    MATERIAL[B[name]] = 'dirt'
for name in ('deepslate', 'deep_iron', 'deep_gold', 'deep_redstone', 'deep_diamond', 'deep_lapis', 'deep_coal',
             'tuff', 'basalt', 'deep_bricks', 'deep_tiles'):
    MATERIAL[B[name]] = 'deep'
for name in ('amethyst', 'budding', 'calcite'):
    MATERIAL[B[name]] = 'crystal'
MATERIAL[B['sculk']] = 'sculk'
MATERIAL[B['gravel']] = 'gravel'


def hit_sound(rng, mat):
    if mat == 'dirt':
        return noise_hit(rng, 0.12, 300, 2200, 0.035, 0.6, 90)
    if mat == 'gravel':
        return noise_hit(rng, 0.12, 500, 3500, 0.03, 0.3, 100)
    if mat == 'deep':
        return noise_hit(rng, 0.14, 600, 3800, 0.03, 0.9, 85)
    if mat == 'crystal':
        x = noise_hit(rng, 0.1, 2000, 7000, 0.02)
        b = AU.bell(int(rng.choice([88, 91, 93, 96])), rng, 0.8)
        y = np.zeros(max(len(x), len(b)))
        y[:len(x)] += x * 0.5
        y[:len(b)] += b * 0.6
        return y
    if mat == 'sculk':
        return noise_hit(rng, 0.1, 200, 1400, 0.04, 0.5, 70)
    return noise_hit(rng, 0.11, 900, 5200, 0.022, 0.7, 110)            # stone: a sharp click


def break_sound(rng, mat):
    if mat == 'crystal':
        x = np.zeros(int(0.9 * SR))
        for k, m in enumerate((86, 90, 93)):
            b = AU.bell(m + int(rng.integers(-2, 3)), rng, 0.8)
            s = int(k * 0.03 * SR)
            x[s:s + len(b)] += b[:len(x) - s] * 0.5
        return x
    base = {'dirt': (200, 1800, 0.07), 'gravel': (300, 3000, 0.08), 'deep': (250, 2600, 0.09),
            'sculk': (150, 1200, 0.1)}.get(mat, (400, 3600, 0.07))
    n = int(0.25 * SR)
    grains = (rng.random(n) < 0.004) * rng.standard_normal(n)
    x = AU.bp(rng.standard_normal(n) * 0.5 + grains * 6, base[0], base[1], 2) * env(n, 0.003, base[2])
    return AU.norm(x, 0.9)


def pop(rng):
    n = int(0.07 * SR)
    t = np.arange(n) / SR
    f = rng.uniform(900, 1500) * (1 + 1.2 * t / 0.07)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.002, 0.018) * 0.7


def oof(rng):
    """The game's hurt sound, roughly: a short low voice-like grunt."""
    n = int(0.28 * SR)
    t = np.arange(n) / SR
    f = 190 * (1 - 0.35 * t / 0.28)
    ph = np.cumsum(f) / SR
    saw = 2 * (ph % 1) - 1
    x = AU.bp(saw, 300, 900, 2) + 0.6 * AU.bp(saw, 1100, 1800, 2)
    return AU.norm(x * env(n, 0.01, 0.09), 0.9)


def groan(rng, dur=1.1, f0=95):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 * (1 + 0.15 * np.sin(2 * np.pi * 2.3 * t)) * (1 - 0.2 * t / dur)
    ph = np.cumsum(f) / SR
    saw = 2 * (ph % 1) - 1
    x = AU.bp(saw, 250, 700, 2) + 0.5 * AU.bp(saw, 900, 1400, 2) + 0.15 * AU.bp(rng.standard_normal(n), 300, 2000, 2)
    return AU.norm(x * np.sin(np.pi * t / dur) ** 0.7, 0.9)


def hiss(rng, dur=0.7):
    n = int(dur * SR)
    x = AU.hp(rng.standard_normal(n), 3000, 2) * (rng.random(n) < 0.3) * np.sin(np.pi * np.arange(n) / n)
    return AU.norm(x, 0.7)


def click(rng):
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    return AU.norm(np.sin(2 * np.pi * 1800 * t) * env(n, 0.0005, 0.006) + 0.3 * rng.standard_normal(n) *
                   env(n, 0.0005, 0.003), 0.8)


def shriek(rng, dur=1.6):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for f0 in (620, 930, 1240):
        f = f0 * (1 + 0.25 * t / dur) * (1 + 0.02 * np.sin(2 * np.pi * 9 * t))
        x += np.sin(2 * np.pi * np.cumsum(f) / SR)
    x += 0.4 * AU.bp(rng.standard_normal(n), 1500, 5000, 2)
    return AU.norm(x * np.clip(t / 0.08, 0, 1) * np.exp(-t / 0.9), 0.85)


def heartbeat(rng):
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * 52 * t) * env(n, 0.004, 0.06)
    y = np.zeros(n)
    s = int(0.16 * SR)
    y[s:] = (np.sin(2 * np.pi * 46 * t) * env(n, 0.004, 0.08))[:n - s] * 0.7
    return AU.norm(x + y, 0.95)


def roar(rng, dur=1.6):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 70 * (1 + 0.3 * np.sin(np.pi * t / dur))
    ph = np.cumsum(f) / SR
    x = np.tanh(3 * np.sin(2 * np.pi * ph)) + 0.8 * AU.bp(rng.standard_normal(n), 100, 900, 2)
    x = AU.lp(x, 1600, 2) * np.clip(t / 0.15, 0, 1) * np.clip((dur - t) / 0.5, 0, 1)
    return AU.norm(x, 0.95)


def sizzle(rng, dur):
    n = int(dur * SR)
    x = AU.hp(rng.standard_normal(n), 2500, 2) * (0.5 + 0.5 * np.abs(AU.lp(rng.standard_normal(n), 8, 1)) * 10)
    pops = (rng.random(n) < 20 / SR) * rng.standard_normal(n) * 6
    return AU.norm(x + AU.bp(pops, 400, 3000, 2), 0.8)


def lava_ambient(rng, n):
    x = AU.lp(rng.standard_normal(n), 300, 2) * 0.6
    pops = np.convolve((rng.random(n) < 3 / SR) * rng.uniform(0.3, 1, n),
                       np.sin(2 * np.pi * 180 * np.arange(int(0.05 * SR)) / SR) * np.exp(
                           -np.arange(int(0.05 * SR)) / (0.012 * SR)))[:n]
    return x + pops


def music(st, dur, rng):
    """Piano and strings in A minor, slow, the game's mood; strings from the mineshaft down; a drone at the city."""
    mus = Mix(dur)
    ev = st.events
    bar = 2.4
    prog = [(45, [57, 60, 64, 69]), (41, [57, 60, 65, 69]), (36, [55, 60, 64, 67]), (43, [55, 59, 62, 67])]
    end = ev['last'] + 0.2
    t = 0.0
    k = 0
    while t < end:
        bass, ch = prog[k % 4]
        dep = min(1.0, max(0.0, (t - ev['cave_land']) / 20.0))
        mus.add(AU.piano(bass, bar, 0.5), t, 0.5)
        mus.add(AU.piano(bass + 12, bar, 0.4), t + 0.01, 0.3)
        pat = [0, 2, 1, 3, 2, 1] if k % 2 == 0 else [0, 1, 2, 3, 2, 3]
        step = bar / 6
        for j, ix in enumerate(pat):
            tn = t + j * step
            if tn < end:
                mus.add(AU.piano(ch[ix] + 12 * (k % 4 == 3 and j == 5), 0.6, 0.35 + 0.15 * dep, seed=k * 7 + j),
                        tn, 0.35)
        if t > ev['mine_land'] - 1:
            fr = [AU.midi_hz(m) for m in [bass + 12] + ch[:3]]
            mus.add(AU.strings(fr, bar + 0.4, rng, attack=0.6, release=0.8), t, 0.18 + 0.25 * dep)
        t += bar
        k += 1
    # the deep dark: a low drone under the city
    d0, d1 = ev['shriek'] - 2.5, ev['diamonds']
    n = int((d1 - d0 + 1.0) * SR)
    tt = np.arange(n) / SR
    dr = (np.sin(2 * np.pi * 41.2 * tt) + 0.7 * np.sin(2 * np.pi * 43.65 * tt) + 0.3 * np.sin(2 * np.pi * 82.4 * tt))
    mus.add(AU.lp(dr, 300, 2) * np.clip(tt / 1.5, 0, 1) * np.clip((d1 - d0 + 1 - tt) / 1.0, 0, 1), d0, 0.5)
    # the diamonds: a bright chord
    for v, m in enumerate((57, 64, 69, 72, 76, 81)):
        mus.add(AU.piano(m, 2.0, 0.8, seed=v), ev['diamonds'] + 0.42 + 0.01 * v, 0.4)
    fr = [AU.midi_hz(m) for m in (57, 64, 69, 73)]
    mus.add(AU.strings(fr, ev['last'] - ev['diamonds'], rng, attack=0.4, release=1.0, bright=3000), ev['diamonds'],
            0.4)
    # death: a low sting
    for v, m in enumerate((38, 44, 50)):
        mus.add(AU.piano(m, 2.5, 0.9, seed=30 + v), ev['lava'] + 0.05, 0.45)
    return mus


def build(st, out_path, seed=7):
    rng = np.random.default_rng(seed)
    dur = st.T
    sfx, amb = Mix(dur), Mix(dur)
    ev = st.events
    hits = {m: [hit_sound(rng, m) for _ in range(6)] for m in ('dirt', 'gravel', 'deep', 'crystal', 'sculk', 'stone')}
    brk = {m: [break_sound(rng, m) for _ in range(4)] for m in hits}
    pops = [pop(rng) for _ in range(8)]
    for (th, x, y, b) in st.hits:
        m = MATERIAL.get(b, 'stone')
        g = 0.42 if th < ev['last'] else 0.7
        sfx.add(hits[m][int(rng.integers(6))], th, g * rng.uniform(0.8, 1.0), 0.1 * (x))
    for (tb, x, y, b) in st.breaks:
        m = MATERIAL.get(b, 'stone')
        sfx.add(brk[m][int(rng.integers(4))], tb, 0.4, 0.1 * x)
        if tb < ev['last']:
            sfx.add(pops[int(rng.integers(8))], tb + 0.42, 0.22, 0.0)
    # falls and hurts
    for (k, t0, t1, d) in st.acts:
        if k == 'fall' and d[0] - d[1] > 2:
            sfx.add(AU.make_whoosh(rng, max(0.3, t1 - t0 + 0.15)), t0, 0.3, 0.0)
        if k == 'torch':
            sfx.add(noise_hit(rng, 0.08, 400, 2500, 0.02, 0.4, 200), t1 - 0.08, 0.35, 0.3)
    for (td, hp) in st.damage:
        sfx.add(oof(rng), td, 0.55, 0.0)
        if td < ev['lava']:
            sfx.add(noise_hit(rng, 0.25, 60, 600, 0.06, 1.0, 55), td, 0.6, 0.0)
    # the cave: the zombie
    sfx.add(groan(rng, 1.2, 90), ev['cave_land'] + 0.5, 0.42, 0.5)
    sfx.add(groan(rng, 1.0, 105), ev['cave_land'] + 2.1, 0.32, 0.4)
    # the mineshaft: spiders
    if 'mine_land' in ev:
        sfx.add(hiss(rng, 0.8), ev['mine_land'] + 0.2, 0.3, -0.5)
        sfx.add(hiss(rng, 0.6), ev['mine_land'] + 1.1, 0.22, -0.6)
    # the geode: a shimmer
    if 'geode_land' in ev:
        for k in range(6):
            sfx.add(AU.bell(int(rng.choice([93, 96, 98, 100])), rng, 1.2), ev['geode_land'] - 0.4 + k * 0.17, 0.16,
                    rng.uniform(-0.7, 0.7))
    # the city
    for (t0, t1, a, b) in st.vibes:
        sfx.add(click(rng), t1, 0.4, 0.5)
    sfx.add(shriek(rng), ev['shriek'], 0.55, 0.5)
    t = ev['warden_rise']
    while t < ev['last']:
        sfx.add(heartbeat(rng), t, 0.6, 0.3)
        t += 1 / 1.4
    sfx.add(roar(rng), ev['warden_roar'], 0.75, 0.4)
    # diamonds
    for (td, n) in st.dings:
        x = np.zeros(int(1.2 * SR))
        for k, m in enumerate((88 + n, 95 + n)):
            b = AU.bell(m, rng, 1.0)
            s = int(k * 0.06 * SR)
            x[s:s + len(b)] += b[:len(x) - s]
        sfx.add(AU.norm(x, 0.8), td, 0.4, 0.0)
    sfx.add(AU.make_levelup(rng), st.dings[-1][0] + 0.1, 0.35, 0.0)
    # the end: heartbeat under the last block, the lava, the death screen's click
    t = ev['last']
    while t < ev['break_last']:
        sfx.add(heartbeat(rng), t, 0.8, 0.0)
        t += 0.6
    sfx.add(AU.make_whoosh(rng, 0.6), ev['break_last'], 0.4, 0.0)
    sfx.add(noise_hit(rng, 0.6, 80, 1200, 0.2, 1.0, 50), ev['lava'], 0.8, 0.0)
    sfx.add(sizzle(rng, ev['dead'] - ev['lava'] + 0.4), ev['lava'], 0.45, 0.0)
    sfx.add(click(rng), ev['dead'] + 2.35, 0.6, 0.0)
    # ambience: birds up top, cave air below, lava near the end
    n = sfx.n
    tt = np.arange(n) / SR
    under = np.clip((tt - 1.6) / 2.0, 0, 1)
    air = AU.bp(rng.standard_normal(n), 80, 500, 2)
    air = air / (np.abs(air).max() + 1e-9)
    amb.L += air * 0.06 * under
    amb.R += np.roll(air, 3000) * 0.06 * under
    near_lava = np.clip((tt - (ev['diamonds'] - 3)) / 3.0, 0, 1) * (tt < ev['dead'] + 0.3)
    la = lava_ambient(rng, n)
    la = la / (np.abs(la).max() + 1e-9)
    amb.L += la * 0.18 * near_lava
    amb.R += np.roll(la, 2000) * 0.18 * near_lava
    mus = music(st, dur, rng)
    mL, mR = AU.reverb(mus.L, mus.R, decay=2.4, wet=0.3)
    sL, sR = AU.reverb(sfx.L, sfx.R, seed=5, decay=1.4, wet=0.18)
    # the music stops dead on the last block
    g = np.ones(n)
    s0 = int(ev['last'] * SR)
    g[s0:] = np.clip(1 - np.arange(n - s0) / (0.3 * SR), 0, 1)
    s1 = int(ev['lava'] * SR)
    g[s1:] = 1.0                                # (the death sting is in the music too)
    L = sL + 0.55 * mL * g + amb.L
    R = sR + 0.55 * mR * g + amb.R
    L, R = L[:int(dur * SR)], R[:int(dur * SR)]
    L, R = AU.hp(L, 30.0, 2), AU.hp(R, 30.0, 2)
    L = L - 0.3 * AU.lp(L, 90.0, 2) + 0.25 * AU.bp(L, 1800.0, 5000.0, 1)
    R = R - 0.3 * AU.lp(R, 90.0, 2) + 0.25 * AU.bp(R, 1800.0, 5000.0, 1)
    gn = 10 ** ((-18.0 - AU.integrated_lufs(L, R)) / 20.0)
    L, R = AU.compress(L * gn, R * gn, thresh_db=-24.0, ratio=2.5)
    gn = 10 ** ((-14.0 - AU.integrated_lufs(L, R)) / 20.0)
    L, R = AU.limiter(L * gn, R * gn, ceiling=10 ** (-1.2 / 20))
    L, R = true_peak(L, R, -2.0)
    f = int(0.01 * SR)
    for ch in (L, R):
        ch[:f] *= np.linspace(0, 1, f)
        ch[-f:] *= np.linspace(1, 0, f)
    AU._write(out_path, L, R)
    print(f'[audio] {dur:.2f}s, {AU.integrated_lufs(L, R):.1f} LUFS', flush=True)


def true_peak(L, R, ceiling_db=-2.0, release=0.08):
    from scipy import signal
    from scipy.ndimage import minimum_filter1d
    c = 10 ** (ceiling_db / 20)
    pk = np.zeros(len(L))
    for x in (L, R):
        up = signal.resample_poly(x, 4, 1)[:4 * len(x)]
        pk = np.maximum(pk, np.abs(up).reshape(-1, 4).max(1))
    g = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    g = minimum_filter1d(g, 2 * int(0.001 * SR) + 1, mode='nearest')
    a = np.exp(-1.0 / (release * SR))
    slow = signal.lfilter([1 - a], [1, -a], g, zi=[g[0] * a])[0]
    out = np.minimum(g, slow)
    return L * out, R * out
