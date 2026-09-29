"""ACT 0: the hook (0:00 - 0:35). No intro, no logo.

Black. "PLAYER JOINED THE GAME / NOAH_404", "PLAYER LEFT THE GAME / NOAH_404", again and again, faster, until it
becomes a scrolling server log (dates from 2018 to yesterday, all at 04:04, two lines hidden in it). Silence. Joined:
NOAH_404. Joined: YOU. Then the desktop of an old computer: the launcher (NOAH_404's account and skin, then his own,
the name blurred), the world list (NOAH_FINAL, last played 1 day ago), loading, the title.
"""
import os

import numpy as np
from PIL import Image, ImageFilter

import anim as A
import film as FM
import ui

WHITE = (235, 235, 235)


def events():
    ev = [(1.2, 'J'), (2.8, 'L'), (4.2, 'J'), (5.2, 'L'), (6.0, 'J'), (6.6, 'L'), (7.1, 'J'), (7.5, 'L'), (7.85, 'J'),
          (8.15, 'L')]
    return ev


def log_entries():
    """The server log shown 8.5-10.5 s: (time it appears, text, colour)."""
    rng = np.random.default_rng(404)
    d0 = np.datetime64('2018-06-14')
    d1 = np.datetime64('2026-04-02')
    n = 64
    days = np.sort(rng.integers(0, int((d1 - d0).astype(int)), n - 2))
    dates = [str(d0 + np.timedelta64(int(k), 'D')) for k in days] + ['2026-04-02', '2026-04-02']
    out = []
    t = 8.5
    for i, d in enumerate(dates):
        sec = int(rng.integers(0, 60))
        what = 'joined the game' if i % 2 == 0 else 'left the game'
        out.append((t, f'[{d} 04:04:{sec:02d}] [Server thread/INFO]: NOAH_404 {what}',
                    (255, 255, 120) if i % 2 == 0 else (200, 200, 200)))
        # hidden lines, only there for a few frames of scroll
        if i == 17:
            out.append((t + 0.01, '[2019-02-02 04:04:04] [Server thread/INFO]: <NOAH_404> is this recording?',
                        (255, 255, 255)))
        if i == 41:
            out.append((t + 0.01, '[2025-07-01 04:04:04] [Server thread/INFO]: ??? joined the game', (255, 90, 90)))
        t += max(0.012, 0.07 * (1 - i / n) ** 1.5)
    return out


LOG = log_entries()


def hook_overlay(img, t, T, ctx, film):
    img[:] = 0
    ev = events()
    if 1.2 <= t < 8.5:
        cur = None
        for (te, kind) in ev:
            if te <= t:
                cur = (te, kind)
        if cur is not None:
            te, kind = cur
            first = 'PLAYER JOINED THE GAME' if kind == 'J' else 'PLAYER LEFT THE GAME'
            lines = [(first, WHITE)]
            if t >= te + (0.45 if te < 3 else 0.12):
                lines.append(('NOAH_404', WHITE))
            ui.big_lines(img, lines)
    elif 8.5 <= t < 10.5:
        shown = [(txt, col) for (ta, txt, col) in LOG if ta <= t]
        ui.server_log(img, shown, alpha=float(np.clip((10.5 - t) / 0.1, 0, 1)))
    elif 11.3 <= t < 12.6:
        lines = [('PLAYER JOINED THE GAME', WHITE)]
        if t >= 11.65:
            lines.append(('NOAH_404', WHITE))
        ui.big_lines(img, lines)
    elif 12.6 <= t < 15.0:
        lines = [('PLAYER JOINED THE GAME', WHITE)]
        if t >= 13.2:
            lines.append(('YOU', WHITE))
        ui.big_lines(img, lines)


BANNER = None


def banner():
    global BANNER
    if BANNER is None:
        p = os.path.join(FM.CACHE, 'banner.png')
        if os.path.exists(p):
            BANNER = np.asarray(Image.open(p).convert('RGB'))
        else:
            BANNER = np.zeros((360, 640, 3), np.uint8) + np.array((70, 100, 60), np.uint8)
    return BANNER


# cursor keyframes (desktop: 15 - 23 s, in local time of the desktop shot)
DESK_CURSOR = A.Keys([(0.0, (980, 620)), (0.9, (80, 170)), (1.3, (80, 170)), (3.0, (80, 170)), (3.9, (470, 244)),
                      (4.3, (470, 244)), (5.2, (500, 285)), (5.8, (500, 285)), (6.6, (1110, 832)), (7.5, (1110, 832))],
                     'smooth')
CLICKS_DESK = [1.1, 1.25, 4.2, 5.8, 7.4]


def desktop_overlay(img, t, T, ctx, film):
    d = ctx.desk
    img[:] = d.base()
    if t >= 1.6:
        switched = t >= 5.8
        acc = 'NOAH_404' if not switched else 'PLAYER_2'
        skin = ctx.skins['noah'] if not switched else ctx.skins['you']
        dropdown = 4.2 <= t < 5.8
        pop = float(np.clip((t - 1.6) / 0.15, 0, 1))
        if pop > 0:
            play = d.launcher(img, acc, skin, banner=banner(), hover_play=t >= 6.6, dropdown=dropdown,
                              accounts=[('NOAH_404', False), ('PLAYER_2', True)])
            del play
            if switched:
                # his name is blurred (like any video would)
                x, y = 360, 150
                reg = img[y + 50:y + 75, x + 66:x + 200]
                img[y + 50:y + 75, x + 66:x + 200] = np.asarray(Image.fromarray(reg).filter(
                    ImageFilter.GaussianBlur(7)))
    cx, cy = DESK_CURSOR(t)
    d.draw_cursor(img, cx, cy)


WS_CURSOR = A.Keys([(0.0, (960, 700)), (0.9, (900, 180)), (5.7, (905, 184)), (6.0, (700, 925)), (6.5, (700, 925))])


def worldselect_overlay(img, t, T, ctx, film):
    icon = np.asarray(Image.fromarray(banner()).resize((64, 64), Image.BILINEAR).convert('RGBA'))
    hover = 'Play Selected World' if t >= 5.95 else None
    ui.world_select(img, icon=icon, hover=hover)
    if t >= 1.2:
        ui.last_played_hint(img, 'Last played: 1 day ago', alpha=float(np.clip((t - 1.2) / 0.2, 0, 1)))
    cx, cy = WS_CURSOR(t)
    ctx.desk.draw_cursor(img, cx, cy)


def loading_overlay(img, t, T, ctx, film):
    text = 'Loading terrain'
    frame = int(t * FM.FPS)
    if frame in (31, 32):
        text = 'Loading YOU'
    ui.loading(img, text, progress=min(1.0, t / 1.7), sub=None)


def title_overlay(img, t, T, ctx, film):
    img[:] = 0
    a = float(np.clip(t / 0.8, 0, 1)) * float(np.clip((4.0 - t) / 0.6, 0, 1))
    g = 0.0
    for (tg, dur) in ((1.7, 0.12), (3.1, 0.08)):
        if tg <= t < tg + dur:
            g = 0.8
    ui.title_card(img, alpha=a, glitch=g, rng=np.random.default_rng(int(t * 24)))


def shots():
    S = FM.Shot
    cues = [(te, 'join' if k == 'J' else 'leave') for (te, k) in events()]
    cues += [(ta, 'log_tick') for (ta, _, _) in LOG[::2]]
    cues += [(11.3, 'join'), (12.6, 'join'), (13.2, 'sub_hit')]
    desk_cues = [(0.0, 'pc_hum_start'), (1.1, 'click'), (1.25, 'click'), (1.6, 'window_open'), (4.2, 'click'),
                 (5.8, 'click'), (7.4, 'click_play')]
    ws_cues = [(0.0, 'mc_menu'), (6.3, 'mc_button')]
    return [
        S('hook_text', 15.0, '2d', overlay=hook_overlay, cues=cues, fx=lambda t: {'grain': 0.02}),
        S('hook_desktop', 8.0, '2d', overlay=desktop_overlay, cues=desk_cues,
          subs=[(0.4, 3.4, 'I found this world on an old computer.'), (3.6, 6.3, 'I think something is wrong with it.')],
          fx=lambda t: {'grain': 0.03, 'fade': float(np.clip(t / 0.3, 0, 1))}),
        S('hook_worldselect', 6.5, '2d', overlay=worldselect_overlay, cues=ws_cues, sub_y=760,
          subs=[(1.8, 4.0, 'Last played... one day ago?'), (4.3, 6.2, "That can't be right.")]),
        S('hook_loading', 2.0, '2d', overlay=loading_overlay, cues=[(0.0, 'loading')]),
        S('hook_title', 4.0, '2d', overlay=title_overlay, cues=[(0.0, 'title_drone'), (1.7, 'glitch'), (3.1, 'glitch')],
          fx=lambda t: {'grain': 0.045}),
    ]
