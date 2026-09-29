"""ACT 10: the final ARG hook (12:21 - 12:50).

His desktop. Normal. No Minecraft, no old_saves, and the date in the corner is tomorrow's. A folder appears:
NOAH_FINAL. Inside, PLAYER_1.mp4 to PLAYER_4.mp4, dated like the houses; PLAYER_4 is dated tomorrow and runs exactly
as long as this video. The cursor rests on PLAYER_1. A notification: NOAH_404 joined the game. Another: NOAH_404:
he's watching too. Black. Four seconds. PLAYER 2 IS ONLINE.
"""
import numpy as np

import anim as A
import film as FM
import screen as SCR
import ui

ICONS = ('Recycle Bin', 'readme.txt')


def mmss(s):
    s = int(round(s))
    return f'{s // 60}:{s % 60:02d}'


def files():
    total = SCR.TOTAL[0] or 770.0
    return [dict(name='PLAYER_1.mp4', date='2018-06-14 04:04', type='MP4 Video', size='2.1 GB', length='19:19:19'),
            dict(name='PLAYER_2.mp4', date='2020-11-03 04:04', type='MP4 Video', size='1.7 GB', length='4:04:04'),
            dict(name='PLAYER_3.mp4', date='2025-07-01 04:04', type='MP4 Video', size='3.4 GB', length='1:04:04'),
            dict(name='PLAYER_4.mp4', date='2026-04-04 04:04', type='MP4 Video', size='404 MB', length=mmss(total))]


def shots(ctx):
    S = FM.Shot
    out = []
    desk = ui.Desktop(clock='4:04 AM', date='4/4/2026')            # (tomorrow)
    folder = ('NOAH_FINAL', ui.icon_folder())
    cur = A.Keys([(0.0, (1180, 640)), (4.5, (1182, 642)), (6.4, (72, 262)), (7.6, (72, 262)), (9.0, (752, 292)),
                  (21.0, (758, 294))])
    t_pop, t_click, t_open = 5.0, 7.3, 7.7

    def desktop(img, t, T, c, f):
        extra = (folder,) if t >= t_pop else ()
        img[:] = desk.base(icons=ICONS, extra_icons=extra)
        if t_pop <= t < t_pop + 0.12:
            img[:] = (img.astype(np.float32) * 1.08).clip(0, 255).astype(np.uint8)
        if t >= t_open:
            desk.explorer(img, 'NOAH_FINAL', files(), hover=0 if t >= 9.0 else None)
        toasts = [(12.0, 'Minecraft', 'NOAH_404 joined the game'), (15.0, 'Minecraft', "NOAH_404: he's watching too")]
        shown = [x for x in toasts if t >= x[0]]
        for k, (t0, title, text) in enumerate(shown):
            slide = float(np.clip((t - t0) / 0.25, 0, 1))
            desk.notification(img, title, text, slide=slide, stack=len(shown) - 1 - k)      # the newest at the bottom
        cx, cy = cur(t)
        desk.draw_cursor(img, cx, cy)

    out.append(S('e1_desktop', 18.5, '2d', overlay=desktop,
                 fx=lambda t: {'grain': 0.03},
                 cues=[(0.0, 'cut_in'), (0.0, 'pc_hum_start'), (t_pop, 'folder_pop'), (t_click, 'click'),
                       (t_click + 0.15, 'click'), (t_open, 'window_open'), (12.0, 'toast'), (15.0, 'toast')]))

    # --- black. Four seconds. PLAYER 2 IS ONLINE ------------------------------------------------------------------------------------
    out.append(S('e2_black', 4.0, 'black', cues=[(0.0, 'silence_all', {'dur': 4.0})]))

    def online(img, t, T, c, f):
        img[:] = 0
        a = float(np.clip((t - 0.2) / 0.08, 0, 1))
        ui.draw_mc_text(img, 'PLAYER 2 IS ONLINE', ui.W // 2, ui.H // 2 - 16, (235, 235, 235), scale=ui.GS * 2,
                        align='center', alpha=a)

    out.append(S('e3_online', 5.0, '2d', overlay=online, fx=lambda t: {'grain': 0.02},
                 cues=[(0.0, 'silence_all', {'dur': 5.0})]))
    out.append(S('e4_end', 1.5, 'black'))
    return out
