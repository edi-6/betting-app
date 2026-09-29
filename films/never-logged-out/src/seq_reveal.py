"""ACT 8: the reveal (10:45 - 11:32).

A book that wasn't there before, dropped on his desk. PLAYER 2: There are always two players. One plays the world.
The other watches. When the watcher enters the world... the world needs a new watcher. It isn't predicting him; it
records him before he does it. It doesn't want to kill him; it wants to replace him. On the monitor (him, from
behind) someone is standing behind him, and he hasn't seen it yet. Footsteps, right behind him. DON'T TURN AROUND.
He turns. Black.
"""
import numpy as np

import anim as A
import common as C
import entities as EN
import film as FM
import ui
from seq_changed import RoomKit

HOTBAR = dict(items=['written_book', 'torch', 'bread', 'clock', 'filled_map', 'torn_page'], selected=6,
              counts={1: 16, 2: 5})
HOTBAR_B = dict(HOTBAR, items=HOTBAR['items'] + ['written_book'], selected=6)       # the new book in slot 7
PAGES = [('There are always two players.', 'PLAYER 2'), ('One plays the world.', None), ('The other watches.', None),
         ('When the watcher enters the world...', None), ('...the world needs a new watcher.', None)]


def shots(ctx):
    S = FM.Shot
    out = []
    kit = RoomKit(ctx)
    P = kit.P
    mon = kit.mon
    floor = kit.door[2]
    behind_chair = np.array([P['room_chair'][0] - 0.1, P['room_chair'][1] - 1.55, floor])
    lean = behind_chair + np.array([0.02, 0.5, 0.0])
    a_lean = A.look_angles(lean + (0, 0, 1.62), mon.centre)
    book = np.array([P['room_desk'][0] + 1.05, P['room_desk'][1] - 0.05, floor + 1.15])

    def dropped_book(t, t_pick=None, eye=None):
        """The book lying on the desk like a dropped item: bobbing and turning. Picked up, it flies to him."""
        p = book + np.array([0.0, 0.0, 0.05 * np.sin(t * 2.0)])
        if t_pick is not None and t >= t_pick:
            u = (t - t_pick) / 0.22
            if u >= 1.0:
                return []
            p = p * (1 - u) + (np.asarray(eye) - np.array([0, 0, 0.4])) * u
        q = EN.qz(t * 1.1)
        return [C.row('item_written_book', p, q, 0.3, C.ITEM_NAMES.index('written_book'))]

    # --- v1: a book on his desk. It wasn't there before ---------------------------------------------------------------------
    near = np.array([book[0] - 0.35, behind_chair[1] + 0.9, floor])
    a_book = A.look_angles(near + (0, 0, 1.62), book)
    t_pick = 4.4
    pov1 = A.POV(pos_keys=[(0, tuple(lean)), (1.2, tuple(lean)), (3.2, tuple(near)), (6.0, tuple(near))],
                 yaw_keys=[(0, a_lean[0]), (1.2, a_lean[0] - 14), (3.2, a_book[0]), (6.0, a_book[0])],
                 pitch_keys=[(0, a_lean[1]), (1.2, -18), (3.2, a_book[1]), (6.0, a_book[1] - 2)], seed=110, jitter=0.06)

    def v1(t, T):
        swing = float(np.clip((t - t_pick + 0.2) / 0.25, 0, 1)) if t < t_pick + 0.05 else 0.0
        cam = pov1.cam(t)
        return kit.scene(pov1, t, T, film_him=True, swing=swing, props=dropped_book(t, t_pick, cam['eye']))

    out.append(S('v1_book', 6.0, '3d', scene=v1, hud=lambda t: HOTBAR_B if t >= t_pick + 0.22 else HOTBAR,
                 subs=[(1.4, 3.6, "That wasn't there before.")],
                 cues=[(1.6, 'step', {'surface': 'carpet'}), (2.4, 'step', {'surface': 'carpet'}),
                       (t_pick + 0.2, 'item_pickup')]))

    # --- v2: PLAYER 2 ------------------------------------------------------------------------------------------------------------
    flips = [0.0, 3.8, 7.4, 11.0, 15.0]

    def v2_late(img, t, T, c, f):
        k = max(i for i, fl in enumerate(flips) if fl <= t)
        text, title = PAGES[k]
        turn = float(np.clip(1.0 - (t - flips[k]) / 0.18, 0, 1)) if k > 0 else 0.0
        ui.book(img, text, k, len(PAGES), title=title, turn=turn)

    out.append(S('v2_player2', 20.0, 'still', scene=v1, still=5.95, late=v2_late, hud=HOTBAR_B,
                 cues=[(0.0, 'book_open')] + [(fl, 'page') for fl in flips[1:]] +
                 [(0.0, 'amb_duck', {'to': 0.2}), (15.0, 'drone', {'level': 0.35, 'dur': 5.0})]))

    # --- v3: it isn't predicting him. On the monitor, someone behind him ------------------------------------------------------------
    pov3 = A.POV(pos_keys=[(0, tuple(near)), (1.8, tuple(lean)), (10.0, tuple(lean + (0.0, 0.08, 0.0)))],
                 yaw_keys=[(0, a_book[0]), (1.8, a_lean[0]), (10.0, a_lean[0])],
                 pitch_keys=[(0, a_book[1]), (1.8, a_lean[1]), (10.0, a_lean[1])], seed=111, jitter=0.05)
    pov3.fov_keys = A.Keys([(0, 70.0), (1.8, 60.0), (6.0, 44.0), (10.0, 38.0)])      # the zoom key: closer
    t_there = 5.2

    WIDE = ((1.25, -2.6, 1.1), (-0.55, 1.0, -0.6), 60.0)        # the camera behind him, up in the other corner

    def other(t, feet, yaw_deg, step=0.0, look=0.0):
        """The one only the camera sees: his skin, right behind him."""
        f = np.asarray(feet, float)
        a = EN.Actor('player', 'you', f + np.array([-0.8 + 0.2 * step, -1.3 + 0.6 * step, 0.0]),
                     yaw=np.radians(yaw_deg + 12.0 * (1 - step)))                 # behind him, on the lit side
        a.idle_t = 0.0
        a.head_yaw = look
        return a

    def v3(t, T):
        extra = [other(t, pov3.feet(t), pov3.angles(t)[0])] if t >= t_there else []
        return kit.scene(pov3, t, T, film_him=True, item='written_book', behind=extra, cam2=WIDE)

    out.append(S('v3_recording', 10.0, '3d', scene=v3, hud=HOTBAR_B,
                 subs=[(0.8, 2.8, "It's not predicting me."), (3.2, 6.0, "It's recording me. Before I do it."),
                       (6.6, 9.8, "It doesn't want to kill me. It wants to replace me.")],
                 cues=[(1.0, 'drone', {'level': 0.35, 'dur': 9.0}), (t_there, 'sub_hit', {'gain': 0.25})]))

    # --- v4: footsteps, right behind him. DON'T TURN AROUND. -------------------------------------------------------------------------------
    pov4 = A.POV(pos_keys=[(0, tuple(lean + (0.0, 0.08, 0.0)))], yaw_keys=[(0, a_lean[0]), (8.0, a_lean[0])],
                 pitch_keys=[(0, a_lean[1]), (8.0, a_lean[1])], seed=112, jitter=0.0)      # he doesn't dare move
    pov4.fov_keys = A.Keys([(0, 38.0), (8.0, 31.0)])

    def v4(t, T):
        step = float(A.smooth((t - 1.2) / 1.4))
        look = float(np.radians(-70.0) * A.smooth((t - 5.2) / 1.6))        # its head turns to the camera
        extra = [other(t, pov4.feet(t), pov4.angles(t)[0], step=step, look=look)]
        return kit.scene(pov4, t, T, film_him=True, item='written_book', behind=extra, cam2=WIDE)

    def v4_rule(img, t, T, c, f):
        a = float(np.clip((t - 3.0) / 0.5, 0, 1))
        if a <= 0:
            return
        rng = np.random.default_rng(int(t * FM.FPS))
        dx, dy = rng.normal(0, 2.0, 2)
        ui.draw_mc_text(img, "DON'T TURN AROUND.", ui.W // 2 + int(dx), 150 + int(dy), (236, 230, 220),
                        scale=ui.GS * 2, align='center', alpha=a * (0.85 + 0.15 * np.sin(t * 23.0)))

    out.append(S('v4_behind', 8.0, '3d', scene=v4, overlay=v4_rule, hud=HOTBAR_B,
                 cues=[(0.0, 'silence_all', {'dur': 1.2}), (1.3, 'step_behind', {'dist': 1.4}),
                       (2.3, 'step_behind', {'dist': 0.8}), (3.0, 'sting_low'),
                       (3.0, 'drone', {'level': 0.55, 'dur': 5.0})]))

    # --- v5: he turns anyway. Black ---------------------------------------------------------------------------------------------------
    t_cut = 0.72
    pov5 = A.POV(pos_keys=[(0, tuple(lean + (0.0, 0.08, 0.0)))],
                 yaw_keys=[(0, a_lean[0]), (0.25, a_lean[0] + 6), (0.9, a_lean[0] + 170)],
                 pitch_keys=[(0, a_lean[1]), (0.9, 0.0)], seed=113, jitter=0.0)

    def v5(t, T):
        return kit.scene(pov5, t, T, film_him=True, item='written_book', cam2=WIDE)

    out.append(S('v5_turn', 0.75, '3d', scene=v5, hud=HOTBAR_B,
                 fx=lambda t: {'fade': 0.0 if t >= t_cut else 1.0},
                 cues=[(0.2, 'turn_whoosh'), (t_cut, 'cut_silence')]))
    return out
