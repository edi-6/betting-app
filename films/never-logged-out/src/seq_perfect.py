"""ACT 1: the perfect world (0:35 - 2:04).

He spawns at sunset south of the village. Wide, beautiful shots: villagers, animals, chimney smoke, the lake in the
low sun (and, far off at the edge of the woods, a figure that shouldn't be there). He walks up the path to the
oldest house, lets himself in: food, tools, books, maps, a bed, a furnace that is still burning. The chest holds a
book called RULES. Five pages. The last one says he already broke rule 4. Hard cut.
"""
import numpy as np

import anim as A
import common as C
import entities as EN
import film as FM
import ui

HOTBAR_BEFORE = dict(items=[], selected=0)
HOTBAR_AFTER = dict(items=['written_book', 'torch', 'bread', 'clock'], selected=0, counts={1: 16, 2: 5})


def smoke(T, base=(-2.5, -23.5, 10.2), n=26, night=False, seed=5):
    """Chimney smoke: soft puffs rising and drifting east."""
    rng = np.random.default_rng(seed)
    out = []
    base = np.array(base, float)
    for k in range(n):
        life = 7.0
        age = (T * 0.9 + k * life / n) % life
        u = age / life
        p = base + np.array([0.9 * age + 0.3 * np.sin(k * 3.1 + T * 0.4), 0.25 * age, 0.75 * age])
        size = 0.7 + 1.8 * u
        a = 0.22 * np.sin(np.pi * min(u * 1.3, 1.0)) * (1 - u)
        col = (0.05, 0.05, 0.06) if night else (0.95, 0.72, 0.55)
        out.append([*p, size, *col, a])
    return np.array(out, np.float32)


def life(ctx, T, skip=()):
    vil = C.Village()
    return vil.villagers(T, skip=skip) + C.animals(T)


def figure_on_ridge(ctx):
    """The one that shouldn't be there: a player standing still at the edge of the woods, facing the village."""
    x, y = -46.5, 44.5
    z = C.ground_z(ctx, x, y)
    a = EN.Actor('player', 'noah', (x, y, z), yaw=np.radians(A.yaw_of((4.0 - x, -10.0 - y))))
    return a


def shots(ctx):
    S = FM.Shot
    P = ctx.worlds['village'].points
    out = []

    # --- p1: spawn. The HUD comes up, he looks at the sunset over the lake, then at the village ---------------
    pov1 = A.POV(pos_keys=[(0, (6.5, -72.5, 0.0))], yaw_keys=[(0, 14), (1.0, 14), (3.6, 70), (5.2, 72), (7.5, 6)],
                 pitch_keys=[(0, -3), (3.6, 1.5), (7.5, -3)], seed=11)

    def p1(t, T):
        return C.pov_scene(pov1, t, 'sunset', actors=life(ctx, T), props=C.world_props(ctx, 'village'),
                           particles={'soft': smoke(T)})

    out.append(S('p1_spawn', 7.5, '3d', scene=p1, hud=lambda t: dict(HOTBAR_BEFORE, alpha=float(np.clip((t - 0.6) / 1.0,
                                                                                                          0, 1))),
                 chat=True, subs=[(3.5, 6.0, "Okay. It's just... a village.")],
                 fx=lambda t: {'fade': float(np.clip(t / 1.2, 0, 1))},
                 cues=[(0.0, 'amb', {'kind': 'day'}), (0.0, 'music', {'piece': 'perfect'}),
                       (0.5, 'chat', {'text': 'YOU joined the game', 'color': ui.YELLOW})]))

    # --- p2: wide crane: the village in the low sun, smoke, villagers, the pen; the figure on the ridge ----------
    eye2 = A.Keys([(0, (46, -46, 12)), (8, (34, -32, 19))])
    tgt2 = A.Keys([(0, (2, -14, 3)), (8, (-6, -2, 5))])

    def p2(t, T):
        cam = dict(eye=eye2(t), target=tgt2(t), fov=55)
        acts = life(ctx, T) + [figure_on_ridge(ctx)]
        return dict(world='village', env=C.env('sunset'), cam=cam, actors=acts, props=C.world_props(ctx, 'village'),
                    particles={'soft': smoke(T)})

    out.append(S('p2_crane', 8.0, '3d', scene=p2, cues=[(1.5, 'villager', {'pan': -0.3, 'dist': 20}),
                                                        (5.0, 'cow', {'pan': 0.5, 'dist': 25})]))

    # --- p3: low through the wheat, the farmer walking, the sun behind --------------------------------------------
    farmer_path = A.Path([(-29.5, -7.4, 0.0), (-34.0, -7.2, 0.0), (-39.5, -7.6, 0.0)])
    farmer = A.Walker(farmer_path, [(0, 0.0), (6.0, 7.5)])
    eye3 = A.Keys([(0, (-26.6, -13.8, 1.22)), (6, (-26.9, -12.6, 1.3))])        # just over the ears of wheat
    tgt3 = A.Keys([(0, (-42.0, -9.5, 0.95)), (6, (-42.0, -8.5, 1.05))])

    def p3(t, T):
        cam = dict(eye=eye3(t), target=tgt3(t), fov=50)
        f = EN.Actor('villager', 'villager_farmer')
        farmer.pose(f, t)
        acts = life(ctx, T, skip=('Tom',)) + [f]
        return dict(world='village', env=C.env('sunset'), cam=cam, actors=acts, props=C.world_props(ctx, 'village'),
                    dof=dict(focus=float(np.linalg.norm(np.array(farmer.pos(t)) - np.array(eye3(t)))), k=2.5, maxr=4))

    out.append(S('p3_wheat', 6.0, '3d', scene=p3, cues=[(2.0, 'villager', {'pan': -0.2, 'dist': 8})]))

    # --- p4: walking up the path to the house ----------------------------------------------------------------------
    path4 = A.Path([(5.2, -58.0, 0.0), (4.4, -50.0, 0.0), (3.0, -43.5, 0.0), (1.4, -39.0, 0.0), (0.6, -37.0, 0.0)])
    pov4 = A.POV(path=path4, s_keys=[(0, 0.0), (8.0, path4.length)],
                 yaw_keys=[(0, 4), (2.5, -12), (4.0, -35), (5.5, -8), (8.0, 0)],
                 pitch_keys=[(0, -4), (4.0, -2), (8.0, -6)], seed=12)
    passer_path = A.Path([(2.2, -40.0, 0.0), (3.6, -46.0, 0.0), (5.2, -54.0, 0.0), (6.0, -62.0, 0.0)])
    passer = A.Walker(passer_path, [(0, 0.0), (8.0, 9.0)])

    def p4(t, T):
        v = EN.Actor('villager', 'villager_shepherd')
        passer.pose(v, t)
        v.head_yaw = 0.6 if 2.5 < t < 5.5 else 0.0            # he looks at the player as they pass
        v.name = None
        return C.pov_scene(pov4, t, 'sunset', actors=life(ctx, T) + [v], props=C.world_props(ctx, 'village'),
                           particles={'soft': smoke(T)})

    steps4 = pov4.steps(0, 8.0)
    out.append(S('p4_walk', 8.0, '3d', scene=p4, hud=HOTBAR_BEFORE, chat=True,
                 subs=[(1.5, 4.0, 'Kind of beautiful, honestly.'), (4.5, 7.2, 'Someone put a lot of work into this.')],
                 cues=[(ts, 'step', {'surface': 'grass'}) for ts in steps4] + [(3.2, 'villager', {'pan': 0.4, 'dist': 3})]))

    # --- p5: the door. He opens it and steps in -------------------------------------------------------------------
    door_keys = A.Keys([(0, 0.0), (1.05, 0.0), (1.35, 88.0, 'out')])
    pov5 = A.POV(pos_keys=[(0, (0.6, -37.0, 0.0)), (0.9, (0.5, -35.9, 0.0)), (3.4, (0.5, -35.8, 0.0)),
                           (5.5, (0.5, -32.8, 0.0)), (6.0, (0.5, -32.6, 0.0))],
                 yaw_keys=[(0, 0), (6.0, 0)], pitch_keys=[(0, -4), (1.0, -8), (3.4, -3), (6.0, -2)], seed=13)

    def p5(t, T):
        swing = float(np.clip((t - 0.9) / 0.25, 0, 1)) if t < 1.15 else 0.0
        return C.pov_scene(pov5, t, 'sunset', actors=life(ctx, T), swing=swing,
                           props=C.world_props(ctx, 'village', door=float(door_keys(t))),
                           particles={'soft': smoke(T)})

    out.append(S('p5_door', 6.0, '3d', scene=p5, hud=HOTBAR_BEFORE,
                 subs=[(2.0, 3.6, 'Hello?')],
                 cues=[(1.05, 'door_open'), (3.8, 'step', {'surface': 'wood'}), (4.4, 'step', {'surface': 'wood'}),
                       (5.0, 'step', {'surface': 'wood'}), (5.5, 'step', {'surface': 'wood'}),
                       (0.0, 'amb_fade', {'to': 'interior', 'dur': 5.0})]))

    # --- p6: inside. He looks around: the bookshelves, the furnace still burning, the maps and the clock -------------
    pov6 = A.POV(pos_keys=[(0, (0.5, -32.6, 0.0)), (6.0, (0.4, -31.0, 0.0)), (13.0, (0.2, -29.6, 0.0))],
                 yaw_keys=[(0, 0), (2.0, -38), (4.2, -52), (6.5, -8), (9.5, 4), (11.5, 28), (13.0, 18)],
                 pitch_keys=[(0, -2), (2.0, -6), (4.2, -4), (6.5, 2), (9.5, -8), (13.0, -10)], seed=14)

    def p6(t, T):
        return C.pov_scene(pov6, t, 'sunset', props=C.world_props(ctx, 'village', door=88.0))

    out.append(S('p6_inside', 13.0, '3d', scene=p6, hud=HOTBAR_BEFORE,
                 subs=[(2.0, 4.5, 'Someone lived here.'), (6.0, 9.5, 'The furnace is still going. Who lit this?'),
                       (10.0, 12.6, "...It's probably just a corrupted save.")],
                 cues=[(0.0, 'fire', {'dur': 13.0, 'pos': 'furnace'}), (5.4, 'step', {'surface': 'wood'}),
                       (6.2, 'step', {'surface': 'wood'})]))

    # --- p7: the chest --------------------------------------------------------------------------------------------
    lid7 = A.Keys([(0, 0.0), (2.45, 0.0), (2.8, 72.0, 'out')])
    pov7 = A.POV(pos_keys=[(0, (0.2, -29.6, 0.0)), (2.0, (-1.5, -26.6, 0.0)), (4.0, (-1.5, -26.5, 0.0))],
                 yaw_keys=[(0, 18), (2.0, 2), (4.0, 0)], pitch_keys=[(0, -10), (2.0, -38), (4.0, -40)], seed=15)

    def p7(t, T):
        swing = float(np.clip((t - 2.25) / 0.25, 0, 1)) if t < 2.5 else 0.0
        return C.pov_scene(pov7, t, 'sunset', swing=swing,
                           props=C.world_props(ctx, 'village', door=88.0, lid=float(lid7(t))))

    out.append(S('p7_chest', 4.0, '3d', scene=p7, hud=HOTBAR_BEFORE,
                 cues=[(0.5, 'step', {'surface': 'wood'}), (1.2, 'step', {'surface': 'wood'}),
                       (1.8, 'step', {'surface': 'wood'}), (2.45, 'chest_open')]))

    # --- p8: the chest's contents; he takes the book --------------------------------------------------------------
    contents = {4: ('written_book', 1), 11: ('torch', 16), 13: ('bread', 5), 15: ('clock', 1)}
    cur8 = A.Keys([(0, (1180, 760)), (1.2, (1020, 440)), (4.0, (1020, 440)), (4.6, (1080, 520)),
                   (6.0, (1080, 520))])

    def p8_late(img, t, T, ctx_, film):
        took = t >= 4.0
        cont = dict(contents)
        if took:
            cont = {}
        hover = None
        if 1.3 <= t < 4.0:
            hover = (4, [('Rules', (255, 255, 255)), ('by NOAH_404', (170, 170, 170)), ('Original', (170, 170, 170))])
        inv = {27: ('written_book', 1), 28: ('torch', 16), 29: ('bread', 5), 30: ('clock', 1)} if took else {}
        ui.chest(img, cont, hover=hover, inventory=inv)
        cx, cy = cur8(t)
        ctx_.desk.draw_cursor(img, cx, cy)

    out.append(S('p8_chestgui', 6.0, 'still', scene=p7, still=3.95, late=p8_late,
                 cues=[(0.0, 'gui_open'), (4.0, 'item_pickup')]))

    # --- p9: the book of rules --------------------------------------------------------------------------------------
    pages = ["RULES\n\n1. Don't go underground after midnight.",
             "2. If you hear footsteps, don't turn around.",
             "3. If someone says your name, leave the world.",
             "4. THERE IS ONLY ONE PLAYER.",
             "You already broke Rule 4."]
    flips = [0.0, 6.0, 11.5, 17.0, 23.0]

    def p9_late(img, t, T, ctx_, film):
        k = max(i for i, f in enumerate(flips) if f <= t)
        turn = float(np.clip(1.0 - (t - flips[k]) / 0.18, 0, 1)) if k > 0 else 0.0
        ui.book(img, pages[k], k, 5, turn=turn)

    out.append(S('p9_rules', 28.0, 'still', scene=p7, still=3.95, late=p9_late, hud=HOTBAR_AFTER,
                 subs=[(19.5, 22.5, "(nervous laugh) Okay. That's creepy.")],
                 cues=[(0.0, 'book_open')] + [(f, 'page') for f in flips[1:]] + [(23.0, 'music_stop'),
                                                                                    (23.0, 'amb_duck', {'to': 0.25})]))
    out.append(S('p10_cut', 1.5, 'black', cues=[(0.0, 'silence')]))
    return out
