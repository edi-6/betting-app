"""ACT 2: the first lie (2:04 - 3:26).

Every villager has a name tag. Tom, Mira, Elias, Rosa, Finn, June... and one standing still, staring: NOAH. He
right-clicks it; it turns at once and walks off to the map house. Inside, a wall of maps of the same village at
different times: BEFORE, ABANDONED, BURNED, TODAY. He takes TODAY. A red mark sits exactly where he stands. Then a
second one appears, behind him. He turns, slowly. Nothing. And the villager is gone.
"""
import numpy as np

import anim as A
import common as C
import entities as EN
import film as FM
import maps as MP

HOTBAR = dict(items=['written_book', 'torch', 'bread', 'clock'], selected=0, counts={1: 16, 2: 5})
HOTBAR_MAP = dict(items=['written_book', 'torch', 'bread', 'clock', 'filled_map'], selected=4, counts={1: 16, 2: 5})

NOAH_SPOT = np.array([4.5, 10.5, 0.0])        # by the well, facing south, staring
CORNER = np.array([17.9, 10.1, 0.0])          # where he stands in the map house


def crowd(T, t_scene=0.0):
    """The plaza: named villagers going about (small loops), for the name-tag scenes."""
    specs = [('Tom', 'villager_farmer', (-0.5, -4.0), 1.6, 0.0), ('Mira', 'villager_librarian', (9.0, -2.5), 1.2, 1.1),
             ('Elias', 'villager_cartographer', (12.5, -1.5), 0.0, 2.2), ('Rosa', 'villager_cleric', (-2.5, 4.5), 1.0, 3.3),
             ('Finn', 'villager_smith', (-3.5, -1.5), 1.4, 4.4), ('June', 'villager_shepherd', (0.5, 9.0), 0.0, 5.5)]
    out = []
    for (name, skin, (x, y), r, ph) in specs:
        a = EN.Actor('villager', skin)
        if r > 0:
            ang = T * 0.35 / r + ph
            a.pos = np.array([x + r * np.cos(ang), y + r * np.sin(ang), 0.0])
            a.yaw = ang + np.pi                    # walking round the circle (tangent)
            a.walk = T * 4.0
            a.walk_amp = 1.0
        else:
            a.pos = np.array([x, y, 0.0])
            a.yaw = np.radians(160 + 40 * np.sin(T * 0.2 + ph))
            a.head_yaw = 0.4 * np.sin(T * 0.6 + ph)
        a.name = name
        a.idle_t = T
        out.append(a)
    return out


def noah_villager(pos, yaw_deg, walk=0.0, amp=0.0, head_yaw=0.0):
    a = EN.Actor('villager', 'villager_noah', pos, yaw=np.radians(yaw_deg))
    a.walk, a.walk_amp, a.head_yaw = walk, amp, head_yaw
    a.name = 'NOAH'
    return a


def map_marks(pos, facing_yaw_deg, second=0.0):
    """TODAY with his mark (and the one behind him, fading in with `second`)."""
    img = MP.to_map_image(np.zeros((1, 1, 3)), np.random.default_rng(0)) if False else None
    del img
    return None


def shots(ctx):
    S = FM.Shot
    out = []
    base_map = ctx.maps['TODAY'].copy()

    def marked_map(p_world, yaw_deg, second):
        img = base_map.copy()
        mx, my = MP.world_to_map(p_world[0], p_world[1])
        # the game's white pointer for the player, then the red marks
        cx, cy = int(round(mx)), int(round(my))
        img[cy - 1:cy + 2, cx - 1:cx + 2] = (240, 240, 240, 255)
        red = (200, 20, 20, 255)
        for d in range(-2, 3):
            if 0 <= cy + d < 128 and 0 <= cx + d < 128:
                img[cy + d, cx + d] = red
                img[cy + d, cx - d] = red
        if second > 0:
            back = -A.dir_from(yaw_deg, 0)[:2]
            bx, by = mx + back[0] * 4.0, my - back[1] * 4.0
            bx, by = int(round(bx)), int(round(by))
            a = float(np.clip(second, 0, 1))
            for d in range(-2, 3):
                for (yy, xx) in ((by + d, bx + d), (by + d, bx - d)):
                    if 0 <= yy < 128 and 0 <= xx < 128:
                        img[yy, xx, :3] = (img[yy, xx, :3] * (1 - a) + np.array(red[:3]) * a).astype(np.uint8)
        return img

    # --- l1: into the plaza; the villagers have names ----------------------------------------------------------
    path1 = A.Path([(6.2, -30.0, 0.0), (6.0, -22.0, 0.0), (5.4, -14.0, 0.0), (5.0, -7.0, 0.0)])
    pov1 = A.POV(path=path1, s_keys=[(0, 0.0), (9.5, path1.length)],
                 yaw_keys=[(0, 6), (3.0, -18), (5.5, 14), (7.5, -6), (9.5, 2)],
                 pitch_keys=[(0, -3), (5.0, -1), (9.5, -4)], seed=21)

    def l1(t, T):
        acts = crowd(T) + [noah_villager(NOAH_SPOT, 180)] + C.animals(T)
        return C.pov_scene(pov1, t, 'sunset', actors=acts, props=C.world_props(ctx, 'village', door=0.0))

    def tags(pov, extra=()):
        def ov(img, t, T, ctx_, film):
            acts = crowd(T) + [noah_villager(NOAH_SPOT, 180)] + list(extra)
            C.name_tags(img, pov.cam(t), acts)
        return ov

    out.append(S('l1_names', 9.5, '3d', scene=l1, overlay=tags(pov1), hud=HOTBAR,
                 subs=[(2.0, 4.5, 'Wait. They all have names.'), (5.0, 8.5, 'Someone name-tagged every villager.')],
                 cues=[(ts, 'step', {'surface': 'path'}) for ts in pov1.steps(0, 9.5)] +
                 [(0.0, 'amb', {'kind': 'day'}), (0.0, 'music', {'piece': 'perfect2'}), (3.0, 'villager', {'pan': -0.4}),
                  (6.5, 'villager', {'pan': 0.5})]))

    # --- l2: he looks around the plaza... and stops on the one staring at him: NOAH ---------------------------------
    pov2 = A.POV(pos_keys=[(0, (5.0, -7.0, 0.0)), (8.0, (5.0, -6.2, 0.0))],
                 yaw_keys=[(0, 2), (2.5, -48), (5.0, 38), (6.8, 2), (8.0, 1)],
                 pitch_keys=[(0, -4), (5.0, -2), (6.8, 1), (8.0, 1)], seed=22)

    def l2(t, T):
        acts = crowd(T) + [noah_villager(NOAH_SPOT, 180)] + C.animals(T)
        return C.pov_scene(pov2, t, 'sunset', actors=acts, props=C.world_props(ctx, 'village'))

    out.append(S('l2_noah', 8.0, '3d', scene=l2, overlay=tags(pov2), hud=HOTBAR,
                 subs=[(5.0, 7.5, '...Noah?')], cues=[(6.6, 'music_stop'), (6.6, 'sting_low')]))

    # --- l3: he walks up and right-clicks it: it turns at once and walks away ---------------------------------------
    pov3 = A.POV(pos_keys=[(0, (5.0, -6.2, 0.0)), (2.4, (4.8, 6.6, 0.0)), (5.0, (5.0, 7.2, 0.0))],
                 yaw_keys=[(0, 1), (2.4, 0), (3.2, 0), (5.0, -38)], pitch_keys=[(0, 1), (2.4, -8), (5.0, -6)], seed=23)
    nv_path = A.Path([(4.5, 10.5, 0.0), (6.5, 9.2, 0.0), (10.0, 7.8, 0.0), (14.8, 6.5, 0.0)])
    nv = A.Walker(nv_path, [(0, 0.0), (2.9, 0.0), (5.0, 3.2)])

    def noah3(t):
        a = noah_villager(nv.pos(t), 180)
        if t < 2.9:
            a.yaw = np.radians(180)
        else:
            nv.pose(a, t)
            k = float(np.clip((t - 2.9) / 0.15, 0, 1))          # snaps round, not like a villager turns
            a.yaw = np.radians(180) * (1 - k) + a.yaw * k
        a.name = 'NOAH'
        return a

    def l3(t, T):
        swing = float(np.clip((t - 2.6) / 0.25, 0, 1)) if t < 2.85 else 0.0
        acts = crowd(T) + [noah3(t)] + C.animals(T)
        return C.pov_scene(pov3, t, 'sunset', actors=acts, swing=swing, props=C.world_props(ctx, 'village'))

    def l3_tags(img, t, T, ctx_, film):
        C.name_tags(img, pov3.cam(t), crowd(T) + [noah3(t)])

    out.append(S('l3_click', 5.0, '3d', scene=l3, overlay=l3_tags, hud=HOTBAR,
                 subs=[(3.1, 4.9, 'Hey- where are you going?')],
                 cues=[(ts, 'step', {'surface': 'path'}) for ts in pov3.steps(0, 2.4)] +
                 [(2.7, 'villager_no', {'pan': 0.0})]))

    # --- l4: third person: he follows the villager to the map house (we see his skin) --------------------------------
    nv4_path = A.Path([(10.0, 7.8, 0.0), (14.8, 6.5, 0.0), (15.2, 6.5, 0.0), (18.5, 6.5, 0.0), (18.0, 9.5, 0.0),
                       CORNER])
    nv4 = A.Walker(nv4_path, [(0, 0.0), (4.0, 4.9), (5.2, 4.9), (7.6, 8.6), (9.4, nv4_path.length)])
    you_path = A.Path([(5.0, 7.2, 0.0), (8.0, 7.0, 0.0), (11.5, 6.6, 0.0), (13.6, 6.5, 0.0)])
    you4 = A.Walker(you_path, [(0, 0.0), (1.2, 0.0), (10.0, you_path.length)])
    door4 = A.Keys([(0, 0.0), (4.05, 0.0), (4.35, 90.0, 'out'), (8.5, 90.0)])
    cam4_eye = A.Keys([(0, (2.2, 3.8, 2.7)), (10.0, (9.8, 4.2, 2.4))])

    def l4(t, T):
        a = EN.Actor('player', 'you')
        you4.pose(a, t)
        n = noah_villager((0, 0, 0), 0)
        nv4.pose(n, t)
        n.name = 'NOAH'
        eye = cam4_eye(t)
        tgt = a.pos + np.array([0.0, 0.0, 1.3]) + (n.pos - a.pos) * 0.45
        cam = dict(eye=eye, target=tgt, fov=52)
        acts = crowd(T) + [a, n] + C.animals(T)
        return dict(world='village', env=C.env('sunset'), cam=cam, actors=acts,
                    props=C.world_props(ctx, 'village', maphouse_door=float(door4(t))),
                    dof=dict(focus=float(np.linalg.norm(a.pos + np.array([0, 0, 1.3]) - eye)), k=6.0, maxr=7))

    def l4_tags(img, t, T, ctx_, film):
        n = noah_villager((0, 0, 0), 0)
        nv4.pose(n, t)
        n.name = 'NOAH'
        C.name_tags(img, l4(t, T)['cam'], crowd(T) + [n])

    out.append(S('l4_follow', 10.0, '3d', scene=l4, overlay=l4_tags,
                 cues=[(4.05, 'door_open', {'pan': 0.3, 'dist': 6})] +
                 [(1.2 + k * 0.45, 'step', {'surface': 'path', 'gain': 0.5}) for k in range(18)]))

    # --- l5: at the door: the villager inside, in the corner. He goes in ---------------------------------------------
    pov5 = A.POV(pos_keys=[(0, (13.6, 6.5, 0.0)), (1.2, (14.6, 6.5, 0.0)), (4.0, (17.6, 6.5, 0.0))],
                 yaw_keys=[(0, -90), (1.5, -78), (3.0, -62), (4.0, -86)], pitch_keys=[(0, -3), (4.0, -2)], seed=24)

    def l5(t, T):
        n = noah_villager(CORNER, -120)
        return C.pov_scene(pov5, t, 'sunset', actors=[n] + crowd(T), props=C.world_props(ctx, 'village',
                                                                                        maphouse_door=90.0))

    def l5_tags(img, t, T, ctx_, film):
        C.name_tags(img, pov5.cam(t), [noah_villager(CORNER, -120)])

    out.append(S('l5_door', 4.0, '3d', scene=l5, overlay=l5_tags, hud=HOTBAR,
                 cues=[(ts, 'step', {'surface': 'wood'}) for ts in pov5.steps(0, 4.0)] +
                 [(0.0, 'amb_fade', {'to': 'interior', 'dur': 3.0})]))

    # --- l6: the wall of maps -------------------------------------------------------------------------------------------
    P = ctx.worlds['village'].points

    def aim(eye, p):
        return A.look_angles(eye, p)

    eye6 = np.array([17.6, 6.5, 1.62])
    targets = {'wall': (23.9, 6.2, 2.0), 'before': (23.9, 7.5, 2.5), 'abandoned': (23.9, 6.5, 2.5),
               'burned': (23.9, 5.5, 2.5), 'today': (23.9, 7.5, 1.5)}
    pov6 = A.POV(pos_keys=[(0, (17.6, 6.5, 0.0)), (4.0, (19.4, 6.5, 0.0)), (15.0, (20.7, 6.6, 0.0))],
                 yaw_keys=[(0, aim(eye6, targets['wall'])[0]), (4.8, aim(eye6 + (1.9, 0, 0), targets['before'])[0]),
                           (8.2, aim(eye6 + (2.6, 0, 0), targets['abandoned'])[0]),
                           (11.2, aim(eye6 + (2.9, 0, 0), targets['burned'])[0]), (15.0, aim(eye6 + (3.1, 0, 0),
                                                                                            targets['burned'])[0])],
                 pitch_keys=[(0, 3), (4.8, 10), (8.2, 13), (11.2, 14), (15.0, 13)], seed=25, jitter=0.12)

    def l6(t, T):
        n = noah_villager(CORNER, -120)
        return C.pov_scene(pov6, t, 'sunset', actors=[n], props=C.world_props(ctx, 'village', maphouse_door=90.0))

    def l6_tags(img, t, T, ctx_, film):
        C.name_tags(img, pov6.cam(t), [noah_villager(CORNER, -120)])

    out.append(S('l6_maps', 15.0, '3d', scene=l6, overlay=l6_tags, hud=HOTBAR,
                 subs=[(1.5, 4.5, 'These are all the same village.'), (5.0, 8.0, "This one's before anything was built."),
                       (8.5, 11.0, "This one's abandoned."), (11.5, 14.5, "And this one's... burned?")],
                 cues=[(ts, 'step', {'surface': 'wood'}) for ts in pov6.steps(0, 15.0)] +
                 [(0.0, 'drone', {'level': 0.25, 'dur': 15.0})]))

    # --- l7: he takes TODAY off the wall ----------------------------------------------------------------------------------
    e7 = np.array([20.7, 6.6, 1.62])
    pov7 = A.POV(pos_keys=[(0, (20.7, 6.6, 0.0))],
                 yaw_keys=[(0, aim(e7, targets['burned'])[0]), (1.6, aim(e7, targets['today'])[0]),
                           (5.0, aim(e7, targets['today'])[0] + 2)],
                 pitch_keys=[(0, 13), (1.6, -1), (3.2, -1), (5.0, -8)], seed=26, jitter=0.1)
    take_at = 2.6
    frame_tag = 'frame:TODAY@23,7,1'

    def l7(t, T):
        swing = float(np.clip((t - take_at + 0.2) / 0.25, 0, 1)) if t < take_at + 0.05 else 0.0
        excl = (frame_tag,) if t >= take_at else ()
        hold = t >= take_at + 0.5
        sc = C.pov_scene(pov7, t, 'sunset', props=C.world_props(ctx, 'village', maphouse_door=90.0), swing=swing,
                         exclude=excl, hold_map=hold, map_layer=0, actors=[noah_villager(CORNER, -120)])
        return sc

    out.append(S('l7_take', 5.0, '3d', scene=l7, hud=lambda t: HOTBAR_MAP if t >= take_at else HOTBAR,
                 cues=[(take_at, 'item_frame_take')]))

    # --- l8: the map in his hands: his mark... and the one behind him --------------------------------------------------------
    pos8 = np.array([20.7, 6.6, 0.0])
    yaw8 = aim(e7, targets['today'])[0] + 2
    pov8 = A.POV(pos_keys=[(0, (20.7, 6.6, 0.0))], yaw_keys=[(0, yaw8), (14.0, yaw8 + 1)],
                 pitch_keys=[(0, -8), (3.0, -14), (14.0, -15)], seed=27, jitter=0.07)

    def l8(t, T):
        second = float(np.clip((t - 7.0) / 1.2, 0, 1))
        img = marked_map(pos8, yaw8, second)

        def prep(r, img=img):
            r.update_kind_layer('held_map', 1, img)
        sc = C.pov_scene(pov8, t, 'sunset', props=C.world_props(ctx, 'village', maphouse_door=90.0),
                         exclude=(frame_tag,), hold_map=True, map_layer=1, actors=[noah_villager(CORNER, -120)])
        sc['prep'] = prep
        return sc

    out.append(S('l8_marks', 14.0, '3d', scene=l8, hud=HOTBAR_MAP,
                 subs=[(2.5, 4.5, "That's me.")],
                 cues=[(7.0, 'sting_note'), (7.0, 'drone', {'level': 0.5, 'dur': 7.0})]))

    # --- l9: he turns around, slowly. Nothing. (For two frames, something in the doorway.) --------------------------
    pov9 = A.POV(pos_keys=[(0, (20.7, 6.6, 0.0))],
                 yaw_keys=[(0, yaw8 + 1), (0.6, yaw8 + 4), (4.6, yaw8 + 178), (8.0, yaw8 + 176)],
                 pitch_keys=[(0, -14), (2.0, -4), (4.6, -2), (8.0, -3)], seed=28, jitter=0.1)

    def l9(t, T):
        acts = []
        fr = int(round(t * FM.FPS))
        if fr in (60, 61):                 # 2.5 s into the turn: the doorway, for two frames
            ghost = EN.Actor('player', 'you', (16.4, 6.5, 0.0), yaw=np.radians(-90))
            acts.append(ghost)
        return C.pov_scene(pov9, t, 'sunset', props=C.world_props(ctx, 'village', maphouse_door=90.0),
                           exclude=(frame_tag,), hold_map=t < 1.0, map_layer=1, actors=acts)

    out.append(S('l9_turn', 8.0, '3d', scene=l9, hud=HOTBAR_MAP, cues=[(0.0, 'silence_all', {'dur': 8.0})]))

    # --- l10: the corner where the villager stood is empty -------------------------------------------------------------
    e10 = np.array([20.7, 6.6, 1.62])
    pov10 = A.POV(pos_keys=[(0, (20.7, 6.6, 0.0))],
                  yaw_keys=[(0, yaw8 + 176), (1.6, aim(e10, CORNER + (0, 0, 1.2))[0]),
                            (4.0, aim(e10, CORNER + (0, 0, 1.0))[0] + 3)],
                  pitch_keys=[(0, -3), (1.6, -6), (4.0, -8)], seed=29, jitter=0.1)

    def l10(t, T):
        return C.pov_scene(pov10, t, 'sunset', props=C.world_props(ctx, 'village', maphouse_door=90.0),
                           exclude=(frame_tag,))

    out.append(S('l10_gone', 4.0, '3d', scene=l10, hud=HOTBAR_MAP,
                 subs=[(0.6, 3.0, '...Where did he go?')], fx=lambda t: {'fade': float(np.clip((4.0 - t) / 1.0, 0, 1))}))
    return out
