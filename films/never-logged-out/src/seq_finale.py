"""ACT 9: the most disturbing moment (11:32 - 12:21).

Black. No music, no sound. Then, whispered: That's not me. The village at night, the house with the rules, and his
character walking up the path, seen from behind by a camera nobody is holding. The keys on the overlay: S held
down, ESC, ESC, ESC; W never lights up, and it keeps walking. It opens the door. Inside, not the house's room but his
real one, and at his desk, in his chair, another him, watching a video player: this video, at this second. It turns
round, slowly, and looks into the lens. <YOU> You stayed.
"""
import numpy as np

import anim as A
import common as C
import entities as EN
import film as FM
import screen as SCR
import ui

W_ = 'finale'
HOTBAR = dict(items=['written_book', 'torch', 'bread', 'clock', 'filled_map', 'torn_page', 'written_book'],
              selected=6, counts={1: 16, 2: 5})


def shots(ctx):
    S = FM.Shot
    out = []
    w = ctx.worlds[W_]
    P = w.points
    door = P['house_door']                          # outside the house's door
    mon = SCR.Monitor((P['room_monitor'][0], P['room_monitor'][1] + 0.38, P['room_monitor'][2]), facing=0)
    chair = P['room_chair']

    def world_scene(cam, env, actors=(), door_deg=0.0, lights=(), prep=None, **kw):
        props = C.world_props(ctx, W_, door=door_deg) + mon.props(glow=0.55)
        sc = dict(world=W_, env=C.env(env, **kw) if isinstance(env, str) else env, cam=cam, actors=list(actors),
                  props=props, lights=list(lights) + mon.light(1.2))
        if prep:
            sc['prep'] = prep
        return sc

    def screen_prep(T):
        # the frame before this one, finished (HUD, subtitles, grain): the video you are watching, as it plays
        img = ctx.last_composed if ctx.last_composed is not None else ctx.last3d

        def prep(r):
            r.update_kind_layer('screen', 0, SCR.Monitor.content(img, T))
        return prep

    # --- x1: black. That's not me. ----------------------------------------------------------------------------------------------
    out.append(S('x1_black', 4.0, 'black', subs=[(1.8, 3.9, "(whispering) That's not me.")],
                 cues=[(0.0, 'silence_all', {'dur': 4.0})]))

    # --- x2: it walks up to the house by itself; the camera follows by itself; S and ESC do nothing ----------------------------------
    path2 = A.Path([(2.6, -54.0, 0.0), (1.9, -47.0, 0.0), (0.9, -41.0, 0.0), (0.5, -36.4, 0.0)])
    walker2 = A.Walker(path2, [(0, 0.0), (9.0, path2.length)])

    def him(t, walker, arm=None):
        a = EN.Actor('player', 'you')
        walker.pose(a, t)
        a.head_yaw = 0.0                        # it doesn't look around
        a.arm_r = arm
        return a

    def follow_cam(walker, t, back=3.4, up=2.3, lag=0.5, sway=0.25, fov=60):
        p = walker.pos(max(t - lag, 0.0))
        d = walker.path.tangent(float(walker.s(max(t - lag, 0.0))))
        side = np.array([d[1], -d[0], 0.0])
        eye = p - d * back + np.array([0, 0, up]) + side * sway * np.sin(t * 0.37)
        tgt = walker.pos(t) + np.array([0, 0, 1.3]) + d * 1.5
        return dict(eye=eye, target=tgt, fov=fov)

    esc2 = [(2.8, 3.0), (3.9, 4.05), (4.7, 4.85), (5.4, 5.5), (6.0, 6.1), (6.6, 6.7), (7.3, 7.45), (8.0, 8.12)]

    def keys2(t):
        k = set()
        if t >= 1.6:
            k.add('S')
        if any(a <= t < b for (a, b) in esc2):
            k.add('ESC')
        return k

    def x2(t, T):
        return world_scene(follow_cam(walker2, t), 'night', actors=[him(t, walker2)])

    def x2_keys(img, t, T, c, f):
        ui.keystrokes(img, keys2(t), alpha=float(np.clip(t / 0.4, 0, 1)))

    out.append(S('x2_walk', 9.0, '3d', scene=x2, late=x2_keys, hud=HOTBAR,
                 fx=lambda t: {'fade': float(np.clip(t / 0.25, 0, 1))},
                 subs=[(2.2, 4.4, "I'm not doing this."), (5.0, 8.2, "I'm pressing back. It won't stop.")],
                 cues=[(0.0, 'amb', {'kind': 'night'})] +
                 [(0.2 + 0.52 * k, 'step', {'surface': 'path', 'gain': 0.7, 'even': 1}) for k in range(17)] +
                 [(a, 'key_click') for (a, b) in esc2] + [(1.6, 'key_click')]))

    # --- x3: it opens the door and goes in; the camera goes after it -----------------------------------------------------------------
    inside = P['house_in']
    path3 = A.Path([(0.5, -36.4, 0.0), (0.5, -35.3, 0.0), (0.5, -34.2, 0.0), (0.5, -33.0, 0.0), (0.5, -31.6, 0.0)])
    walker3 = A.Walker(path3, [(0, 0.0), (0.8, 0.9), (2.4, 0.9), (7.0, path3.length), (8.0, path3.length)])
    t_door = 1.4
    door3 = A.Keys([(0, 0.0), (t_door, 0.0), (t_door + 0.3, 90.0, 'out')])

    def x3(t, T):
        arm = (-1.35, 0.0) if 0.9 <= t < t_door + 0.35 else None
        cam = follow_cam(walker3, t, back=3.0, up=1.9, lag=0.35, sway=0.1, fov=62)
        return world_scene(cam, 'interior_night' if t > 3.2 else 'night', actors=[him(t, walker3, arm)],
                           door_deg=float(door3(t)), prep=screen_prep(T))

    out.append(S('x3_door', 8.0, '3d', scene=x3, late=lambda img, t, T, c, f: ui.keystrokes(img, keys2(t + 9.0)),
                 hud=HOTBAR,
                 cues=[(t_door, 'door_open'), (2.6, 'amb_fade', {'to': 'room', 'dur': 2.0})] +
                 [(2.6 + 0.6 * k, 'step', {'surface': 'wood', 'gain': 0.7}) for k in range(7)]))

    # --- x4: the room: his real room. At the desk, in his chair, another him, watching -------------------------------------------------
    sitter_pos = np.array([chair[0], chair[1] + 0.12, chair[2] + 0.46])

    def sitter(yaw_deg=0.0, head=0.0, skin='you_wrong'):
        a = EN.Actor('player', skin, sitter_pos, yaw=np.radians(yaw_deg))
        a.sit = True
        a.idle_t = 0.0
        a.head_yaw = head
        return a

    stood = np.array([0.15, -31.2, 0.0])                        # it walked in and stopped, just inside
    eye4 = A.Keys([(0, (-1.55, -31.85, 2.4)), (9.0, (-1.3, -31.35, 2.25))])     # in the room's corner, behind it
    tgt4 = A.Keys([(0, (0.8, -25.6, 1.1)), (9.0, (0.65, -25.6, 1.25))])

    def x4(t, T):
        body = EN.Actor('player', 'you', stood, yaw=0.0)
        body.idle_t = 0.0
        return world_scene(dict(eye=eye4(t), target=tgt4(t), fov=58), 'interior_night', actors=[body, sitter()],
                           door_deg=90.0, prep=screen_prep(T), exposure=1.1)

    out.append(S('x4_room', 9.0, '3d', scene=x4, hud=HOTBAR, feedback=True,
                 cues=[(0.0, 'silence_all', {'dur': 1.5}), (1.5, 'pc_hum_start'), (4.0, 'drone', {'level': 0.3,
                                                                                                'dur': 5.0})]))

    # --- x5: over its shoulder to the screen: the video player, this video, this second --------------------------------------------------
    eye5 = A.Keys([(0, (1.75, -29.4, 2.0)), (8.0, (1.45, -27.3, 1.85))], 'smooth')      # past its right shoulder
    tgt5 = A.Keys([(0, mon.centre + (0, 0, -0.05)), (8.0, mon.centre)])

    def x5(t, T):
        return world_scene(dict(eye=eye5(t), target=tgt5(t), fov=54), 'interior_night', actors=[sitter()],
                           door_deg=90.0, prep=screen_prep(T), exposure=1.1)

    out.append(S('x5_screen', 8.0, '3d', scene=x5, hud=HOTBAR, feedback=True,
                 cues=[(0.0, 'drone', {'level': 0.35, 'dur': 8.0})]))

    # --- x6: it turns round in the chair and looks into the lens. You stayed. ------------------------------------------------------------
    t_turn0, t_turn1, t_say = 1.5, 6.0, 7.2
    eye6 = A.Keys([(0, (0.5, -28.9, 1.75)), (11.0, (0.5, -27.9, 1.62))])

    def x6(t, T):
        u = float(A.smoother((t - t_turn0) / (t_turn1 - t_turn0)))
        body_yaw = 180.0 * u                                     # from facing the desk to facing us
        head = np.radians(-35.0) * np.sin(np.pi * u) + np.radians(0.0)
        s6 = sitter(body_yaw, head)
        s6.head_pitch = -0.05 * u
        face = sitter_pos + np.array([0, 0, 1.02])
        fov = 50.0 - 14.0 * float(A.smooth((t - t_turn1) / 5.0))
        glow = [[*(eye6(t) + np.array([0, -0.3, 0])), 5.0, 0.30 * u, 0.36 * u, 0.48 * u]]   # your screen, on its face
        return world_scene(dict(eye=eye6(t), target=face, fov=fov), 'interior_night', actors=[s6], door_deg=90.0,
                           prep=screen_prep(T), exposure=1.1, lights=glow)

    out.append(S('x6_stayed', 11.0, '3d', scene=x6, hud=HOTBAR, chat=True, feedback=True,
                 cues=[(t_turn0, 'chair_creak'), (t_turn1, 'silence_all', {'dur': 1.2}),
                       (t_say, 'chat', {'text': '<YOU> You stayed.', 'color': (255, 255, 255)}),
                       (t_say, 'chat_msg'), (t_say + 0.4, 'silence_all', {'dur': 3.4})]))
    return out
