"""ACT 3: the footsteps (3:26 - 4:59).

Night. He stays inside (rule 2 in his head). Steps outside, one... pause... another. They circle the house. The
windows show only the moonlit path (and, far off, a torch moving by itself). The steps stop. Silence. Then steps
inside, on the floor; he looks down: the moonlight on the boards, his shadow... and a second one beside it. The steps
go up, above him: there is no upstairs. The ceiling block cracks and breaks. Nothing. A sign falls through:
GOOD. YOU DIDN'T TURN AROUND.
"""
import numpy as np

import anim as A
import common as C
import entities as EN
import film as FM
import gfx
import looks
import ui

HOTBAR = dict(items=['written_book', 'torch', 'bread', 'clock', 'filled_map'], selected=5, counts={1: 16, 2: 5})
MOON = gfx.sun_dir(looks.MOON_AZ, looks.MOON_EL)


def self_shadow(eye_feet, yaw_deg, skin='you'):
    """His own body, seen only as a shadow (first person hides it, the shaders still cast it)."""
    a = EN.Actor('player', skin, eye_feet, yaw=np.radians(yaw_deg))
    a.mat = gfx.MAT_SHADOW
    return a


def interior_env(**kw):
    return looks.get('interior_night', light_col=(0.42, 0.52, 0.78), **kw)


def room_fill(ceil, k=1.0):
    """The torches' and the furnace's light spilling across the room, as a soft light over his head, and a little
    more on the ceiling right above him: without them that ceiling (and the sign at his feet) would be black."""
    return [[ceil[0] + 0.5, ceil[1] - 0.2, 2.0, 6.0, 0.55 * k, 0.47 * k, 0.4 * k],
            [ceil[0] + 0.5, ceil[1] + 0.1, 2.55, 2.6, 0.9 * k, 0.78 * k, 0.66 * k]]


def room_env(k=1.0):
    """The night interior with its darkest corners lifted a little (neutral, so they don't turn red)."""
    return dict(light_col=(0.42, 0.52, 0.78), min_amb=(0.018 * k, 0.019 * k, 0.024 * k))


def shots(ctx):
    S = FM.Shot
    out = []
    ceil = ctx.worlds['village'].dyn['house_ceiling']['pos']

    # --- f1: the village at night -----------------------------------------------------------------------------------
    eye1 = A.Keys([(0, (-16.0, -50.0, 6.5)), (7, (-10.5, -45.0, 5.0))])
    tgt1 = A.Keys([(0, (0.0, -29.0, 1.5)), (7, (0.0, -29.0, 2.0))])

    def f1(t, T):
        return dict(world='village', env=looks.get('night'), cam=dict(eye=eye1(t), target=tgt1(t), fov=50),
                    props=C.world_props(ctx, 'village'), particles={'soft': __import__('seq_perfect').smoke(T, night=True)})

    out.append(S('f1_night', 7.0, '3d', scene=f1, cues=[(0.0, 'amb', {'kind': 'night'})],
                 fx=lambda t: {'fade': float(np.clip(t / 1.5, 0, 1))}))

    # --- f2: inside by the bed; rule 2 in his head --------------------------------------------------------------------
    feet2 = (1.6, -27.4, 0.0)
    pov2 = A.POV(pos_keys=[(0, feet2)], yaw_keys=[(0, 172), (4.0, 186), (9.0, 178)], pitch_keys=[(0, -8), (9, -4)],
                 seed=31, jitter=0.12)

    def f2(t, T):
        return C.pov_scene(pov2, t, 'interior_night', props=C.world_props(ctx, 'village'),
                           actors=[self_shadow(feet2, 178)], envkw=dict(light_col=(0.42, 0.52, 0.78)))

    def f2_over(img, t, T, ctx_, film):
        a = float(np.clip((t - 5.0) / 0.6, 0, 1) * np.clip((8.2 - t) / 0.8, 0, 1)) * 0.55
        if a > 0:
            ui.draw_mc_text(img, "2. If you hear footsteps, don't turn around.", ui.W // 2, ui.H // 2 - 180,
                            (215, 205, 190), align='center', alpha=a)

    out.append(S('f2_inside', 9.0, '3d', scene=f2, overlay=f2_over, hud=HOTBAR,
                 subs=[(1.0, 4.0, "Okay. It's night. I'm staying inside.")],
                 cues=[(0.0, 'amb', {'kind': 'night_in'}), (0.0, 'fire_off')]))

    # --- f3: steps outside, circling ------------------------------------------------------------------------------------
    pov3 = A.POV(pos_keys=[(0, feet2)], yaw_keys=[(0, 178), (1.3, 196), (3.8, 150), (6.3, 128), (9.6, 222),
                                                  (11.2, 236), (14.0, 200)],
                 pitch_keys=[(0, -4), (6.0, -2), (14.0, -5)], seed=32, jitter=0.14)
    steps_out = [(1.0, -0.8), (3.5, -0.6), (6.0, -0.3), (9.5, 0.6), (11.0, 0.8), (12.5, 0.2), (14.0, -0.5)]

    def f3(t, T):
        return C.pov_scene(pov3, t, 'interior_night', props=C.world_props(ctx, 'village'),
                           actors=[self_shadow(feet2, 178)], envkw=dict(light_col=(0.42, 0.52, 0.78)))

    out.append(S('f3_steps_out', 14.0, '3d', scene=f3, hud=HOTBAR,
                 subs=[(7.0, 9.0, '...Did you hear that?')],
                 cues=[(ts, 'step_out', {'pan': p}) for (ts, p) in steps_out]))

    # --- f4: the windows: nothing. Far off, a torch floats along the path by itself -------------------------------------
    win = np.array([-3.9, -30.0, 1.9])
    pov4 = A.POV(pos_keys=[(0, feet2), (3.0, (-1.9, -30.2, 0.0)), (14.0, (-2.2, -30.3, 0.0))],
                 yaw_keys=[(0, 196), (2.0, 150), (3.4, A.look_angles((-1.9, -30.2, 1.62), win)[0]),
                           (8.0, 95), (11.0, 82), (14.0, 88)],
                 pitch_keys=[(0, -4), (3.4, 2), (8.0, 0), (14.0, 1)], seed=33, jitter=0.12)
    ftorch = A.Path([(-26.0, -44.0, 1.3), (-24.5, -36.0, 1.35), (-23.0, -28.0, 1.3), (-22.5, -20.0, 1.3)])

    def floating_torch(t):
        p = ftorch.at(ftorch.length * float(np.clip((t - 3.0) / 11.0, 0, 1)))
        p = p + np.array([0.0, 0.0, 0.05 * np.sin(t * 5.1)])
        flick = 0.85 + 0.15 * np.sin(t * 17.0)
        light = [*p, 9.0, 2.2 * flick, 1.4 * flick, 0.7 * flick]
        q = EN.qmul(EN.qz(0.4), EN.qx(0.2))
        return C.row('item_torch', p, q, 0.5, C.ITEM_NAMES.index('torch'), emit=0.0), light

    def f4(t, T):
        tr, light = floating_torch(t)
        feet = pov4.feet(t)
        return C.pov_scene(pov4, t, 'interior_night', props=C.world_props(ctx, 'village') + [tr], lights=[light],
                           actors=[self_shadow(feet, pov4.angles(t)[0])], envkw=dict(light_col=(0.42, 0.52, 0.78)))

    out.append(S('f4_windows', 14.0, '3d', scene=f4, hud=HOTBAR,
                 cues=[(0.6, 'step', {'surface': 'wood'}), (1.4, 'step', {'surface': 'wood'}),
                       (2.2, 'step', {'surface': 'wood'}), (4.0, 'step_out', {'pan': 0.7}),
                       (6.5, 'step_out', {'pan': 0.9}), (8.0, 'silence_all', {'dur': 6.0})]))

    # --- f5: steps inside. The moon throws the window onto the far wall: his silhouette... and one beside it -------
    feet5 = np.array([1.8, -31.3, 0.0])
    # he walks to the lit patch on the east wall; at 5 s he sways (his shadow moves... the other one doesn't)
    pov5 = A.POV(pos_keys=[(0, (-2.2, -30.3, 0.0)), (3.0, tuple(feet5)), (5.0, tuple(feet5)),
                           (5.8, tuple(feet5 + (0.0, -0.18, 0.0))), (6.8, tuple(feet5 + (0.0, 0.05, 0.0))),
                           (7.6, tuple(feet5)), (12.0, tuple(feet5))],
                 yaw_keys=[(0, 88), (1.3, 20), (3.0, -86), (12.0, -90)],
                 pitch_keys=[(0, 1), (3.0, -9), (7.0, -12), (12.0, -13)], seed=34, jitter=0.06)
    steps_in = [(0.8, 0.5, 'near'), (2.6, 0.2, 'near'), (4.4, -0.2, 'near'), (6.2, -0.4, 'near'), (8.0, 0.0, 'near')]

    def f5(t, T):
        k = float(A.smooth((t - 8.0) / 3.0))
        other = self_shadow((1.6, -29.9 - 0.3 * k, 0.0), -90.0, skin='you')   # beside him, unseen: it edges closer
        other.idle_t = t
        other.head_yaw = 0.45 * k                                               # ...and its head turns towards him
        # the clouds part as he turns from the window: the moonlight swells (from f4's level)
        m = float(A.smooth((t - 0.5) / 3.0))
        lc = tuple(np.array([0.42, 0.52, 0.78]) * (1 - m) + np.array([1.1, 1.5, 3.2]) * m)
        return C.pov_scene(pov5, t, 'interior_night', props=C.world_props(ctx, 'village'),
                           actors=[self_shadow(pov5.feet(t), pov5.angles(t)[0]), other],
                           envkw=dict(light_col=lc, fog_sun=(0.16, 0.16, 0.22), shadow_pen=0.25))

    out.append(S('f5_shadows', 12.0, '3d', scene=f5, hud=HOTBAR,
                 subs=[(5.0, 7.5, '(whispering) No no no no...')],
                 cues=[(ts, 'step_in', {'pan': p}) for (ts, p, _) in steps_in] +
                 [(5.0, 'drone', {'level': 0.35, 'dur': 7.0})]))

    # --- f6: the steps go up. There is no upstairs --------------------------------------------------------------------------
    feet6 = np.array([0.5, -29.6, 0.0])
    pov6 = A.POV(pos_keys=[(0, tuple(feet5)), (2.0, tuple(feet6)), (10.0, tuple(feet6))],
                 yaw_keys=[(0, -90), (2.0, -30), (4.0, 2), (10.0, 0)],
                 pitch_keys=[(0, -10), (2.0, 5), (4.0, 35), (8.0, 62), (10.0, 66)], seed=35, jitter=0.1)
    steps_up = [(0.6, 0.0), (2.2, 0.3), (3.8, -0.2), (5.4, 0.1), (7.0, 0.0), (8.0, 0.0)]

    def f6(t, T):
        k = float(np.clip((t - 2.0) / 4.0, 0, 1))            # fades in as he looks up (f5 has none)
        return C.pov_scene(pov6, t, 'interior_night', props=C.world_props(ctx, 'village'),
                           actors=[self_shadow(pov6.feet(t), pov6.angles(t)[0])], lights=room_fill(ceil, k * k),
                           envkw=room_env(k * k))

    out.append(S('f6_above', 10.0, '3d', scene=f6, hud=HOTBAR,
                 subs=[(2.0, 5.5, "There's no upstairs. There's no upstairs in this house.")],
                 cues=[(ts, 'step_above', {'pan': p}) for (ts, p) in steps_up]))

    # --- f7: the block above him cracks and breaks. Nothing there ----------------------------------------------------------
    pov7 = A.POV(pos_keys=[(0, tuple(feet6))], yaw_keys=[(0, 0), (10.0, 1)], pitch_keys=[(0, 66), (10.0, 64)],
                 seed=36, jitter=0.08)
    t_break = 6.0

    def f7(t, T):
        if t < 0.5:
            state = None
        elif t < t_break:
            state = (t - 0.5) / (t_break - 0.5) * 10.0
        else:
            state = 'gone'
        props = C.world_props(ctx, 'village', ceiling=state)
        props += C.breaking_particles(ceil, t - t_break, 'spruce_planks')
        return C.pov_scene(pov7, t, 'interior_night', props=props, actors=[self_shadow(feet6, 0.0)],
                           lights=room_fill(ceil), envkw=room_env())

    out.append(S('f7_crack', 10.0, '3d', scene=f7, hud=HOTBAR,
                 cues=[(0.5 + k * 0.55, 'dig', {'k': k}) for k in range(10)] + [(t_break, 'block_break'),
                                                                                (t_break, 'silence_all', {'dur': 4.0})]))

    # --- f8: a sign falls through the hole: GOOD. YOU DIDN'T TURN AROUND. -------------------------------------------------
    fall_t0, fall_t1 = 0.5, 0.95
    sign_pos = (ceil[0], ceil[1], 0)
    # he looks down at it, then crouches to read it
    pov8 = A.POV(pos_keys=[(0, tuple(feet6)), (3.0, (0.5, -29.9, 0.0)), (6.0, (0.5, -29.9, 0.0)),
                           (9.0, (0.5, -29.62, 0.0))],
                 yaw_keys=[(0, 0), (14.0, 1)],
                 pitch_keys=[(0, 60), (0.9, 30), (2.2, -45), (4.0, -46), (6.0, -33), (14.0, -31)],
                 seed=37, jitter=0.07)
    pov8.eye_keys = A.Keys([(0, 1.62), (4.0, 1.62), (5.6, 1.27)])

    def f8(t, T):
        props = C.world_props(ctx, 'village', ceiling='gone')
        u = float(np.clip((t - fall_t0) / (fall_t1 - fall_t0), 0, 1))
        z = 3.2 * (1 - u * u)
        if t >= fall_t0 - 0.3:
            props += C.sign_prop(ctx, 'village', 'sign_good', (sign_pos[0], sign_pos[1], z), 0)
        return C.pov_scene(pov8, t, 'interior_night', props=props, actors=[self_shadow(pov8.feet(t), 0.0)],
                           lights=room_fill(ceil), envkw=room_env())

    out.append(S('f8_sign', 14.0, '3d', scene=f8, hud=HOTBAR,
                 subs=[(10.0, 13.0, "...Who's there?")],
                 cues=[(fall_t1, 'sign_land'), (2.0, 'drone', {'level': 0.5, 'dur': 12.0})]))
    out.append(S('f9_black', 3.0, 'black', cues=[(0.0, 'silence_all', {'dur': 3.0})]))
    return out
