"""The rider's point of view: where his eyes are in the cart, where he looks, how the view shakes and widens with
speed.

The eye sits in the minecart, above the rails along the track's (banked) up vector. The head turns into what is
coming: the view direction blends the track's tangent with the direction to a point further along the track (a
look-ahead that grows with speed), and the head keeps part of its own sense of up against the banking. The field of
view widens with speed, like the game's sprint FOV, and the view shakes a little, more at speed and at the bottoms
of drops where the g-force peaks.
"""
import numpy as np

import track as TK

EYE_UP = 1.34            # eye height above the rails (seated in the cart)
EYE_BACK = -0.08         # (negative: a little ahead of the cart's centre, leaning forward)


def _hash_noise(t, seed):
    """Smooth 1-D value noise in -1..1."""
    i = np.floor(t)
    f = t - i
    def h(n):
        return np.sin(n * 127.1 + seed * 311.7) * 43758.5453 % 1.0 * 2 - 1
    u = f * f * (3 - 2 * f)
    return h(i) * (1 - u) + h(i + 1) * u


def smoothstep(a, b, x):
    u = np.clip((x - a) / (b - a), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def view(tr, s, t, look=1.0, shake=1.0, fov_base=88.0, fov_fast=18.0, extra_pitch=0.0, look_dist=None,
         attend=None, jolt=0.0):
    """Camera for the cart at arc length s on track tr (t: ride time, for the shake). attend: (point, weight[,
    narrowing of the view in degrees]) the rider glances at; jolt: a hard landing (the head dips, then bounces).
    Returns (cam dict for the renderer, eye, (F, R, U) camera axes, (T, Rt, Ut) track frame, speed, g)."""
    s = float(np.clip(s, 0.0, tr.length - 0.02))
    P = tr.pos(s)
    T, Rt, Ut = tr.frame(s)
    v = tr.speed(s)
    eye = P + Ut * EYE_UP - T * EYE_BACK
    # look-ahead: a point further along the track, a little above the rails
    la = float(np.clip(4.0 + v * 0.26, 4.0, 14.0)) if look_dist is None else float(look_dist)
    sa = min(s + la, tr.length - 0.02)
    Pa = tr.pos(sa)
    Ta, Ra, Ua = tr.frame(sa)
    to = Pa + Ua * 0.9 - eye
    to /= np.linalg.norm(to)
    F = T * (1 - 0.55 * look) + to * 0.55 * look
    F /= np.linalg.norm(F)
    if extra_pitch:
        F = F * np.cos(np.radians(extra_pitch)) + Ut * np.sin(np.radians(extra_pitch))
        F /= np.linalg.norm(F)
    if attend is not None and attend[1] > 0:
        # the head turns towards something worth looking at (the dragon going over, the ghast firing)
        d = np.asarray(attend[0], float) - eye
        d /= np.linalg.norm(d)
        F = F * (1 - attend[1]) + d * attend[1]
        F /= np.linalg.norm(F)
    # the head keeps some of its own up against the banking
    Z = np.array([0.0, 0.0, 1.0])
    up = Ut * 0.8 + Z * 0.2
    up = up - np.dot(up, F) * F
    if np.linalg.norm(up) < 1e-4:
        up = Ut - np.dot(Ut, F) * F
    up /= np.linalg.norm(up)
    R = np.cross(F, up)
    # g-force: the curvature along up (the bottoms of drops) pushes the rider into the seat
    sb = max(s - 0.5, 0.0)
    sf = min(s + 0.5, tr.length - 0.02)
    Tb, _, _ = tr.frame(sb)
    Tf, _, _ = tr.frame(sf)
    kv = np.dot((Tf - Tb) / max(sf - sb, 1e-3), Ut)                # vertical curvature (+ = valley)
    g = 1.0 + v * v * kv / TK.G
    # shake: mostly vertical, faster and stronger at speed and under g
    amp = shake * (0.004 + 0.018 * smoothstep(10, 45, v) + 0.012 * np.clip(g - 1.5, 0, 2))
    fq = 5.0 + v * 0.35
    jx = _hash_noise(t * fq, 1.0) * amp
    jy = _hash_noise(t * fq * 1.13, 2.0) * amp
    eye_s = eye + R * jx * 0.6 + up * jy - Ut * 0.03 * np.clip(g - 1.0, -1, 2) - up * 0.16 * jolt
    F2 = F + R * jx * 0.25 + up * jy * 0.35 - up * 0.05 * jolt
    F2 /= np.linalg.norm(F2)
    fov = fov_base + fov_fast * smoothstep(12.0, 46.0, v)
    if attend is not None and len(attend) > 2:
        fov -= attend[2]
    cam = dict(eye=eye_s, target=eye_s + F2, fov=float(fov), up=up)
    return cam, eye_s, (F2, np.cross(F2, up), up), (T, Rt, Ut), v, g
