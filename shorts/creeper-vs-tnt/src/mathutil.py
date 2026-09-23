"""Quaternion helpers (x, y, z, w convention, matching the shaders)."""
import numpy as np


def mat_to_quat(R):
    """Rotation matrix (..., 3, 3) -> quaternion (..., 4) [x, y, z, w]. Vectorised."""
    R = np.asarray(R, np.float64)
    shp = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    c0 = tr > 0
    c1 = ~c0 & (m00 > m11) & (m00 > m22)
    c2 = ~c0 & ~c1 & (m11 > m22)
    c3 = ~(c0 | c1 | c2)
    q = np.zeros((len(R), 4))
    with np.errstate(invalid='ignore', divide='ignore'):
        s = np.sqrt(np.maximum(tr + 1.0, 1e-12)) * 2
        f = np.stack([(R[:, 2, 1] - R[:, 1, 2]) / s, (R[:, 0, 2] - R[:, 2, 0]) / s,
                      (R[:, 1, 0] - R[:, 0, 1]) / s, 0.25 * s], -1)
        q[c0] = f[c0]
        s = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 1e-12)) * 2
        f = np.stack([0.25 * s, (R[:, 0, 1] + R[:, 1, 0]) / s, (R[:, 0, 2] + R[:, 2, 0]) / s,
                      (R[:, 2, 1] - R[:, 1, 2]) / s], -1)
        q[c1] = f[c1]
        s = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 1e-12)) * 2
        f = np.stack([(R[:, 0, 1] + R[:, 1, 0]) / s, 0.25 * s, (R[:, 1, 2] + R[:, 2, 1]) / s,
                      (R[:, 0, 2] - R[:, 2, 0]) / s], -1)
        q[c2] = f[c2]
        s = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 1e-12)) * 2
        f = np.stack([(R[:, 0, 2] + R[:, 2, 0]) / s, (R[:, 1, 2] + R[:, 2, 1]) / s, 0.25 * s,
                      (R[:, 1, 0] - R[:, 0, 1]) / s], -1)
        q[c3] = f[c3]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(shp + (4,))


def quat_mul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], -1)


def quat_rotate(q, v):
    qv = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def axis_angle_quat(axis, angle):
    axis = np.asarray(axis, np.float64)
    n = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = axis / np.maximum(n, 1e-12)
    h = np.asarray(angle)[..., None] * 0.5
    return np.concatenate([axis * np.sin(h), np.cos(h)], -1)


def integrate_quat(q, omega, dt):
    """Advance orientation q by world-space angular velocity omega over dt."""
    ang = np.linalg.norm(omega, axis=-1)
    dq = axis_angle_quat(np.where(ang[..., None] > 1e-9, omega, np.array([1.0, 0, 0])), ang * dt)
    out = quat_mul(dq, q)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def quat_from_basis(x_axis, z_axis):
    """Orientation whose local X maps to x_axis and local Z to (orthogonalised) z_axis. Vectorised."""
    x = np.asarray(x_axis, np.float64)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    z = np.asarray(z_axis, np.float64) * np.ones_like(x)
    z = z - x * np.sum(z * x, -1, keepdims=True)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    R = np.stack([x, y, z], -1)
    return mat_to_quat(R)


def slerp(a, b, t):
    d = np.sum(a * b, -1, keepdims=True)
    b = np.where(d < 0, -b, b)
    d = np.abs(d)
    t = np.asarray(t)[..., None] if np.ndim(t) else t
    th = np.arccos(np.clip(d, -1, 1))
    s = np.sin(th)
    small = s < 1e-5
    wa = np.where(small, 1 - t, np.sin((1 - t) * th) / np.where(small, 1, s))
    wb = np.where(small, t, np.sin(t * th) / np.where(small, 1, s))
    out = wa * a + wb * b
    return out / np.linalg.norm(out, axis=-1, keepdims=True)
