"""The player's arm in first person: Steve's right arm from the earlier videos (own pixel art: a cyan sleeve over
skin), 4 x 4 x 12 voxels, held in the bottom right of the view like in the game, with the game's swing: a quick arc
forwards and down (0.3 s). The fist lands halfway through.
"""
import numpy as np

from creeper import VOXEL_DTYPE, _qmul
from mathutil import quat_rotate

PX = 1.3 / 16.0                   # the arm is drawn a little larger than life, like the game does
SLEEVE = [(22, 178, 182), (16, 168, 173), (30, 188, 190), (12, 158, 165)]
SLEEVE_DK = [(10, 140, 148), (6, 132, 140)]
SKIN = [(200, 146, 104), (194, 140, 99), (206, 152, 110), (189, 136, 96)]
ROWS = ["TTTT", "TTTT", "TTTT", "tTTt"] + ["SSSS"] * 8          # from the shoulder (row 0) to the hand
PAL = {'T': SLEEVE, 't': SLEEVE_DK, 'S': SKIN}
SWING = 0.3


def _quat(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.array([*(axis * np.sin(ang / 2)), np.cos(ang / 2)])


class Arm:
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        ii, jj, kk = np.meshgrid(np.arange(4), np.arange(4), np.arange(12), indexing='ij')
        surf = (ii == 0) | (ii == 3) | (jj == 0) | (jj == 3) | (kk == 0) | (kk == 11)
        self.ii, self.jj, self.kk = ii[surf], jj[surf], kk[surf]
        n = len(self.ii)
        col = np.zeros((n, 3))
        for m in range(n):
            ch = ROWS[11 - self.kk[m]][self.ii[m]]          # kk = 11 is the shoulder end
            opts = PAL[ch]
            col[m] = opts[rng.integers(len(opts))]
        col *= 1.0 + (rng.random((n, 1)) - 0.5) * 0.08
        self.col = np.clip(col, 0, 255)
        vis = np.zeros(n, np.uint8)
        for bit, (arr, v) in enumerate(((self.ii, 3), (self.ii, 0), (self.jj, 3), (self.jj, 0), (self.kk, 11),
                                        (self.kk, 0))):
            vis[arr == v] |= 1 << bit
        self.vis = vis
        # local frame: the arm's length along +z from the fist (z = 0) back to the shoulder (z = 12 px)
        self.local = np.stack([self.ii - 1.5, self.jj - 1.5, self.kk + 0.5], -1) * PX

    def instances(self, cam_eye, cam_target, t_swing, t, bob=0.0):
        """Voxels of the arm for a camera, swinging if t is within SWING of t_swing (the start of the swing)."""
        f = np.asarray(cam_target, float) - np.asarray(cam_eye, float)
        f /= np.linalg.norm(f)
        r = np.cross(f, [0.0, 0.0, 1.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        # rest pose in camera space (right, up, forwards): the shoulder off the bottom-right corner, the fist a
        # little right of and below the middle of the view; the swing drives the fist forwards and down
        u_s = np.clip((t - t_swing) / SWING, 0.0, 1.0) if t_swing is not None else 0.0
        s = np.sin(np.pi * u_s)
        shoulder = np.array([0.52, -0.98 + 0.015 * np.sin(bob), 0.62]) + np.array([-0.06, 0.05, 0.22]) * s
        fist = np.array([0.20, -0.36 + 0.015 * np.sin(bob), 1.22]) + np.array([-0.16, -0.30, 0.34]) * s
        q_cam = _aim(fist - shoulder, roll=np.radians(-25.0))
        del shoulder
        n = len(self.local)
        P = quat_rotate(np.tile(q_cam, (n, 1)), self.local)
        # camera space (x right, y forwards, z up) -> world
        P = P + fist[[0, 2, 1]]
        R = np.stack([r, f, u], 1)                # columns: right, forwards, up
        W = P @ R.T + np.asarray(cam_eye, float)
        # voxel orientation: camera basis quaternion times the arm's own
        q_basis = _mat_to_quat(R)
        q = _qmul(q_basis, q_cam)
        inst = np.zeros(n, VOXEL_DTYPE)
        inst['pos'] = W
        inst['quat'] = q
        inst['scale'] = PX
        c8 = self.col.astype(np.uint8)
        for key in ('cx', 'cy', 'cz', 'inner'):
            inst[key][:, :3] = c8
        inst['cx'][:, 3] = 0b111111
        inst['cy'][:, 3] = self.vis
        inst['cz'][:, 3] = 0
        return inst


def _aim(d_cam, roll=0.0):
    """Rotation (camera frame x right, y forwards, z up) that turns the arm's +z (fist -> shoulder) to point
    from the fist back along -d_cam (d_cam given as right, up, forwards)."""
    back = -np.array([d_cam[0], d_cam[2], d_cam[1]], float)
    back /= np.linalg.norm(back)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, back)
    ang = np.arccos(np.clip(np.dot(z, back), -1, 1))
    q = _quat(axis, ang) if np.linalg.norm(axis) > 1e-9 else np.array([0.0, 0.0, 0.0, 1.0])
    return _qmul(q, _quat([0, 0, 1], roll))


def _mat_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s])
