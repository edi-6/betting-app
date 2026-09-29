"""His computer, in the game: a monitor on the desk whose screen is a glowing quad in the 3D scene (so whoever sits in
front of it hides it properly), textured every frame with a video player showing a picture: a second camera's view
(act 7: him, filmed from behind) or the film's own previous frame (the finale: the video you are watching)."""
import numpy as np

import common as C
import decals as DC
import entities as EN
import ui

SW, SH = 512, 288                     # the screen texture
TOTAL = [None]                        # the film's length, set by story.film() once every shot exists


def register(r):
    if 'screen' not in r.kinds:
        r.add_kind('screen', DC.quad_mesh(), [np.zeros((SH, SW, 4), np.uint8)], filt=__import__('moderngl').LINEAR)


class Monitor:
    """A monitor on a desk: `pos` is the middle of the desk's back edge at desk height, `facing` the way the screen
    looks (0: -y, 1: +x, 2: +y, 3: -x)."""

    def __init__(self, pos, facing=0, width=0.96):
        self.pos = np.asarray(pos, float)
        self.f = facing
        self.w = width
        self.h = width * 9 / 16
        a = np.radians(90.0 * facing)
        self.fwd = np.array([np.sin(a), -np.cos(a), 0.0])            # facing 0 looks towards -y
        self.right = np.array([np.cos(a), np.sin(a), 0.0])
        self.q = EN.qz(a)
        self.z0 = self.pos[2] + 0.26                                    # bottom of the screen
        self.centre = self.pos + np.array([0, 0, self.z0 - self.pos[2] + self.h / 2]) + self.fwd * 0.12

    def props(self, glow=0.28):
        at = C.atlas()
        k = at['black_concrete']
        out = []
        body = self.centre - self.fwd * 0.03
        out.append(C.row('prop_cube', body, self.q, (self.w + 0.06, 0.05, self.h + 0.06), k))
        out.append(C.row('prop_cube', self.pos + self.fwd * 0.08 + np.array([0, 0, 0.14]), self.q, (0.08, 0.05, 0.26),
                         k))
        out.append(C.row('prop_cube', self.pos + self.fwd * 0.10 + np.array([0, 0, 0.012]), self.q, (0.42, 0.24, 0.025),
                         k))
        scr = self.centre + self.fwd * 0.0
        out.append(EN.inst('screen', scr, self.q, (self.w, 1.0, self.h), 0, (1, 1, 1), glow))
        return out

    def light(self, strength=1.0):
        p = self.centre + self.fwd * 0.5
        return [[*p, 6.0, 0.45 * strength, 0.55 * strength, 0.75 * strength]]

    @staticmethod
    def content(frame, t_global, title='THE PLAYER WHO NEVER LOGGED OUT'):
        """The screen texture: a video player at the film's own time."""
        if frame is None:
            frame = np.zeros((360, 640, 3), np.uint8)
        total = TOTAL[0] or 816.0
        return ui.video_player(frame, t_global, total, title=title, size=(SW, SH))
