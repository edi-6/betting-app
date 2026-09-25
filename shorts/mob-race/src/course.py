"""The race course: a marble run built down the face of a cliff, in the plane of the cliff (x to the right, z up;
one unit is one Minecraft block). It is a list of pieces the simulation turns into collision shapes and the renderer
turns into blocks, plus the machinery:

  box(x0, z0, x1, z1, mat)         an axis-aligned run of blocks
  beam(x0, z0, x1, z1, mat, t)     a tilted plank / ice / stone beam from (x0, z0) to (x1, z1), t thick
  peg(x, z, r, mat)                a round post (the plinko)

From the top: eight starting stalls -> round 1 PLINKO -> round 2 SLIME & ICE -> round 3 PISTONS & TNT -> THE FINAL.
Every round ends on a slope with a trapdoor in it over a lava basin, then a holding pen: the leaders roll over the
trapdoor into the pen, and once all but the last two are across (the last one in the final) the trapdoor swings
open and drops the rest into the lava. Then the pen's floor opens and the next round starts, all together.
"""
import numpy as np

R = 0.62                  # marble radius
HALF = 7.6                # the course is 2 * HALF wide between its side walls
N = 8


class Course:
    def __init__(self):
        self.pieces = []          # static geometry
        self.traps = []
        self.pens = []            # holding pens after the trapdoors: floors that open to start the next round
        self.pistons = []
        self.tnt = []
        self.lava = []            # (x0, z0, x1, z1): whatever falls in is out
        self.stages = []          # dict(name, z_top, z_bot)
        self.start = None
        self.finish = None
        self.z_end = 0.0
        self.lake = None

    def box(self, x0, z0, x1, z1, mat='stone_bricks'):
        self.pieces.append(dict(kind='box', mat=mat, x0=min(x0, x1), z0=min(z0, z1), x1=max(x0, x1),
                                z1=max(z0, z1)))

    def beam(self, x0, z0, x1, z1, mat='planks', t=0.5):
        self.pieces.append(dict(kind='beam', mat=mat, a=(float(x0), float(z0)), b=(float(x1), float(z1)), t=t))

    def peg(self, x, z, r=0.26, mat='fence'):
        self.pieces.append(dict(kind='peg', mat=mat, c=(float(x), float(z)), r=r))

    def walls(self, z_top, z_bot, mat='stone_bricks'):
        self.box(-HALF - 1.0, z_bot, -HALF, z_top, mat)
        self.box(HALF, z_bot, HALF + 1.0, z_top, mat)

    def round_end(self, z_hi, a, name, keep, mat='planks', slope=0.2):
        """The end of a round. The marbles come in at the wall on side a (+1: right) and roll away from it down a
        slope with a trapdoor in it (a lava basin under it) into a holding pen against the other wall. Once
        `keep` are across, the trapdoor swings open and drops whoever comes after; when the losers are in the
        lava, the pen floor opens and the survivors drop onto the next round together. Returns the height under
        the pen where the next round starts."""
        d = -a                                      # direction of travel
        x_hi = a * HALF
        x_t0 = a * (HALF - 2.6)                     # trapdoor, in travel order
        x_t1 = a * (HALF - 6.4)
        x_end = -a * HALF

        def zat(x):
            return z_hi - abs(x - x_hi) * slope

        self.beam(x_hi, zat(x_hi), x_t0, zat(x_t0), mat)
        self.traps.append(dict(name=name, hinge=(x_t0, zat(x_t0)), length=abs(x_t1 - x_t0) - 0.1, dir=d,
                               angle=float(np.arctan2(zat(x_t1) - zat(x_t0), x_t1 - x_t0)), keep=keep,
                               count_x=x_t1 + d * (R + 0.3)))
        # the pen: a fixed ledge past the hole, then a floor sloping down to the far wall that opens to start the
        # next round (well clear of the lava basin)
        z_p0 = zat(x_t1) - 0.05
        x_l = x_t1 + d * 1.7
        z_l = z_p0 - 0.14 * 1.6
        self.beam(x_t1 + d * 0.1, z_p0, x_l, z_l, mat)
        z_p1 = z_l - 0.12 * abs(x_end - x_l)
        self.pens.append(dict(name=name, a=(x_l, z_l), b=(x_end, z_p1), mat=mat, keep=keep))
        # the lava basin under the trapdoor
        lo, hi = sorted((x_t0, x_t1))
        zb = zat(x_t1) - 4.4
        self.lava.append((lo + 0.1, zb, hi - 0.1, zb + 1.8))
        self.box(lo - 0.9, zb - 0.8, hi + 0.9, zb, 'netherrack')
        self.box(lo - 0.9, zb, lo, zb + 2.6, 'netherrack')
        self.box(hi, zb, hi + 0.9, zb + 2.6, 'netherrack')
        return min(z_p1, zb - 0.8) - 3.0


def build():
    c = Course()
    # ---- the starting stalls: eight marbles side by side on a floor that opens at GO
    xs = -6.3 + 1.8 * np.arange(N)
    z0 = 0.0
    for k in range(N + 1):
        x = -7.2 + 1.8 * k
        c.box(x - 0.12, z0, x + 0.12, z0 + 2.0, 'iron_bars')
    c.box(-HALF - 1.0, z0 + 2.0, HALF + 1.0, z0 + 3.0, 'stone_bricks')
    c.start = dict(xs=xs, z=z0 + R, floor=(-7.4, z0 - 0.5, 7.4, z0))
    # ---- ROUND 1: PLINKO
    bot = -15.4
    rows = 0
    z = -3.4                                     # a drop first, so they arrive with some speed
    while z > bot:
        off = 0.5 if rows % 2 == 0 else -0.5     # never a peg right under a stall
        for x in np.arange(-6.5 + off, 7.01, 2.0):
            if abs(x) < HALF - 0.4:
                c.peg(x, z)
        rows += 1
        z -= 1.9
    a = -1                                       # the pen is on the right
    z = c.round_end(z - 0.6, a, 'TRAPDOOR 1', 6)
    c.stages.append(dict(name='PLINKO', z_top=z0 + 3.0, z_bot=z))
    a = -a                                       # they drop out of the pen at this wall
    # ---- ROUND 2: SLIME & ICE: ice legs in a zigzag; slime walls at the turns throw them onto the next leg
    zt = z
    for i in range(3):
        c.beam(a * (HALF - 0.3), z, -a * (HALF - 2.8), z - 4.2, 'ice')
        c.box(-a * (HALF - 0.9), z - 7.4, -a * HALF, z - 2.6, 'slime')
        z -= 7.2
        a = -a
    z = c.round_end(z + 0.8, a, 'TRAPDOOR 2', 4)
    a = -a
    c.stages.append(dict(name='SLIME & ICE', z_top=zt + 1.0, z_bot=z))
    # ---- ROUND 3: PISTONS & TNT: stone legs; pistons in the far wall punch out over the drop, TNT on the others
    zt = z
    for i in range(3):
        xa, xb = a * (HALF - 0.3), -a * (HALF - 2.5)
        c.beam(xa, z, xb, z - 3.2, 'stone', 0.6)
        if i % 2 == 0:
            c.pistons.append(dict(x=-a * HALF, z=z - 3.2 + 1.0, dir=a, reach=2.4, period=2.0, phase=0.37 * i))
        else:
            u = 0.5
            c.tnt.append(dict(x=xa + (xb - xa) * u, z=z - 3.2 * u + 0.8))
        z -= 7.0
        a = -a
    z = c.round_end(z + 0.8, a, 'TRAPDOOR 3', 2, mat='stone')
    a = -a
    c.stages.append(dict(name='PISTONS & TNT', z_top=zt + 1.0, z_bot=z))
    # ---- THE FINAL: a long steep ice drop, a slime turn, the last trapdoor over the lava lake; the pen is the podium
    zt = z
    c.beam(a * (HALF - 0.3), z, -a * (HALF - 2.8), z - 6.6, 'ice')
    c.box(-a * (HALF - 0.9), z - 10.4, -a * HALF, z - 5.0, 'slime')
    a = -a
    z -= 9.6
    z = c.round_end(z, a, 'FINAL TRAPDOOR', 1, mat='gold', slope=0.16)
    c.pens[-1]['podium'] = True
    c.stages.append(dict(name='FINAL', z_top=zt + 1.0, z_bot=z))
    c.walls(z0 + 3.0, z - 1.0)
    c.z_end = z - 1.0
    # the lava lake at the foot of the cliff (for the look; the course's basins are above it)
    c.lake = (-20.0, z - 6.0, 20.0, z - 4.2)
    return c
