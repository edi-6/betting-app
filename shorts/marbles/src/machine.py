"""The marble machine's geometry, shared by the physics, the world and the renderer.

The machine is a wall of black concrete standing in the plains, facing south (towards -y), with a slot between the
board and a glass front exactly one marble deep. In that slot the marbles move in a plane, so the physics is 2D:
sim coordinates (x, z) are blocks across and up the board from the middle of the tank's floor, and map to the world
as (MX + x, MY, MZ + z).

From the top: a hopper of iron (3D, full of 10,000 marbles) with a hole in its floor, a chute that feeds them into
the slot through a throat, a field of end rods sticking out of the board (the pegs), and below them the tank, the
part of the slot behind the gold frame, where they pile up into the picture.
"""
import numpy as np

N = 10000                          # marbles
R = 0.25                           # marble radius (half a block across)
HALF = 17.0                        # the slot's inner half width
BOARD_TOP = 98.0                   # the top of the slot
THROAT_HALF = 3.7                  # the chute's throat into the slot (x in +-THROAT_HALF at the top)
THROAT_Z = 97.2                    # where the marbles come out of the chute
LANES = 14                         # marbles side by side coming through the throat
PEG_R = 0.08                       # an end rod, seen end on
PEG_DX, PEG_DZ = 2.2, 1.9
PEG_TOP, PEG_BOTTOM = 92.4, 75.0
FRAME_TOP = 68.0                   # the top of the picture frame (the fill ends between 62 and 66)

# world placement
MX, MY, MZ = 0.0, 180.0, 6.0       # the tank's floor sits on a pedestal 6 blocks high
SLOT = 0.30                        # the slot runs from y = MY - SLOT (glass) to MY + SLOT (board)

# the hopper (world coordinates): an iron box on top of the board, its floor funnelling to a hole over the chute
HOP_X, HOP_Y = 9.0, 4.5            # half sizes
HOP_Z0 = MZ + BOARD_TOP + 3.0      # its floor (the chute is under it)
HOP_Z1 = HOP_Z0 + 10.0             # its rim
HOLE = 2.0                         # half size of the hole in its floor
GOLD_R = 2.4                       # the golden ball (too big for the hole)


def pegs():
    """End rods: staggered rows, one row every PEG_DZ, PEG_DX apart, clear of the walls."""
    out = []
    zs = np.arange(PEG_TOP, PEG_BOTTOM - 1e-6, -PEG_DZ)
    for k, z in enumerate(zs):
        off = 0.0 if k % 2 == 0 else PEG_DX / 2
        for x in np.arange(-HALF + 1.0 + off, HALF - 0.9, PEG_DX):
            out.append((float(x), float(z)))
    return np.array(out)


def to_world(x, z):
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    return np.stack([MX + x, np.full_like(x, MY), MZ + z], -1)
