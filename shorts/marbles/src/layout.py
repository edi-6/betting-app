"""Where things are in the world (blocks; x east, y north, z up): the plains from the dominoes videos, with the river
and its bridge to the south, the cliff to the north, and the marble machine (machine.py) standing in the meadow
between them, facing south.
"""
import machine as M

RIVER_Y = (-30.0, -22.0)          # the river's banks (south, north), running east-west
BRIDGE = (-33.0, -19.0)
CLIFF_Y = 258.0                   # the cliff's face, north of the machine
CLIFF_Z = 72                      # its flat top
CLIFF_SPOT = (2.5, CLIFF_Y + 1.0, float(CLIFF_Z))      # someone stands on its edge at the very end

# kept clear of trees: the machine, the meadow in front of it (the goat, the cameras, where the marbles land) and
# the spot on the cliff's edge
CLEAR = [(-46.0, M.MY - 110.0, 46.0, M.MY + 22.0),
         (-14.0, CLIFF_Y - 2.0, 14.0, CLIFF_Y + 20.0)]
