"""The ride: one track per dimension, steered segment by segment (see track.PathBuilder).

OVERWORLD  the lift hill up from the station on the mesa's rim, the crest, the drop into the badlands canyon, the
           river bends, through the waterfall, under the arch, an airtime hill, into the ruined portal in the cliff.
NETHER     out of a portal on a ledge high over the lava ocean, a second drop, low over the lava between pillars and
           lava falls, the hump where the ghast's fireball breaks the track (the cart flies the gap), up onto the
           fortress bridge and down into the End portal at its end.
THE END    off the obsidian platform over the void, up to the ring of obsidian pillars, a slalom between them while
           the dragon swoops, a corkscrew round a pillar, and the dive into the exit portal under the perched dragon.
DEEP DARK  out of the dark over the ancient city, round its tower, and straight at the great portal as it wakes.
THE SIFT   out of the portal on a siftslate cliff, down into the Singer's Meadow, through the white trees and the blue
           grass, low over an ichor lake, under the ribs of a giant fossil, and into the rift that leads home.

Heights (z): overworld river 20, mesa rim 100; nether lava sea 31; end island top about 62, void below.
"""
import track as TK

RIVER_Z = 20
RIM_Z = 100
LAVA_Z = 31
END_TOP = 62


def overworld():
    pb = TK.PathBuilder((0.0, -186.0, RIM_Z + 1.0), yaw=0.0, pitch=0.0, v=7.0)
    pb.mark('station')
    pb.chain_on(7.0)
    pb.seg(8, pitch=24)
    pb.mark('lift')
    pb.seg(36, pitch=24)
    pb.seg(9, pitch=0)
    pb.chain_off()
    pb.mark('crest')
    pb.seg(9, pitch=-30)                     # tipping over: the canyon opens up below
    pb.mark('drop')
    pb.seg(26, pitch=-76)
    pb.seg(58, pitch=-76)
    pb.seg(34, pitch=0)                      # pull-out just over the water
    pb.mark('bottom')
    pb.seg(52, turn=58)                      # a long right bend along the river
    pb.mark('bend2')
    pb.seg(62, turn=-92, pitch=7)            # left, climbing to the waterfall ledge
    pb.seg(12, pitch=0)
    pb.mark('waterfall')
    pb.seg(30, turn=28, pitch=-7)
    pb.mark('arch')
    pb.seg(16, pitch=0)
    pb.seg(16, pitch=24)                     # airtime hill
    pb.seg(20, pitch=-26)
    pb.seg(14, pitch=0)
    pb.mark('approach')
    pb.powered_on(36.0, 30.0)
    pb.seg(30, turn=6)                       # square on to the portal (facing north)
    pb.powered_off()
    pb.mark('portal')
    pb.seg(10)
    return pb.build(name='overworld')


def nether():
    pb = TK.PathBuilder((0.0, -150.0, 96.0), yaw=0.0, pitch=0.0, v=30.0)
    pb.mark('start')
    pb.powered_on(30.0, 30.0)
    pb.seg(14)
    pb.powered_off()
    pb.mark('drop')
    pb.seg(12, pitch=-38)
    pb.seg(48, pitch=-68)
    pb.seg(34, pitch=0)                      # down to just over the lava
    pb.mark('lava')
    pb.seg(56, turn=-55)
    pb.mark('falls')
    pb.seg(56, turn=70)
    pb.mark('ghast')
    pb.seg(34, turn=-15, pitch=6)
    pb.seg(22, pitch=13)                     # up the hump the fireball will break
    pb.mark('takeoff')
    pb.ballistic(until_z=None, duration=1.30)
    pb.mark('land')
    pb.seg(18, pitch=-6)
    pb.powered_on(34.0, 24.0)
    pb.seg(30, turn=40, pitch=12)
    pb.powered_off()
    pb.mark('bridge')
    pb.seg(46, pitch=0)
    pb.mark('portal')
    pb.seg(8, pitch=-60)
    pb.seg(6)
    return pb.build(name='nether')


def the_end():
    pb = TK.PathBuilder((100.0, 0.0, 50.0), yaw=-90.0, pitch=0.0, v=30.0)
    pb.mark('start')
    pb.powered_on(32.0, 24.0)
    pb.seg(14)
    pb.seg(26, pitch=16)                     # up towards the island's edge
    pb.seg(14, pitch=0)
    pb.powered_off()
    pb.mark('ring')
    pb.seg(36, turn=-26, pitch=-6)           # into the ring, between two pillars
    pb.seg(30, turn=40, pitch=0)
    pb.mark('dragon')
    pb.seg(30, turn=-30)
    pb.mark('helix')
    pb.powered_on(30.0, 20.0)
    pb.seg(80, turn=200, pitch=-9)           # a banked spiral out over the void and back in
    pb.powered_off()
    pb.seg(12, pitch=0)
    pb.mark('approach')
    pb.seg(34, turn=-20)
    pb.mark('portal')
    pb.seg(10, pitch=-55)
    pb.seg(8)
    return pb.build(name='end')


def deep():
    pb = TK.PathBuilder((0.0, -60.0, 40.0), yaw=0.0, pitch=0.0, v=31.0)
    pb.mark('start')
    pb.seg(22, pitch=-8)                     # out over the city, dipping
    pb.seg(26, turn=35, pitch=0)             # round the tower
    pb.mark('tower')
    pb.seg(24, turn=-35)                     # square onto the portal
    pb.mark('approach')
    pb.powered_on(36.0, 20.0)
    pb.seg(26)
    pb.powered_off()
    pb.mark('portal')
    pb.seg(10)
    return pb.build(name='deep')


def sift():
    pb = TK.PathBuilder((0.0, -150.0, 88.0), yaw=0.0, pitch=0.0, v=30.0)
    pb.mark('start')
    pb.powered_on(32.0, 24.0)
    pb.seg(16)
    pb.powered_off()
    pb.mark('drop')
    pb.seg(10, pitch=-34)
    pb.seg(36, pitch=-56)
    pb.seg(26, pitch=0)                      # pull-out over the meadow
    pb.mark('meadow')
    pb.seg(46, turn=-55)                     # left, through the white trees
    pb.seg(46, turn=70)                      # right, down to the lake
    pb.mark('lake')
    pb.seg(40, turn=-15)                     # low over the ichor
    pb.mark('fossil')
    pb.seg(34, pitch=3)                      # under the fossil's ribs
    pb.seg(16, pitch=16)
    pb.seg(14, pitch=0)
    pb.mark('rift')
    pb.seg(10)
    return pb.build(name='sift')


if __name__ == '__main__':
    for f in (overworld, nether, the_end, deep, sift):
        tr = f()
        print(tr.summary())
