"""The block registry and the shapes of the non-cube blocks.

Axes: x east, y north, z up; 1 unit = 1 block. A voxel (i, j, k) of the world spans [x, x+1) x [y, y+1) x [z, z+1).
Models are written like the game's JSON block models: boxes in 1/16 units, faces with a texture and a uv rect (in
texture pixels, v down), for the block facing SOUTH (its front towards -y). The voxel's state rotates them:
facing 0 = front towards -y, 1 = +x, 2 = +y, 3 = -x (state & 3). Other state bits are per block (below).
"""
import numpy as np

# state bits
FACING = 3
HALF_TOP = 4         # slabs/trapdoors in the upper half, the head half of a bed, the upper half of a door
OPEN = 8             # doors, trapdoors
WALL = 16            # torches, signs and lanterns on a wall (or hanging, for lanterns)
HINGE_R = 32         # doors: hinge on the right


class Block:
    def __init__(self, name, shape='cube', tex=None, opaque=True, light=0, soul=0, cutout=False, model=None,
                 connects=False, material='terrain'):
        self.name = name
        self.shape = shape               # cube | cross | model | none | liquid
        self.tex = tex or {'all': name}
        self.opaque = opaque             # hides neighbours' faces and blocks light
        self.light = light               # warm block light it gives off (0..15)
        self.soul = soul                 # cold (soul fire) light it gives off
        self.cutout = cutout             # has see-through texels (drawn with alpha test)
        self.model = model               # f(state, neighbours) -> list of boxes, for shape 'model'
        self.connects = connects         # fences and panes join up with their neighbours
        self.material = material
        self.id = None

    def face_tex(self, face, state=0):
        """Texture of a cube face ('px','nx','py','ny','pz','nz') for a block in `state`."""
        t = self.tex
        if face == 'pz':
            return t.get('top', t.get('all'))
        if face == 'nz':
            return t.get('bottom', t.get('top', t.get('all')))
        if 'front' in t:
            front = {0: 'ny', 1: 'px', 2: 'py', 3: 'nx'}[state & FACING]
            if face == front:
                return t['front']
        return t.get('side', t.get('all'))


# ---------------------------------------------------------------------------------------------
# model helpers
# ---------------------------------------------------------------------------------------------
def box(x0, y0, z0, x1, y1, z1, faces, rot=None):
    """faces: dict face -> (texture, (u0, v0, u1, v1)) in pixels, or texture name (uv from the box's extent)."""
    return {'from': (x0, y0, z0), 'to': (x1, y1, z1), 'faces': faces, 'rot': rot}


def auto_faces(tex, x0, y0, z0, x1, y1, z1, skip=(), per_face=None):
    """All six faces with uv taken from the box's own extent on each face (like the game's default uvs)."""
    f = {}
    per_face = per_face or {}
    for face in ('px', 'nx', 'py', 'ny', 'pz', 'nz'):
        if face in skip:
            continue
        t = per_face.get(face, tex)
        if face in ('px', 'nx'):
            uv = (y0, 16 - z1, y1, 16 - z0) if face == 'px' else (16 - y1, 16 - z1, 16 - y0, 16 - z0)
        elif face in ('py', 'ny'):
            uv = (16 - x1, 16 - z1, 16 - x0, 16 - z0) if face == 'py' else (x0, 16 - z1, x1, 16 - z0)
        else:
            uv = (x0, 16 - y1, x1, 16 - y0)
        f[face] = (t, uv)
    return f


def cube_box(x0, y0, z0, x1, y1, z1, tex, skip=(), per_face=None, rot=None):
    return box(x0, y0, z0, x1, y1, z1, auto_faces(tex, x0, y0, z0, x1, y1, z1, skip, per_face), rot)


# ---------------------------------------------------------------------------------------------
# models (facing south; rotated by the mesher)
# ---------------------------------------------------------------------------------------------
def m_torch(tex):
    def f(state, nb):
        stick = {'px': (tex, (7, 4, 9, 16)), 'nx': (tex, (7, 4, 9, 16)), 'py': (tex, (7, 4, 9, 16)),
                 'ny': (tex, (7, 4, 9, 16)), 'pz': (tex, (7, 4, 9, 6)), 'nz': (tex, (7, 14, 9, 16))}
        if state & WALL:
            # leaning out of the wall behind it (+y), 22.5 degrees
            return [box(7, 11, 3.5, 9, 13, 15.5, stick, rot=('x', -22.5, (8, 16, 3.5)))]
        return [box(7, 7, 0, 9, 9, 12, stick)]
    return f


def m_lantern(tex):
    def f(state, nb):
        z = 0.0
        out = []
        if state & WALL:           # hanging from the block above
            z = 1.0
            out.append(box(7.5, 7.5, 10 + z, 8.5, 8.5, 16, {k: ('chain', (7, 0, 9, 6)) for k in
                                                            ('px', 'nx', 'py', 'ny')}))
        out.append(box(5, 5, z, 11, 11, z + 1, {k: (tex, (5, 9, 11, 10)) for k in ('px', 'nx', 'py', 'ny')} |
                       {'nz': (tex, (5, 9, 11, 10)), 'pz': (tex, (5, 9, 11, 10))}))
        out.append(box(5, 5, z + 1, 11, 11, z + 8, {k: (tex, (5, 2, 11, 9)) for k in ('px', 'nx', 'py', 'ny')}))
        out.append(box(6, 6, z + 8, 10, 10, z + 10, {k: (tex, (6, 0, 10, 2)) for k in
                                                     ('px', 'nx', 'py', 'ny', 'pz')}))
        return out
    return f


def m_chain(state, nb):
    return [box(7, 7, 0, 9, 9, 16, {k: ('chain', (6, 0, 10, 16)) for k in ('px', 'nx', 'py', 'ny')})]


def m_bed(color):
    def f(state, nb):
        head = state & HALF_TOP
        top = f'{color}_bed_head' if head else f'{color}_bed_foot'
        side = f'{color}_bed_side'
        faces = {'pz': (top, (0, 0, 16, 16)), 'nz': ('oak_planks', (0, 0, 16, 16)),
                 'px': (side, (0, 3, 16, 9)), 'nx': (side, (0, 3, 16, 9))}
        # head half: the pillow end is the block's back (+y); the foot half's end faces -y
        if head:
            faces['py'] = (side, (0, 3, 16, 9))
        else:
            faces['ny'] = (side, (0, 3, 16, 9))
        out = [box(0, 0, 3, 16, 16, 9, faces)]
        legs = [(0, 13), (13, 13)] if head else [(0, 0), (13, 0)]
        for (lx, ly) in legs:
            out.append(cube_box(lx, ly, 0, lx + 3, ly + 3, 3, 'oak_planks'))
        return out
    return f


def m_chest(state, nb):
    per = {'ny': 'chest_front'}
    return [cube_box(1, 1, 0, 15, 15, 14, 'chest_side', per_face=per | {'pz': 'chest_top', 'nz': 'chest_top'}),
            cube_box(7, 0, 7, 9, 1, 11, 'chest_front')]


def m_door(wood):
    def f(state, nb):
        part = 'top' if state & HALF_TOP else 'bottom'
        tex = f'{wood}_door_{part}'
        if state & OPEN:
            # swung open against the side wall (hinge side)
            if state & HINGE_R:
                return [box(13, 0, 0, 16, 16, 16, auto_faces(tex, 13, 0, 0, 16, 16, 16))]
            return [box(0, 0, 0, 3, 16, 16, auto_faces(tex, 0, 0, 0, 3, 16, 16))]
        fs = auto_faces(tex, 0, 0, 0, 16, 3, 16)
        fs['ny'] = (tex, (0, 0, 16, 16))
        fs['py'] = (tex, (16, 0, 0, 16))
        return [box(0, 0, 0, 16, 3, 16, fs)]
    return f


def m_trapdoor(state, nb):
    t = 'oak_trapdoor'
    if state & OPEN:                            # against the back wall
        fs = auto_faces(t, 0, 13, 0, 16, 16, 16)
        fs['ny'] = (t, (0, 0, 16, 16))
        fs['py'] = (t, (0, 0, 16, 16))
        return [box(0, 13, 0, 16, 16, 16, fs)]
    z0 = 13 if state & HALF_TOP else 0
    fs = auto_faces(t, 0, 0, z0, 16, 16, z0 + 3)
    fs['pz'] = (t, (0, 0, 16, 16))
    fs['nz'] = (t, (0, 0, 16, 16))
    return [box(0, 0, z0, 16, 16, z0 + 3, fs)]


def m_ladder(state, nb):
    # flat against the wall behind it (+y)
    return [box(0, 15.2, 0, 16, 15.4, 16, {'ny': ('ladder', (0, 0, 16, 16)), 'py': ('ladder', (16, 0, 0, 16))})]


def m_fence(tex):
    def f(state, nb):
        out = [cube_box(6, 6, 0, 10, 10, 16, tex)]
        for d, (dx, dy) in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0))):
            if nb.get((dx, dy)):
                for (z0, z1) in ((6, 9), (12, 15)):
                    if dx == 0:
                        ya, yb = (0, 6) if dy < 0 else (10, 16)
                        out.append(cube_box(7, ya, z0, 9, yb, z1, tex))
                    else:
                        xa, xb = (0, 6) if dx < 0 else (10, 16)
                        out.append(cube_box(xa, 7, z0, xb, 9, z1, tex))
        return out
    return f


def m_pane(state, nb):
    g = 'glass'
    out = [box(7, 7, 0, 9, 9, 16, {k: (g, (7, 0, 9, 16)) for k in ('px', 'nx', 'py', 'ny')} |
               {'pz': (g, (7, 7, 9, 9)), 'nz': (g, (7, 7, 9, 9))})]
    for (dx, dy) in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        if nb.get((dx, dy)):
            if dx == 0:
                ya, yb = (0, 7) if dy < 0 else (9, 16)
                out.append(box(7, ya, 0, 9, yb, 16, {'px': (g, (ya, 0, yb, 16)), 'nx': (g, (16 - yb, 0, 16 - ya, 16)),
                                                     'pz': (g, (7, 16 - yb, 9, 16 - ya))}))
            else:
                xa, xb = (0, 7) if dx < 0 else (9, 16)
                out.append(box(xa, 7, 0, xb, 9, 16, {'ny': (g, (xa, 0, xb, 16)), 'py': (g, (16 - xb, 0, 16 - xa, 16)),
                                                     'pz': (g, (xa, 7, xb, 9))}))
    return out


def m_sign(state, nb):
    t = 'sign'
    if state & WALL:            # on the wall behind it (+y)
        return [cube_box(0, 14, 4, 16, 16, 12, t)]
    return [cube_box(7, 7, 0, 9, 9, 9, 'oak_log'), cube_box(0, 7, 9, 16, 9, 17, t)]


def m_lectern(state, nb):
    out = [cube_box(0, 0, 0, 16, 16, 2, 'lectern_base', per_face={'pz': 'lectern_base'}),
           cube_box(4, 4, 2, 12, 12, 13, 'lectern_side'),
           cube_box(0, 1, 12, 16, 15, 16, 'lectern_side', per_face={'pz': 'lectern_top'},
                    rot=('x', 22.5, (8, 8, 14)))]
    return out


def m_item_frame(state, nb):
    # flat on the wall behind it (+y)
    return [cube_box(2, 15, 2, 14, 16, 14, 'item_frame')]


def m_carpet(tex):
    def f(state, nb):
        return [cube_box(0, 0, 0, 16, 16, 1, tex)]
    return f


def m_slab(tex, top_tex=None):
    def f(state, nb):
        z0 = 8 if state & HALF_TOP else 0
        return [cube_box(0, 0, z0, 16, 16, z0 + 8, tex, per_face={'pz': top_tex or tex, 'nz': top_tex or tex})]
    return f


def m_stairs(tex):
    def f(state, nb):
        # low step at the front (-y), high part at the back
        return [cube_box(0, 0, 0, 16, 16, 8, tex), cube_box(0, 8, 8, 16, 16, 16, tex)]
    return f


def m_flower_pot(plant):
    def f(state, nb):
        out = [cube_box(5, 5, 0, 11, 11, 6, 'terracotta_pot')]
        return out
    return f


def m_crafting_plate(state, nb):
    return []


# ---------------------------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------------------------
REG = []
BY_NAME = {}


def add(b):
    b.id = len(REG)
    REG.append(b)
    BY_NAME[b.name] = b
    return b


add(Block('air', 'none', opaque=False))
add(Block('grass_block', tex={'top': 'grass_top', 'bottom': 'dirt', 'side': 'grass_side'}, material='ground'))
add(Block('dirt', material='ground'))
add(Block('dirt_path', tex={'top': 'dirt_path_top', 'bottom': 'dirt', 'side': 'dirt_path_side'}, material='ground'))
add(Block('farmland', tex={'top': 'farmland', 'bottom': 'dirt', 'side': 'dirt'}, material='ground'))
for n in ('stone', 'smooth_stone', 'cobblestone', 'mossy_cobblestone', 'stone_bricks', 'mossy_stone_bricks',
          'cracked_stone_bricks', 'gravel', 'sand', 'cobbled_deepslate', 'deepslate_tiles', 'obsidian',
          'oak_planks', 'spruce_planks', 'dark_oak_planks', 'bookshelf', 'glowstone',
          'white_wool', 'red_wool', 'gray_wool', 'blue_wool', 'light_gray_wool', 'brown_wool', 'green_wool',
          'white_concrete', 'light_gray_concrete', 'gray_concrete', 'black_concrete', 'blue_concrete', 'quartz'):
    add(Block(n, light=15 if n == 'glowstone' else 0))
BY_NAME['bookshelf'].tex = {'top': 'oak_planks', 'side': 'bookshelf'}
add(Block('deepslate', tex={'top': 'deepslate_top', 'side': 'deepslate'}))
add(Block('redstone_lamp', tex={'all': 'redstone_lamp_on'}, light=15))
for wood, bark in (('oak', 'oak'), ('spruce', 'spruce'), ('birch', 'birch'), ('dead', 'dead')):
    add(Block(f'{wood}_log', tex={'top': f'{bark}_log_top' if wood != 'dead' else 'oak_log_top',
                                  'side': f'{bark}_log'}))
for lv in ('oak', 'spruce', 'birch'):
    add(Block(f'{lv}_leaves', opaque=False, cutout=True, material='leaf'))
add(Block('glass', opaque=False, cutout=True))
add(Block('water', 'liquid', opaque=False, material='water'))
add(Block('hay_block', tex={'top': 'hay_top', 'side': 'hay_side'}))
add(Block('pumpkin', tex={'top': 'pumpkin_top', 'side': 'pumpkin_side'}))
add(Block('crafting_table', tex={'top': 'crafting_table_top', 'bottom': 'oak_planks', 'side': 'crafting_table_side',
                                 'front': 'crafting_table_front'}))
add(Block('furnace', tex={'top': 'furnace_top', 'side': 'furnace_side', 'front': 'furnace_front'}))
add(Block('lit_furnace', tex={'top': 'furnace_top', 'side': 'furnace_side', 'front': 'furnace_front_on'}, light=13))
add(Block('screen_block', tex={'all': 'black_concrete'}))
# models
add(Block('torch', 'model', opaque=False, cutout=True, light=14, model=m_torch('torch')))
add(Block('soul_torch', 'model', opaque=False, cutout=True, soul=10, model=m_torch('soul_torch')))
add(Block('lantern', 'model', opaque=False, cutout=True, light=15, model=m_lantern('lantern')))
add(Block('soul_lantern', 'model', opaque=False, cutout=True, soul=10, model=m_lantern('soul_lantern')))
add(Block('chain', 'model', opaque=False, cutout=True, model=m_chain))
add(Block('red_bed', 'model', opaque=False, model=m_bed('red')))
add(Block('blue_bed', 'model', opaque=False, model=m_bed('blue')))
add(Block('chest', 'model', opaque=False, model=m_chest))
add(Block('oak_door', 'model', opaque=False, cutout=True, model=m_door('oak')))
add(Block('spruce_door', 'model', opaque=False, cutout=True, model=m_door('spruce')))
add(Block('oak_trapdoor', 'model', opaque=False, cutout=True, model=m_trapdoor))
add(Block('ladder', 'model', opaque=False, cutout=True, model=m_ladder))
add(Block('oak_fence', 'model', opaque=False, model=m_fence('oak_planks'), connects=True))
add(Block('spruce_fence', 'model', opaque=False, model=m_fence('spruce_planks'), connects=True))
add(Block('glass_pane', 'model', opaque=False, cutout=True, model=m_pane, connects=True))
add(Block('oak_sign', 'model', opaque=False, model=m_sign))
add(Block('lectern', 'model', opaque=False, model=m_lectern))
add(Block('item_frame', 'model', opaque=False, model=m_item_frame))
add(Block('red_carpet', 'model', opaque=False, model=m_carpet('red_wool')))
add(Block('gray_carpet', 'model', opaque=False, model=m_carpet('gray_wool')))
add(Block('brown_carpet', 'model', opaque=False, model=m_carpet('brown_wool')))
add(Block('oak_slab', 'model', opaque=False, model=m_slab('oak_planks')))
add(Block('spruce_slab', 'model', opaque=False, model=m_slab('spruce_planks')))
add(Block('stone_slab', 'model', opaque=False, model=m_slab('smooth_stone')))
add(Block('cobblestone_slab', 'model', opaque=False, model=m_slab('cobblestone')))
add(Block('white_slab', 'model', opaque=False, model=m_slab('white_concrete')))
add(Block('oak_stairs', 'model', opaque=False, model=m_stairs('oak_planks')))
add(Block('spruce_stairs', 'model', opaque=False, model=m_stairs('spruce_planks')))
add(Block('cobblestone_stairs', 'model', opaque=False, model=m_stairs('cobblestone')))
add(Block('stone_brick_stairs', 'model', opaque=False, model=m_stairs('stone_bricks')))
# plants (two crossed quads)
for p in ('tall_grass', 'poppy', 'dandelion', 'cornflower', 'wheat', 'wheat_young', 'dead_bush', 'sugar_cane',
          'cobweb'):
    add(Block(p, 'cross', opaque=False, cutout=True, material='plant' if p != 'cobweb' else 'terrain'))

B = {b.name: b.id for b in REG}
NBLOCKS = len(REG)
OPAQUE = np.array([b.opaque and b.shape in ('cube',) for b in REG], bool)
CUTOUT = np.array([b.cutout for b in REG], bool)
SHAPE = np.array([b.shape for b in REG])
LIGHT = np.array([b.light for b in REG], np.uint8)
SOUL = np.array([b.soul for b in REG], np.uint8)
IS_CUBE = np.array([b.shape == 'cube' for b in REG], bool)          # full cubes (maybe see-through)
CONNECTS = np.array([b.connects for b in REG], bool)


def block(name):
    return B[name]


if __name__ == '__main__':
    print(NBLOCKS, 'blocks')
    for b in REG:
        print(b.id, b.name, b.shape)
