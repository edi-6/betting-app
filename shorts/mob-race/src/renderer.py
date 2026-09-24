"""Deferred OpenGL renderer (moderngl, runs headless on Mesa llvmpipe).

Pipeline per frame:
  near shadow map (dynamic objects + nearby trees)  ->  G-buffer (albedo, normal+material, depth)
  -> SSAO (half res) + bilateral blur -> lighting (sun w/ PCSS-ish soft shadows, sky ambient,
  metal reflections, foliage translucency, aerial fog) -> tonemap/grade -> FXAA -> downsample -> read back.
"""
import numpy as np
import moderngl

from sky import EL_MIN
SUN_DIR = np.array([0.45, -0.6, 0.66]) / np.linalg.norm([0.45, -0.6, 0.66])

MAT_TERRAIN, MAT_LEAF, MAT_SKIN, MAT_FLESH, MAT_BONE, MAT_GROUND, MAT_PROP, MAT_GLOW = 1, 2, 3, 4, 5, 7, 8, 9
MAT_TNT, MAT_ARROW = 10, 11      # MAT_TNT: albedo alpha carries an unlit white flash (primed TNT, a creeper)
MAT_HOT = 12                     # heated by the accretion disk: albedo alpha carries the heat (0..1), it glows
MAX_GIANTS = 8
SCULK_LAYER = -1         # (no sculk in the studio)
LEAVES_LAYER = -1        # (no leaves in the studio)


# ---------------------------------------------------------------------------------------------
# math helpers
# ---------------------------------------------------------------------------------------------
def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def look_at(eye, target, up=(0, 0, 1)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def ortho(l, r, b, t, n, f):
    m = np.eye(4)
    m[0, 0] = 2 / (r - l)
    m[1, 1] = 2 / (t - b)
    m[2, 2] = -2 / (f - n)
    m[0, 3] = -(r + l) / (r - l)
    m[1, 3] = -(t + b) / (t - b)
    m[2, 3] = -(f + n) / (f - n)
    return m


def light_matrix(center, half, depth=420.0):
    c = np.asarray(center, float)
    eye = c + SUN_DIR * depth * 0.5
    up = (0, 0, 1) if abs(SUN_DIR[2]) < 0.99 else (0, 1, 0)
    return ortho(-half, half, -half, half, 1.0, depth) @ look_at(eye, c, up)


def frustum_planes(vp):
    m = vp
    planes = [m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1], m[3] + m[2], m[3] - m[2]]
    out = []
    for p in planes:
        n = np.linalg.norm(p[:3])
        out.append(p / n)
    return np.array(out)


def aabb_visible(planes, lo, hi):
    for p in planes:
        # positive vertex
        v = np.where(p[:3] >= 0, hi, lo)
        if p[:3] @ v + p[3] < 0:
            return False
    return True


def m4(m):
    return np.ascontiguousarray(m.T, dtype=np.float32).tobytes()


# ---------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------
def unit_cube():
    """36 vertices: pos(3) normal(3) face(1). face: 0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z."""
    faces = [
        ((1, 0, 0), [(.5, -.5, -.5), (.5, .5, -.5), (.5, .5, .5), (.5, -.5, .5)]),
        ((-1, 0, 0), [(-.5, .5, -.5), (-.5, -.5, -.5), (-.5, -.5, .5), (-.5, .5, .5)]),
        ((0, 1, 0), [(.5, .5, -.5), (-.5, .5, -.5), (-.5, .5, .5), (.5, .5, .5)]),
        ((0, -1, 0), [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, -.5, .5), (-.5, -.5, .5)]),
        ((0, 0, 1), [(-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)]),
        ((0, 0, -1), [(-.5, .5, -.5), (.5, .5, -.5), (.5, -.5, -.5), (-.5, -.5, -.5)]),
    ]
    out = []
    for fi, (n, q) in enumerate(faces):
        for k in (0, 1, 2, 0, 2, 3):
            out.append((*q[k], *n, fi))
    return np.array(out, np.float32)


# ---------------------------------------------------------------------------------------------
# shaders
# ---------------------------------------------------------------------------------------------
QROT = """
vec3 qrot(vec4 q, vec3 v){ return v + 2.0*cross(q.xyz, cross(q.xyz, v) + q.w*v); }
"""

STATIC_VS = """
#version 430
uniform mat4 u_vp;
uniform vec4 u_bh;           // black hole position and how hard it pulls on the tree tops
uniform float u_time;
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv; in float in_layer; in vec3 in_tint;
out vec3 v_nrm; out vec2 v_uv; flat out float v_layer; out vec3 v_tint; out vec3 v_wpos;
void main(){
    vec3 p = in_pos;
    if (u_bh.w > 0.0 && int(in_layer + 0.5) == %d) {
        vec2 d = u_bh.xy - p.xy;
        float dist = max(length(d), 1e-3);
        float k = u_bh.w * clamp(p.z / 11.0, 0.0, 1.0) * clamp(70.0 / (dist + 25.0), 0.0, 1.4);
        float w = 0.72 + 0.28 * sin(u_time * 7.3 + p.x * 0.61 + p.y * 0.37);
        p.xy += d / dist * k * w;
        p.z += 0.12 * k * w;
    }
    v_nrm = in_nrm; v_uv = in_uv; v_layer = in_layer; v_tint = in_tint; v_wpos = p;
    gl_Position = u_vp * vec4(p, 1.0);
}
""" % LEAVES_LAYER

STATIC_FS = """
#version 430
uniform sampler2DArray u_blocks;
uniform sampler2D u_tintnoise;
uniform float u_is_ground;
uniform float u_sculk_r;
uniform vec2 u_sculk_c;
in vec3 v_nrm; in vec2 v_uv; flat in float v_layer; in vec3 v_tint; in vec3 v_wpos;
layout(location=0) out vec4 o_albedo;
layout(location=1) out vec4 o_normal;
void main(){
    float layer = v_layer;
    if (u_is_ground > 0.5 && u_sculk_r > 0.0) {
        // sculk has spread over the ground around the Warden: whole blocks, ragged edge, a few strays
        vec2 cell = floor(v_wpos.xy - u_sculk_c);
        float n1 = texture(u_tintnoise, (cell + 0.5) / 53.0).r;
        float n2 = texture(u_tintnoise, (cell + 0.5) / 7.0 + 0.37).r;
        float r = length(cell + 0.5);
        if (r < u_sculk_r * (0.7 + 0.6 * n1) || (r < u_sculk_r * 1.7 && n2 > 0.78)) layer = %d.0;
    }
    vec4 tx = texture(u_blocks, vec3(v_uv, layer));
    vec3 c = pow(tx.rgb, vec3(2.2)) * v_tint;
    int lay = int(layer + 0.5);
    float mat = %d.0;
    if (lay == -1) mat = %d.0;
    if (false) {                                // (no grass in the studio)
        // large scale colour variation on grass so fields don't look uniform
        float n = texture(u_tintnoise, v_wpos.xy / 97.0).r;
        float n2 = texture(u_tintnoise, v_wpos.xy / 23.0 + 0.37).r;
        vec3 t = mix(vec3(0.90, 0.95, 0.86), vec3(1.07, 1.05, 0.93), n) * mix(0.95, 1.04, n2);
        if (lay == 0) c *= t;
        else if (fract(-v_wpos.z) < 0.3) c *= t;
    }
    if (u_is_ground > 0.5) mat = %d.0;
    if (tx.a > 0.35) mat = %d.0;                 // glowing texels (sculk)
    o_albedo = vec4(sqrt(clamp(c, 0.0, 1.0)), 0.0);
    o_normal = vec4(v_nrm * 0.5 + 0.5, mat / 255.0);
}
""" % (SCULK_LAYER, MAT_TERRAIN, MAT_LEAF, MAT_GROUND, MAT_GLOW)

GIANT_POSE = """
uniform int u_ng;            // standing giants: instance ranges [x, y), fx = (hurt, white flash, swell, heat)
uniform ivec2 u_range[%d];
uniform vec4 u_gfx[%d];
uniform vec3 u_gcenter[%d];
uniform vec4 u_gq[%d];       // rigid pose of the giant: rotation about the pivot, then translation
uniform vec3 u_gt[%d];
uniform vec3 u_gp[%d];
uniform vec4 u_gs[%d];       // stretch (spaghettification): world axis xyz, factor w, about u_gsc
uniform vec3 u_gsc[%d];
uniform vec4 u_gsq[%d];      // squash under the press: height factor, width factor, bulge, rest height
uniform vec3 u_gsqc[%d];     // ... about the centre of the base
""" % ((MAX_GIANTS,) * 10)

VOXEL_VS = """
#version 430
uniform mat4 u_vp;
uniform vec4 u_hole;         // black hole centre and radius: voxels glow as they reach its edge
""" + GIANT_POSE + """
in vec3 in_pos; in vec3 in_nrm; in float in_face;
in vec3 i_pos; in vec4 i_quat; in float i_scale; in vec4 i_cx; in vec4 i_cy; in vec4 i_cz; in vec4 i_inner;
out vec3 v_nrm; flat out vec3 v_col; flat out float v_mat; flat out float v_white;
""" + QROT + """
void main(){
    vec3 p = qrot(i_quat, in_pos * i_scale) + i_pos;
    float hurt = 0.0, white = 0.0, gheat = 0.0;
    vec4 gq = vec4(0.0, 0.0, 0.0, 1.0);
    for (int g = 0; g < u_ng; g++) {
        if (gl_InstanceID >= u_range[g].x && gl_InstanceID < u_range[g].y) {
            vec4 sq = u_gsq[g];
            if (sq.x != 1.0 || sq.y != 1.0) {
                vec3 d = p - u_gsqc[g];
                float h = clamp(d.z / max(sq.w, 1e-3), 0.0, 1.0);
                float bul = 1.0 + sq.z * 4.0 * h * (1.0 - h);
                p = u_gsqc[g] + vec3(d.xy * sq.y * bul, d.z * sq.x);
            }
            p = u_gcenter[g] + (p - u_gcenter[g]) * u_gfx[g].z;      // a primed creeper swells
            hurt = u_gfx[g].x;
            white = u_gfx[g].y;
            gheat = u_gfx[g].w;
            gq = u_gq[g];
            p = u_gp[g] + u_gt[g] + qrot(gq, p - u_gp[g]);
            vec3 dp = p - u_gsc[g];                  // spaghettified: the half facing the hole is drawn out
            float al = dot(dp, u_gs[g].xyz);           // towards it, thinner and thinner, like a noodle
            float fw = clamp(al / 4.0, 0.0, 1.0);
            p = u_gsc[g] + u_gs[g].xyz * (al * (al > 0.0 ? u_gs[g].w : 1.0))
                + (dp - u_gs[g].xyz * al) * (1.0 - (1.0 - pow(u_gs[g].w, -0.4)) * fw);
        }
    }
    v_white = white;
    v_nrm = qrot(gq, qrot(i_quat, in_nrm));
    int face = int(in_face + 0.5);
    int vis = int(i_cy.a * 255.0 + 0.5);
    if (((vis >> face) & 1) == 0) { gl_Position = vec4(0.0, 0.0, -10.0, 1.0); v_nrm = vec3(0.0); v_col = vec3(0.0); v_mat = 0.0; v_white = 0.0; return; }
    int mask = int(i_cx.a * 255.0 + 0.5);
    vec3 skin = face < 2 ? i_cx.rgb : (face < 4 ? i_cy.rgb : i_cz.rgb);
    bool isSkin = ((mask >> face) & 1) == 1;
    v_col = isSkin ? skin : i_inner.rgb;
    v_col = v_col * vec3(1.0, 1.0 - 0.6 * hurt, 1.0 - 0.6 * hurt) + vec3(0.22 * hurt, 0.0, 0.0);
    // inner alpha: 255 bone, 128 glowing soul speck; the z colour's alpha: 255 glowing skin, 1..254 heat
    float innerMat = i_inner.a > 0.75 ? %d.0 : (i_inner.a > 0.25 ? %d.0 : %d.0);
    v_mat = isSkin ? (i_cz.a > 0.998 ? %d.0 : %d.0) : innerMat;
    float heat = max(gheat, (i_cz.a > 0.002 && i_cz.a <= 0.998) ? i_cz.a : 0.0);
    if (u_hole.w > 0.0) {
        float hd = length(p - u_hole.xyz);
        heat = max(heat, pow(clamp((1.5 * u_hole.w + 1.0 - hd) / (0.5 * u_hole.w + 1.0), 0.0, 1.0), 1.5));
    }
    if (heat > 0.0 && v_mat != %d.0) { v_mat = %d.0; white = heat; }
    if (white > 0.0 && v_mat != %d.0) v_mat = %d.0;
    v_white = white;
    gl_Position = u_vp * vec4(p, 1.0);
}
""" % (MAT_BONE, MAT_GLOW, MAT_FLESH, MAT_GLOW, MAT_SKIN, MAT_GLOW, MAT_HOT, MAT_HOT, MAT_TNT)

VOXEL_FS = """
#version 430
in vec3 v_nrm; flat in vec3 v_col; flat in float v_mat; flat in float v_white;
layout(location=0) out vec4 o_albedo;
layout(location=1) out vec4 o_normal;
void main(){
    vec3 c = pow(v_col, vec3(2.2));
    float spec = 0.0;
    if (int(v_mat + 0.5) == %d) spec = 0.09;   // wet flesh
    if (int(v_mat + 0.5) == %d) spec = 0.06;
    if (int(v_mat + 0.5) == %d) spec = v_white;
    if (int(v_mat + 0.5) == %d) spec = v_white;
    o_albedo = vec4(sqrt(clamp(c, 0.0, 1.0)), spec);
    o_normal = vec4(normalize(v_nrm) * 0.5 + 0.5, v_mat / 255.0);
}
""" % (MAT_FLESH, MAT_BONE, MAT_TNT, MAT_HOT)

PROP_VS = """
#version 430
uniform mat4 u_vp;
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv; in float in_layer;
in vec3 i_pos; in vec4 i_quat; in float i_scale; in float i_var; in float i_fade; in float i_flash;
out vec3 v_nrm; out vec2 v_uv; flat out float v_layer; flat out float v_fade; flat out float v_flash;
""" + QROT + """
void main(){
    v_nrm = qrot(i_quat, in_nrm);
    v_uv = in_uv;
    v_layer = in_layer >= 100.0 ? in_layer - 100.0 + i_var : in_layer;   // worn working face per instance
    v_fade = i_fade;
    v_flash = i_flash;
    gl_Position = u_vp * vec4(qrot(i_quat, in_pos * i_scale) + i_pos, 1.0);
}
"""

PROP_FS = """
#version 430
uniform sampler2DArray u_props;
uniform float u_sheen;
in vec3 v_nrm; in vec2 v_uv; flat in float v_layer; flat in float v_fade; flat in float v_flash;
layout(location=0) out vec4 o_albedo;
layout(location=1) out vec4 o_normal;
const float BAYER[16] = float[16](0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
void main(){
    if (v_fade < 0.999) {                          // screen-door fade for props right next to the lens
        ivec2 q = ivec2(mod(gl_FragCoord.xy, 4.0));
        if (v_fade <= (BAYER[q.x + q.y * 4] + 0.5) / 16.0) discard;
    }
    vec3 c = texture(u_props, vec3(v_uv, v_layer)).rgb;
    // primed TNT: its unlit white flash rides in the albedo alpha of MAT_TNT; iron gets a little sheen;
    // torn-out blocks (u_sheen < -1.5) are terrain until the accretion disk heats them (heat in i_flash)
    bool block = u_sheen < -1.5;
    bool tnt = u_sheen < 0.0 && !block;
    float mat = tnt ? %d.0 : (block ? (v_flash > 0.002 ? %d.0 : %d.0) : %d.0);
    o_albedo = vec4(sqrt(clamp(pow(c, vec3(2.2)), 0.0, 1.0)), (tnt || block) ? v_flash : u_sheen);
    o_normal = vec4(normalize(v_nrm) * 0.5 + 0.5, mat / 255.0);
}
""" % (MAT_TNT, MAT_HOT, MAT_TERRAIN, MAT_PROP)

ARROW_TEX = """
// Hand-built mip chain laid out side by side (see arrows.py): pick the level from the derivatives,
// then texelFetch so thin shafts never average away.
vec4 arrow_tex(sampler2DArray tex, vec2 uv, float layer){
    vec2 st = uv * vec2(16.0, 8.0);
    float rho = max(length(dFdx(st)), length(dFdy(st)));
    int lod = clamp(int(floor(log2(max(rho, 1e-6)))), 0, 4);
    ivec2 size = ivec2(max(16 >> lod, 1), max(8 >> lod, 1));
    int off = lod == 0 ? 0 : (lod == 1 ? 16 : (lod == 2 ? 24 : (lod == 3 ? 28 : 30)));
    ivec2 tc = clamp(ivec2(floor(uv * vec2(size))), ivec2(0), size - 1);
    return texelFetch(tex, ivec3(tc.x + off, tc.y, int(layer + 0.5)), 0);
}
"""

ARROW_VS = """
#version 430
uniform mat4 u_vp;
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv;
in vec3 i_pos; in vec4 i_quat; in float i_scale; in float i_var; in float i_fade;
out vec3 v_nrm; out vec2 v_uv; flat out float v_layer; flat out float v_fade;
""" + QROT + """
void main(){
    v_nrm = qrot(i_quat, in_nrm);
    v_uv = in_uv;
    v_layer = i_var;
    v_fade = i_fade;
    gl_Position = u_vp * vec4(qrot(i_quat, in_pos * i_scale) + i_pos, 1.0);
}
"""

ARROW_FS = """
#version 430
uniform sampler2DArray u_arrows;
in vec3 v_nrm; in vec2 v_uv; flat in float v_layer; flat in float v_fade;
layout(location=0) out vec4 o_albedo;
layout(location=1) out vec4 o_normal;
""" + ARROW_TEX + """
const float BAYER[16] = float[16](0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
void main(){
    vec4 t = arrow_tex(u_arrows, v_uv, v_layer);
    if (t.a < 0.5) discard;
    if (v_fade < 0.999) {                          // screen-door fade for arrows right next to the lens
        ivec2 q = ivec2(mod(gl_FragCoord.xy, 4.0));
        if (v_fade <= (BAYER[q.x + q.y * 4] + 0.5) / 16.0) discard;
    }
    vec3 n = normalize(v_nrm);
    if (!gl_FrontFacing) n = -n;                    // crossed quads are two-sided
    n = normalize(n + vec3(0.0, 0.0, 0.35));        // a little sky light on both planes, like the game
    o_albedo = vec4(sqrt(clamp(pow(t.rgb, vec3(2.2)), 0.0, 1.0)), 0.0);
    o_normal = vec4(n * 0.5 + 0.5, %d.0 / 255.0);
}
""" % MAT_ARROW

SHADOW_STATIC_VS = """
#version 430
uniform mat4 u_lvp;
in vec3 in_pos;
void main(){ gl_Position = u_lvp * vec4(in_pos, 1.0); }
"""
SHADOW_VOXEL_VS = """
#version 430
uniform mat4 u_lvp;
""" + GIANT_POSE + """
in vec3 in_pos; in float in_face; in vec3 i_pos; in vec4 i_quat; in float i_scale; in vec4 i_cy;
""" + QROT + """
void main(){
    int vis = int(i_cy.a * 255.0 + 0.5);
    if (((vis >> int(in_face + 0.5)) & 1) == 0) { gl_Position = vec4(0.0, 0.0, -10.0, 1.0); return; }
    vec3 p = qrot(i_quat, in_pos * i_scale) + i_pos;
    for (int g = 0; g < u_ng; g++)
        if (gl_InstanceID >= u_range[g].x && gl_InstanceID < u_range[g].y) {
            vec4 sq = u_gsq[g];
            if (sq.x != 1.0 || sq.y != 1.0) {
                vec3 d = p - u_gsqc[g];
                float h = clamp(d.z / max(sq.w, 1e-3), 0.0, 1.0);
                float bul = 1.0 + sq.z * 4.0 * h * (1.0 - h);
                p = u_gsqc[g] + vec3(d.xy * sq.y * bul, d.z * sq.x);
            }
            p = u_gcenter[g] + (p - u_gcenter[g]) * u_gfx[g].z;
            p = u_gp[g] + u_gt[g] + qrot(u_gq[g], p - u_gp[g]);
            vec3 dp = p - u_gsc[g];                  // spaghettified: the half facing the hole is drawn out
            float al = dot(dp, u_gs[g].xyz);           // towards it, thinner and thinner, like a noodle
            float fw = clamp(al / 4.0, 0.0, 1.0);
            p = u_gsc[g] + u_gs[g].xyz * (al * (al > 0.0 ? u_gs[g].w : 1.0))
                + (dp - u_gs[g].xyz * al) * (1.0 - (1.0 - pow(u_gs[g].w, -0.4)) * fw);
        }
    gl_Position = u_lvp * vec4(p, 1.0);
}
"""
SHADOW_PROP_VS = """
#version 430
uniform mat4 u_lvp;
in vec3 in_pos; in vec3 i_pos; in vec4 i_quat; in float i_scale;
""" + QROT + """
void main(){ gl_Position = u_lvp * vec4(qrot(i_quat, in_pos * i_scale) + i_pos, 1.0); }
"""

SHADOW_ARROW_VS = """
#version 430
uniform mat4 u_lvp;
in vec3 in_pos; in vec2 in_uv; in vec3 i_pos; in vec4 i_quat; in float i_scale; in float i_var;
out vec2 v_uv; flat out float v_layer;
""" + QROT + """
void main(){ v_uv = in_uv; v_layer = i_var; gl_Position = u_lvp * vec4(qrot(i_quat, in_pos * i_scale) + i_pos, 1.0); }
"""
SHADOW_ARROW_FS = """
#version 430
uniform sampler2DArray u_arrows;
in vec2 v_uv; flat in float v_layer;
""" + ARROW_TEX + """
void main(){ if (arrow_tex(u_arrows, v_uv, v_layer).a < 0.5) discard; }
"""
DEPTH_FS = """
#version 430
void main(){}
"""

FULLSCREEN_VS = """
#version 430
in vec2 in_pos;
out vec2 v_uv;
void main(){ v_uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

SSAO_FS = """
#version 430
uniform sampler2D u_depth;
uniform sampler2D u_normal;
uniform mat4 u_proj;
uniform mat4 u_invproj;
uniform mat3 u_viewrot;
uniform vec2 u_res;
uniform float u_radius;
in vec2 v_uv;
out float o_ao;
vec3 vpos(vec2 uv){
    float d = texture(u_depth, uv).r;
    vec4 p = u_invproj * vec4(uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
    return p.xyz / p.w;
}
const int NS = 16;
void main(){
    float d = texture(u_depth, v_uv).r;
    if (d >= 1.0) { o_ao = 1.0; return; }
    vec3 P = vpos(v_uv);
    vec3 N = normalize(u_viewrot * (texture(u_normal, v_uv).xyz * 2.0 - 1.0));
    float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    float ang = ign * 6.2831853;
    vec3 rv = vec3(cos(ang), sin(ang), 0.0);
    vec3 T = normalize(rv - N * dot(rv, N));
    if (any(isnan(T))) T = normalize(cross(N, vec3(0.0, 1.0, 0.0)));
    vec3 B = cross(N, T);
    // scale radius a bit with distance so far geometry still gets some AO
    float rad = u_radius * clamp(-P.z / 45.0, 0.6, 2.5);
    float occ = 0.0;
    for (int i = 0; i < NS; i++){
        float fi = float(i) + ign;
        float r = sqrt((fi + 0.5) / float(NS));
        float th = fi * 2.399963;
        vec2 dk = vec2(cos(th), sin(th)) * r;
        float h = sqrt(max(0.0, 1.0 - r * r));
        float sc = mix(0.15, 1.0, (fi / float(NS)) * (fi / float(NS)));
        vec3 s = (T * dk.x + B * dk.y + N * h) * rad * sc;
        vec3 Q = P + s;
        vec4 q = u_proj * vec4(Q, 1.0);
        vec2 quv = q.xy / q.w * 0.5 + 0.5;
        if (quv.x < 0.0 || quv.x > 1.0 || quv.y < 0.0 || quv.y > 1.0) continue;
        float sz = vpos(quv).z;
        float range = smoothstep(0.0, 1.0, rad / max(abs(P.z - sz), 1e-4));
        occ += (sz >= Q.z + 0.02 * rad ? 1.0 : 0.0) * range;
    }
    o_ao = clamp(1.0 - occ / float(NS), 0.0, 1.0);
}
"""

BLUR_FS = """
#version 430
uniform sampler2D u_src;
uniform sampler2D u_depth;
uniform vec2 u_dir;
uniform vec2 u_texel;
uniform float u_near;
uniform float u_far;
in vec2 v_uv;
out float o_v;
float lin(float d){ float z = d * 2.0 - 1.0; return 2.0 * u_near * u_far / (u_far + u_near - z * (u_far - u_near)); }
void main(){
    float dc = lin(texture(u_depth, v_uv).r);
    float sum = 0.0, wsum = 0.0;
    for (int i = -4; i <= 4; i++){
        vec2 uv = v_uv + u_dir * u_texel * float(i);
        float dz = lin(texture(u_depth, uv).r);
        float w = exp(-float(i * i) / 12.0) * exp(-abs(dz - dc) / (0.02 * dc + 0.05));
        sum += texture(u_src, uv).r * w;
        wsum += w;
    }
    o_v = sum / max(wsum, 1e-5);
}
"""

LIGHT_FS = """
#version 430
uniform sampler2D u_albedo;
uniform sampler2D u_normal;
uniform sampler2D u_depth;
uniform sampler2D u_ao;
uniform sampler2DShadow u_sh_near;
uniform sampler2D u_sh_near_raw;
uniform sampler2DShadow u_sh_far;
uniform sampler2D u_sky;
uniform mat4 u_invvp;
uniform vec3 u_cam;
uniform mat4 u_lvp_near;
uniform mat4 u_lvp_far;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_col;
uniform vec3 u_sky_amb;
uniform vec3 u_gnd_amb;
uniform float u_fog;
uniform vec3 u_fog_col;      // studio: fog colour (negative r = the outdoor sky haze)
uniform vec3 u_bounce;       // light bounced sideways off the floor
uniform float u_near_half;
uniform float u_el_min;
uniform float u_shadow_res;
uniform int u_nl;
uniform vec4 u_lpos[16];
uniform vec4 u_lcol[16];
uniform float u_glow;       // emission strength of glowing materials (pulses with the Warden's heartbeat)
uniform float u_glow_dim;   // 1 while he lives; glowing parts go dull when he dies
uniform vec3 u_sky_mul;     // the sky darkens and reddens as the black hole grows
uniform float u_hot;        // emission of whatever the accretion disk has heated
in vec2 v_uv;
out vec4 o_col;

const vec2 POISSON[12] = vec2[](
    vec2(-0.326, -0.406), vec2(-0.840, -0.074), vec2(-0.696, 0.457), vec2(-0.203, 0.621),
    vec2(0.962, -0.195), vec2(0.473, -0.480), vec2(0.519, 0.767), vec2(0.185, -0.893),
    vec2(0.507, 0.064), vec2(0.896, 0.412), vec2(-0.322, -0.933), vec2(-0.792, -0.598));

vec3 clear_sky(vec3 d){
    float t = clamp(d.z, 0.0, 1.0);
    float k = pow(t, 0.42);
    vec3 s = mix(vec3(0.62, 0.78, 0.98), vec3(0.10, 0.28, 0.78), k);
    float mu = max(dot(d, u_sun_dir), 0.0);
    s += (0.22 * pow(mu, 6.0)) * vec3(1.0, 0.92, 0.78);
    float hz = exp(-max(d.z, 0.0) * 9.0);
    return s * (1.0 - 0.35 * hz) + 0.35 * hz * vec3(0.80, 0.88, 0.98);
}

vec3 sky(vec3 d){
    float phi = atan(d.y, d.x);
    float u = fract(phi / 6.2831853);
    float el = degrees(asin(clamp(d.z, -1.0, 1.0)));
    float v = (90.0 - el) / (90.0 - u_el_min);
    return textureLod(u_sky, vec2(u, clamp(v, 0.0, 1.0)), 0.0).rgb;
}

float shadow_near(vec3 wp, vec3 n, float ign){
    vec3 p = wp + n * 0.035;
    vec4 l = u_lvp_near * vec4(p, 1.0);
    vec3 s = l.xyz / l.w * 0.5 + 0.5;
    if (any(lessThan(s.xy, vec2(0.002))) || any(greaterThan(s.xy, vec2(0.998)))) return -1.0;
    float texel = 1.0 / u_shadow_res;
    float ca = cos(ign * 6.283), sa = sin(ign * 6.283);
    mat2 rot = mat2(ca, sa, -sa, ca);
    // blocker search
    float searchR = 28.0 * texel;
    float bsum = 0.0; float bn = 0.0;
    for (int i = 0; i < 8; i++){
        float bd = texture(u_sh_near_raw, s.xy + rot * POISSON[i] * searchR).r;
        if (bd < s.z - 0.0006) { bsum += bd; bn += 1.0; }
    }
    if (bn < 0.5) return 1.0;
    float avg = bsum / bn;
    float depth_range = 420.0;
    float dist = (s.z - avg) * depth_range;           // world units between blocker and receiver
    float pen = clamp(dist * 0.022 + 0.035, 0.035, 0.9);  // penumbra width in world units
    float r = pen / (2.0 * u_near_half);                 // in shadow-map uv
    r = max(r, 1.2 * texel);
    float sum = 0.0;
    for (int i = 0; i < 12; i++){
        sum += texture(u_sh_near, vec3(s.xy + rot * POISSON[i] * r, s.z - 0.00012));
    }
    return sum / 12.0;
}

float shadow_far(vec3 wp, vec3 n, float ign){
    vec3 p = wp + n * 0.18;
    vec4 l = u_lvp_far * vec4(p, 1.0);
    vec3 s = l.xyz / l.w * 0.5 + 0.5;
    if (any(lessThan(s.xy, vec2(0.0))) || any(greaterThan(s.xy, vec2(1.0)))) return 1.0;
    float ca = cos(ign * 6.283), sa = sin(ign * 6.283);
    mat2 rot = mat2(ca, sa, -sa, ca);
    float r = 2.2 / u_shadow_res;
    float sum = 0.0;
    for (int i = 0; i < 8; i++) sum += texture(u_sh_far, vec3(s.xy + rot * POISSON[i] * r, s.z - 0.0004));
    return sum / 8.0;
}

void main(){
    float d = texture(u_depth, v_uv).r;
    vec4 hp = u_invvp * vec4(v_uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
    vec3 wp = hp.xyz / hp.w;
    vec4 fp = u_invvp * vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 vdir = normalize(fp.xyz / fp.w - u_cam);
    if (d >= 1.0) { o_col = vec4(sky(vdir) * u_sky_mul, 1.0); return; }

    vec4 nm = texture(u_normal, v_uv);
    vec3 N = normalize(nm.xyz * 2.0 - 1.0);
    int mat = int(nm.a * 255.0 + 0.5);
    vec4 al = texture(u_albedo, v_uv);
    vec3 A = al.rgb * al.rgb;
    if (mat == %d) A *= u_glow_dim;
    float flash = 0.0, heat = 0.0;
    float spec = al.a;
    if (mat == %d) { flash = spec; spec = 0.0; }
    if (mat == %d) { heat = spec; spec = 0.0; }
    float ao = texture(u_ao, v_uv).r;
    float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));

    float sh = shadow_near(wp, N, ign);
    float farw = 0.0;
    if (sh < 0.0) { sh = shadow_far(wp, N, ign); }
    else {
        // blend towards far cascade near the border of the near one
        vec4 l = u_lvp_near * vec4(wp, 1.0);
        vec2 s = abs(l.xy / l.w);
        farw = smoothstep(0.86, 0.98, max(s.x, s.y));
        if (farw > 0.0) sh = mix(sh, min(sh, shadow_far(wp, N, ign)), farw);
    }
    // static far cascade also darkens near objects under tree canopies etc.
    float ndl = dot(N, u_sun_dir);
    float diff = max(ndl, 0.0);
    vec3 V = -vdir;
    vec3 col;
    float hemi = N.z * 0.5 + 0.5;
    vec3 amb = mix(u_gnd_amb, u_sky_amb, hemi);
    // cheap bounce: faces looking sideways get a touch of warm light from the sunlit ground
    amb += u_bounce * (1.0 - abs(N.z));
    if (mat == %d) {
        // leaves: soft wrap lighting + translucency
        float wrap = max((ndl + 0.45) / 1.45, 0.0);
        col = A * (u_sun_col * wrap * sh + amb * ao * 1.05);
        col += A * u_sun_col * 0.18 * max(-ndl, 0.0) * sh;
    } else {
        col = A * (u_sun_col * diff * sh + amb * ao);
    }
    if (spec > 0.0) {
        vec3 H = normalize(u_sun_dir + V);
        float fres = 0.04 + 0.96 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
        col += u_sun_col * spec * fres * 3.0 * pow(max(dot(N, H), 0.0), 40.0) * sh;     // wet flesh sheen
        col += spec * 0.5 * fres * sky(reflect(-V, N)) * ao;
    }
    // point lights (explosions)
    vec3 pl = vec3(0.0);
    for (int i = 0; i < u_nl; i++){
        vec3 Lv = u_lpos[i].xyz - wp;
        float d = length(Lv);
        float rad = u_lpos[i].w;
        if (d < rad) {
            float att = (1.0 - d / rad) * (1.0 - d / rad) / (1.0 + d * d * 0.015);
            float nd = max(dot(N, Lv / max(d, 1e-3)), 0.0) * 0.8 + 0.2;
            pl += u_lcol[i].rgb * att * nd;
        }
    }
    col += A * pl;
    if (mat == %d) col += A * u_glow;
    if (heat > 0.0) {
        // heated by the accretion disk: glows red, then orange, then yellow-white
        vec3 hc = mix(vec3(0.9, 0.16, 0.02), vec3(1.0, 0.38, 0.06), heat * heat);
        float tex = 0.35 + 1.3 * dot(A, vec3(0.3333));          // keep the block's texture as embers
        col = mix(col, hc * tex * u_hot * (0.3 + 0.7 * heat), clamp(heat * 1.4, 0.0, 1.0));
    }
    col = mix(col, vec3(1.35), flash);                 // primed TNT / creeper: unlit white flash, like the game
    // aerial perspective
    float dist = length(wp - u_cam);
    float hfac = exp(-max(wp.z, 0.0) / 60.0);
    float f = 1.0 - exp(-dist * u_fog * mix(0.7, 1.0, hfac));
    vec3 hz = u_fog_col.r >= 0.0 ? u_fog_col : clear_sky(normalize(vec3(vdir.xy, 0.035))) * u_sky_mul;
    col = mix(col, hz, clamp(f, 0.0, 1.0) * (1.0 - 0.8 * heat));
    o_col = vec4(col, 1.0);
}
""" % (MAT_GLOW, MAT_TNT, MAT_HOT, MAT_LEAF, MAT_GLOW)

POST_FS = """
#version 430
uniform sampler2D u_hdr;
uniform float u_exposure;
uniform float u_sat;
uniform float u_contrast;
uniform float u_vignette;
uniform sampler2D u_bloom1;
uniform sampler2D u_bloom2;
uniform float u_bloom;
uniform vec2 u_res;
uniform float u_frame;
in vec2 v_uv;
out vec4 o_col;
vec3 aces(vec3 x){
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
void main(){
    vec3 c = (texture(u_hdr, v_uv).rgb + u_bloom * (texture(u_bloom1, v_uv).rgb + 0.8 * texture(u_bloom2, v_uv).rgb))
             * u_exposure;
    c = aces(c);
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(l), c, u_sat);
    // evening grade: cool teal shadows, warm highlights
    c *= mix(vec3(0.92, 1.0, 1.07), vec3(1.07, 1.0, 0.89), smoothstep(0.12, 0.72, l));
    c = clamp((c - 0.5) * u_contrast + 0.5, 0.0, 1.0);
    vec2 q = v_uv - 0.5;
    q.x *= u_res.x / u_res.y;
    c *= 1.0 - u_vignette * smoothstep(0.35, 0.95, length(q) * 1.3);
    c = pow(c, vec3(1.0 / 2.2));
    // luma in alpha for FXAA
    o_col = vec4(c, dot(c, vec3(0.299, 0.587, 0.114)));
}
"""

FXAA_FS = """
#version 430
uniform sampler2D u_src;
uniform vec2 u_texel;
uniform float u_frame;
in vec2 v_uv;
out vec4 o_col;
// FXAA 3.11 style (simplified quality preset)
float L(vec2 uv){ return texture(u_src, uv).a; }
void main(){
    vec2 uv = v_uv;
    vec3 rgbM = texture(u_src, uv).rgb;
    float lM = L(uv);
    float lN = L(uv + vec2(0.0, u_texel.y));
    float lS = L(uv - vec2(0.0, u_texel.y));
    float lE = L(uv + vec2(u_texel.x, 0.0));
    float lW = L(uv - vec2(u_texel.x, 0.0));
    float lmin = min(lM, min(min(lN, lS), min(lE, lW)));
    float lmax = max(lM, max(max(lN, lS), max(lE, lW)));
    float range = lmax - lmin;
    vec3 outc = rgbM;
    if (range >= max(0.0312, lmax * 0.125)) {
        float lNW = L(uv + vec2(-u_texel.x, u_texel.y));
        float lNE = L(uv + u_texel);
        float lSW = L(uv - u_texel);
        float lSE = L(uv + vec2(u_texel.x, -u_texel.y));
        float edgeH = abs(-2.0 * lW + lNW + lSW) + 2.0 * abs(-2.0 * lM + lN + lS) + abs(-2.0 * lE + lNE + lSE);
        float edgeV = abs(-2.0 * lS + lSW + lSE) + 2.0 * abs(-2.0 * lM + lW + lE) + abs(-2.0 * lN + lNW + lNE);
        bool horz = edgeH >= edgeV;
        float l1 = horz ? lS : lW;
        float l2 = horz ? lN : lE;
        float g1 = l1 - lM, g2 = l2 - lM;
        bool steep1 = abs(g1) >= abs(g2);
        float gs = 0.25 * max(abs(g1), abs(g2));
        float stepL = horz ? u_texel.y : u_texel.x;
        float lavg;
        if (steep1) { stepL = -stepL; lavg = 0.5 * (l1 + lM); } else { lavg = 0.5 * (l2 + lM); }
        vec2 cuv = uv;
        if (horz) cuv.y += stepL * 0.5; else cuv.x += stepL * 0.5;
        vec2 off = horz ? vec2(u_texel.x, 0.0) : vec2(0.0, u_texel.y);
        vec2 uv1 = cuv - off, uv2 = cuv + off;
        float e1 = L(uv1) - lavg, e2 = L(uv2) - lavg;
        bool r1 = abs(e1) >= gs, r2 = abs(e2) >= gs;
        const float QS[10] = float[](1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 4.0, 8.0, 8.0);
        for (int i = 0; i < 10 && !(r1 && r2); i++){
            if (!r1) { uv1 -= off * QS[i]; e1 = L(uv1) - lavg; r1 = abs(e1) >= gs; }
            if (!r2) { uv2 += off * QS[i]; e2 = L(uv2) - lavg; r2 = abs(e2) >= gs; }
        }
        float d1 = horz ? uv.x - uv1.x : uv.y - uv1.y;
        float d2 = horz ? uv2.x - uv.x : uv2.y - uv.y;
        bool dir1 = d1 < d2;
        float dmin = min(d1, d2);
        float elen = d1 + d2;
        float pixOff = -dmin / elen + 0.5;
        bool mLess = lM < lavg;
        bool good = ((dir1 ? e1 : e2) < 0.0) != mLess;
        float fo = good ? pixOff : 0.0;
        float lA = (2.0 * (lN + lS + lE + lW) + lNW + lNE + lSW + lSE) / 12.0;
        float sub = clamp(abs(lA - lM) / range, 0.0, 1.0);
        sub = (-2.0 * sub + 3.0) * sub * sub;
        fo = max(fo, sub * sub * 0.75);
        vec2 fuv = uv;
        if (horz) fuv.y += fo * stepL; else fuv.x += fo * stepL;
        outc = texture(u_src, fuv).rgb;
    }
    // triangular dither to kill banding after 8-bit + video encode
    float n1 = fract(sin(dot(gl_FragCoord.xy + u_frame * 0.618, vec2(12.9898, 78.233))) * 43758.5453);
    float n2 = fract(sin(dot(gl_FragCoord.xy + u_frame * 1.618 + 3.1, vec2(39.3468, 11.135))) * 24634.6345);
    outc += (n1 + n2 - 1.0) / 255.0;
    o_col = vec4(outc, 1.0);
}
"""

DOWN_FS = """
#version 430
uniform sampler2D u_src;
uniform vec2 u_src_texel;
uniform float u_scale;
in vec2 v_uv;
out vec4 o_col;
void main(){
    // tent filter over the source footprint of one destination pixel
    vec3 acc = vec3(0.0); float w = 0.0;
    float r = u_scale * 0.5 + 0.5;
    for (int y = -2; y <= 2; y++) for (int x = -2; x <= 2; x++){
        vec2 o = vec2(x, y) * 0.5 * u_scale;
        float wt = max(0.0, 1.0 - abs(o.x) / r) * max(0.0, 1.0 - abs(o.y) / r);
        acc += texture(u_src, v_uv + o * u_src_texel).rgb * wt;
        w += wt;
    }
    o_col = vec4(acc / w, 1.0);
}
"""


PARTICLE_VS = """
#version 430
uniform mat4 u_view;
uniform mat4 u_proj;
in vec2 in_corner;
in vec3 i_pos; in float i_size; in float i_alpha; in float i_fire; in float i_var; in float i_age;
out vec2 v_uv; out vec2 v_corner; out float v_vdepth;
flat out float v_alpha; flat out float v_fire; flat out float v_var;
void main(){
    vec4 vc = u_view * vec4(i_pos, 1.0);
    float ang = i_var * 1.9 + i_age * 0.8 + i_pos.x * 0.37;
    mat2 R = mat2(cos(ang), sin(ang), -sin(ang), cos(ang));
    vc.xy += R * in_corner * i_size * 0.5;
    v_vdepth = -vc.z;
    v_corner = in_corner;
    v_uv = in_corner * 0.5 + 0.5;
    v_alpha = i_alpha; v_fire = i_fire; v_var = i_var;
    gl_Position = u_proj * vc;
}
"""

PUFF_FS = """
#version 430
uniform sampler2DArray u_puff;
uniform sampler2D u_depth;
uniform vec2 u_res;
uniform float u_near;
uniform float u_far;
uniform vec3 u_light;       // smoke lighting colour (sun + sky)
uniform vec3 u_shade;       // colour on the side facing away from the sun
uniform vec2 u_sun2d;       // sun direction in screen space
uniform vec3 u_tint;
in vec2 v_uv; in vec2 v_corner; in float v_vdepth;
flat in float v_alpha; flat in float v_fire; flat in float v_var;
out vec4 o;
float lin(float d){ float z = d * 2.0 - 1.0; return 2.0 * u_near * u_far / (u_far + u_near - z * (u_far - u_near)); }
void main(){
    vec4 t = texture(u_puff, vec3(v_uv, v_var));
    float dens = t.r;
    float a = dens * v_alpha;
    float sd = lin(texture(u_depth, gl_FragCoord.xy / u_res).r);
    a *= clamp((sd - v_vdepth) / 1.5, 0.0, 1.0);
    if (a < 0.002) discard;
    // billowy self-shading: brighter towards the sun and on top, darker underneath, texture adds lumps
    float lit = clamp(0.42 + 0.45 * dot(v_corner, u_sun2d) + 0.22 * v_corner.y + 0.4 * (t.g - 0.5), 0.0, 1.0);
    lit = lit * lit * (3.0 - 2.0 * lit);
    vec3 smoke = mix(u_shade, u_light, lit) * u_tint;
    vec3 fire = mix(vec3(1.0, 0.30, 0.05), vec3(1.0, 0.78, 0.40), v_fire * t.g) * (9.0 * v_fire * v_fire);
    vec3 c = smoke * (1.0 - 0.6 * v_fire) + fire;
    o = vec4(c * a, a);
}
"""

FLASH_FS = """
#version 430
uniform sampler2D u_depth;
uniform vec2 u_res;
uniform float u_near;
uniform float u_far;
in vec2 v_uv; in vec2 v_corner; in float v_vdepth;
flat in float v_alpha; flat in float v_fire; flat in float v_var;
out vec4 o;
float lin(float d){ float z = d * 2.0 - 1.0; return 2.0 * u_near * u_far / (u_far + u_near - z * (u_far - u_near)); }
void main(){
    float r2 = dot(v_corner, v_corner);
    if (r2 > 1.0) discard;
    float sd = lin(texture(u_depth, gl_FragCoord.xy / u_res).r);
    float vis = clamp((sd - v_vdepth + 1.0) / 2.5, 0.0, 1.0);
    float core = exp(-r2 * 9.0) * 1.6 + exp(-r2 * 3.0) * 0.6;
    o = vec4(vec3(1.0, 0.82, 0.55) * 14.0 * core * v_alpha * vis, 0.0);
}
"""

RING_VS = """
#version 430
uniform mat4 u_vp;
in vec2 in_corner;
in vec3 i_pos; in vec3 i_axis; in float i_radius; in float i_alpha;
out vec2 v_local; flat out float v_alpha;
void main(){
    vec3 a = normalize(i_axis);
    vec3 t = abs(a.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 u = normalize(cross(a, t));
    vec3 v = cross(a, u);
    v_local = in_corner * 1.3;
    v_alpha = i_alpha;
    gl_Position = u_vp * vec4(i_pos + (u * v_local.x + v * v_local.y) * i_radius, 1.0);
}
"""

RING_FS = """
#version 430
uniform sampler2D u_depth;
uniform vec2 u_res;
uniform float u_near;
uniform float u_far;
uniform vec3 u_col;
in vec2 v_local; flat in float v_alpha;
out vec4 o;
float lin(float d){ float z = d * 2.0 - 1.0; return 2.0 * u_near * u_far / (u_far + u_near - z * (u_far - u_near)); }
void main(){
    float r = length(v_local);
    float ring = exp(-pow((r - 1.0) / 0.06, 2.0)) + 0.45 * exp(-pow((r - 0.9) / 0.2, 2.0));
    if (ring * v_alpha < 0.004) discard;
    float sd = lin(texture(u_depth, gl_FragCoord.xy / u_res).r);
    float vis = clamp((sd - lin(gl_FragCoord.z) + 0.4) / 0.8, 0.0, 1.0);
    o = vec4(u_col * ring * v_alpha * vis, 0.0);
}
"""

STREAK_VS = """
#version 430
uniform mat4 u_vp;
uniform mat4 u_view;
uniform vec3 u_cam;
in vec2 in_corner;                 // x: -1 head .. 1 tail, y: -1..1 across
in vec3 i_head; in vec3 i_tail; in float i_width; in float i_alpha;
out vec2 v_c; out float v_vdepth; flat out float v_alpha;
void main(){
    float t = in_corner.x * 0.5 + 0.5;
    vec3 P = mix(i_head, i_tail, t);
    vec3 side = cross(i_tail - i_head, P - u_cam);
    side /= max(length(side), 1e-6);
    P += side * in_corner.y * i_width * 0.5;
    v_c = vec2(t, in_corner.y);
    v_alpha = i_alpha;
    v_vdepth = -(u_view * vec4(P, 1.0)).z;
    gl_Position = u_vp * vec4(P, 1.0);
}
"""

STREAK_FS = """
#version 430
uniform sampler2D u_depth;
uniform vec2 u_res;
uniform float u_near;
uniform float u_far;
uniform vec3 u_col;
in vec2 v_c; in float v_vdepth; flat in float v_alpha;
out vec4 o;
float lin(float d){ float z = d * 2.0 - 1.0; return 2.0 * u_near * u_far / (u_far + u_near - z * (u_far - u_near)); }
void main(){
    float sd = lin(texture(u_depth, gl_FragCoord.xy / u_res).r);
    if (sd < v_vdepth - 0.05) discard;
    float a = v_alpha * (1.0 - v_c.x) * (1.0 - v_c.y * v_c.y);
    o = vec4(u_col * a, a);
}
"""

DISK_VS = """
#version 430
uniform mat4 u_vp;
uniform vec3 u_c;            // centre of the accretion disk
uniform vec3 u_n;            // its normal
uniform float u_rout;
in vec2 in_pos;
out vec2 v_local; out vec3 v_wp;
void main(){
    vec3 n = normalize(u_n);
    vec3 t = abs(n.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 a = normalize(cross(t, n));
    vec3 b = cross(n, a);
    v_local = in_pos * 1.02;
    v_wp = u_c + (a * v_local.x + b * v_local.y) * u_rout;
    gl_Position = u_vp * vec4(v_wp, 1.0);
}
"""

DISK_FS = """
#version 430
uniform sampler2D u_depth;
uniform vec2 u_res;
uniform float u_near;
uniform float u_far;
uniform vec3 u_cam;
uniform float u_rin, u_rout, u_rot, u_bright;
uniform vec2 u_beam;         // in-plane direction of the side moving towards the camera (Doppler beaming)
uniform float u_side;        // 0: only the half behind the hole (gets lensed), 1: only the half in front
uniform float u_hole_dist;
uniform vec3 u_n;
in vec2 v_local; in vec3 v_wp;
out vec4 o;
float lin(float d){ float z = d * 2.0 - 1.0; return 2.0 * u_near * u_far / (u_far + u_near - z * (u_far - u_near)); }
void main(){
    float r = length(v_local) * u_rout;
    if (r < u_rin * 0.97 || r > u_rout) discard;
    bool front = length(v_wp - u_cam) < u_hole_dist;
    if (front != (u_side > 0.5)) discard;
    float u = (r - u_rin) / (u_rout - u_rin);
    float phi = atan(v_local.y, v_local.x);
    float spin = u_rot * pow(max(u_rin, 0.05) / r, 1.5);
    float lr = log(r);
    float bands = 0.5 + 0.28 * sin(phi * 3.0 - spin + lr * 11.0) + 0.16 * sin(phi * 7.0 - spin * 1.4 + lr * 23.0)
                + 0.08 * sin(phi * 17.0 - spin * 2.1 + lr * 41.0);
    bands = pow(clamp(bands, 0.0, 1.2), 1.8) * 1.4;
    float edge = exp(-u / 0.06);                               // the thin white-hot inner edge
    float prof = (pow(1.0 - u, 2.8) * 0.75 + 1.25 * edge) * smoothstep(0.0, 0.03, u);
    vec3 hot = mix(vec3(1.0, 0.72, 0.42), vec3(1.0, 0.34, 0.05), smoothstep(0.0, 0.2, u));
    hot = mix(hot, vec3(0.42, 0.04, 0.02), smoothstep(0.2, 1.0, u));
    float beam = 1.0 + 0.5 * dot(normalize(v_local), u_beam);
    float I = u_bright * prof * bands * beam;
    // thin and glowing seen edge-on, faint seen face-on
    float cv = abs(dot(normalize(v_wp - u_cam), normalize(u_n)));
    I *= mix(1.15, 0.3, cv * cv);
    float sd = lin(texture(u_depth, gl_FragCoord.xy / u_res).r);
    float vis = clamp((sd - lin(gl_FragCoord.z) + 0.6) / 1.2, 0.0, 1.0);
    o = vec4(hot * I * vis, 0.0);
}
"""

LENS_FS = """
#version 430
uniform sampler2D u_hdr;
uniform sampler2D u_depth;
uniform sampler2D u_sky;
uniform mat4 u_invvp;
uniform mat4 u_vp;
uniform vec3 u_cam;
uniform vec3 u_bh;
uniform float u_rs;          // angular radius of the black shadow (radians)
uniform float u_re;          // Einstein angle (radians)
uniform float u_front;       // nothing closer to the camera than this is bent (or hidden)
uniform vec3 u_ring_col;
uniform vec3 u_sky_mul;
uniform float u_el_min;
uniform vec2 u_fade;         // the bending fades out between these many Einstein angles
in vec2 v_uv;
out vec4 o;
vec3 sky(vec3 d){
    float phi = atan(d.y, d.x);
    float u = fract(phi / 6.2831853);
    float el = degrees(asin(clamp(d.z, -1.0, 1.0)));
    float v = (90.0 - el) / (90.0 - u_el_min);
    return textureLod(u_sky, vec2(u, clamp(v, 0.0, 1.0)), 0.0).rgb;
}
float dist_at(vec2 uv){
    float d = texture(u_depth, uv).r;
    if (d >= 1.0) return 1e9;
    vec4 hp = u_invvp * vec4(uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
    return length(hp.xyz / hp.w - u_cam);
}
void main(){
    vec3 c0 = texture(u_hdr, v_uv).rgb;
    vec4 fp = u_invvp * vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 d = normalize(fp.xyz / fp.w - u_cam);
    vec3 h = normalize(u_bh - u_cam);
    float cth = clamp(dot(d, h), -1.0, 1.0);
    float th = acos(cth);
    if (cth < 0.0 || th > u_fade.y * u_re || dist_at(v_uv) < u_front) { o = vec4(c0, 1.0); return; }
    vec3 col = c0;
    if (th < u_rs) col = vec3(0.0);
    else {
        // point-mass lens: light seen at angle th comes from beta = th - re^2 / th (the other side if < 0);
        // the bending fades out between u_fade.x and u_fade.y Einstein angles
        float fade = 1.0 - smoothstep(u_fade.x * u_re, u_fade.y * u_re, th);
        float beta = th - fade * u_re * u_re / th;
        vec3 perp = d - h * cth;
        float pl = length(perp);
        vec3 ax = pl > 1e-6 ? perp / pl : vec3(0.0, 0.0, 1.0);
        vec3 sdir = normalize(h * cos(beta) + ax * sin(beta));
        vec4 cs = u_vp * vec4(u_cam + sdir * 900.0, 1.0);
        bool ok = false;
        if (cs.w > 0.0) {
            vec2 uv = cs.xy / cs.w * 0.5 + 0.5;
            if (all(greaterThan(uv, vec2(0.001))) && all(lessThan(uv, vec2(0.999))) && dist_at(uv) >= u_front) {
                col = texture(u_hdr, uv).rgb;
                ok = true;
            }
        }
        if (!ok) {
            // the light would come from behind something in front of the hole (or off screen)
            col = (cs.w > 0.0 && beta > 0.0) ? c0 : sky(sdir) * u_sky_mul;
        }
    }
    float x = (th - u_rs * 1.035) / (u_rs * 0.05);
    col += u_ring_col * exp(-x * x);
    o = vec4(col, 1.0);
}
"""

DOF_COC_FS = """
#version 430
uniform sampler2D u_hdr;
uniform sampler2D u_depth;
uniform float u_near;
uniform float u_far;
uniform float u_focus;       // focus distance (world units)
uniform float u_k;           // blur disc radius in pixels = u_k * |d - focus| / d
uniform float u_maxr;
in vec2 v_uv;
out vec4 o;
float lin(float d){ float z = d * 2.0 - 1.0; return 2.0 * u_near * u_far / (u_far + u_near - z * (u_far - u_near)); }
void main(){
    // colour and the signed circle of confusion (negative in front of the focus; it grows with depth)
    float d = lin(texture(u_depth, v_uv).r);
    float s = clamp(u_k * (d - u_focus) / max(d, 0.1), -4.0 * u_maxr, 4.0 * u_maxr);
    o = vec4(texture(u_hdr, v_uv).rgb, s);
}
"""

DOF_FS = """
#version 430
uniform sampler2D u_src;     // rgb + signed circle of confusion
uniform float u_maxr;        // cap (pixels)
uniform float u_step;        // spiral density: one sample per 2 pi u_step square pixels
uniform vec2 u_texel;
in vec2 v_uv;
out vec4 o;
void main(){
    vec4 c = texture(u_src, v_uv);
    vec3 col = c.rgb;
    float s0 = c.a;
    float c0 = min(u_maxr, abs(s0));
    float tot = 1.0;
    float radius = u_step;
    float band = 0.4 * u_step;
    for (float ang = 0.0; radius < u_maxr; ang += 2.39996323) {
        vec4 sm = texture(u_src, v_uv + vec2(cos(ang), sin(ang)) * u_texel * radius);
        float ss = min(u_maxr, abs(sm.a));
        if (sm.a > s0) ss = clamp(ss, 0.0, c0 * 2.0);        // background can't bleed over what's in front
        float m = smoothstep(radius - band, radius + band, ss);
        col += mix(col / tot, sm.rgb, m);
        tot += 1.0;
        radius += u_step / radius;
    }
    o = vec4(col / tot, 1.0);
}
"""

BRIGHT_FS = """
#version 430
uniform sampler2D u_src;
uniform vec2 u_texel;
uniform float u_thresh;
in vec2 v_uv;
out vec4 o;
void main(){
    vec3 c = vec3(0.0);
    for (int y = -1; y <= 1; y += 2) for (int x = -1; x <= 1; x += 2)
        c += texture(u_src, v_uv + vec2(x, y) * u_texel).rgb;
    c *= 0.25;
    float l = max(c.r, max(c.g, c.b));
    float k = max(l - u_thresh, 0.0) / max(l, 1e-4);
    o = vec4(min(c * k, vec3(60.0)), 1.0);
}
"""

GAUSS_FS = """
#version 430
uniform sampler2D u_src;
uniform vec2 u_dir;          // texel size * direction
in vec2 v_uv;
out vec4 o;
const float O[3] = float[](0.0, 1.3846153846, 3.2307692308);
const float W[3] = float[](0.2270270270, 0.3162162162, 0.0702702703);
void main(){
    vec3 c = texture(u_src, v_uv).rgb * W[0];
    for (int i = 1; i < 3; i++){
        c += texture(u_src, v_uv + u_dir * O[i]).rgb * W[i];
        c += texture(u_src, v_uv - u_dir * O[i]).rgb * W[i];
    }
    o = vec4(c, 1.0);
}
"""


class Renderer:
    def __init__(self, width, height, ss=1.0, shadow_res=4096, near_half=46.0, far_half=300.0):
        self.W, self.H = width, height
        self.ss = ss
        self.iw, self.ih = int(round(width * ss)), int(round(height * ss))
        self.shadow_res = shadow_res
        self.near_half = near_half
        self.far_half = far_half
        self.far_center = (0.0, 0.0, 10.0)                           # centre of the static far shadow map
        ctx = moderngl.create_standalone_context(backend='egl', require=430)
        self.ctx = ctx
        self.progs = {}
        self._build_programs()
        self.quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32).tobytes())
        self.fs_vaos = {}
        self._build_targets()
        cube = unit_cube()
        self.cube_vbo = ctx.buffer(cube.tobytes())
        self.static = None
        self.frame_counter = 0
        self.exposure = 0.74
        self.sat = 1.2
        self.contrast = 1.05
        self.vignette = 0.16
        self.fog = 0.0013
        self.sun_col = np.array([1.0, 0.64, 0.36]) * 3.3          # low golden sun
        self.sky_amb = np.array([0.40, 0.44, 0.70]) * 0.95
        self.gnd_amb = np.array([0.34, 0.31, 0.20]) * 0.85
        self.glow = 5.0                                              # emission of glowing materials
        self.dof = None                                              # dict(focus, k, maxr) or None
        self.fog_col = (-1.0, 0.0, 0.0)
        self.bounce = (0.06, 0.072, 0.03)
        self.sculk_r = 11.0                                          # radius of the sculk patch
        self.ring_col = np.array([0.12, 0.80, 1.0]) * 6.5
        self.ssao_radius = 1.4
        self.bloom = 0.3
        self.bloom_thresh = 3.2
        self.light_col = np.array([1.0, 0.58, 0.26]) * 16.0
        self.puff_tint = np.array([0.96, 0.80, 0.62])        # dust in warm evening light
        self.streak_col = np.array([0.85, 0.88, 0.95])
        self.corner_vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32).tobytes())
        self.region = None
        self.prop_kinds = {}

    # -- setup ------------------------------------------------------------------------------
    def _prog(self, name, vs, fs):
        self.progs[name] = self.ctx.program(vertex_shader=vs, fragment_shader=fs)
        return self.progs[name]

    def _build_programs(self):
        self._prog('static', STATIC_VS, STATIC_FS)
        self._prog('voxel', VOXEL_VS, VOXEL_FS)
        self._prog('prop', PROP_VS, PROP_FS)
        self._prog('arrow', ARROW_VS, ARROW_FS)
        self._prog('sh_arrow', SHADOW_ARROW_VS, SHADOW_ARROW_FS)
        self._prog('sh_static', SHADOW_STATIC_VS, DEPTH_FS)
        self._prog('sh_voxel', SHADOW_VOXEL_VS, DEPTH_FS)
        self._prog('sh_prop', SHADOW_PROP_VS, DEPTH_FS)
        self._prog('streak', STREAK_VS, STREAK_FS)
        self._prog('ring', RING_VS, RING_FS)
        self._prog('puff', PARTICLE_VS, PUFF_FS)
        self._prog('flash', PARTICLE_VS, FLASH_FS)
        self._prog('bright', FULLSCREEN_VS, BRIGHT_FS)
        self._prog('gauss', FULLSCREEN_VS, GAUSS_FS)
        self._prog('ssao', FULLSCREEN_VS, SSAO_FS)
        self._prog('blur', FULLSCREEN_VS, BLUR_FS)
        self._prog('light', FULLSCREEN_VS, LIGHT_FS)
        self._prog('post', FULLSCREEN_VS, POST_FS)
        self._prog('fxaa', FULLSCREEN_VS, FXAA_FS)
        self._prog('down', FULLSCREEN_VS, DOWN_FS)
        self._prog('disk', DISK_VS, DISK_FS)
        self._prog('lens', FULLSCREEN_VS, LENS_FS)
        self._prog('dof', FULLSCREEN_VS, DOF_FS)
        self._prog('dof_coc', FULLSCREEN_VS, DOF_COC_FS)

    def _fs(self, name):
        if name not in self.fs_vaos:
            self.fs_vaos[name] = self.ctx.vertex_array(self.progs[name], [(self.quad, '2f', 'in_pos')])
        return self.fs_vaos[name]

    def _build_targets(self):
        ctx = self.ctx
        iw, ih = self.iw, self.ih
        self.g_albedo = ctx.texture((iw, ih), 4)
        self.g_normal = ctx.texture((iw, ih), 4)
        self.g_depth = ctx.depth_texture((iw, ih))
        for t in (self.g_albedo, self.g_normal, self.g_depth):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            t.repeat_x = t.repeat_y = False
        self.g_depth.compare_func = ''      # read raw depth values in the shaders
        self.gbuf = ctx.framebuffer(color_attachments=[self.g_albedo, self.g_normal], depth_attachment=self.g_depth)
        hw, hh = max(1, iw // 2), max(1, ih // 2)
        self.ao_a = ctx.texture((hw, hh), 1, dtype='f2')
        self.ao_b = ctx.texture((hw, hh), 1, dtype='f2')
        for t in (self.ao_a, self.ao_b):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
        self.ao_fbo_a = ctx.framebuffer(color_attachments=[self.ao_a])
        self.ao_fbo_b = ctx.framebuffer(color_attachments=[self.ao_b])
        self.hdr = ctx.texture((iw, ih), 4, dtype='f2')
        self.hdr.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hdr.repeat_x = self.hdr.repeat_y = False
        self.hdr_fbo = ctx.framebuffer(color_attachments=[self.hdr])
        self.hdr2 = ctx.texture((iw, ih), 4, dtype='f2')        # after gravitational lensing
        self.hdr2.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hdr2.repeat_x = self.hdr2.repeat_y = False
        self.hdr2_fbo = ctx.framebuffer(color_attachments=[self.hdr2])
        self.out_hdr = self.hdr
        self.ldr = ctx.texture((iw, ih), 4)
        self.ldr.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.ldr.repeat_x = self.ldr.repeat_y = False
        self.ldr_fbo = ctx.framebuffer(color_attachments=[self.ldr])
        self.aa = ctx.texture((iw, ih), 4)
        self.aa.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.aa.repeat_x = self.aa.repeat_y = False
        self.aa_fbo = ctx.framebuffer(color_attachments=[self.aa])
        self.out = ctx.texture((self.W, self.H), 4)
        self.out_fbo = ctx.framebuffer(color_attachments=[self.out])
        self.bloom_tex = []
        for div in (4, 8):
            pair = []
            for _ in range(2):
                t = ctx.texture((max(1, iw // div), max(1, ih // div)), 4, dtype='f2')
                t.filter = (moderngl.LINEAR, moderngl.LINEAR)
                t.repeat_x = t.repeat_y = False
                pair.append((t, ctx.framebuffer(color_attachments=[t])))
            self.bloom_tex.append(pair)
        S = self.shadow_res
        self.sh_near = ctx.depth_texture((S, S))
        self.sh_far = ctx.depth_texture((S, S))
        for t in (self.sh_near, self.sh_far):
            t.repeat_x = t.repeat_y = False
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.sh_near_fbo = ctx.framebuffer(depth_attachment=self.sh_near)
        self.sh_far_fbo = ctx.framebuffer(depth_attachment=self.sh_far)
        self.smp_near_cmp = ctx.sampler(texture=self.sh_near, filter=(moderngl.LINEAR, moderngl.LINEAR),
                                        compare_func='<=', repeat_x=False, repeat_y=False)
        self.smp_near_raw = ctx.sampler(texture=self.sh_near, filter=(moderngl.NEAREST, moderngl.NEAREST),
                                        compare_func='', repeat_x=False, repeat_y=False)
        self.smp_far_cmp = ctx.sampler(texture=self.sh_far, filter=(moderngl.LINEAR, moderngl.LINEAR),
                                       compare_func='<=', repeat_x=False, repeat_y=False)

    def set_textures(self, block_tex_list, sky, tint_noise, puffs):
        """block_tex_list: 16x16 RGB or RGBA block textures (alpha = glow mask)."""
        ctx = self.ctx
        blocks = []
        for t in block_tex_list:
            t = np.clip(t, 0, 255).astype(np.uint8)
            if t.shape[-1] == 3:
                t = np.concatenate([t, np.zeros(t.shape[:2] + (1,), np.uint8)], -1)
            blocks.append(t)
        blocks = np.stack(blocks)                                                       # (L,16,16,4)
        n = blocks.shape[0]
        self.t_blocks = ctx.texture_array((16, 16, n), 4, blocks.tobytes())
        self.t_blocks.build_mipmaps()
        self.t_blocks.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.NEAREST)
        self.t_blocks.anisotropy = 1.0
        pf = np.ascontiguousarray(puffs, np.float32)                                # (4, 128, 128, 2)
        self.t_puff = ctx.texture_array((pf.shape[2], pf.shape[1], pf.shape[0]), 2, pf.tobytes(), dtype='f4')
        self.t_puff.build_mipmaps()
        self.t_puff.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.t_puff.repeat_x = self.t_puff.repeat_y = False
        h, w = sky.shape[:2]
        self.t_sky = ctx.texture((w, h), 3, np.ascontiguousarray(sky, np.float32).tobytes(), dtype='f4')
        self.t_sky.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.t_sky.repeat_x = True
        self.t_sky.repeat_y = False
        tn = np.ascontiguousarray(tint_noise, np.float32)
        self.t_tint = ctx.texture(tn.shape[::-1], 1, tn.tobytes(), dtype='f4')
        self.t_tint.filter = (moderngl.LINEAR, moderngl.LINEAR)

    def set_static(self, vertices, indices, chunks, ground_half=1600.0, hole=None):
        ctx = self.ctx
        self.static_vbo = ctx.buffer(np.ascontiguousarray(vertices, np.float32).tobytes())
        self.static_ibo = ctx.buffer(np.ascontiguousarray(indices, np.uint32).tobytes())
        self.static_chunks = chunks
        fmt = (self.static_vbo, '3f 3f 2f 1f 3f', 'in_pos', 'in_nrm', 'in_uv', 'in_layer', 'in_tint')
        self.static_vao = ctx.vertex_array(self.progs['static'], [fmt], index_buffer=self.static_ibo, index_element_size=4)
        self.static_sh_vao = ctx.vertex_array(self.progs['sh_static'], [(self.static_vbo, '3f 36x', 'in_pos')],
                                              index_buffer=self.static_ibo, index_element_size=4)
        g = ground_half
        h = hole
        if h is None:
            rects = [(-g, -g, g, g)]
        else:
            rects = [(-g, -g, g, -h), (-g, h, g, g), (-g, -h, -h, h), (h, -h, g, h)]   # frame around a hole
        verts, inds = [], []
        for (x0, y0, x1, y1) in rects:
            b = len(verts)
            for (x, y) in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                verts.append((x, y, 0, 0, 0, 1, x, -y, 0, 1, 1, 1))
            inds.extend((b, b + 1, b + 2, b, b + 2, b + 3))
        gv = np.array(verts, np.float32)
        gi = np.array(inds, np.uint32)
        self.ground_vbo = ctx.buffer(gv.tobytes())
        self.ground_ibo = ctx.buffer(gi.tobytes())
        self.ground_vao = ctx.vertex_array(self.progs['static'],
                                           [(self.ground_vbo, '3f 3f 2f 1f 3f', 'in_pos', 'in_nrm', 'in_uv', 'in_layer', 'in_tint')],
                                           index_buffer=self.ground_ibo, index_element_size=4)
        # static far shadow map (once)
        self.lvp_far = light_matrix(self.far_center, self.far_half, depth=700.0)
        self.sh_far_fbo.use()
        self.sh_far_fbo.clear(depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        ctx.polygon_offset = (2.0, 4.0)
        self.progs['sh_static']['u_lvp'].write(m4(self.lvp_far))
        self.static_sh_vao.render()
        ctx.polygon_offset = (0.0, 0.0)
        # which chunks can cast into the near cascade
        self.near_chunks = [c for c in chunks if np.hypot(*np.clip(0, c[2][:2], c[3][:2])) < self.near_half + 40]

    def set_ground_region(self, vertices, indices):
        """Upload the destructible ground mesh (same vertex layout as the static terrain)."""
        ctx = self.ctx
        if self.region is not None:
            for o in self.region:
                o.release()
        vbo = ctx.buffer(np.ascontiguousarray(vertices, np.float32).tobytes())
        ibo = ctx.buffer(np.ascontiguousarray(indices, np.uint32).tobytes())
        vao = ctx.vertex_array(self.progs['static'], [(vbo, '3f 3f 2f 1f 3f', 'in_pos', 'in_nrm', 'in_uv', 'in_layer',
                                                        'in_tint')], index_buffer=ibo, index_element_size=4)
        svao = ctx.vertex_array(self.progs['sh_static'], [(vbo, '3f 36x', 'in_pos')], index_buffer=ibo,
                                index_element_size=4)
        self.region = (vao, svao, vbo, ibo)

    # -- per frame ------------------------------------------------------------------------------
    def _voxel_vaos(self, inst_buf):
        ctx = self.ctx
        g = ctx.vertex_array(self.progs['voxel'], [
            (self.cube_vbo, '3f 3f 1f', 'in_pos', 'in_nrm', 'in_face'),
            (inst_buf, '3f 4f 1f 4f1 4f1 4f1 4f1/i', 'i_pos', 'i_quat', 'i_scale', 'i_cx', 'i_cy', 'i_cz', 'i_inner'),
        ])
        s = ctx.vertex_array(self.progs['sh_voxel'], [
            (self.cube_vbo, '3f 12x 1f', 'in_pos', 'in_face'),
            (inst_buf, '3f 4f 1f 4x 4f1 8x/i', 'i_pos', 'i_quat', 'i_scale', 'i_cy'),
        ])
        return g, s

    def add_prop_kind(self, name, mesh, layers, sheen=0.22):
        """A kind of instanced prop: triangles (pos3 normal3 uv2 layer1), 16x16 RGBA texture layers and the
        material: sheen >= 0 is lit iron with that much gloss, sheen < 0 is TNT (instances flash white)."""
        ctx = self.ctx
        vbo = ctx.buffer(np.ascontiguousarray(mesh, np.float32).tobytes())
        pl = np.stack([np.ascontiguousarray(a, np.uint8) for a in layers])
        tex = ctx.texture_array((pl.shape[2], pl.shape[1], pl.shape[0]), 4, pl.tobytes())
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.NEAREST)
        self.prop_kinds[name] = {'vbo': vbo, 'tex': tex, 'sheen': float(sheen)}

    def _prop_vaos(self, kind, inst_buf):
        ctx = self.ctx
        k = self.prop_kinds[kind]
        g = ctx.vertex_array(self.progs['prop'], [
            (k['vbo'], '3f 3f 2f 1f', 'in_pos', 'in_nrm', 'in_uv', 'in_layer'),
            (inst_buf, '3f 4f 1f 1f 1f 1f/i', 'i_pos', 'i_quat', 'i_scale', 'i_var', 'i_fade', 'i_flash'),
        ])
        s = ctx.vertex_array(self.progs['sh_prop'], [
            (k['vbo'], '3f 24x', 'in_pos'),
            (inst_buf, '3f 4f 1f 12x/i', 'i_pos', 'i_quat', 'i_scale'),
        ])
        return g, s

    def set_arrow_textures(self, layers):
        al = np.stack([np.ascontiguousarray(a, np.uint8) for a in layers])            # (V, 8, 32, 4) mip atlases
        self.t_arrows = self.ctx.texture_array((al.shape[2], al.shape[1], al.shape[0]), 4, al.tobytes())
        self.t_arrows.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.t_arrows.repeat_x = self.t_arrows.repeat_y = False

    def set_arrow_mesh(self, mesh):
        self.arrow_vbo = self.ctx.buffer(np.ascontiguousarray(mesh, np.float32).tobytes())

    def _arrow_vaos(self, inst_buf):
        ctx = self.ctx
        g = ctx.vertex_array(self.progs['arrow'], [
            (self.arrow_vbo, '3f 3f 2f', 'in_pos', 'in_nrm', 'in_uv'),
            (inst_buf, '3f 4f 1f 1f 1f/i', 'i_pos', 'i_quat', 'i_scale', 'i_var', 'i_fade'),
        ])
        s = ctx.vertex_array(self.progs['sh_arrow'], [
            (self.arrow_vbo, '3f 12x 2f', 'in_pos', 'in_uv'),
            (inst_buf, '3f 4f 1f 1f 4x/i', 'i_pos', 'i_quat', 'i_scale', 'i_var'),
        ])
        return g, s

    def _giant_uniforms(self, giants):
        """giants: list of (start, end, hurt, white, swell, centre3[, pose]) for the giants' instance ranges;
        pose: dict(q, t, pivot, axis, stretch, scentre, heat) moves, turns and stretches the whole giant."""
        rng = np.zeros((MAX_GIANTS, 2), np.int32)
        fx = np.zeros((MAX_GIANTS, 4), np.float32)
        fx[:, 2] = 1.0
        cen = np.zeros((MAX_GIANTS, 3), np.float32)
        gq = np.zeros((MAX_GIANTS, 4), np.float32)
        gq[:, 3] = 1.0
        gt = np.zeros((MAX_GIANTS, 3), np.float32)
        gp = np.zeros((MAX_GIANTS, 3), np.float32)
        gs = np.zeros((MAX_GIANTS, 4), np.float32)
        gs[:, 2] = 1.0
        gs[:, 3] = 1.0
        gsc = np.zeros((MAX_GIANTS, 3), np.float32)
        gsq = np.zeros((MAX_GIANTS, 4), np.float32)
        gsq[:, 0] = 1.0
        gsq[:, 1] = 1.0
        gsq[:, 3] = 1.0
        gsqc = np.zeros((MAX_GIANTS, 3), np.float32)
        n = 0
        for g in (giants or [])[:MAX_GIANTS]:
            a, b, hurt, white, swell, c = g[:6]
            pose = g[6] if len(g) > 6 else None
            rng[n] = (a, b)
            fx[n] = (hurt, white, swell, 0.0)
            cen[n] = c
            if pose:
                gq[n] = pose['q']
                gt[n] = pose['t']
                gp[n] = pose['pivot']
                gs[n, :3] = pose['axis']
                gs[n, 3] = pose['stretch']
                gsc[n] = pose['scentre']
                fx[n, 3] = pose.get('heat', 0.0)
                if 'squash' in pose:
                    sz_, sxy_, bul_, h_, c_ = pose['squash']
                    gsq[n] = (sz_, sxy_, bul_, h_)
                    gsqc[n] = c_
            n += 1
        for name in ('voxel', 'sh_voxel'):
            P = self.progs[name]
            P['u_ng'] = n
            P['u_range'].write(rng.tobytes())
            P['u_gfx'].write(fx.tobytes())
            P['u_gcenter'].write(cen.tobytes())
            P['u_gq'].write(gq.tobytes())
            P['u_gt'].write(np.ascontiguousarray(gt).tobytes())
            P['u_gp'].write(np.ascontiguousarray(gp).tobytes())
            P['u_gs'].write(gs.tobytes())
            P['u_gsc'].write(np.ascontiguousarray(gsc).tobytes())
            P['u_gsq'].write(gsq.tobytes())
            P['u_gsqc'].write(np.ascontiguousarray(gsqc).tobytes())

    def render(self, cam, voxels=None, props=None, arrows=None, fx=None, near_center=(0.0, 0.0, 14.0),
               giants=None, glow=None, glow_dim=1.0, sculk=(24.0, 0.0), bh=None):
        """cam: dict(eye, target, fov, up(optional), roll(optional)).
        voxels: VOXEL_DTYPE instances; props: float32 (M, 10) pos3 quat4 scale1 variant1 fade1 (anvils);
        fx: dict(puffs, streaks, rings, lights); hurt: (overlay amount 0..1, number of leading voxel instances it
        tints); glow: emission strength of glowing materials (default self.glow)."""
        ctx = self.ctx
        iw, ih = self.iw, self.ih
        aspect = iw / ih
        near, far = 0.3, 1500.0
        up = np.array(cam.get('up', (0, 0, 1)), float)
        view = look_at(cam['eye'], cam['target'], up)
        if cam.get('roll', 0.0):
            rr = np.radians(cam['roll'])
            rz = np.eye(4)
            rz[0, 0], rz[0, 1], rz[1, 0], rz[1, 1] = np.cos(rr), -np.sin(rr), np.sin(rr), np.cos(rr)
            view = rz @ view
        proj = perspective(cam['fov'], aspect, near, far)
        vp = proj @ view
        planes = frustum_planes(vp)
        lvp_near = light_matrix(near_center, self.near_half)
        fx = fx or {}

        vox_g = vox_s = None
        n_vox = 0
        if voxels is not None and len(voxels):
            n_vox = len(voxels)
            vbuf = ctx.buffer(np.ascontiguousarray(voxels).tobytes())
            vox_g, vox_s = self._voxel_vaos(vbuf)
        pvaos = []                                     # (kind, g, s, n, buf)
        for kind, arr in (props or {}).items():
            if arr is not None and len(arr):
                pbuf = ctx.buffer(np.ascontiguousarray(arr, np.float32).tobytes())
                g_, s_ = self._prop_vaos(kind, pbuf)
                pvaos.append((kind, g_, s_, len(arr), pbuf))
        arr_g = arr_s = None
        n_arr = 0
        if arrows is not None and len(arrows):
            n_arr = len(arrows)
            abuf = ctx.buffer(np.ascontiguousarray(arrows, np.float32).tobytes())
            arr_g, arr_s = self._arrow_vaos(abuf)
        self._giant_uniforms(giants)

        # 1. near shadow map
        self.sh_near_fbo.use()
        self.sh_near_fbo.clear(depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        ctx.polygon_offset = (1.5, 3.0)
        self.progs['sh_static']['u_lvp'].write(m4(lvp_near))
        for (first, count, lo, hi) in self.near_chunks:
            self.static_sh_vao.render(first=first, vertices=count)
        if self.region is not None:
            self.region[1].render()
        if vox_s is not None:
            self.progs['sh_voxel']['u_lvp'].write(m4(lvp_near))
            vox_s.render(instances=n_vox)
        if pvaos:
            self.progs['sh_prop']['u_lvp'].write(m4(lvp_near))
            for (_, _, s_, n_, _) in pvaos:
                s_.render(instances=n_)
        if arr_s is not None:
            self.progs['sh_arrow']['u_lvp'].write(m4(lvp_near))
            self.t_arrows.use(0)
            self.progs['sh_arrow']['u_arrows'] = 0
            arr_s.render(instances=n_arr)
        ctx.polygon_offset = (0.0, 0.0)

        # 2. G-buffer
        self.gbuf.use()
        self.gbuf.clear(0, 0, 0, 0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        P = self.progs['static']
        P['u_vp'].write(m4(vp))
        self.t_blocks.use(0)
        self.t_tint.use(1)
        P['u_blocks'] = 0
        P['u_tintnoise'] = 1
        P['u_sculk_r'] = float(self.sculk_r)
        P['u_sculk_c'] = (float(sculk[0]), float(sculk[1]))
        if bh is not None:
            P['u_bh'] = (*[float(v) for v in bh['c']], float(bh.get('sway', 0.0)))
        else:
            P['u_bh'] = (0.0, 0.0, 0.0, 0.0)
        P['u_time'] = float(bh.get('time', 0.0)) if bh is not None else 0.0
        if vox_g is not None:
            self.progs['voxel']['u_vp'].write(m4(vp))
            self.progs['voxel']['u_hole'] = ((*[float(v) for v in bh['c']], float(bh.get('r', 0.0)))
                                             if bh is not None else (0.0, 0.0, 0.0, 0.0))
            vox_g.render(instances=n_vox)
        for (kind, g_, _, n_, _) in pvaos:
            Pp_ = self.progs['prop']
            Pp_['u_vp'].write(m4(vp))
            self.prop_kinds[kind]['tex'].use(2)
            Pp_['u_props'] = 2
            Pp_['u_sheen'] = self.prop_kinds[kind]['sheen']
            g_.render(instances=n_)
        if arr_g is not None:
            ctx.disable(moderngl.CULL_FACE)
            self.progs['arrow']['u_vp'].write(m4(vp))
            self.t_arrows.use(2)
            self.progs['arrow']['u_arrows'] = 2
            arr_g.render(instances=n_arr)
            ctx.enable(moderngl.CULL_FACE)
        P['u_is_ground'] = 0.0
        for (first, count, lo, hi) in self.static_chunks:
            if aabb_visible(planes, lo, hi):
                self.static_vao.render(first=first, vertices=count)
        if self.region is not None:
            self.region[0].render()
        P['u_is_ground'] = 1.0
        self.ground_vao.render()

        # 3. SSAO
        invproj = np.linalg.inv(proj)
        self.ao_fbo_a.use()
        S = self.progs['ssao']
        self.g_depth.use(0)
        self.g_normal.use(1)
        S['u_depth'] = 0
        S['u_normal'] = 1
        S['u_proj'].write(m4(proj))
        S['u_invproj'].write(m4(invproj))
        S['u_viewrot'].write(np.ascontiguousarray(view[:3, :3].T, np.float32).tobytes())
        S['u_radius'] = self.ssao_radius
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self._fs('ssao').render(moderngl.TRIANGLE_STRIP)
        B = self.progs['blur']
        B['u_near'] = near
        B['u_far'] = far
        hw, hh = self.ao_a.size
        B['u_texel'] = (1.0 / hw, 1.0 / hh)
        self.g_depth.use(1)
        B['u_depth'] = 1
        B['u_src'] = 0
        self.ao_fbo_b.use()
        self.ao_a.use(0)
        B['u_dir'] = (1.0, 0.0)
        self._fs('blur').render(moderngl.TRIANGLE_STRIP)
        self.ao_fbo_a.use()
        self.ao_b.use(0)
        B['u_dir'] = (0.0, 1.0)
        self._fs('blur').render(moderngl.TRIANGLE_STRIP)

        # 4. lighting
        self.hdr_fbo.use()
        Lp = self.progs['light']
        self.g_albedo.use(0)
        self.g_normal.use(1)
        self.g_depth.use(2)
        self.ao_a.use(3)
        self.smp_near_cmp.use(4)
        self.smp_near_raw.use(5)
        self.smp_far_cmp.use(6)
        self.t_sky.use(7)
        for k, u in (('u_albedo', 0), ('u_normal', 1), ('u_depth', 2), ('u_ao', 3), ('u_sh_near', 4),
                     ('u_sh_near_raw', 5), ('u_sh_far', 6), ('u_sky', 7)):
            Lp[k] = u
        Lp['u_invvp'].write(m4(np.linalg.inv(vp)))
        Lp['u_cam'] = tuple(np.asarray(cam['eye'], float))
        Lp['u_lvp_near'].write(m4(lvp_near))
        Lp['u_lvp_far'].write(m4(self.lvp_far))
        Lp['u_sun_dir'] = tuple(SUN_DIR)
        light_mul = np.asarray(bh.get('light_mul', (1.0, 1.0, 1.0)), float) if bh is not None else np.ones(3)
        Lp['u_sun_col'] = tuple(self.sun_col * light_mul)
        Lp['u_sky_amb'] = tuple(self.sky_amb * light_mul)
        Lp['u_gnd_amb'] = tuple(self.gnd_amb * light_mul)
        Lp['u_sky_mul'] = tuple(float(v) for v in (bh.get('sky_mul', (1.0, 1.0, 1.0)) if bh is not None else (1.0, 1.0, 1.0)))
        Lp['u_hot'] = float(bh.get('hot', 6.0)) if bh is not None else 6.0
        Lp['u_fog'] = self.fog
        Lp['u_fog_col'] = tuple(float(v) for v in self.fog_col)
        Lp['u_bounce'] = tuple(float(v) for v in self.bounce)
        Lp['u_near_half'] = self.near_half
        Lp['u_el_min'] = EL_MIN
        Lp['u_shadow_res'] = float(self.shadow_res)
        lights = fx.get('lights')
        lpos = np.zeros((16, 4), np.float32)
        lcol = np.zeros((16, 4), np.float32)
        nl = 0
        if lights is not None and len(lights):
            nl = min(16, len(lights))
            lpos[:nl, :3] = lights[:nl, :3]
            lpos[:nl, 3] = lights[:nl, 4]
            if lights.shape[1] >= 8:
                lcol[:nl, :3] = lights[:nl, 5:8] * lights[:nl, 3:4]
            else:
                lcol[:nl, :3] = self.light_col[None, :] * lights[:nl, 3:4]
        Lp['u_nl'] = nl
        Lp['u_glow'] = float(self.glow if glow is None else glow)
        Lp['u_glow_dim'] = float(glow_dim)
        Lp['u_lpos'].write(lpos.tobytes())
        Lp['u_lcol'].write(lcol.tobytes())
        self._fs('light').render(moderngl.TRIANGLE_STRIP)
        for smp, unit in ((self.smp_near_cmp, 4), (self.smp_near_raw, 5), (self.smp_far_cmp, 6)):
            smp.clear(unit)

        # 5. particles (smoke / fireball puffs sorted back to front, then additive flashes)
        self._particles(fx, view, proj, near, far, vp, cam)

        # 6. the black hole: accretion disk (far half), gravitational lensing + shadow, disk (near half)
        self.out_hdr = self.hdr
        if bh is not None and bh.get('r', 0.0) > 0.0:
            self._black_hole(bh, cam, vp, near, far)
        # 7. depth of field (a macro lens: focus on the block under the press)
        if self.dof is not None and self.dof.get('k', 0.0) > 0.0:
            src = self.out_hdr
            dst, dst_fbo = (self.hdr2, self.hdr2_fbo) if src is self.hdr else (self.hdr, self.hdr_fbo)
            dst_fbo.use()
            ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
            sc = self.iw / 1080.0                    # sizes are given for a 1080 wide frame
            maxr = float(self.dof.get('maxr', 12.0)) * sc
            Cp = self.progs['dof_coc']
            src.use(0)
            self.g_depth.use(1)
            Cp['u_hdr'] = 0
            Cp['u_depth'] = 1
            Cp['u_near'] = near
            Cp['u_far'] = far
            Cp['u_focus'] = float(self.dof['focus'])
            Cp['u_k'] = float(self.dof['k']) * sc
            Cp['u_maxr'] = maxr
            self._fs('dof_coc').render(moderngl.TRIANGLE_STRIP)
            # gather from the colour + CoC buffer back into the first one
            src_fbo = self.hdr_fbo if src is self.hdr else self.hdr2_fbo
            src_fbo.use()
            Dp = self.progs['dof']
            dst.use(0)
            Dp['u_src'] = 0
            Dp['u_maxr'] = maxr
            Dp['u_step'] = 1.25 * sc
            Dp['u_texel'] = (1.0 / self.iw, 1.0 / self.ih)
            self._fs('dof').render(moderngl.TRIANGLE_STRIP)
            self.out_hdr = src

        # release per-frame buffers
        if vox_g is not None:
            vox_g.release()
            vox_s.release()
            vbuf.release()
        for (_, g_, s_, _, b_) in pvaos:
            g_.release()
            s_.release()
            b_.release()
        if arr_g is not None:
            arr_g.release()
            arr_s.release()
            abuf.release()
        return self.out_hdr

    def _disk(self, bh, cam, vp, near, far, side):
        d = bh.get('disk')
        if not d or d.get('bright', 0.0) <= 0.0:
            return
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        D = self.progs['disk']
        D['u_vp'].write(m4(vp))
        D['u_c'] = tuple(float(v) for v in bh['c'])
        D['u_n'] = tuple(float(v) for v in d['n'])
        D['u_rout'] = float(d['rout'])
        D['u_rin'] = float(d['rin'])
        D['u_rot'] = float(d.get('rot', 0.0))
        D['u_bright'] = float(d['bright'])
        D['u_beam'] = tuple(float(v) for v in d.get('beam', (0.0, 0.0)))
        D['u_side'] = float(side)
        D['u_cam'] = tuple(float(v) for v in cam['eye'])
        D['u_hole_dist'] = float(np.linalg.norm(np.asarray(bh['c'], float) - np.asarray(cam['eye'], float)))
        self.g_depth.use(0)
        D['u_depth'] = 0
        D['u_res'] = (float(self.iw), float(self.ih))
        D['u_near'] = near
        D['u_far'] = far
        self._fs('disk').render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)

    def _black_hole(self, bh, cam, vp, near, far):
        """Far half of the disk into the scene, bend the scene around the hole into hdr2 (shadow + photon
        ring), then the near half of the disk on top, unbent."""
        ctx = self.ctx
        self.hdr_fbo.use()
        self._disk(bh, cam, vp, near, far, 0)
        eye = np.asarray(cam['eye'], float)
        c = np.asarray(bh['c'], float)
        dist = float(np.linalg.norm(c - eye))
        r = float(bh['r'])
        rs = float(np.arcsin(min(0.999, r / max(dist, 1e-3))))
        re = float(np.arcsin(min(0.999, r * bh.get('lens', 1.6) / max(dist, 1e-3))))
        self.hdr2_fbo.use()
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        L = self.progs['lens']
        self.hdr.use(0)
        self.g_depth.use(1)
        self.t_sky.use(2)
        L['u_hdr'] = 0
        L['u_depth'] = 1
        L['u_sky'] = 2
        L['u_invvp'].write(m4(np.linalg.inv(vp)))
        L['u_vp'].write(m4(vp))
        L['u_cam'] = tuple(eye)
        L['u_bh'] = tuple(c)
        L['u_rs'] = rs
        L['u_re'] = max(re, rs * 1.05)
        L['u_front'] = max(0.0, dist - r)
        L['u_ring_col'] = tuple(float(v) for v in np.asarray(bh.get('ring', (5.0, 3.6, 2.2)), float))
        L['u_sky_mul'] = tuple(float(v) for v in bh.get('sky_mul', (1.0, 1.0, 1.0)))
        L['u_el_min'] = EL_MIN
        L['u_fade'] = tuple(float(v) for v in bh.get('lens_fade', (1.9, 3.4)))
        self._fs('lens').render(moderngl.TRIANGLE_STRIP)
        self._disk(bh, cam, vp, near, far, 1)
        self.out_hdr = self.hdr2

    def _particles(self, fx, view, proj, near, far, vp, cam):
        ctx = self.ctx
        puffs = fx.get('puffs')
        flashes = fx.get('flashes')
        streaks = fx.get('streaks')
        rings = fx.get('rings')
        if all(x is None or not len(x) for x in (puffs, flashes, streaks, rings)):
            return
        self.hdr_fbo.use()
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        vmat = m4(view)
        pmat = m4(proj)
        # sun direction projected into the screen plane (for fake self-shadowing of the puffs)
        sv = view[:3, :3] @ SUN_DIR
        s2 = sv[:2] / max(np.linalg.norm(sv[:2]), 1e-6)
        self.g_depth.use(0)
        if puffs is not None and len(puffs):
            vz = (np.c_[puffs[:, :3], np.ones(len(puffs))] @ view.T)[:, 2]
            order = np.argsort(vz)                   # most negative z = farthest first
            data = np.ascontiguousarray(puffs[order], np.float32)
            buf = ctx.buffer(data.tobytes())
            Pp = self.progs['puff']
            vao = ctx.vertex_array(Pp, [(self.corner_vbo, '2f', 'in_corner'),
                                        (buf, '3f 1f 1f 1f 1f 1f/i', 'i_pos', 'i_size', 'i_alpha', 'i_fire',
                                         'i_var', 'i_age')])
            Pp['u_view'].write(vmat)
            Pp['u_proj'].write(pmat)
            Pp['u_depth'] = 0
            self.t_puff.use(1)
            Pp['u_puff'] = 1
            Pp['u_res'] = (float(self.iw), float(self.ih))
            Pp['u_near'] = near
            Pp['u_far'] = far
            Pp['u_light'] = tuple(self.sun_col * 0.30 + self.sky_amb * 0.40)
            Pp['u_shade'] = tuple(self.sky_amb * 0.22 + np.array([0.08, 0.08, 0.08]))
            Pp['u_sun2d'] = (float(s2[0]), float(s2[1]))
            Pp['u_tint'] = tuple(self.puff_tint)
            ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
            vao.render(moderngl.TRIANGLE_STRIP, instances=len(data))
            vao.release()
            buf.release()
        if streaks is not None and len(streaks):
            data = np.ascontiguousarray(streaks, np.float32)
            buf = ctx.buffer(data.tobytes())
            Sp = self.progs['streak']
            vao = ctx.vertex_array(Sp, [(self.corner_vbo, '2f', 'in_corner'),
                                        (buf, '3f 3f 1f 1f/i', 'i_head', 'i_tail', 'i_width', 'i_alpha')])
            Sp['u_vp'].write(m4(vp))
            Sp['u_view'].write(vmat)
            Sp['u_cam'] = tuple(np.asarray(cam['eye'], float))
            Sp['u_depth'] = 0
            Sp['u_res'] = (float(self.iw), float(self.ih))
            Sp['u_near'] = near
            Sp['u_far'] = far
            Sp['u_col'] = tuple(self.streak_col)
            ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
            vao.render(moderngl.TRIANGLE_STRIP, instances=len(data))
            vao.release()
            buf.release()
        if rings is not None and len(rings):
            data = np.ascontiguousarray(rings, np.float32)                  # (R, 8) pos3 axis3 radius alpha
            buf = ctx.buffer(data.tobytes())
            Rp = self.progs['ring']
            vao = ctx.vertex_array(Rp, [(self.corner_vbo, '2f', 'in_corner'),
                                        (buf, '3f 3f 1f 1f/i', 'i_pos', 'i_axis', 'i_radius', 'i_alpha')])
            Rp['u_vp'].write(m4(vp))
            Rp['u_depth'] = 0
            Rp['u_res'] = (float(self.iw), float(self.ih))
            Rp['u_near'] = near
            Rp['u_far'] = far
            Rp['u_col'] = tuple(self.ring_col)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            vao.render(moderngl.TRIANGLE_STRIP, instances=len(data))
            vao.release()
            buf.release()
        if flashes is not None and len(flashes):
            data = np.zeros((len(flashes), 8), np.float32)
            data[:, 0:4] = flashes[:, 0:4]
            data[:, 4] = flashes[:, 4]
            buf = ctx.buffer(data.tobytes())
            Fp = self.progs['flash']
            vao = ctx.vertex_array(Fp, [(self.corner_vbo, '2f', 'in_corner'),
                                        (buf, '3f 1f 1f 1f 1f 1f/i', 'i_pos', 'i_size', 'i_alpha', 'i_fire',
                                         'i_var', 'i_age')])
            Fp['u_view'].write(vmat)
            Fp['u_proj'].write(pmat)
            Fp['u_depth'] = 0
            Fp['u_res'] = (float(self.iw), float(self.ih))
            Fp['u_near'] = near
            Fp['u_far'] = far
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            vao.render(moderngl.TRIANGLE_STRIP, instances=len(data))
            vao.release()
            buf.release()
        ctx.disable(moderngl.BLEND)

    def _bloom(self, src=None):
        """Threshold the HDR buffer and blur it at 1/4 and 1/8 resolution."""
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        (a4, fa4), (b4, fb4) = self.bloom_tex[0]
        (a8, fa8), (b8, fb8) = self.bloom_tex[1]
        Br = self.progs['bright']
        fa4.use()
        (src or self.hdr).use(0)
        Br['u_src'] = 0
        Br['u_texel'] = (1.0 / self.iw, 1.0 / self.ih)
        Br['u_thresh'] = self.bloom_thresh
        self._fs('bright').render(moderngl.TRIANGLE_STRIP)
        Gp = self.progs['gauss']
        Gp['u_src'] = 0
        for (src, dst_fbo, dst_tex, texel, d) in (
                (a4, fb4, b4, 1.0 / a4.size[0], (1, 0)),
                (b4, fa4, a4, 1.0 / a4.size[1], (0, 1))):
            dst_fbo.use()
            src.use(0)
            Gp['u_dir'] = (d[0] * 1.5 / a4.size[0], d[1] * 1.5 / a4.size[1])
            self._fs('gauss').render(moderngl.TRIANGLE_STRIP)
        # second, wider level from the first
        fa8.use()
        a4.use(0)
        Gp['u_dir'] = (0.0, 0.0)
        self._fs('gauss').render(moderngl.TRIANGLE_STRIP)
        for (src, dst_fbo, d) in ((a8, fb8, (1, 0)), (b8, fa8, (0, 1))):
            dst_fbo.use()
            src.use(0)
            Gp['u_dir'] = (d[0] * 2.0 / a8.size[0], d[1] * 2.0 / a8.size[1])
            self._fs('gauss').render(moderngl.TRIANGLE_STRIP)
        return a4, a8

    def finish(self, hdr_tex=None):
        """Tonemap + FXAA + downsample current HDR buffer; return (H, W, 3) uint8."""
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        src = hdr_tex or self.out_hdr
        b1, b2 = self._bloom(src)
        self.ldr_fbo.use()
        Pp = self.progs['post']
        src.use(0)
        b1.use(1)
        b2.use(2)
        Pp['u_hdr'] = 0
        Pp['u_bloom1'] = 1
        Pp['u_bloom2'] = 2
        Pp['u_bloom'] = self.bloom
        Pp['u_exposure'] = self.exposure
        Pp['u_sat'] = self.sat
        Pp['u_contrast'] = self.contrast
        Pp['u_vignette'] = self.vignette
        Pp['u_res'] = (float(self.iw), float(self.ih))
        self._fs('post').render(moderngl.TRIANGLE_STRIP)
        self.aa_fbo.use()
        F = self.progs['fxaa']
        self.ldr.use(0)
        F['u_src'] = 0
        F['u_texel'] = (1.0 / self.iw, 1.0 / self.ih)
        F['u_frame'] = float(self.frame_counter % 1000)
        self._fs('fxaa').render(moderngl.TRIANGLE_STRIP)
        self.frame_counter += 1
        if self.ss != 1.0:
            self.out_fbo.use()
            D = self.progs['down']
            self.aa.use(0)
            D['u_src'] = 0
            D['u_src_texel'] = (1.0 / self.iw, 1.0 / self.ih)
            D['u_scale'] = float(self.ss)
            self._fs('down').render(moderngl.TRIANGLE_STRIP)
            data = self.out_fbo.read(components=3, alignment=1)
            img = np.frombuffer(data, np.uint8).reshape(self.H, self.W, 3)
        else:
            data = self.aa_fbo.read(components=3, alignment=1)
            img = np.frombuffer(data, np.uint8).reshape(self.ih, self.iw, 3)
        return img[::-1].copy()

