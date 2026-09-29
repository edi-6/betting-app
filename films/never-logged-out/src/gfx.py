"""Deferred renderer for the film (moderngl, GL 4.3, headless on Mesa llvmpipe).

Per frame:
  near shadow map (sun or moon)  ->  G-buffer (albedo+spec, normal+material, emission)  ->  SSAO (half res)
  ->  volumetric fog (half res raymarch: moon/sun shafts through the shadow map, halos from the block light volume)
  ->  lighting (light volume: sky light, warm block light, cold soul light; the sun/moon with soft shadows; point
      lights; emission; sky panorama with a square moon, sun and stars; aerial fog)  ->  particles
  ->  depth of field (optional)  ->  bloom, tonemap, grade, vignette  ->  FXAA  ->  downsample  ->  read back.

World units are blocks; z is up.
"""
import numpy as np
import moderngl

MAT_TERRAIN, MAT_LEAF, MAT_PLANT, MAT_WATER, MAT_ENTITY, MAT_HAND, MAT_GLOW, MAT_GROUND = 1, 2, 3, 4, 5, 6, 7, 8


# ---------------------------------------------------------------------------------------------
# math
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
    if np.linalg.norm(s) < 1e-6:
        s = np.cross(f, (0, 1, 0))
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


def light_matrix(center, half, sdir, depth=400.0, res=None):
    c = np.asarray(center, float)
    sdir = np.asarray(sdir, float)
    up = (0, 0, 1) if abs(sdir[2]) < 0.99 else (0, 1, 0)
    view = look_at(c + sdir * depth * 0.5, c, up)
    if res:
        # snap the centre to whole shadow texels so shadows don't crawl as the camera moves
        texel = 2 * half / res
        cl = view @ np.array([*c, 1.0])
        dx = np.round(cl[0] / texel) * texel - cl[0]
        dy = np.round(cl[1] / texel) * texel - cl[1]
        T = np.eye(4)
        T[0, 3] = dx
        T[1, 3] = dy
        view = T @ view
    return ortho(-half, half, -half, half, 1.0, depth) @ view


def frustum_planes(vp):
    m = vp
    planes = [m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1], m[3] + m[2], m[3] - m[2]]
    return np.array([p / np.linalg.norm(p[:3]) for p in planes])


def aabb_visible(planes, lo, hi):
    for p in planes:
        v = np.where(p[:3] >= 0, hi, lo)
        if p[:3] @ v + p[3] < 0:
            return False
    return True


def m4(m):
    return np.ascontiguousarray(m.T, dtype=np.float32).tobytes()


def quat_from_mat(R):
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
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


# ---------------------------------------------------------------------------------------------
# shaders
# ---------------------------------------------------------------------------------------------
QROT = "vec3 qrot(vec4 q, vec3 v){ return v + 2.0*cross(q.xyz, cross(q.xyz, v) + q.w*v); }\n"

STATIC_VS = """
#version 430
uniform mat4 u_vp;
uniform float u_time;
uniform float u_wind;
uniform ivec4 u_sway;        // texture layers that sway in the wind: leaves x3, tall grass
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv; in float in_layer; in vec3 in_tint;
out vec3 v_nrm; out vec2 v_uv; flat out float v_layer; out vec3 v_tint; out vec3 v_wpos;
void main(){
    vec3 p = in_pos;
    int lay = int(in_layer + 0.5);
    if (u_wind > 0.0) {
        float ph = u_time * 1.7 + p.x * 0.37 + p.y * 0.29;
        if (lay == u_sway.x || lay == u_sway.y || lay == u_sway.z) {
            p.xy += u_wind * 0.035 * vec2(sin(ph + p.z * 0.5), cos(ph * 0.83 + p.z * 0.4));
        } else if (lay == u_sway.w) {
            float top = fract(p.z) > 0.01 ? 1.0 : 0.0;          // only the top edge of the blades moves
            p.xy += top * u_wind * 0.09 * vec2(sin(ph * 1.3), cos(ph * 1.1));
        }
    }
    v_nrm = in_nrm; v_uv = in_uv; v_layer = in_layer; v_tint = in_tint; v_wpos = p;
    gl_Position = u_vp * vec4(p, 1.0);
}
"""

STATIC_FS = """
#version 430
uniform sampler2DArray u_blocks;
uniform sampler2DArray u_emit;
uniform sampler2D u_noise;
uniform float u_cutout;
uniform float u_time;
uniform int u_water;
uniform ivec4 u_leafy;       // leaves x3, tall grass (material)
uniform int u_grass_top;
uniform int u_grass_side;
uniform float u_emit_scale;
in vec3 v_nrm; in vec2 v_uv; flat in float v_layer; in vec3 v_tint; in vec3 v_wpos;
layout(location=0) out vec4 o_albedo;
layout(location=1) out vec4 o_normal;
layout(location=2) out vec4 o_extra;
void main(){
    int lay = int(v_layer + 0.5);
    vec2 uv = v_uv;
    vec3 nrm = normalize(v_nrm);
    float spec = 0.0;
    float mat = %d.0;
    if (lay == u_water) {
        uv += vec2(u_time * 0.05, u_time * 0.02);
        vec2 q = v_wpos.xy * 0.23 + vec2(u_time * 0.11, u_time * 0.05);
        float e = 0.02;
        float h0 = texture(u_noise, q).r + 0.5 * texture(u_noise, q * 2.3 + 0.41).r;
        float hx = texture(u_noise, q + vec2(e, 0.0)).r + 0.5 * texture(u_noise, (q + vec2(e, 0.0)) * 2.3 + 0.41).r;
        float hy = texture(u_noise, q + vec2(0.0, e)).r + 0.5 * texture(u_noise, (q + vec2(0.0, e)) * 2.3 + 0.41).r;
        nrm = normalize(vec3(-(hx - h0) / e * 0.015, -(hy - h0) / e * 0.015, 1.0));
        spec = 0.9;
        mat = %d.0;
    }
    vec4 tx = texture(u_blocks, vec3(uv, v_layer));
    if (u_cutout > 0.5 && tx.a < 0.5) discard;
    vec3 c = pow(tx.rgb, vec3(2.2)) * v_tint;
    if (lay == u_leafy.x || lay == u_leafy.y || lay == u_leafy.z) mat = %d.0;
    if (lay == u_leafy.w) mat = %d.0;
    if (lay == u_grass_top || (lay == u_grass_side && fract(-v_wpos.z) < 0.3) || lay == u_leafy.w) {
        float n = texture(u_noise, v_wpos.xy / 97.0).r;
        float n2 = texture(u_noise, v_wpos.xy / 23.0 + 0.37).r;
        c *= mix(vec3(0.88, 0.96, 0.84), vec3(1.08, 1.04, 0.90), n) * mix(0.94, 1.05, n2);
    }
    if (lay == u_leafy.x || lay == u_leafy.y || lay == u_leafy.z) {
        float n = texture(u_noise, floor(v_wpos.xy) / 41.0 + 0.13).r;
        c *= mix(vec3(0.86, 0.98, 0.84), vec3(1.10, 1.06, 0.92), n);
    }
    if (lay == u_grass_top && nrm.z > 0.5) mat = %d.0;
    float em = texture(u_emit, vec3(uv, v_layer)).r;
    o_albedo = vec4(sqrt(clamp(c, 0.0, 1.0)), spec);
    o_normal = vec4(nrm * 0.5 + 0.5, mat / 255.0);
    o_extra = vec4(em * u_emit_scale / 16.0, 0.0, 0.0, 1.0);
}
""" % (MAT_TERRAIN, MAT_WATER, MAT_LEAF, MAT_PLANT, MAT_GROUND)

# instanced meshes (characters, block props, decals): pos3 quat4 scale3 layer1 tint3 emit1 mat1 = 16 floats
INST_FMT = '3f 4f 3f 1f 3f 1f 1f/i'
INST_NAMES = ('i_pos', 'i_quat', 'i_scale', 'i_layer', 'i_tint', 'i_emit', 'i_mat')

PROP_VS = """
#version 430
uniform mat4 u_vp;
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv; in float in_layer;
in vec3 i_pos; in vec4 i_quat; in vec3 i_scale; in float i_layer; in vec3 i_tint; in float i_emit; in float i_mat;
out vec3 v_nrm; out vec2 v_uv; flat out float v_layer; flat out vec3 v_tint; flat out float v_emit; flat out float v_mat;
""" + QROT + """
void main(){
    v_nrm = qrot(i_quat, in_nrm / i_scale);
    v_uv = in_uv;
    v_layer = in_layer < 0.0 ? i_layer : in_layer;     // meshes with layer -1 take the instance's layer
    v_tint = i_tint; v_emit = i_emit; v_mat = i_mat;
    gl_Position = u_vp * vec4(qrot(i_quat, in_pos * i_scale) + i_pos, 1.0);
}
"""

PROP_FS = """
#version 430
uniform sampler2DArray u_tex;
uniform sampler2DArray u_emit;
uniform int u_has_emit;
in vec3 v_nrm; in vec2 v_uv; flat in float v_layer; flat in vec3 v_tint; flat in float v_emit; flat in float v_mat;
layout(location=0) out vec4 o_albedo;
layout(location=1) out vec4 o_normal;
layout(location=2) out vec4 o_extra;
void main(){
    vec4 t = texture(u_tex, vec3(v_uv, v_layer));
    if (t.a < 0.5) discard;
    vec3 c = pow(t.rgb, vec3(2.2)) * v_tint;
    float em = v_emit;
    if (u_has_emit > 0) em += texture(u_emit, vec3(v_uv, v_layer)).r * 1.0;
    // glowing props (a screen) keep their colour as emission: store it with full albedo
    o_albedo = vec4(sqrt(clamp(c, 0.0, 1.0)), 0.0);
    o_normal = vec4(normalize(v_nrm) * 0.5 + 0.5, v_mat / 255.0);
    o_extra = vec4(em / 16.0, 0.0, 0.0, 1.0);
}
"""

SH_STATIC_VS = """
#version 430
uniform mat4 u_lvp;
in vec3 in_pos; in vec2 in_uv; in float in_layer;
out vec2 v_uv; flat out float v_layer;
void main(){ v_uv = in_uv; v_layer = in_layer; gl_Position = u_lvp * vec4(in_pos, 1.0); }
"""
SH_CUT_FS = """
#version 430
uniform sampler2DArray u_blocks;
uniform float u_cutout;
in vec2 v_uv; flat in float v_layer;
void main(){ if (u_cutout > 0.5 && texture(u_blocks, vec3(v_uv, v_layer)).a < 0.5) discard; }
"""
SH_PROP_VS = """
#version 430
uniform mat4 u_lvp;
in vec3 in_pos; in vec2 in_uv; in float in_layer; in vec3 i_pos; in vec4 i_quat; in vec3 i_scale; in float i_layer;
out vec2 v_uv; flat out float v_layer;
""" + QROT + """
void main(){ v_uv = in_uv; v_layer = in_layer < 0.0 ? i_layer : in_layer;
             gl_Position = u_lvp * vec4(qrot(i_quat, in_pos * i_scale) + i_pos, 1.0); }
"""
SH_PROP_FS = """
#version 430
uniform sampler2DArray u_tex;
in vec2 v_uv; flat in float v_layer;
void main(){ if (texture(u_tex, vec3(v_uv, v_layer)).a < 0.5) discard; }
"""

DEPTHCOPY_FS = """
#version 430
uniform sampler2D u_src;
in vec2 v_uv;
void main(){ gl_FragDepth = texture(u_src, v_uv).r; }
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
uniform float u_radius;
in vec2 v_uv;
out float o_ao;
vec3 vpos(vec2 uv){
    float d = texture(u_depth, uv).r;
    vec4 p = u_invproj * vec4(uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
    return p.xyz / p.w;
}
const int NS = 12;
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
    float rad = u_radius * clamp(-P.z / 30.0, 0.5, 2.0);
    float occ = 0.0;
    for (int i = 0; i < NS; i++){
        float fi = float(i) + ign;
        float r = sqrt((fi + 0.5) / float(NS));
        float th = fi * 2.399963;
        vec2 dk = vec2(cos(th), sin(th)) * r;
        float h = sqrt(max(0.0, 1.0 - r * r));
        float sc = mix(0.15, 1.0, (fi / float(NS)) * (fi / float(NS)));
        vec3 Q = P + (T * dk.x + B * dk.y + N * h) * rad * sc;
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
out vec4 o_v;
float lin(float d){ float z = d * 2.0 - 1.0; return 2.0 * u_near * u_far / (u_far + u_near - z * (u_far - u_near)); }
void main(){
    float dc = lin(texture(u_depth, v_uv).r);
    vec4 sum = vec4(0.0); float wsum = 0.0;
    for (int i = -4; i <= 4; i++){
        vec2 uv = v_uv + u_dir * u_texel * float(i);
        float dz = lin(texture(u_depth, uv).r);
        float w = exp(-float(i * i) / 12.0) * exp(-abs(dz - dc) / (0.03 * dc + 0.08));
        sum += texture(u_src, uv) * w;
        wsum += w;
    }
    o_v = sum / max(wsum, 1e-5);
}
"""

COMMON_LIGHT = """
uniform sampler3D u_lvol;
uniform vec3 u_lvol_org;
uniform vec3 u_lvol_size;
uniform sampler2DShadow u_sh_near;
uniform sampler2D u_sh_near_raw;
uniform sampler2DShadow u_sh_far;
uniform mat4 u_lvp_near;
uniform mat4 u_lvp_far;
uniform float u_near_half;
uniform float u_shadow_res;
uniform vec3 u_sun_dir;
uniform float u_blk_flicker;
vec3 lightvol(vec3 p){
    vec3 t = (p - u_lvol_org) / u_lvol_size;
    if (any(lessThan(t, vec3(0.0))) || any(greaterThan(t, vec3(1.0)))) return vec3(1.0, 0.0, 0.0);
    return texture(u_lvol, t).rgb;
}
float mc_curve(float l){ float c = l / (4.0 - 3.0 * l); return c * sqrt(c); }   // the game's level -> brightness, steeper
const vec2 POISSON[12] = vec2[](
    vec2(-0.326, -0.406), vec2(-0.840, -0.074), vec2(-0.696, 0.457), vec2(-0.203, 0.621),
    vec2(0.962, -0.195), vec2(0.473, -0.480), vec2(0.519, 0.767), vec2(0.185, -0.893),
    vec2(0.507, 0.064), vec2(0.896, 0.412), vec2(-0.322, -0.933), vec2(-0.792, -0.598));
float shadow_far(vec3 wp, vec3 n){
    vec4 l = u_lvp_far * vec4(wp + n * 0.2, 1.0);
    vec3 s = l.xyz / l.w * 0.5 + 0.5;
    if (any(lessThan(s.xy, vec2(0.0))) || any(greaterThan(s.xy, vec2(1.0)))) return 1.0;
    return texture(u_sh_far, vec3(s.xy, s.z - 0.0006));
}
float shadow_near_1(vec3 wp){
    vec4 l = u_lvp_near * vec4(wp, 1.0);
    vec3 s = l.xyz / l.w * 0.5 + 0.5;
    if (any(lessThan(s.xy, vec2(0.002))) || any(greaterThan(s.xy, vec2(0.998)))) return -1.0;
    return texture(u_sh_near, vec3(s.xy, s.z - 0.0003));
}
float shadow_soft(vec3 wp, vec3 n, float ign){
    vec3 p = wp + n * 0.04;
    vec4 l = u_lvp_near * vec4(p, 1.0);
    vec3 s = l.xyz / l.w * 0.5 + 0.5;
    if (any(lessThan(s.xy, vec2(0.002))) || any(greaterThan(s.xy, vec2(0.998)))) return shadow_far(wp, n);
    float texel = 1.0 / u_shadow_res;
    float ca = cos(ign * 6.283), sa = sin(ign * 6.283);
    mat2 rot = mat2(ca, sa, -sa, ca);
    float bsum = 0.0, bn = 0.0;
    for (int i = 0; i < 6; i++){
        float bd = texture(u_sh_near_raw, s.xy + rot * POISSON[i] * 20.0 * texel).r;
        if (bd < s.z - 0.0005) { bsum += bd; bn += 1.0; }
    }
    if (bn < 0.5) return 1.0;
    float dist = (s.z - bsum / bn) * 400.0;
    float pen = clamp(dist * 0.02 + 0.03, 0.03, 0.8);
    float r = max(pen / (2.0 * u_near_half), 1.2 * texel);
    float sum = 0.0;
    for (int i = 0; i < 12; i++) sum += texture(u_sh_near, vec3(s.xy + rot * POISSON[i] * r, s.z - 0.00015));
    float sh = sum / 12.0;
    vec2 e = abs(s.xy * 2.0 - 1.0);
    float farw = smoothstep(0.86, 0.98, max(e.x, e.y));
    return mix(sh, min(sh, shadow_far(wp, n)), farw);
}
"""

VOLFOG_FS = """
#version 430
uniform sampler2D u_depth;
uniform mat4 u_invvp;
uniform vec3 u_cam;
uniform vec3 u_sun_col;        // sun or moon radiance for the fog
uniform vec3 u_blk_col;
uniform vec3 u_soul_col;
uniform vec3 u_amb_col;
uniform vec4 u_fog;            // density at ground, height falloff (blocks), base height, max distance
uniform float u_g;             // forward scattering
uniform int u_steps;
uniform float u_frame;
in vec2 v_uv;
out vec4 o;
""" + COMMON_LIGHT + """
float hg(float c, float g){ float g2 = g * g; return (1.0 - g2) / (4.0 * 3.14159 * pow(1.0 + g2 - 2.0 * g * c, 1.5)); }
void main(){
    float d = texture(u_depth, v_uv).r;
    vec4 fp = u_invvp * vec4(v_uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
    vec3 wp = fp.xyz / fp.w;
    vec3 rd = wp - u_cam;
    float len = length(rd);
    rd /= max(len, 1e-4);
    float tmax = min(len, u_fog.w);
    float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy + u_frame * 5.588238, vec2(0.06711056, 0.00583715))));
    float dt = tmax / float(u_steps);
    float cosT = dot(rd, u_sun_dir);
    float ph = mix(hg(cosT, u_g), 1.0 / 12.566, 0.35) * 12.566;
    vec3 S = vec3(0.0);
    float T = 1.0;
    for (int i = 0; i < u_steps; i++){
        float t = (float(i) + ign) * dt;
        vec3 p = u_cam + rd * t;
        float dens = u_fog.x * exp(-max(p.z - u_fog.z, 0.0) / u_fog.y);
        if (dens < 1e-5) continue;
        vec3 lv = lightvol(p);
        float sh = shadow_near_1(p);
        if (sh < 0.0) sh = shadow_far(p, vec3(0.0));
        sh *= smoothstep(0.35, 0.9, lv.r);
        vec3 inl = u_sun_col * sh * ph + u_amb_col * lv.r * lv.r
                   + u_blk_col * mc_curve(lv.g) * u_blk_flicker + u_soul_col * mc_curve(lv.b);
        float tr = exp(-dens * dt);
        S += T * inl * (1.0 - tr);
        T *= tr;
    }
    o = vec4(S, T);
}
"""

LIGHT_FS = """
#version 430
uniform sampler2D u_albedo;
uniform sampler2D u_normal;
uniform sampler2D u_extra;
uniform sampler2D u_depth;
uniform sampler2D u_ao;
uniform sampler2D u_fogtex;
uniform sampler2D u_sky;
uniform sampler2D u_moon;
uniform mat4 u_invvp;
uniform vec3 u_cam;
uniform vec3 u_sun_col;
uniform vec3 u_sky_amb;
uniform vec3 u_gnd_amb;
uniform vec3 u_blk_col;
uniform vec3 u_soul_col;
uniform vec3 u_min_amb;        // a floor so the darkest caves aren't pure black (cinematic)
uniform float u_emit_gain;
uniform vec3 u_sky_mul;
uniform float u_aerial;        // plain distance fog density
uniform vec3 u_aerial_col;
uniform int u_use_vfog;
uniform vec3 u_moon_dir;
uniform float u_moon_size;     // half angle (radians) of the square moon
uniform float u_moon_bright;
uniform vec3 u_sun_disc_dir;
uniform float u_sun_size;
uniform vec3 u_sun_disc_col;
uniform float u_stars;
uniform float u_time;
uniform int u_nl;
uniform vec4 u_lpos[16];
uniform vec4 u_lcol[16];
uniform float u_flash;         // lightning
in vec2 v_uv;
out vec4 o_col;
""" + COMMON_LIGHT + """
vec3 sky(vec3 d){
    float phi = atan(d.y, d.x);
    float u = fract(phi / 6.2831853);
    float el = degrees(asin(clamp(d.z, -1.0, 1.0)));
    float v = clamp((90.0 - el) / 102.0, 0.0, 1.0);
    return textureLod(u_sky, vec2(u, v), 0.0).rgb;
}
float hash13(vec3 p){ p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }
vec3 sky_full(vec3 d){
    vec3 c = sky(d) * u_sky_mul;
    // stars: a fixed hash grid over the sky, fading at the horizon
    if (u_stars > 0.0 && d.z > 0.0) {
        vec3 q = d * 420.0;
        vec3 cell = floor(q);
        float h = hash13(cell);
        if (h > 0.9965) {
            vec3 f = fract(q) - 0.5;
            float s = exp(-dot(f, f) * 30.0);
            float tw = 0.75 + 0.25 * sin(u_time * (2.0 + h * 9.0) + h * 80.0);
            c += vec3(0.9, 0.93, 1.0) * s * tw * u_stars * smoothstep(0.0, 0.25, d.z) * (h - 0.9965) / 0.0035 * 3.0;
        }
    }
    // the square moon (textured), with a soft halo
    if (u_moon_bright > 0.0) {
        vec3 m = normalize(u_moon_dir);
        vec3 a = normalize(cross(m, vec3(0.0, 0.0, 1.0)));
        vec3 b = cross(a, m);
        float cm = dot(d, m);
        if (cm > 0.0) {
            vec2 q = vec2(dot(d, a), dot(d, b)) / cm;
            vec2 uv = q / (2.0 * tan(u_moon_size)) + 0.5;
            float dd = acos(clamp(cm, -1.0, 1.0));
            c += vec3(0.55, 0.62, 0.8) * u_moon_bright * 0.05 * exp(-dd / (u_moon_size * 3.5));
            c += vec3(0.4, 0.45, 0.6) * u_moon_bright * 0.02 * exp(-dd / (u_moon_size * 12.0));
            if (all(greaterThan(uv, vec2(0.0))) && all(lessThan(uv, vec2(1.0)))) {
                vec4 mt = texture(u_moon, uv);
                c = mix(c, mt.rgb * u_moon_bright, mt.a);
            }
        }
    }
    if (u_sun_size > 0.0) {
        vec3 s = normalize(u_sun_disc_dir);
        vec3 a = normalize(cross(s, vec3(0.0, 0.0, 1.0)));
        vec3 b = cross(a, s);
        float cs = dot(d, s);
        if (cs > 0.0) {
            vec2 q = abs(vec2(dot(d, a), dot(d, b)) / cs);
            float r = max(q.x, q.y) / tan(u_sun_size);
            c += u_sun_disc_col * (r < 1.0 ? 1.0 : 0.0);
            float dd = acos(clamp(cs, -1.0, 1.0));
            c += u_sun_disc_col * 0.03 * exp(-dd / (u_sun_size * 3.0));
        }
    }
    return c;
}
void main(){
    float d = texture(u_depth, v_uv).r;
    vec4 hp = u_invvp * vec4(v_uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
    vec3 wp = hp.xyz / hp.w;
    vec4 fp = u_invvp * vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 vdir = normalize(fp.xyz / fp.w - u_cam);
    vec4 fogv = u_use_vfog > 0 ? texture(u_fogtex, v_uv) : vec4(0.0, 0.0, 0.0, 1.0);
    if (d >= 1.0) {
        vec3 c = sky_full(vdir) + vec3(u_flash * 0.6);
        o_col = vec4(c * fogv.a + fogv.rgb, 1.0);
        return;
    }
    vec4 nm = texture(u_normal, v_uv);
    vec3 N = normalize(nm.xyz * 2.0 - 1.0);
    int mat = int(nm.a * 255.0 + 0.5);
    vec4 al = texture(u_albedo, v_uv);
    vec3 A = al.rgb * al.rgb;
    float spec = al.a;
    float em = texture(u_extra, v_uv).r * 16.0;
    float ao = texture(u_ao, v_uv).r;
    float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    vec3 lp = (mat == %d) ? u_cam + vec3(0.0, 0.0, -0.2) : wp + N * 0.5;
    if (mat == %d) lp = wp + N * 0.3;
    vec3 L = lightvol(lp);
    float skyl = L.r, blk = L.g, soul = L.b;
    float sh = (mat == %d) ? max(shadow_near_1(u_cam - vec3(0.0, 0.0, 0.3)), 0.0) : shadow_soft(wp, N, ign);
    sh *= smoothstep(0.3, 0.85, skyl);
    float ndl = dot(N, u_sun_dir);
    vec3 V = -vdir;
    float hemi = N.z * 0.5 + 0.5;
    vec3 amb = mix(u_gnd_amb, u_sky_amb, hemi) * (skyl * skyl) * ao;
    vec3 blkc = u_blk_col * mc_curve(blk) * u_blk_flicker * mix(1.0, ao, 0.6);
    vec3 soulc = u_soul_col * mc_curve(soul) * mix(1.0, ao, 0.6);
    vec3 direct;
    if (mat == %d || mat == %d) {
        float wrap = max((ndl + 0.5) / 1.5, 0.0);
        direct = u_sun_col * wrap * sh + u_sun_col * 0.2 * max(-ndl, 0.0) * sh;
    } else {
        direct = u_sun_col * max(ndl, 0.0) * sh;
    }
    vec3 col = A * (direct + amb + blkc + soulc + u_min_amb * ao);
    if (spec > 0.0) {
        vec3 H = normalize(u_sun_dir + V);
        float fres = 0.02 + 0.98 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
        col += u_sun_col * spec * fres * 4.0 * pow(max(dot(N, H), 0.0), 60.0) * sh;
        col += spec * fres * sky_full(reflect(-V, N)) * smoothstep(0.4, 0.9, skyl);
        col += spec * fres * 0.3 * (blkc + soulc);
    }
    vec3 pl = vec3(0.0);
    for (int i = 0; i < u_nl; i++){
        vec3 Lv = u_lpos[i].xyz - wp;
        float dl = length(Lv);
        float rad = u_lpos[i].w;
        if (dl < rad) {
            float att = (1.0 - dl / rad) * (1.0 - dl / rad) / (1.0 + dl * dl * 0.1);
            float nd = max(dot(N, Lv / max(dl, 1e-3)), 0.0) * 0.85 + 0.15;
            pl += u_lcol[i].rgb * att * nd;
        }
    }
    col += A * pl;
    col += A * em * u_emit_gain;
    col += A * u_flash * 0.25 * skyl;
    float dist = length(wp - u_cam);
    float f = 1.0 - exp(-dist * u_aerial);
    col = mix(col, u_aerial_col, clamp(f, 0.0, 1.0));
    col = col * fogv.a + fogv.rgb;
    o_col = vec4(col, 1.0);
}
""" % (MAT_HAND, MAT_ENTITY, MAT_HAND, MAT_LEAF, MAT_PLANT)

PARTICLE_VS = """
#version 430
uniform mat4 u_view;
uniform mat4 u_proj;
in vec2 in_corner;
in vec3 i_pos; in float i_size; in vec4 i_col;
out vec2 v_c; out float v_vdepth; flat out vec4 v_col;
void main(){
    vec4 vc = u_view * vec4(i_pos, 1.0);
    vc.xy += in_corner * i_size * 0.5;
    v_vdepth = -vc.z;
    v_c = in_corner;
    v_col = i_col;
    gl_Position = u_proj * vc;
}
"""
PARTICLE_FS = """
#version 430
uniform sampler2D u_depth;
uniform vec2 u_res;
uniform float u_near;
uniform float u_far;
uniform int u_soft;          // 1: smoke puff (alpha blended), 0: glowing speck (additive)
in vec2 v_c; in float v_vdepth; flat in vec4 v_col;
out vec4 o;
float lin(float d){ float z = d * 2.0 - 1.0; return 2.0 * u_near * u_far / (u_far + u_near - z * (u_far - u_near)); }
void main(){
    float r2 = dot(v_c, v_c);
    if (r2 > 1.0) discard;
    float sd = lin(texture(u_depth, gl_FragCoord.xy / u_res).r);
    float vis = clamp((sd - v_vdepth) / 0.6, 0.0, 1.0);
    float a = (u_soft > 0 ? (1.0 - r2) * (1.0 - r2) : exp(-r2 * 4.0)) * v_col.a * vis;
    if (a < 0.002) discard;
    o = u_soft > 0 ? vec4(v_col.rgb * a, a) : vec4(v_col.rgb * a, 0.0);
}
"""

STREAK_VS = """
#version 430
uniform mat4 u_vp;
uniform mat4 u_view;
uniform vec3 u_cam;
in vec2 in_corner;
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
    float a = v_alpha * (1.0 - abs(v_c.x * 2.0 - 1.0)) * (1.0 - v_c.y * v_c.y);
    o = vec4(u_col * a, a);
}
"""

DOF_COC_FS = """
#version 430
uniform sampler2D u_hdr;
uniform sampler2D u_depth;
uniform float u_near;
uniform float u_far;
uniform float u_focus;
uniform float u_k;
uniform float u_maxr;
in vec2 v_uv;
out vec4 o;
float lin(float d){ float z = d * 2.0 - 1.0; return 2.0 * u_near * u_far / (u_far + u_near - z * (u_far - u_near)); }
void main(){
    float d = lin(texture(u_depth, v_uv).r);
    float s = clamp(u_k * (d - u_focus) / max(d, 0.1), -4.0 * u_maxr, 4.0 * u_maxr);
    o = vec4(texture(u_hdr, v_uv).rgb, s);
}
"""

DOF_FS = """
#version 430
uniform sampler2D u_src;
uniform float u_maxr;
uniform float u_step;
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
        if (sm.a > s0) ss = clamp(ss, 0.0, c0 * 2.0);
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
uniform vec2 u_dir;
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

POST_FS = """
#version 430
uniform sampler2D u_hdr;
uniform sampler2D u_bloom1;
uniform sampler2D u_bloom2;
uniform float u_bloom;
uniform float u_exposure;
uniform float u_sat;
uniform float u_contrast;
uniform float u_vignette;
uniform vec3 u_lift;
uniform vec3 u_gain;
uniform vec2 u_res;
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
    c = c * u_gain + u_lift * (1.0 - c);
    c = clamp((c - 0.5) * u_contrast + 0.5, 0.0, 1.0);
    vec2 q = v_uv - 0.5;
    q.x *= u_res.x / u_res.y;
    c *= 1.0 - u_vignette * smoothstep(0.3, 1.0, length(q) * 1.25);
    c = pow(c, vec3(1.0 / 2.2));
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


# ---------------------------------------------------------------------------------------------
# environment presets
# ---------------------------------------------------------------------------------------------
def sun_dir(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


class Env(dict):
    """Lighting and look for a shot. Keys are documented in DEFAULT."""
    def __getattr__(self, k):
        return self[k]


DEFAULT = dict(
    sky='sunset',                       # panorama key
    sky_mul=(1.0, 1.0, 1.0),
    light_dir=sun_dir(200.0, 7.0),      # sun or moon (the one casting shadows)
    light_col=(3.0, 1.7, 0.9),
    sky_amb=(0.35, 0.40, 0.55),
    gnd_amb=(0.25, 0.20, 0.14),
    blk_col=(2.6, 1.55, 0.78),          # torch light at level 15
    soul_col=(0.35, 1.3, 1.6),
    min_amb=(0.004, 0.004, 0.006),
    emit_gain=3.0,
    aerial=0.0022,
    aerial_col=(0.9, 0.6, 0.45),
    fog=(0.012, 18.0, 0.0, 120.0),       # volumetric: density, height falloff, base z, max distance (0 = off)
    fog_sun=(1.0, 1.0, 1.0),            # tint/scale of the sun's in-scattering
    fog_amb=(0.06, 0.07, 0.1),
    fog_g=0.55,
    moon_dir=sun_dir(20.0, 35.0), moon_size=0.035, moon_bright=0.0,
    sun_disc_dir=sun_dir(200.0, 7.0), sun_size=0.0, sun_disc_col=(30.0, 16.0, 6.0),
    stars=0.0,
    exposure=0.8, sat=1.05, contrast=1.04, vignette=0.28, lift=(0.0, 0.0, 0.0), gain=(1.0, 1.0, 1.0),
    bloom=0.35, bloom_thresh=2.0, ssao=1.0, wind=1.0, flicker=1.0, flash=0.0,
)


def env(**kw):
    e = Env(DEFAULT)
    e.update(kw)
    return e


# ---------------------------------------------------------------------------------------------
# the renderer
# ---------------------------------------------------------------------------------------------
class Renderer:
    def __init__(self, width, height, ss=1.0, shadow_res=4096, near_half=40.0, far_half=220.0):
        self.W, self.H = width, height
        self.ss = ss
        self.iw, self.ih = int(round(width * ss)), int(round(height * ss))
        self.shadow_res = shadow_res
        self.near_half = near_half
        self.far_half = far_half
        ctx = moderngl.create_standalone_context(backend='egl', require=430)
        self.ctx = ctx
        self.progs = {}
        self._build_programs()
        self.quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32).tobytes())
        self.corner_vbo = self.quad
        self.fs_vaos = {}
        self._build_targets()
        self.worlds = {}
        self.world = None
        self.skies = {}
        self.kinds = {}
        self.frame_counter = 0
        self.time = 0.0
        self.dof = None
        self.far_cache = {}
        self.max_dist = 400.0

    # -- setup ------------------------------------------------------------------------------
    def _prog(self, name, vs, fs):
        self.progs[name] = self.ctx.program(vertex_shader=vs, fragment_shader=fs)
        return self.progs[name]

    def _build_programs(self):
        P = self._prog
        P('static', STATIC_VS, STATIC_FS)
        P('prop', PROP_VS, PROP_FS)
        P('sh_static', SH_STATIC_VS, SH_CUT_FS)
        P('sh_prop', SH_PROP_VS, SH_PROP_FS)
        P('ssao', FULLSCREEN_VS, SSAO_FS)
        P('blur', FULLSCREEN_VS, BLUR_FS)
        P('vfog', FULLSCREEN_VS, VOLFOG_FS)
        P('light', FULLSCREEN_VS, LIGHT_FS)
        P('particle', PARTICLE_VS, PARTICLE_FS)
        P('streak', STREAK_VS, STREAK_FS)
        P('dof_coc', FULLSCREEN_VS, DOF_COC_FS)
        P('dof', FULLSCREEN_VS, DOF_FS)
        P('bright', FULLSCREEN_VS, BRIGHT_FS)
        P('gauss', FULLSCREEN_VS, GAUSS_FS)
        P('post', FULLSCREEN_VS, POST_FS)
        P('fxaa', FULLSCREEN_VS, FXAA_FS)
        P('down', FULLSCREEN_VS, DOWN_FS)
        P('depthcopy', FULLSCREEN_VS, DEPTHCOPY_FS)

    def _fs(self, name):
        if name not in self.fs_vaos:
            self.fs_vaos[name] = self.ctx.vertex_array(self.progs[name], [(self.quad, '2f', 'in_pos')])
        return self.fs_vaos[name]

    def _tex(self, size, comps, dtype='f1', filt=moderngl.LINEAR):
        t = self.ctx.texture(size, comps, dtype=dtype)
        t.filter = (filt, filt)
        t.repeat_x = t.repeat_y = False
        return t

    def _build_targets(self):
        ctx = self.ctx
        iw, ih = self.iw, self.ih
        N = moderngl.NEAREST
        self.g_albedo = self._tex((iw, ih), 4, filt=N)
        self.g_normal = self._tex((iw, ih), 4, filt=N)
        self.g_extra = self._tex((iw, ih), 4, filt=N)
        self.g_depth = ctx.depth_texture((iw, ih))
        self.g_depth.filter = (N, N)
        self.g_depth.repeat_x = self.g_depth.repeat_y = False
        self.g_depth.compare_func = ''
        self.gbuf = ctx.framebuffer(color_attachments=[self.g_albedo, self.g_normal, self.g_extra],
                                    depth_attachment=self.g_depth)
        hw, hh = max(1, iw // 2), max(1, ih // 2)
        self.ao_a = self._tex((hw, hh), 1, 'f2')
        self.ao_b = self._tex((hw, hh), 1, 'f2')
        self.ao_fbo_a = ctx.framebuffer(color_attachments=[self.ao_a])
        self.ao_fbo_b = ctx.framebuffer(color_attachments=[self.ao_b])
        fw, fh = max(1, iw // 3), max(1, ih // 3)
        self.fog_a = self._tex((fw, fh), 4, 'f2')
        self.fog_b = self._tex((fw, fh), 4, 'f2')
        self.fog_fbo_a = ctx.framebuffer(color_attachments=[self.fog_a])
        self.fog_fbo_b = ctx.framebuffer(color_attachments=[self.fog_b])
        self.hdr = self._tex((iw, ih), 4, 'f2')
        self.hdr_fbo = ctx.framebuffer(color_attachments=[self.hdr])
        self.hdr2 = self._tex((iw, ih), 4, 'f2')
        self.hdr2_fbo = ctx.framebuffer(color_attachments=[self.hdr2])
        self.ldr = self._tex((iw, ih), 4)
        self.ldr_fbo = ctx.framebuffer(color_attachments=[self.ldr])
        self.aa = self._tex((iw, ih), 4)
        self.aa_fbo = ctx.framebuffer(color_attachments=[self.aa])
        self.out = self._tex((self.W, self.H), 4)
        self.out_fbo = ctx.framebuffer(color_attachments=[self.out])
        self.bloom_tex = []
        for div in (4, 8):
            pair = []
            for _ in range(2):
                t = self._tex((max(1, iw // div), max(1, ih // div)), 4, 'f2')
                pair.append((t, ctx.framebuffer(color_attachments=[t])))
            self.bloom_tex.append(pair)
        S = self.shadow_res
        self.sh_near = ctx.depth_texture((S, S))
        self.sh_near.repeat_x = self.sh_near.repeat_y = False
        self.sh_near_fbo = ctx.framebuffer(depth_attachment=self.sh_near)
        self.sh_static = ctx.depth_texture((S, S))
        self.sh_static.repeat_x = self.sh_static.repeat_y = False
        self.sh_static.filter = (N, N)
        self.sh_static.compare_func = ''
        self.sh_static_fbo = ctx.framebuffer(depth_attachment=self.sh_static)
        self.static_key = None
        self.sh_far = ctx.depth_texture((S, S))
        self.sh_far.repeat_x = self.sh_far.repeat_y = False
        self.sh_far_fbo = ctx.framebuffer(depth_attachment=self.sh_far)
        L = moderngl.LINEAR
        self.smp_near_cmp = ctx.sampler(texture=self.sh_near, filter=(L, L), compare_func='<=', repeat_x=False,
                                        repeat_y=False)
        self.smp_near_raw = ctx.sampler(texture=self.sh_near, filter=(N, N), compare_func='', repeat_x=False,
                                        repeat_y=False)
        self.smp_far_cmp = ctx.sampler(texture=self.sh_far, filter=(L, L), compare_func='<=', repeat_x=False,
                                       repeat_y=False)

    def set_blocks(self, rgba, emit, noise, layer_ids):
        """rgba: (L,16,16,4) uint8; emit: (L,16,16) float; noise: (n,n) float tileable; layer_ids: dict of the
        special layers (water, grass_top, grass_side, oak_leaves, spruce_leaves, birch_leaves, tall_grass)."""
        ctx = self.ctx
        n = rgba.shape[0]
        self.t_blocks = ctx.texture_array((16, 16, n), 4, np.ascontiguousarray(rgba).tobytes())
        self.t_blocks.build_mipmaps(0, 2)
        self.t_blocks.filter = (moderngl.NEAREST_MIPMAP_LINEAR, moderngl.NEAREST)
        self.t_emit = ctx.texture_array((16, 16, n), 1, np.ascontiguousarray(emit, np.float32).tobytes(), dtype='f4')
        self.t_emit.filter = (moderngl.NEAREST, moderngl.NEAREST)
        nz = np.ascontiguousarray(noise, np.float32)
        self.t_noise = ctx.texture(nz.shape[::-1], 1, nz.tobytes(), dtype='f4')
        self.t_noise.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.layer_ids = layer_ids

    def set_moon(self, img):
        img = np.ascontiguousarray(img, np.uint8)
        self.t_moon = self.ctx.texture((img.shape[1], img.shape[0]), 4, img[::-1].tobytes())
        self.t_moon.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.t_moon.repeat_x = self.t_moon.repeat_y = False

    def set_sky(self, key, pano):
        """Equirectangular panorama (H, W, 3) linear: row 0 = zenith, last row = 12 degrees below the horizon."""
        h, w = pano.shape[:2]
        t = self.ctx.texture((w, h), 3, np.ascontiguousarray(pano, np.float32).tobytes(), dtype='f4')
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = True
        t.repeat_y = False
        self.skies[key] = t

    def add_world(self, key, mesh, light, light_origin):
        """mesh: dict bucket -> (V, I, sections); light: (X,Y,Z,4) uint8 volume starting at light_origin."""
        ctx = self.ctx
        w = {'buckets': {}}
        for bucket, (V, I, secs) in mesh.items():
            if not len(I):
                continue
            vbo = ctx.buffer(np.ascontiguousarray(V, np.float32).tobytes())
            ibo = ctx.buffer(np.ascontiguousarray(I, np.uint32).tobytes())
            fmt = (vbo, '3f 3f 2f 1f 3f', 'in_pos', 'in_nrm', 'in_uv', 'in_layer', 'in_tint')
            vao = ctx.vertex_array(self.progs['static'], [fmt], index_buffer=ibo, index_element_size=4)
            svao = ctx.vertex_array(self.progs['sh_static'], [(vbo, '3f 12x 2f 1f 12x', 'in_pos', 'in_uv', 'in_layer')],
                                    index_buffer=ibo, index_element_size=4)
            w['buckets'][bucket] = {'vao': vao, 'svao': svao, 'secs': secs, 'vbo': vbo, 'ibo': ibo}
        X, Y, Z = light.shape[:3]
        data = np.ascontiguousarray(np.transpose(light, (2, 1, 0, 3)))
        t = ctx.texture3d((X, Y, Z), 4, data.tobytes())
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = t.repeat_y = t.repeat_z = False
        w['lvol'] = t
        w['lorg'] = np.asarray(light_origin, float)
        w['lsize'] = np.array([X, Y, Z], float)
        self.worlds[key] = w

    def use_world(self, key):
        self.world = self.worlds[key]
        self.world_key = key

    def add_kind(self, name, mesh, layers=None, tex=None, emit=None, filt=moderngl.NEAREST):
        """An instanced mesh (pos3 nrm3 uv2 layer1 per vertex; layer -1 = the instance's) with its own texture array
        (layers: list of (h, w, 4) uint8, all the same size) or tex='blocks' to use the block atlas."""
        ctx = self.ctx
        vbo = ctx.buffer(np.ascontiguousarray(mesh, np.float32).tobytes())
        k = {'vbo': vbo, 'n': len(mesh)}
        if tex == 'blocks':
            k['tex'] = None
        elif isinstance(tex, str):
            k['tex'] = self.kinds[tex]['tex']                  # share another kind's texture array
            k['size'] = self.kinds[tex].get('size')
        else:
            pl = np.stack([np.ascontiguousarray(a, np.uint8) for a in layers])
            t = ctx.texture_array((pl.shape[2], pl.shape[1], pl.shape[0]), 4, pl.tobytes())
            t.filter = (filt, filt)
            k['tex'] = t
            k['size'] = (pl.shape[2], pl.shape[1])
        k['emit'] = None
        if emit is not None:
            e = np.stack([np.ascontiguousarray(a, np.float32) for a in emit])
            te = ctx.texture_array((e.shape[2], e.shape[1], e.shape[0]), 1, e.tobytes(), dtype='f4')
            te.filter = (moderngl.NEAREST, moderngl.NEAREST)
            k['emit'] = te
        self.kinds[name] = k

    def update_kind_layer(self, name, layer, img):
        """Replace one texture layer of a kind (e.g. the monitor's screen) with an (h, w, 4) uint8 image."""
        k = self.kinds[name]
        w, h = k['size']
        k['tex'].write(np.ascontiguousarray(img, np.uint8).tobytes(), viewport=(0, 0, layer, w, h, 1))

    # -- far shadow map -------------------------------------------------------------------------
    def _far_shadow(self, e, center):
        key = (self.world_key, tuple(np.round(e['light_dir'], 4)), tuple(np.round(center, 0)))
        if self.far_cache.get('key') == key:
            return self.far_cache['lvp']
        lvp = light_matrix(center, self.far_half, e['light_dir'], depth=600.0)
        ctx = self.ctx
        self.sh_far_fbo.use()
        self.sh_far_fbo.clear(depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        ctx.polygon_offset = (2.0, 4.0)
        P = self.progs['sh_static']
        P['u_lvp'].write(m4(lvp))
        self.t_blocks.use(0)
        P['u_blocks'] = 0
        for bucket, b in self.world['buckets'].items():
            if bucket == 'plants':
                continue
            P['u_cutout'] = 1.0 if bucket == 'cutout' else 0.0
            for (first, count, lo, hi) in b['secs']:
                if hi[2] >= -8.0:
                    b['svao'].render(first=first, vertices=count)
        ctx.polygon_offset = (0.0, 0.0)
        self.far_cache = {'key': key, 'lvp': lvp}
        return lvp

    # -- render ---------------------------------------------------------------------------------
    def render(self, cam, e, instances=None, lights=None, particles=None, streaks=None, far_center=None,
               near_center=None, clip=None, clip_z=None, plant_dist=48.0):
        """cam: dict(eye, target, fov, up, roll); e: Env; instances: dict kind -> (M, 16) float32;
        lights: (L, 7) x y z radius r g b; particles: dict(soft=(P,8), glow=(P,8)) pos3 size1 rgba4;
        streaks: (S, 8) head3 tail3 width alpha; clip: (lo, hi) boxes of sections to draw (None = all)."""
        ctx = self.ctx
        W = self.world
        iw, ih = self.iw, self.ih
        near, far = 0.05, 1200.0
        eye = np.asarray(cam['eye'], float)
        up = np.array(cam.get('up', (0, 0, 1)), float)
        view = look_at(eye, cam['target'], up)
        if cam.get('roll', 0.0):
            rr = np.radians(cam['roll'])
            rz = np.eye(4)
            rz[0, 0], rz[0, 1], rz[1, 0], rz[1, 1] = np.cos(rr), -np.sin(rr), np.sin(rr), np.cos(rr)
            view = rz @ view
        proj = perspective(cam['fov'], iw / ih, near, far)
        vp = proj @ view
        planes = frustum_planes(vp)
        fwd = np.asarray(cam['target'], float) - eye
        fwd /= np.linalg.norm(fwd)
        nc = near_center if near_center is not None else eye + fwd * (self.near_half * 0.55)
        ldir = np.asarray(e['light_dir'], float)
        ldir = ldir / np.linalg.norm(ldir)
        if clip_z is None:
            clip_z = (-8.0, 1e9) if eye[2] > -3.0 else (-1e9, 0.0)
        lvp_far = self._far_shadow(e, far_center if far_center is not None else np.round(eye / 32.0) * 32.0)

        def sec_ok(lo, hi, pl):
            if clip is not None:
                ok = False
                for (a, b) in clip:
                    if np.all(hi >= a) and np.all(lo <= b):
                        ok = True
                        break
                if not ok:
                    return False
            return aabb_visible(pl, lo, hi)

        # instance buffers
        inst = []
        for kind, arr in (instances or {}).items():
            if arr is None or not len(arr):
                continue
            k = self.kinds[kind]
            buf = ctx.buffer(np.ascontiguousarray(arr, np.float32).tobytes())
            g = ctx.vertex_array(self.progs['prop'], [
                (k['vbo'], '3f 3f 2f 1f', 'in_pos', 'in_nrm', 'in_uv', 'in_layer'),
                (buf, INST_FMT, *INST_NAMES)])
            s = ctx.vertex_array(self.progs['sh_prop'], [
                (k['vbo'], '3f 12x 2f 1f', 'in_pos', 'in_uv', 'in_layer'),
                (buf, '3f 4f 3f 1f 20x/i', 'i_pos', 'i_quat', 'i_scale', 'i_layer')])
            shadow = arr[:, 15] != MAT_HAND
            inst.append((kind, g, s, len(arr), buf, bool(shadow.any())))

        prof = getattr(self, 'profile', None)
        import time as _t

        def mark(name):
            if prof is not None:
                ctx.finish()
                prof.append((name, _t.time()))
        mark('start')
        # 1. near shadow map: the static world is cached (re-rendered when the snapped centre or the light moves),
        # copied in, and the moving things drawn on top
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        lit = float(np.max(e['light_col'])) > 1e-3
        ncs = np.round(np.asarray(nc, float) / 8.0) * 8.0
        lvp_near = light_matrix(ncs, self.near_half, ldir, depth=400.0, res=self.shadow_res)
        skey = (self.world_key, tuple(np.round(ldir, 4)), tuple(ncs), clip_z)
        if lit and skey != self.static_key:
            self.sh_static_fbo.use()
            self.sh_static_fbo.clear(depth=1.0)
            ctx.polygon_offset = (1.5, 3.0)
            Ps = self.progs['sh_static']
            Ps['u_lvp'].write(m4(lvp_near))
            self.t_blocks.use(0)
            Ps['u_blocks'] = 0
            lplanes = frustum_planes(lvp_near)
            for bucket, b in W['buckets'].items():
                if bucket == 'plants':
                    continue
                Ps['u_cutout'] = 1.0 if bucket == 'cutout' else 0.0
                for (first, count, lo, hi) in b['secs']:
                    if clip_z is not None and (hi[2] < clip_z[0] or lo[2] > clip_z[1]):
                        continue
                    if aabb_visible(lplanes, lo, hi):
                        b['svao'].render(first=first, vertices=count)
            ctx.polygon_offset = (0.0, 0.0)
            self.static_key = skey
        self.sh_near_fbo.use()
        if lit:
            ctx.depth_func = '1'
            self.sh_static.use(0)
            self.progs['depthcopy']['u_src'] = 0
            self._fs('depthcopy').render(moderngl.TRIANGLE_STRIP)
            ctx.depth_func = '<'
            ctx.polygon_offset = (1.5, 3.0)
            Pp = self.progs['sh_prop']
            Pp['u_lvp'].write(m4(lvp_near))
            for (kind, g, s, n, buf, cast) in inst:
                if not cast:
                    continue
                k = self.kinds[kind]
                (k['tex'] or self.t_blocks).use(0)
                Pp['u_tex'] = 0
                s.render(instances=n)
            ctx.polygon_offset = (0.0, 0.0)
        else:
            self.sh_near_fbo.clear(depth=1.0)
        mark('shadow')
        # 2. G-buffer
        self.gbuf.use()
        self.gbuf.clear(0, 0, 0, 0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        P = self.progs['static']
        P['u_vp'].write(m4(vp))
        self.t_blocks.use(0)
        self.t_emit.use(1)
        self.t_noise.use(2)
        P['u_blocks'] = 0
        P['u_emit'] = 1
        P['u_noise'] = 2
        P['u_time'] = float(self.time)
        P['u_wind'] = float(e['wind'])
        li = self.layer_ids
        P['u_sway'] = (li['oak_leaves'], li['spruce_leaves'], li['birch_leaves'], li['tall_grass'])
        P['u_leafy'] = (li['oak_leaves'], li['spruce_leaves'], li['birch_leaves'], li['tall_grass'])
        P['u_water'] = li['water']
        P['u_grass_top'] = li['grass_top']
        P['u_grass_side'] = li['grass_side']
        P['u_emit_scale'] = 1.0
        maxd2 = float(e.get('max_dist', self.max_dist)) ** 2
        for bucket in ('solid', 'cutout', 'plants'):
            if bucket not in W['buckets']:
                continue
            b = W['buckets'][bucket]
            P['u_cutout'] = 0.0 if bucket == 'solid' else 1.0
            if bucket != 'solid':
                ctx.disable(moderngl.CULL_FACE)
            md2 = maxd2 if bucket != 'plants' else plant_dist ** 2
            for (first, count, lo, hi) in b['secs']:
                if hi[2] < clip_z[0] or lo[2] > clip_z[1]:
                    continue
                c = np.clip(eye, lo, hi)
                if np.sum((c - eye) ** 2) > md2:
                    continue
                if sec_ok(lo, hi, planes):
                    b['vao'].render(first=first, vertices=count)
            ctx.enable(moderngl.CULL_FACE)
        Pr = self.progs['prop']
        Pr['u_vp'].write(m4(vp))
        ctx.disable(moderngl.CULL_FACE)
        for (kind, g, s, n, buf, cast) in inst:
            k = self.kinds[kind]
            (k['tex'] or self.t_blocks).use(3)
            Pr['u_tex'] = 3
            if k['emit'] is not None or k['tex'] is None:
                (k['emit'] if k['emit'] is not None else self.t_emit).use(4)
                Pr['u_emit'] = 4
                Pr['u_has_emit'] = 1
            else:
                Pr['u_has_emit'] = 0
            g.render(instances=n)
        ctx.enable(moderngl.CULL_FACE)

        mark('gbuffer')
        # 3. SSAO
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        invproj = np.linalg.inv(proj)
        if e['ssao'] > 0:
            self.ao_fbo_a.use()
            S = self.progs['ssao']
            self.g_depth.use(0)
            self.g_normal.use(1)
            S['u_depth'] = 0
            S['u_normal'] = 1
            S['u_proj'].write(m4(proj))
            S['u_invproj'].write(m4(invproj))
            S['u_viewrot'].write(np.ascontiguousarray(view[:3, :3].T, np.float32).tobytes())
            S['u_radius'] = 0.9 * float(e['ssao'])
            self._fs('ssao').render(moderngl.TRIANGLE_STRIP)
            self._blur(self.ao_a, self.ao_fbo_b, self.ao_b, self.ao_fbo_a, near, far)
        else:
            self.ao_fbo_a.use()
            self.ao_fbo_a.clear(1.0, 1.0, 1.0, 1.0)

        # common light uniforms
        def light_common(Pl, units):
            def put(k, v):
                if k in Pl:
                    Pl[k] = v
            W['lvol'].use(units[0])
            put('u_lvol', units[0])
            put('u_lvol_org', tuple(W['lorg']))
            put('u_lvol_size', tuple(W['lsize']))
            self.smp_near_cmp.use(units[1])
            self.smp_near_raw.use(units[2])
            self.smp_far_cmp.use(units[3])
            put('u_sh_near', units[1])
            put('u_sh_near_raw', units[2])
            put('u_sh_far', units[3])
            if 'u_lvp_near' in Pl:
                Pl['u_lvp_near'].write(m4(lvp_near))
            if 'u_lvp_far' in Pl:
                Pl['u_lvp_far'].write(m4(lvp_far))
            put('u_near_half', self.near_half)
            put('u_shadow_res', float(self.shadow_res))
            put('u_sun_dir', tuple(ldir))
            put('u_blk_flicker', float(e['flicker']))

        mark('ssao')
        invvp = np.linalg.inv(vp)
        # 4. volumetric fog
        fog = e['fog']
        use_vfog = fog[0] > 0 and fog[3] > 0
        if use_vfog:
            self.fog_fbo_a.use()
            F = self.progs['vfog']
            self.g_depth.use(0)
            F['u_depth'] = 0
            light_common(F, (1, 2, 3, 4))
            F['u_invvp'].write(m4(invvp))
            F['u_cam'] = tuple(eye)
            F['u_sun_col'] = tuple(np.asarray(e['light_col'], float) * np.asarray(e['fog_sun'], float))
            F['u_blk_col'] = tuple(np.asarray(e['blk_col'], float) * 0.35)
            F['u_soul_col'] = tuple(np.asarray(e['soul_col'], float) * 0.35)
            F['u_amb_col'] = tuple(np.asarray(e['fog_amb'], float))
            F['u_fog'] = tuple(float(v) for v in fog)
            F['u_g'] = float(e['fog_g'])
            F['u_steps'] = int(e.get('fog_steps', 18))
            F['u_frame'] = float(self.frame_counter % 64)
            self._fs('vfog').render(moderngl.TRIANGLE_STRIP)
            for smp, unit in ((self.smp_near_cmp, 2), (self.smp_near_raw, 3), (self.smp_far_cmp, 4)):
                smp.clear(unit)
            self._blur(self.fog_a, self.fog_fbo_b, self.fog_b, self.fog_fbo_a, near, far)

        mark('vfog')
        # 5. lighting
        self.hdr_fbo.use()
        ctx.disable(moderngl.DEPTH_TEST)
        L = self.progs['light']
        for kname, unit, tex in (('u_albedo', 0, self.g_albedo), ('u_normal', 1, self.g_normal),
                                 ('u_extra', 2, self.g_extra), ('u_depth', 3, self.g_depth), ('u_ao', 4, self.ao_a),
                                 ('u_fogtex', 5, self.fog_a), ('u_sky', 6, self.skies[e['sky']]),
                                 ('u_moon', 7, self.t_moon)):
            tex.use(unit)
            L[kname] = unit
        light_common(L, (8, 9, 10, 11))
        L['u_invvp'].write(m4(invvp))
        L['u_cam'] = tuple(eye)
        for kname in ('sky_amb', 'gnd_amb', 'blk_col', 'soul_col', 'min_amb', 'sky_mul', 'aerial_col', 'moon_dir',
                      'sun_disc_dir', 'sun_disc_col'):
            L['u_' + kname] = tuple(float(v) for v in e[kname])
        L['u_sun_col'] = tuple(float(v) for v in e['light_col'])
        L['u_emit_gain'] = float(e['emit_gain'])
        L['u_aerial'] = float(e['aerial'])
        L['u_use_vfog'] = 1 if use_vfog else 0
        L['u_moon_size'] = float(e['moon_size'])
        L['u_moon_bright'] = float(e['moon_bright'])
        L['u_sun_size'] = float(e['sun_size'])
        L['u_stars'] = float(e['stars'])
        L['u_time'] = float(self.time)
        L['u_flash'] = float(e['flash'])
        lpos = np.zeros((16, 4), np.float32)
        lcol = np.zeros((16, 4), np.float32)
        nl = 0
        if lights is not None and len(lights):
            lights = np.asarray(lights, np.float32)
            nl = min(16, len(lights))
            lpos[:nl] = lights[:nl, :4]
            lcol[:nl, :3] = lights[:nl, 4:7]
        L['u_nl'] = nl
        L['u_lpos'].write(lpos.tobytes())
        L['u_lcol'].write(lcol.tobytes())
        self._fs('light').render(moderngl.TRIANGLE_STRIP)
        for smp, unit in ((self.smp_near_cmp, 9), (self.smp_near_raw, 10), (self.smp_far_cmp, 11)):
            smp.clear(unit)

        mark('light')
        # 6. particles and streaks (rain)
        self._particles(particles, streaks, view, proj, vp, eye, near, far)

        # 7. depth of field
        self.out_hdr = self.hdr
        if self.dof is not None and self.dof.get('k', 0.0) > 0.0:
            self._dof(near, far)

        for (kind, g, s, n, buf, cast) in inst:
            g.release()
            s.release()
            buf.release()
        self.cam_view, self.cam_proj = view, proj
        return self.out_hdr

    def _blur(self, src, fbo_b, tex_b, fbo_a, near, far):
        B = self.progs['blur']
        B['u_near'] = near
        B['u_far'] = far
        hw, hh = src.size
        B['u_texel'] = (1.0 / hw, 1.0 / hh)
        self.g_depth.use(1)
        B['u_depth'] = 1
        B['u_src'] = 0
        fbo_b.use()
        src.use(0)
        B['u_dir'] = (1.0, 0.0)
        self._fs('blur').render(moderngl.TRIANGLE_STRIP)
        fbo_a.use()
        tex_b.use(0)
        B['u_dir'] = (0.0, 1.0)
        self._fs('blur').render(moderngl.TRIANGLE_STRIP)

    def _particles(self, particles, streaks, view, proj, vp, eye, near, far):
        ctx = self.ctx
        if not particles and streaks is None:
            return
        self.hdr_fbo.use()
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        self.g_depth.use(0)
        for key, soft in (('soft', 1), ('glow', 0)):
            arr = (particles or {}).get(key)
            if arr is None or not len(arr):
                continue
            if soft:
                vz = (np.c_[arr[:, :3], np.ones(len(arr))] @ view.T)[:, 2]
                arr = arr[np.argsort(vz)]
            buf = ctx.buffer(np.ascontiguousarray(arr, np.float32).tobytes())
            Pp = self.progs['particle']
            vao = ctx.vertex_array(Pp, [(self.corner_vbo, '2f', 'in_corner'),
                                        (buf, '3f 1f 4f/i', 'i_pos', 'i_size', 'i_col')])
            Pp['u_view'].write(m4(view))
            Pp['u_proj'].write(m4(proj))
            Pp['u_depth'] = 0
            Pp['u_res'] = (float(self.iw), float(self.ih))
            Pp['u_near'] = near
            Pp['u_far'] = far
            Pp['u_soft'] = soft
            ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA) if soft else (moderngl.ONE, moderngl.ONE)
            vao.render(moderngl.TRIANGLE_STRIP, instances=len(arr))
            vao.release()
            buf.release()
        if streaks is not None and len(streaks):
            data = np.ascontiguousarray(streaks[:, :8], np.float32)
            buf = ctx.buffer(data.tobytes())
            Sp = self.progs['streak']
            vao = ctx.vertex_array(Sp, [(self.corner_vbo, '2f', 'in_corner'),
                                        (buf, '3f 3f 1f 1f/i', 'i_head', 'i_tail', 'i_width', 'i_alpha')])
            Sp['u_vp'].write(m4(vp))
            Sp['u_view'].write(m4(view))
            Sp['u_cam'] = tuple(eye)
            Sp['u_depth'] = 0
            Sp['u_res'] = (float(self.iw), float(self.ih))
            Sp['u_near'] = near
            Sp['u_far'] = far
            Sp['u_col'] = tuple(self.streak_col) if hasattr(self, 'streak_col') else (0.5, 0.55, 0.65)
            ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
            vao.render(moderngl.TRIANGLE_STRIP, instances=len(data))
            vao.release()
            buf.release()
        ctx.disable(moderngl.BLEND)

    def _dof(self, near, far):
        ctx = self.ctx
        src = self.out_hdr
        self.hdr2_fbo.use()
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        sc = self.iw / 1920.0
        maxr = float(self.dof.get('maxr', 10.0)) * sc
        C = self.progs['dof_coc']
        src.use(0)
        self.g_depth.use(1)
        C['u_hdr'] = 0
        C['u_depth'] = 1
        C['u_near'] = near
        C['u_far'] = far
        C['u_focus'] = float(self.dof['focus'])
        C['u_k'] = float(self.dof['k']) * sc
        C['u_maxr'] = maxr
        self._fs('dof_coc').render(moderngl.TRIANGLE_STRIP)
        self.hdr_fbo.use()
        D = self.progs['dof']
        self.hdr2.use(0)
        D['u_src'] = 0
        D['u_maxr'] = maxr
        D['u_step'] = 1.25 * sc
        D['u_texel'] = (1.0 / self.iw, 1.0 / self.ih)
        self._fs('dof').render(moderngl.TRIANGLE_STRIP)
        self.out_hdr = self.hdr

    def _bloom(self, src, thresh):
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        (a4, fa4), (b4, fb4) = self.bloom_tex[0]
        (a8, fa8), (b8, fb8) = self.bloom_tex[1]
        Br = self.progs['bright']
        fa4.use()
        src.use(0)
        Br['u_src'] = 0
        Br['u_texel'] = (1.0 / self.iw, 1.0 / self.ih)
        Br['u_thresh'] = thresh
        self._fs('bright').render(moderngl.TRIANGLE_STRIP)
        Gp = self.progs['gauss']
        Gp['u_src'] = 0
        for (s_, dst_fbo, d) in ((a4, fb4, (1, 0)), (b4, fa4, (0, 1))):
            dst_fbo.use()
            s_.use(0)
            Gp['u_dir'] = (d[0] * 1.5 / a4.size[0], d[1] * 1.5 / a4.size[1])
            self._fs('gauss').render(moderngl.TRIANGLE_STRIP)
        fa8.use()
        a4.use(0)
        Gp['u_dir'] = (0.0, 0.0)
        self._fs('gauss').render(moderngl.TRIANGLE_STRIP)
        for (s_, dst_fbo, d) in ((a8, fb8, (1, 0)), (b8, fa8, (0, 1))):
            dst_fbo.use()
            s_.use(0)
            Gp['u_dir'] = (d[0] * 2.0 / a8.size[0], d[1] * 2.0 / a8.size[1])
            self._fs('gauss').render(moderngl.TRIANGLE_STRIP)
        return a4, a8

    def finish(self, e):
        """Tonemap + grade + FXAA + downsample the current HDR buffer; (H, W, 3) uint8."""
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        src = self.out_hdr
        b1, b2 = self._bloom(src, float(e['bloom_thresh']))
        self.ldr_fbo.use()
        Pp = self.progs['post']
        src.use(0)
        b1.use(1)
        b2.use(2)
        Pp['u_hdr'] = 0
        Pp['u_bloom1'] = 1
        Pp['u_bloom2'] = 2
        Pp['u_bloom'] = float(e['bloom'])
        Pp['u_exposure'] = float(e['exposure'])
        Pp['u_sat'] = float(e['sat'])
        Pp['u_contrast'] = float(e['contrast'])
        Pp['u_vignette'] = float(e['vignette'])
        Pp['u_lift'] = tuple(float(v) for v in e['lift'])
        Pp['u_gain'] = tuple(float(v) for v in e['gain'])
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

    def project(self, pts):
        """World points (N,3) -> pixel coords (N,2) in the output image and view depth (N,) of the last render."""
        pts = np.asarray(pts, float).reshape(-1, 3)
        h = np.c_[pts, np.ones(len(pts))]
        v = h @ self.cam_view.T
        c = v @ self.cam_proj.T
        ndc = c[:, :2] / c[:, 3:4]
        x = (ndc[:, 0] * 0.5 + 0.5) * self.W
        y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * self.H
        return np.stack([x, y], -1), -v[:, 2]
