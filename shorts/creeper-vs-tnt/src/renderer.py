"""Deferred OpenGL renderer (moderngl, runs headless on Mesa llvmpipe).

Pipeline per frame:
  near shadow map (dynamic objects + nearby trees)  ->  G-buffer (albedo, normal+material, depth)
  -> SSAO (half res) + bilateral blur -> lighting (sun w/ PCSS-ish soft shadows, sky ambient,
  metal reflections, foliage translucency, aerial fog) -> tonemap/grade -> FXAA -> downsample -> read back.
"""
import numpy as np
import moderngl

from sky import SUN_DIR, EL_MIN

MAT_TERRAIN, MAT_LEAF, MAT_SKIN, MAT_FLESH, MAT_BONE, MAT_GROUND, MAT_TNT = 1, 2, 3, 4, 5, 7, 8


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
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv; in float in_layer; in vec3 in_tint;
out vec3 v_nrm; out vec2 v_uv; flat out float v_layer; out vec3 v_tint; out vec3 v_wpos;
void main(){
    v_nrm = in_nrm; v_uv = in_uv; v_layer = in_layer; v_tint = in_tint; v_wpos = in_pos;
    gl_Position = u_vp * vec4(in_pos, 1.0);
}
"""

STATIC_FS = """
#version 430
uniform sampler2DArray u_blocks;
uniform sampler2D u_tintnoise;
uniform float u_is_ground;
in vec3 v_nrm; in vec2 v_uv; flat in float v_layer; in vec3 v_tint; in vec3 v_wpos;
layout(location=0) out vec4 o_albedo;
layout(location=1) out vec4 o_normal;
void main(){
    vec3 c = texture(u_blocks, vec3(v_uv, v_layer)).rgb;
    c = pow(c, vec3(2.2)) * v_tint;
    int lay = int(v_layer + 0.5);
    float mat = %d.0;
    if (lay == 7) mat = %d.0;
    if (lay == 0 || lay == 2) {
        // large scale colour variation on grass so fields don't look uniform
        float n = texture(u_tintnoise, v_wpos.xy / 97.0).r;
        float n2 = texture(u_tintnoise, v_wpos.xy / 23.0 + 0.37).r;
        vec3 t = mix(vec3(0.90, 0.95, 0.86), vec3(1.07, 1.05, 0.93), n) * mix(0.95, 1.04, n2);
        if (lay == 0) c *= t;
        else if (fract(-v_wpos.z) < 0.3) c *= t;
    }
    if (u_is_ground > 0.5) mat = %d.0;
    o_albedo = vec4(sqrt(clamp(c, 0.0, 1.0)), 0.0);
    o_normal = vec4(v_nrm * 0.5 + 0.5, mat / 255.0);
}
""" % (MAT_TERRAIN, MAT_LEAF, MAT_GROUND)

VOXEL_VS = """
#version 430
uniform mat4 u_vp;
in vec3 in_pos; in vec3 in_nrm; in float in_face;
in vec3 i_pos; in vec4 i_quat; in float i_scale; in vec4 i_cx; in vec4 i_cy; in vec4 i_cz; in vec4 i_inner;
out vec3 v_nrm; flat out vec3 v_col; flat out float v_mat;
""" + QROT + """
void main(){
    vec3 p = qrot(i_quat, in_pos * i_scale) + i_pos;
    v_nrm = qrot(i_quat, in_nrm);
    int face = int(in_face + 0.5);
    int vis = int(i_cy.a * 255.0 + 0.5);
    if (((vis >> face) & 1) == 0) { gl_Position = vec4(0.0, 0.0, -10.0, 1.0); v_nrm = vec3(0.0); v_col = vec3(0.0); v_mat = 0.0; return; }
    int mask = int(i_cx.a * 255.0 + 0.5);
    vec3 skin = face < 2 ? i_cx.rgb : (face < 4 ? i_cy.rgb : i_cz.rgb);
    bool isSkin = ((mask >> face) & 1) == 1;
    v_col = isSkin ? skin : i_inner.rgb;
    float innerMat = i_inner.a > 0.5 ? %d.0 : %d.0;
    v_mat = isSkin ? %d.0 : innerMat;
    gl_Position = u_vp * vec4(p, 1.0);
}
""" % (MAT_BONE, MAT_FLESH, MAT_SKIN)

VOXEL_FS = """
#version 430
in vec3 v_nrm; flat in vec3 v_col; flat in float v_mat;
layout(location=0) out vec4 o_albedo;
layout(location=1) out vec4 o_normal;
void main(){
    vec3 c = pow(v_col, vec3(2.2));
    float spec = 0.0;
    if (int(v_mat + 0.5) == %d) spec = 0.09;   // wet flesh
    if (int(v_mat + 0.5) == %d) spec = 0.06;
    o_albedo = vec4(sqrt(clamp(c, 0.0, 1.0)), spec);
    o_normal = vec4(normalize(v_nrm) * 0.5 + 0.5, v_mat / 255.0);
}
""" % (MAT_FLESH, MAT_BONE)

TNT_VS = """
#version 430
uniform mat4 u_vp;
in vec3 in_pos; in vec3 in_nrm; in float in_face;
in vec3 i_pos; in float i_scale; in float i_flash;
out vec3 v_nrm; out vec2 v_uv; flat out float v_layer; flat out float v_flash;
void main(){
    int f = int(in_face + 0.5);
    vec3 lp = in_pos;
    vec2 uv;
    if (f == 0) uv = vec2(lp.y + 0.5, 0.5 - lp.z);
    else if (f == 1) uv = vec2(0.5 - lp.y, 0.5 - lp.z);
    else if (f == 2) uv = vec2(0.5 - lp.x, 0.5 - lp.z);
    else if (f == 3) uv = vec2(lp.x + 0.5, 0.5 - lp.z);
    else if (f == 4) uv = vec2(lp.x + 0.5, 0.5 - lp.y);
    else uv = vec2(lp.x + 0.5, lp.y + 0.5);
    v_uv = uv;
    v_layer = f < 4 ? 0.0 : (f == 4 ? 1.0 : 2.0);
    v_nrm = in_nrm;
    v_flash = i_flash;
    gl_Position = u_vp * vec4(lp * i_scale + i_pos, 1.0);
}
"""

TNT_FS = """
#version 430
uniform sampler2DArray u_tnt;
in vec3 v_nrm; in vec2 v_uv; flat in float v_layer; flat in float v_flash;
layout(location=0) out vec4 o_albedo;
layout(location=1) out vec4 o_normal;
void main(){
    vec3 c = pow(texture(u_tnt, vec3(v_uv, v_layer)).rgb, vec3(2.2));
    o_albedo = vec4(sqrt(clamp(c, 0.0, 1.0)), v_flash);     // alpha carries the white fuse flash
    o_normal = vec4(normalize(v_nrm) * 0.5 + 0.5, %d.0 / 255.0);
}
""" % MAT_TNT

SHADOW_STATIC_VS = """
#version 430
uniform mat4 u_lvp;
in vec3 in_pos;
void main(){ gl_Position = u_lvp * vec4(in_pos, 1.0); }
"""
SHADOW_VOXEL_VS = """
#version 430
uniform mat4 u_lvp;
in vec3 in_pos; in float in_face; in vec3 i_pos; in vec4 i_quat; in float i_scale; in vec4 i_cy;
""" + QROT + """
void main(){
    int vis = int(i_cy.a * 255.0 + 0.5);
    if (((vis >> int(in_face + 0.5)) & 1) == 0) { gl_Position = vec4(0.0, 0.0, -10.0, 1.0); return; }
    gl_Position = u_lvp * vec4(qrot(i_quat, in_pos * i_scale) + i_pos, 1.0);
}
"""
SHADOW_TNT_VS = """
#version 430
uniform mat4 u_lvp;
in vec3 in_pos; in vec3 i_pos; in float i_scale;
void main(){ gl_Position = u_lvp * vec4(in_pos * i_scale + i_pos, 1.0); }
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
uniform float u_near_half;
uniform float u_el_min;
uniform float u_shadow_res;
uniform int u_nl;
uniform vec4 u_lpos[16];
uniform vec4 u_lcol[16];
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
    if (d >= 1.0) { o_col = vec4(sky(vdir), 1.0); return; }

    vec4 nm = texture(u_normal, v_uv);
    vec3 N = normalize(nm.xyz * 2.0 - 1.0);
    int mat = int(nm.a * 255.0 + 0.5);
    vec4 al = texture(u_albedo, v_uv);
    vec3 A = al.rgb * al.rgb;
    float spec = al.a;
    float flash = 0.0;
    if (mat == 8) { flash = spec; spec = 0.0; }
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
    amb += vec3(0.10, 0.12, 0.05) * (1.0 - abs(N.z)) * 0.6;
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
    // explosion light
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
    // primed TNT flashes white (unlit overlay, like the game)
    col = mix(col, vec3(1.9), flash);
    // aerial perspective
    float dist = length(wp - u_cam);
    float hfac = exp(-max(wp.z, 0.0) / 60.0);
    float f = 1.0 - exp(-dist * u_fog * mix(0.7, 1.0, hfac));
    vec3 hz = clear_sky(normalize(vec3(vdir.xy, 0.035)));
    col = mix(col, hz, clamp(f, 0.0, 1.0));
    o_col = vec4(col, 1.0);
}
""" % MAT_LEAF

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
    vec3 smoke = mix(u_shade, u_light, lit) * vec3(1.0, 0.97, 0.93);     // faint warm dust tint
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
        self.exposure = 0.68
        self.sat = 1.2
        self.contrast = 1.05
        self.vignette = 0.16
        self.fog = 0.0013
        self.sun_col = np.array([1.0, 0.95, 0.86]) * 3.1
        self.sky_amb = np.array([0.40, 0.55, 0.86]) * 1.05
        self.gnd_amb = np.array([0.30, 0.36, 0.20]) * 0.95
        self.ssao_radius = 1.4
        self.bloom = 0.3
        self.bloom_thresh = 3.2
        self.light_col = np.array([1.0, 0.58, 0.26]) * 16.0
        self.corner_vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32).tobytes())
        self.region = None

    # -- setup ------------------------------------------------------------------------------
    def _prog(self, name, vs, fs):
        self.progs[name] = self.ctx.program(vertex_shader=vs, fragment_shader=fs)
        return self.progs[name]

    def _build_programs(self):
        self._prog('static', STATIC_VS, STATIC_FS)
        self._prog('voxel', VOXEL_VS, VOXEL_FS)
        self._prog('tnt', TNT_VS, TNT_FS)
        self._prog('sh_static', SHADOW_STATIC_VS, DEPTH_FS)
        self._prog('sh_voxel', SHADOW_VOXEL_VS, DEPTH_FS)
        self._prog('sh_tnt', SHADOW_TNT_VS, DEPTH_FS)
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

    def set_textures(self, block_tex_list, tnt_layers, sky, tint_noise, puffs):
        ctx = self.ctx
        blocks = np.stack([np.clip(t, 0, 255).astype(np.uint8) for t in block_tex_list])  # (L,16,16,3)
        n = blocks.shape[0]
        self.t_blocks = ctx.texture_array((16, 16, n), 3, blocks.tobytes())
        self.t_blocks.build_mipmaps()
        self.t_blocks.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.NEAREST)
        self.t_blocks.anisotropy = 1.0
        tl = np.stack([np.clip(t, 0, 255).astype(np.uint8) for t in tnt_layers])   # (3,16,16,3)
        self.t_tnt = ctx.texture_array((16, 16, tl.shape[0]), 3, tl.tobytes())
        self.t_tnt.build_mipmaps(max_level=3)
        self.t_tnt.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.NEAREST)
        self.t_tnt.repeat_x = self.t_tnt.repeat_y = False
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

    def set_static(self, vertices, indices, chunks, ground_half=1600.0, hole=28.0):
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
        rects = [(-g, -g, g, -h), (-g, h, g, g), (-g, -h, -h, h), (h, -h, g, h)]   # frame around the hole
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
        self.lvp_far = light_matrix((0, 0, 10), self.far_half, depth=700.0)
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

    def _tnt_vaos(self, inst_buf):
        ctx = self.ctx
        g = ctx.vertex_array(self.progs['tnt'], [
            (self.cube_vbo, '3f 3f 1f', 'in_pos', 'in_nrm', 'in_face'),
            (inst_buf, '3f 1f 1f/i', 'i_pos', 'i_scale', 'i_flash'),
        ])
        s = ctx.vertex_array(self.progs['sh_tnt'], [
            (self.cube_vbo, '3f 16x', 'in_pos'),
            (inst_buf, '3f 1f 4x/i', 'i_pos', 'i_scale'),
        ])
        return g, s

    def render(self, cam, voxels=None, tnt=None, fx=None, near_center=(0.0, 0.0, 13.0)):
        """cam: dict(eye, target, fov, up(optional), roll(optional)).
        voxels: VOXEL_DTYPE instances; tnt: float32 (M, 5) pos3 scale flash; fx: dict(puffs, flashes, lights)."""
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
        tnt_g = tnt_s = None
        n_tnt = 0
        if tnt is not None and len(tnt):
            n_tnt = len(tnt)
            tbuf = ctx.buffer(np.ascontiguousarray(tnt, np.float32).tobytes())
            tnt_g, tnt_s = self._tnt_vaos(tbuf)

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
        if tnt_s is not None:
            self.progs['sh_tnt']['u_lvp'].write(m4(lvp_near))
            tnt_s.render(instances=n_tnt)
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
        if vox_g is not None:
            self.progs['voxel']['u_vp'].write(m4(vp))
            vox_g.render(instances=n_vox)
        if tnt_g is not None:
            self.progs['tnt']['u_vp'].write(m4(vp))
            self.t_tnt.use(2)
            self.progs['tnt']['u_tnt'] = 2
            tnt_g.render(instances=n_tnt)
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
        Lp['u_sun_col'] = tuple(self.sun_col)
        Lp['u_sky_amb'] = tuple(self.sky_amb)
        Lp['u_gnd_amb'] = tuple(self.gnd_amb)
        Lp['u_fog'] = self.fog
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
            lcol[:nl, :3] = self.light_col[None, :] * lights[:nl, 3:4]
        Lp['u_nl'] = nl
        Lp['u_lpos'].write(lpos.tobytes())
        Lp['u_lcol'].write(lcol.tobytes())
        self._fs('light').render(moderngl.TRIANGLE_STRIP)
        for smp, unit in ((self.smp_near_cmp, 4), (self.smp_near_raw, 5), (self.smp_far_cmp, 6)):
            smp.clear(unit)

        # 5. particles (smoke / fireball puffs sorted back to front, then additive flashes)
        self._particles(fx, view, proj, near, far)

        # release per-frame buffers
        if vox_g is not None:
            vox_g.release()
            vox_s.release()
            vbuf.release()
        if tnt_g is not None:
            tnt_g.release()
            tnt_s.release()
            tbuf.release()
        return self.hdr

    def _particles(self, fx, view, proj, near, far):
        ctx = self.ctx
        puffs = fx.get('puffs')
        flashes = fx.get('flashes')
        if (puffs is None or not len(puffs)) and (flashes is None or not len(flashes)):
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
            ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
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

    def _bloom(self):
        """Threshold the HDR buffer and blur it at 1/4 and 1/8 resolution."""
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        (a4, fa4), (b4, fb4) = self.bloom_tex[0]
        (a8, fa8), (b8, fb8) = self.bloom_tex[1]
        Br = self.progs['bright']
        fa4.use()
        self.hdr.use(0)
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
        src = hdr_tex or self.hdr
        b1, b2 = self._bloom()
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

