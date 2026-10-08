"""One-time bake of volumetric cumulus clouds into the sky panorama (raymarched in GLSL on llvmpipe).

Technique: tileable 3D Perlin-Worley base noise + Worley detail noise (as 3D textures), a 2D coverage
map, cumulus height profile, Beer-Lambert + powder, dual-lobe HG phase, and a cheap multiple
scattering approximation. Output is composited over the analytic clear-sky gradient from sky.py.
"""
import numpy as np
import moderngl

from sky import SUN_DIR, EL_MIN, make_sky, CLOUD_SUN, CLOUD_AMB_LO, CLOUD_AMB_HI

NOISE_COMMON = """
#version 430
uniform float u_z;        // slice coordinate in [0,1)
uniform int u_res;
in vec2 v_uv;
out vec4 o;
uint hash(uvec3 x){
    x = ((x >> 8u) ^ x.yzx) * 1103515245u;
    x = ((x >> 8u) ^ x.yzx) * 1103515245u;
    x = ((x >> 8u) ^ x.yzx) * 1103515245u;
    return x.x;
}
vec3 hash3(ivec3 c, int seed){
    uvec3 u = uvec3(c + ivec3(seed * 131, seed * 71, seed * 29) + 1024);
    uint h1 = hash(u), h2 = hash(u + 17u), h3 = hash(u + 91u);
    return vec3(h1, h2, h3) / 4294967295.0;
}
float worley(vec3 p, int period, int seed){
    vec3 q = p * float(period);
    ivec3 c = ivec3(floor(q));
    vec3 f = fract(q);
    float md = 1e9;
    for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++){
        ivec3 o = ivec3(x, y, z);
        ivec3 cc = (c + o + period * 64) % period;
        vec3 fp = vec3(o) + hash3(cc, seed);
        md = min(md, dot(fp - f, fp - f));
    }
    return clamp(sqrt(md), 0.0, 1.0);
}
vec3 grad3(ivec3 c, int seed){
    vec3 h = hash3(c, seed) * 2.0 - 1.0;
    return normalize(h + 1e-4);
}
float perlin(vec3 p, int period, int seed){
    vec3 q = p * float(period);
    ivec3 c = ivec3(floor(q));
    vec3 f = fract(q);
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float n[8];
    for (int i = 0; i < 8; i++){
        ivec3 o = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        ivec3 cc = (c + o + period * 64) % period;
        n[i] = dot(grad3(cc, seed), f - vec3(o));
    }
    float x00 = mix(n[0], n[1], u.x), x10 = mix(n[2], n[3], u.x);
    float x01 = mix(n[4], n[5], u.x), x11 = mix(n[6], n[7], u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}
float remap(float v, float lo, float hi, float nlo, float nhi){ return nlo + (v - lo) / (hi - lo) * (nhi - nlo); }
"""

BASE_FS = NOISE_COMMON + """
void main(){
    vec3 p = vec3(v_uv, u_z);
    float pf = 0.0, amp = 1.0, norm = 0.0;
    for (int i = 0; i < 4; i++){ pf += amp * perlin(p, 4 << i, 3 + i); norm += amp; amp *= 0.5; }
    pf = pf / norm * 0.5 + 0.5;
    float w = 1.0 - (worley(p, 4, 11) * 0.625 + worley(p, 8, 12) * 0.25 + worley(p, 16, 13) * 0.125);
    // Perlin-Worley: perlin dilated by worley
    float pw = clamp(remap(pf, w - 1.0, 1.0, 0.0, 1.0), 0.0, 1.0);
    o = vec4(pw, w, 0.0, 1.0);
}
"""

DETAIL_FS = NOISE_COMMON + """
void main(){
    vec3 p = vec3(v_uv, u_z);
    float w = 1.0 - (worley(p, 4, 21) * 0.625 + worley(p, 8, 22) * 0.25 + worley(p, 16, 23) * 0.125);
    o = vec4(w, 0.0, 0.0, 1.0);
}
"""

FS_VS = """
#version 430
in vec2 in_pos; out vec2 v_uv;
void main(){ v_uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

MARCH_FS = """
#version 430
uniform sampler3D u_base;
uniform sampler3D u_detail;
uniform vec3 u_sun;
uniform vec3 u_sun_col;
uniform vec3 u_amb_lo;
uniform vec3 u_amb_hi;
uniform float u_el_min;
uniform vec2 u_tile0;     // panorama uv origin of this tile
uniform vec2 u_tile_size;
uniform float u_coverage;
in vec2 v_uv;
out vec4 o;

const float BASE = 1.35;   // km
const float TOP = 3.4;     // km

float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float remap(float v, float lo, float hi, float nlo, float nhi){ return nlo + (v - lo) / (hi - lo) * (nhi - nlo); }
float sat(float v){ return clamp(v, 0.0, 1.0); }

float coverage(vec2 xy){
    // large scale cloud placement (km): separate clusters with clear sky between
    float c = texture(u_base, vec3(xy / 38.0, 0.37)).g;
    float c2 = texture(u_base, vec3(xy / 11.0 + 0.3, 0.71)).r;
    float v = c * 0.7 + c2 * 0.3;
    return sat(remap(v, 1.0 - u_coverage, 1.0 - u_coverage + 0.22, 0.0, 1.0));
}

float density(vec3 p, bool detail){
    float h = (p.z - BASE) / (TOP - BASE);
    if (h < 0.0 || h > 1.0) return 0.0;
    float cov = coverage(p.xy);
    if (cov <= 0.0) return 0.0;
    // cumulus profile: sharp flat base, rounded top that is higher where coverage is strong
    float topH = mix(0.35, 1.0, cov);
    float g = sat(remap(h, 0.0, 0.06, 0.0, 1.0)) * sat(remap(h, topH * 0.35, topH, 1.0, 0.0));
    vec4 b = texture(u_base, p / 4.2 + vec3(0.0, 0.0, 0.13));
    float shape = mix(b.g, b.r, 0.35);
    float d = sat(remap(shape * g, 1.0 - cov * 0.85, 1.0, 0.0, 1.0));
    if (d <= 0.0 || !detail) return d;
    float det = texture(u_detail, p / 0.7 + vec3(0.2, 0.0, 0.0)).r;
    float erode = mix(1.0 - det, det, sat(h * 3.0));
    d = sat(remap(d, erode * 0.38, 1.0, 0.0, 1.0));
    return d * 1.4;
}

float hg(float c, float g){ float g2 = g * g; return (1.0 - g2) / (4.0 * 3.14159 * pow(1.0 + g2 - 2.0 * g * c, 1.5)); }

void main(){
    vec2 puv = u_tile0 + v_uv * u_tile_size;
    float phi = puv.x * 6.2831853;
    float el = radians(90.0 - puv.y * (90.0 - u_el_min));
    vec3 rd = vec3(cos(el) * cos(phi), cos(el) * sin(phi), sin(el));
    o = vec4(0.0, 0.0, 0.0, 1.0);   // rgb = in-scattered light, a = transmittance
    if (rd.z < 0.004) return;
    float t0 = BASE / rd.z, t1 = TOP / rd.z;
    float maxd = 90.0;
    if (t0 > maxd) return;
    t1 = min(t1, maxd + 25.0);
    const int N = 112;
    float dt = (t1 - t0) / float(N);
    float jitter = hash12(gl_FragCoord.xy + u_tile0 * 9173.0);
    float T = 1.0;
    vec3 L = vec3(0.0);
    float cosT = dot(rd, u_sun);
    float phase = mix(hg(cosT, 0.62), hg(cosT, -0.25), 0.55);
    vec3 sunc = u_sun_col * 9.5;
    float sigma = 22.0;  // extinction per km at density 1
    for (int i = 0; i < N; i++){
        float t = t0 + (float(i) + jitter) * dt;
        vec3 p = rd * t;
        float d = density(p, true);
        if (d <= 0.001) continue;
        // light march towards the sun
        float ls = 0.0;
        float st = 0.07;
        vec3 q = p;
        for (int j = 0; j < 6; j++){
            q += u_sun * st;
            ls += density(q, j < 3) * st;
            st *= 1.7;
        }
        float h = sat((p.z - BASE) / (TOP - BASE));
        // multiple scattering approximation (3 octaves)
        float beer = 0.0, a = 1.0, b = 1.0, c = 1.0;
        for (int k = 0; k < 3; k++){
            beer += b * exp(-sigma * ls * a) * mix(1.0, phase * 12.566, c);
            a *= 0.3; b *= 0.55; c *= 0.4;
        }
        float powder = 1.0 - 0.65 * exp(-sigma * 1.8 * ls - d * 2.0);
        vec3 amb = mix(u_amb_lo, u_amb_hi, h) * 2.2;
        vec3 S = sunc * beer * powder * 0.09 + amb * (0.55 + 0.45 * h);
        float ext = sigma * d;
        float Ts = exp(-ext * dt);
        L += T * S * (1.0 - Ts);
        T *= Ts;
        if (T < 0.01) break;
    }
    // aerial perspective on distant clouds
    float fade = exp(-max(t0 - 6.0, 0.0) / 55.0);
    o = vec4(L * fade, mix(1.0, T, fade));
}
"""


def bake(width=8192, height=2048, tile=512, coverage=0.53, seed=0, verbose=True):
    ctx = moderngl.create_standalone_context(backend='egl', require=430)
    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32).tobytes())

    def volume(fs, res):
        prog = ctx.program(vertex_shader=FS_VS, fragment_shader=fs)
        vao = ctx.vertex_array(prog, [(quad, '2f', 'in_pos')])
        tex = ctx.texture((res, res), 4, dtype='f4')
        fbo = ctx.framebuffer(color_attachments=[tex])
        fbo.use()
        vol = np.zeros((res, res, res, 2), np.float32)
        for z in range(res):
            prog['u_z'] = (z + 0.5) / res
            vao.render(moderngl.TRIANGLE_STRIP)
            vol[z] = np.frombuffer(fbo.read(components=4, dtype='f4'), np.float32).reshape(res, res, 4)[..., :2]
        return vol

    base = volume(BASE_FS, 128)
    detail = volume(DETAIL_FS, 64)

    def stretch(a):
        lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
        return np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32)
    base[..., 0] = stretch(base[..., 0])
    base[..., 1] = stretch(base[..., 1])
    detail[..., 0] = stretch(detail[..., 0])
    tb = ctx.texture3d((128, 128, 128), 2, np.ascontiguousarray(base).tobytes(), dtype='f4')
    td = ctx.texture3d((64, 64, 64), 1, np.ascontiguousarray(detail[..., 0]).tobytes(), dtype='f4')
    for t in (tb, td):
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = t.repeat_y = t.repeat_z = True
    prog = ctx.program(vertex_shader=FS_VS, fragment_shader=MARCH_FS)
    vao = ctx.vertex_array(prog, [(quad, '2f', 'in_pos')])
    tb.use(0)
    td.use(1)
    prog['u_base'] = 0
    prog['u_detail'] = 1
    prog['u_sun'] = tuple(SUN_DIR)
    prog['u_sun_col'] = tuple(float(v) for v in CLOUD_SUN)
    prog['u_amb_lo'] = tuple(float(v) for v in CLOUD_AMB_LO)
    prog['u_amb_hi'] = tuple(float(v) for v in CLOUD_AMB_HI)
    prog['u_el_min'] = EL_MIN
    prog['u_coverage'] = coverage
    out = np.zeros((height, width, 4), np.float32)
    tex = ctx.texture((tile, tile), 4, dtype='f4')
    fbo = ctx.framebuffer(color_attachments=[tex])
    fbo.use()
    import time
    t_start = time.time()
    # rows near the zenith/nadir that the camera never sees are cheap anyway (early outs)
    for ty in range(0, height, tile):
        for tx in range(0, width, tile):
            prog['u_tile0'] = (tx / width, ty / height)
            prog['u_tile_size'] = (tile / width, tile / height)
            vao.render(moderngl.TRIANGLE_STRIP)
            out[ty:ty + tile, tx:tx + tile] = np.frombuffer(fbo.read(components=4, dtype='f4'), np.float32).reshape(tile, tile, 4)
        if verbose:
            print(f'  clouds row {ty // tile + 1}/{height // tile}  {time.time() - t_start:.0f}s', flush=True)
    # (tile row r read from GL maps to panorama row ty + r, so no flip is needed)
    ctx.release()
    return out


def make_sky_with_clouds(width=8192, height=2048, coverage=0.53):
    clear = make_sky(width, height, clouds=False)
    c = bake(width, height, coverage=coverage)
    return (clear * c[..., 3:4] + c[..., :3]).astype(np.float32)
