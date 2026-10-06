"""The Minecraft world from above at any scale: a world generator evaluated per pixel on the GPU (moderngl, a
fragment shader), coloured like a satellite view of a Minecraft world.

The view is always centred on the world's origin (spawn, where Steve lies), so the noise coordinates p / wavelength
stay small for every octave that's evaluated and float32 is plenty from 0.2 m a pixel to 60,000 km across. Octaves
finer than a few pixels are faded out (they average to nothing), so the map never shimmers.

The world: continents and oceans with coastlines at every scale; land heights (inland plateaus, ranges of ridged
mountains, hills, small bumps); temperature and humidity at two scales giving the biomes (plains, forest, birch,
dark forest, taiga, snowy plains, desert, badlands, savanna, jungle, swamp, stone and snow on the peaks, beaches);
rivers; trees as canopies on a jittered grid (averaged into the forest's colour once they're smaller than a pixel);
block-by-block variation; hillshading from the same sun as the 3D renders; Minecraft's blocky clouds at Y=192 with
their shadows; around spawn a meadow, a forest and a river by design. Inside the spawn area the map shows the
satellite capture of the 3D world instead, blended at its edge. Optionally the Earth, to scale, and the world border.

A 'fields' mode writes height / water / biome / tree at one pixel a block, so the 3D spawn area can be built from
exactly the same world.
"""
import numpy as np
import moderngl

WORLD_HALF = 30_000_000.0          # the world border (blocks from spawn)
SUN = np.array([-0.45, 0.55, 0.70]) / np.linalg.norm([-0.45, 0.55, 0.70])     # from the north-west

VS = """
#version 430
in vec2 in_pos;
void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

FS = """
#version 430
uniform vec2 u_res;
uniform float u_s;            // metres a pixel
uniform int u_mode;           // 0 colour, 1 fields
uniform float u_time;
uniform vec3 u_sun;
uniform sampler2D u_sat;      // the satellite pictures of the 3D spawn area: a wide one
uniform float u_sat_half;
uniform sampler2D u_sat2;     // ... and a sharper one of the middle
uniform float u_sat2_half;
uniform int u_sat_on;
uniform sampler2D u_earth_tex;  // land mask, equirectangular
uniform vec3 u_earth_c;       // centre x, y (m), radius (m)
uniform float u_earth_a;
uniform float u_cloud_a;
uniform float u_border;       // the world border's glow (0..1)
uniform int u_clip;           // nothing (alpha 0) outside the world border
uniform float u_wt = -1.0;    // (calibration: force the trees' detail weight)
out vec4 o;

uint hash3(uint x, uint y, uint z) {
    uint h = x * 0x8da6b343u ^ y * 0xd8163841u ^ z * 0xcb1ab31fu;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15; h *= 0x27d4eb2du; h ^= h >> 16;
    return h;
}
float hval(ivec2 c, uint seed) { return float(hash3(uint(c.x), uint(c.y), seed) & 0xffffffu) / 16777215.0; }
vec2 hval2(ivec2 c, uint seed) { return vec2(hval(c, seed), hval(c, seed + 977u)); }

float vnoise(vec2 u, uint seed) {
    vec2 fl = floor(u);
    vec2 f = u - fl;
    ivec2 c = ivec2(fl);
    vec2 w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float a = hval(c, seed), b = hval(c + ivec2(1, 0), seed);
    float cc = hval(c + ivec2(0, 1), seed), d = hval(c + ivec2(1, 1), seed);
    return mix(mix(a, b, w.x), mix(cc, d, w.x), w.y) * 2.0 - 1.0;
}

// fractal noise from wavelength lam0 down, octaves finer than a few pixels faded out (they average to nothing)
float octw(float lam) { return smoothstep(2.5, 6.0, lam / u_s); }
float octwc(float lam) { return smoothstep(20.0, 50.0, lam / u_s); }    // for fields that get thresholded
float fbmk(vec2 p, float lam0, float gain, uint seed, float lam_min, float k0, float k1) {
    float v = 0.0, amp = 1.0, lam = lam0;
    for (int k = 0; k < 30; k++) {
        if (lam < lam_min) break;
        float w = smoothstep(k0, k1, lam / u_s);
        if (w <= 0.0) break;
        v += amp * w * vnoise(p / lam, seed + uint(k) * 7919u);
        amp *= gain;
        lam *= 0.5;
    }
    return v;
}
float fbm(vec2 p, float lam0, float gain, uint seed, float lam_min) { return fbmk(p, lam0, gain, seed, lam_min, 2.5, 6.0); }
float fbmc(vec2 p, float lam0, float gain, uint seed, float lam_min) { return fbmk(p, lam0, gain, seed, lam_min, 20.0, 50.0); }
// the local patchwork of biomes: faded out sooner, so from far away the biomes are big smooth regions, not speckle
float fbmb(vec2 p, float lam0, float gain, uint seed, float lam_min) { return fbmk(p, lam0, gain, seed, lam_min, 60.0, 160.0); }
// the same noise at the origin (where every octave is just its lattice value): for the biases round spawn
float fbm0k(float lam0, float gain, uint seed, float lam_min, float k0, float k1) {
    float v = 0.0, amp = 1.0, lam = lam0;
    for (int k = 0; k < 30; k++) {
        if (lam < lam_min) break;
        float w = smoothstep(k0, k1, lam / u_s);
        if (w <= 0.0) break;
        v += amp * w * (hval(ivec2(0, 0), seed + uint(k) * 7919u) * 2.0 - 1.0);
        amp *= gain;
        lam *= 0.5;
    }
    return v;
}
float fbm0(float lam0, float gain, uint seed, float lam_min) { return fbm0k(lam0, gain, seed, lam_min, 20.0, 50.0); }
float fbm0b(float lam0, float gain, uint seed, float lam_min) { return fbm0k(lam0, gain, seed, lam_min, 60.0, 160.0); }

const float SAT_R0 = 180.0;          // the satellite pictures fade into the generated map from here out
const float TREE_SHADOW = 0.34;     // ground in a tree's shadow
const float CANOPY_SELF = 0.55;     // leaves in the shadow of the canopy's upper layers
const float CANOPY_TOP = 0.10;      // the upper layers are a little brighter
const float AVG_SHADE = 0.473;      // how much of the ground between trees is in their shadow (per unit density)
const float AVG_CANOPY = 0.736;     // a canopy's average shade
const float AVG_COVER = 0.72;       // canopy cover per unit tree density
float region(vec2 p, float R) { return exp(-dot(p, p) / (R * R)); }
float spawnw(vec2 p) { return 1.0 - smoothstep(380.0, 1100.0, length(p)); }
const vec2 VILLAGE = vec2(-12.0, -88.0);

// ---- the world's fields ---------------------------------------------------------------------------
// each is natural noise plus a smooth bias round spawn (centred on the origin, so no edges anywhere)
float continent(vec2 p) {
    float c = fbmc(p, 9.0e6, 0.63, 11u, 2.0) * 0.55 + 0.04;
    float c0 = fbm0(9.0e6, 0.63, 11u, 2.0) * 0.55 + 0.04;
    return c + (0.78 - c0) * region(p, 1.6e6);          // spawn is deep in a big continent
}
float temperature(vec2 p) {
    float t = fbmc(p, 7.0e6, 0.6, 51u, 400.0) * 0.55 + fbmb(p, 2.4e4, 0.5, 52u, 400.0) * 0.32;
    float t0 = fbm0(7.0e6, 0.6, 51u, 400.0) * 0.55 + fbm0b(2.4e4, 0.5, 52u, 400.0) * 0.32;
    return t + (0.08 - t0) * max(region(p, 9.0e3), 0.7 * region(p, 3.5e5));
}
float humidity(vec2 p) {
    float h = fbmc(p, 6.0e6, 0.6, 61u, 400.0) * 0.55 + fbmb(p, 1.8e4, 0.5, 62u, 400.0) * 0.36;
    float h0 = fbm0(6.0e6, 0.6, 61u, 400.0) * 0.55 + fbm0b(1.8e4, 0.5, 62u, 400.0) * 0.36;
    h += (-0.07 - h0) * region(p, 7.0e3);
    // round spawn: patches of forest in the meadow, and the village's fields kept open
    float near = region(p, 1800.0);
    h += near * 0.2 * fbm(p, 520.0, 0.5, 63u, 20.0);
    h -= 0.35 * region(p - VILLAGE, 95.0) + 0.3 * region(p, 40.0);
    return h;
}
// the mountains' ridges (with their snow) are drawn only while they're big enough not to turn into speckle
float ridge_vis() { return smoothstep(40.0, 110.0, 4.5e4 / u_s); }
float mountains(vec2 p) {
    float reg = smoothstep(0.05, 0.45, fbmc(p, 2.5e5, 0.55, 23u, 2000.0)) * (1.0 - region(p, 9000.0));
    // massifs (the snow sits on them in patches) with ridges cut into them
    float vis = ridge_vis();
    float m = smoothstep(-0.15, 0.75, fbmc(p, 6.0e4, 0.5, 33u, 30.0));
    float r = 1.0 - abs(fbmc(p, 2.2e4, 0.45, 31u, 30.0));
    return reg * 230.0 * mix(0.4, m * (0.72 + 0.28 * r * r), vis);
}
float mountain_haze(vec2 p) {
    // where the ridges are too small to see, a mountain range still reads as grey
    float reg = smoothstep(0.05, 0.45, fbmc(p, 2.5e5, 0.55, 23u, 2000.0)) * (1.0 - region(p, 9000.0));
    return reg * (1.0 - ridge_vis());
}
float height(vec2 p, float c) {
    // metres above sea level (negative: under water)
    float base = 3.0 + 70.0 * smoothstep(0.0, 0.7, c) + mountains(p);
    float hills = 24.0 * fbm(p, 900.0, 0.5, 41u, 4.0) + 2.2 * fbm(p, 40.0, 0.5, 43u, 2.0);
    float meadow = 1.6 * fbm(p, 160.0, 0.5, 45u, 8.0) + 0.6 * fbm(p, 40.0, 0.5, 46u, 8.0);
    float w = spawnw(p);
    float land = base + mix(hills, meadow, w);
    float sea = c * 260.0;
    return c > 0.0 ? mix(sea, land, smoothstep(0.0, 0.03, c)) : sea;
}
// distance (m) to the nearest river centre, and its half-width
vec2 river(vec2 p) {
    float rv = fbm(p, 7000.0, 0.5, 71u, 15.0);
    float g = 1.0 / 7000.0 * 2.2;
    float d = abs(rv) / g;
    float hw = 7.0;
    // spawn's river: north-south, east of the meadow (and no other river near spawn)
    float xr = 52.0 + 16.0 * sin(p.y / 70.0) + 6.0 * sin(p.y / 23.0 + 1.0);
    float ds = abs(p.x - xr) + 1.0e9 * smoothstep(2500.0, 2600.0, abs(p.y));
    d += 1.0e5 * region(p, 1500.0);
    return vec2(min(d, ds), hw);
}

// biome ids: 0 ocean 1 beach 2 plains 3 forest 4 birch 5 dark forest 6 taiga 7 snowy 8 desert 9 badlands
// 10 savanna 11 jungle 12 swamp 13 stone 14 snow peak
int biome(vec2 p, float h, float c, float T, float H) {
    if (h < 0.0) return 0;
    if (h < 2.5 && c < 0.05) return 1;
    if (h > 228.0) return 14;
    if (h > 165.0) return 13;
    if (T < -0.30) return H > 0.05 ? 6 : 7;
    if (T < -0.12) return H > 0.0 ? 6 : 2;
    if (T > 0.24) {
        if (H < -0.12) return T > 0.4 && fbmc(p, 6000.0, 0.5, 81u, 400.0) > 0.05 ? 9 : 8;
        if (H < 0.1) return 10;
        return 11;
    }
    if (H > 0.38 && h < 14.0) return 12;
    if (H > 0.30) return 5;
    if (H > 0.08) return (fbmc(p, 3000.0, 0.5, 91u, 400.0) > 0.15) ? 4 : 3;
    if (H > -0.02) return 3;
    return 2;
}
float tree_prob(int b) {
    if (b == 3) return 0.80; if (b == 4) return 0.75; if (b == 5) return 0.97; if (b == 6) return 0.78;
    if (b == 11) return 0.95; if (b == 12) return 0.45; if (b == 10) return 0.12; if (b == 2) return 0.035;
    if (b == 7) return 0.03;
    return 0.0;
}
vec3 ground_col(int b, vec2 p) {
    if (b == 1) return vec3(222, 210, 164);
    if (b == 2) return vec3(130, 183, 76);
    if (b == 3) return vec3(124, 176, 72);
    if (b == 4) return vec3(126, 178, 74);
    if (b == 5) return vec3(70, 118, 46);
    if (b == 6) return vec3(88, 124, 76);
    if (b == 7) return vec3(236, 242, 246);
    if (b == 8) return vec3(219, 205, 150);
    if (b == 9) {
        float band = fract(p.y * 0.0 + (fbm(p, 900.0, 0.5, 83u, 50.0) * 3.0 + p.x * 0.0005));
        return band < 0.33 ? vec3(190, 102, 54) : (band < 0.66 ? vec3(206, 132, 74) : vec3(166, 84, 48));
    }
    if (b == 10) return vec3(168, 160, 84);
    if (b == 11) return vec3(72, 150, 42);
    if (b == 12) return vec3(88, 104, 64);
    if (b == 13) return vec3(128, 128, 132);
    if (b == 14) return vec3(240, 245, 250);
    return vec3(130, 183, 76);
}
vec3 canopy_col(int b) {
    if (b == 4) return vec3(118, 156, 72);
    if (b == 5) return vec3(42, 86, 32);
    if (b == 6) return vec3(54, 88, 62);
    if (b == 7) return vec3(214, 226, 232);
    if (b == 10) return vec3(124, 140, 52);
    if (b == 11) return vec3(44, 132, 34);
    if (b == 12) return vec3(70, 96, 46);
    return vec3(54, 120, 39);
}

// trees on a jittered grid of 7 m cells: (coverage, shade, shadow) at p
bool canopy_at(vec2 p, ivec2 c, float pr, float grow, out float d, out float r, out vec2 q) {
    d = 1e9; r = 0.0; q = vec2(0.0);
    if (hval(c, 301u) > pr) return false;
    vec2 tc = (vec2(c) + 0.2 + 0.6 * hval2(c, 303u)) * 7.0;
    r = 2.2 + 1.1 * hval(c, 307u) + grow;
    // the canopy is made of blocks
    q = floor(p) + 0.5 - floor(tc) - 0.5;
    d = max(abs(q.x), abs(q.y)) * 0.55 + length(q) * 0.45;
    return d < r;
}
vec3 trees(vec2 p, int b) {
    float pr = tree_prob(b);
    if (pr <= 0.0) return vec3(0.0);
    vec2 cell = floor(p / 7.0);
    float cov = 0.0, sh = 0.0, shadow = 0.0;
    vec2 sd = normalize(u_sun.xy);
    for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        ivec2 c = ivec2(cell) + ivec2(dx, dy);
        float d, r; vec2 q;
        if (canopy_at(p, c, pr, 0.0, d, r, q)) {
            cov = 1.0;
            // the canopy is a mound of leaf blocks in three layers: a block whose neighbour towards the sun is
            // a layer higher is in its shadow
            float lv = d < r - 2.0 ? 2.0 : (d < r - 1.0 ? 1.0 : 0.0);
            vec2 qs = q + sd * 1.2;
            float ds = max(abs(qs.x), abs(qs.y)) * 0.55 + length(qs) * 0.45;
            float lvs = ds < r - 2.0 ? 2.0 : (ds < r - 1.0 ? 1.0 : 0.0);
            float s = (lvs > lv ? CANOPY_SELF : 1.0) * (1.0 + CANOPY_TOP * lv);
            sh = max(sh, s);
        }
    }
    if (cov < 0.5) {
        // the canopies' shadows on the ground (the leaves are 3 to 7 blocks up, the sun 45 degrees high)
        vec2 ps = p + sd * 5.0;
        vec2 cs = floor(ps / 7.0);
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
            float d, r; vec2 q;
            if (canopy_at(ps, ivec2(cs) + ivec2(dx, dy), pr, 1.0, d, r, q)) shadow = 1.0;
        }
    }
    return vec3(cov, sh, shadow);
}

float clouds(vec2 p) {
    vec2 q = p - vec2(u_time * 1.6, 0.0);
    vec2 cell = floor(q / 12.0);
    float n = vnoise(cell / 5.0, 501u) * 0.7 + vnoise(cell / 2.3, 502u) * 0.3;
    return step(0.36, n);
}

// the Earth (land mask, nearest texel: pixel art), seen from above the Sahara
vec4 earth(vec2 p) {
    vec2 d = (p - u_earth_c.xy) / u_earth_c.z;
    float r2 = dot(d, d);
    float ps = u_s / u_earth_c.z;
    if (r2 > (1.0 + 0.08) * (1.0 + 0.08)) return vec4(0.0);
    if (r2 > 1.0) {
        float a = 1.0 - (sqrt(r2) - 1.0) / 0.08;
        return vec4(0.45, 0.7, 1.0, 0.65 * a * a);
    }
    float rho = sqrt(r2);
    float cz = sqrt(max(1.0 - r2, 0.0));
    float phi0 = radians(18.0), lam0 = radians(18.0);
    float c = asin(clamp(rho, 0.0, 1.0));
    float lat = asin(cos(c) * sin(phi0) + (rho > 1e-6 ? d.y * sin(c) * cos(phi0) / rho : 0.0));
    float lon = lam0 + atan(d.x * sin(c), rho * cos(c) * cos(phi0) - d.y * sin(c) * sin(phi0));
    vec2 uv = vec2(fract((lon + 3.14159265) / 6.2831853), clamp(0.5 - lat / 3.14159265, 0.0, 1.0));
    float land = texture(u_earth_tex, uv).r;
    vec3 col = land > 0.5 ? (abs(degrees(lat)) < 32.0 && abs(degrees(lat)) > 15.0 ? vec3(0.78, 0.66, 0.38)
                                                                                    : vec3(0.26, 0.56, 0.24))
                          : vec3(0.12, 0.30, 0.72);
    if (abs(degrees(lat)) > 66.0) col = vec3(0.92, 0.95, 0.98);
    float lit = 0.35 + 0.75 * max(dot(normalize(vec3(d.x, d.y, cz)), normalize(vec3(-0.4, 0.5, 0.75))), 0.0);
    col *= lit;
    col = mix(col, vec3(0.55, 0.75, 1.0), 0.35 * pow(rho, 6.0));
    float edge = clamp((1.0 - rho) / max(ps, 1e-9), 0.0, 1.0);
    return vec4(col, edge);
}

void main() {
    vec2 px = gl_FragCoord.xy - u_res * 0.5;
    vec2 p = px * u_s;
    float c = continent(p);
    float h = height(p, c);
    float T = temperature(p);
    float Hm = humidity(p);
    int b = biome(p, h, c, T, Hm);
    vec2 rv = river(p);
    bool riv = h >= 0.0 && h < 110.0 && rv.x < rv.y;
    if (u_mode == 1) {
        // fields at one pixel a block: height, water (0 land, 1 river, 2 sea), biome, tree trunk here
        vec2 cell = floor(p / 7.0);
        ivec2 ci = ivec2(cell);
        float trunk = 0.0;
        if (hval(ci, 301u) <= tree_prob(b)) {
            vec2 tc = (vec2(ci) + 0.2 + 0.6 * hval2(ci, 303u)) * 7.0;
            if (floor(tc) == floor(p)) trunk = 2.2 + 1.1 * hval(ci, 307u);
        }
        o = vec4(h, riv ? 1.0 : (h < 0.0 ? 2.0 : 0.0), float(b), trunk);
        return;
    }
    // ---- colour
    vec3 col;
    if (h < 0.0) {
        float dep = clamp(-h / 60.0, 0.0, 1.0);
        col = mix(vec3(58, 118, 196), vec3(26, 52, 126), dep);
        if (T > 0.3) col = mix(col, vec3(48, 150, 196), 0.4 * (1.0 - dep));
            if (T < -0.55) col = mix(col, vec3(200, 220, 236), 0.55);           // frozen ocean
    } else {
        col = ground_col(b, p);
        // trees: canopies while they're bigger than a pixel, then their average
        float tp = tree_prob(b);
        vec3 cc = canopy_col(b);
        if (tp > 0.0) {
            float wt = u_wt >= 0.0 ? u_wt : smoothstep(2.5, 0.9, u_s);
            vec3 tr = wt > 0.0 ? trees(p, b) : vec3(0.0);
            vec3 det = tr.x > 0.0 ? cc * tr.y : col * (tr.z > 0.5 ? TREE_SHADOW : 1.0);
            vec3 avg = mix(col * mix(1.0, TREE_SHADOW, AVG_SHADE * tp), cc * AVG_CANOPY, clamp(tp * AVG_COVER, 0.0, 1.0));
            col = mix(avg, det, wt);
        }
        if (h > 210.0 && b != 14) col = mix(col, vec3(240, 245, 250), smoothstep(210.0, 228.0, h));
        col = mix(col, vec3(132, 132, 136), 0.45 * mountain_haze(p));
    }
    // block by block
    vec2 blk = floor(p);
    float bw = smoothstep(1.6, 0.6, u_s);
    col *= 1.0 + bw * (hval(ivec2(blk), 401u) - 0.5) * 0.10;
    // rivers
    if (riv) {
        float cover = clamp((rv.y - rv.x) / max(u_s, 0.5) + 0.5, 0.0, 1.0) * smoothstep(40.0, 4.0, u_s);
        col = mix(col, vec3(56, 126, 208), cover);
    }
    col /= 255.0;
    // hillshade from the height's slope
    float hs = h;
    vec2 g = vec2(dFdx(hs), dFdy(hs)) / u_s;
    vec3 n = normalize(vec3(-g, 1.0));
    float lam = dot(n, u_sun);
    col *= h >= 0.0 ? (0.62 + 0.52 * lam) : (0.9 + 0.1 * lam);
    // the satellite pictures of the 3D spawn area (sampled outside any branch, for their mipmaps)
    if (u_sat_on == 1) {
        vec3 sw = texture(u_sat, p / (2.0 * u_sat_half) + 0.5).rgb;
        vec3 sn = texture(u_sat2, p / (2.0 * u_sat2_half) + 0.5).rgb;
        float e2 = max(abs(p.x), abs(p.y));
        vec3 sc = mix(sw, sn, 1.0 - smoothstep(u_sat2_half - 24.0, u_sat2_half - 4.0, e2));
        float a = (1.0 - smoothstep(SAT_R0, u_sat_half - 10.0, length(p))) * (1.0 - smoothstep(1.5, 3.0, u_s));
        col = mix(col, sc, a);
    }
    // Minecraft's clouds at Y=192, and their shadows
    if (u_cloud_a > 0.0) {
        vec2 off = u_sun.xy / u_sun.z * 190.0;
        float cs = clouds(p + off);
        col *= 1.0 - 0.28 * cs * u_cloud_a;
        float cl = clouds(p);
        col = mix(col, vec3(0.96, 0.97, 0.99), 0.82 * cl * u_cloud_a);
    }
    // the Earth, to scale
    if (u_earth_a > 0.0) {
        vec4 e = earth(p);
        col = mix(col, e.rgb, e.a * u_earth_a);
    }
    // the world border
    float e = max(abs(p.x), abs(p.y));
    float d = (30000000.0 - e) / u_s;
    if (u_clip == 1 && d < 0.0) { o = vec4(0.0); return; }
    if (u_border > 0.0) {
        float line = smoothstep(7.0, 0.0, d);
        float stripes = step(0.5, fract((p.x + p.y) / (u_s * 14.0)));
        col = mix(col, vec3(0.35, 0.75, 1.0) * (0.75 + 0.5 * stripes), line * u_border);
    }
    o = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


def earth_mask():
    """A 72 x 36 equirectangular land mask (5 degree cells), drawn from memory: rows from 90N, columns from 180W."""
    rows = {
        1: [(-90, -70), (-60, -20), (10, 25), (45, 60), (95, 110)],
        2: [(-125, -65), (-70, -20), (10, 30), (50, 60), (90, 115)],
        3: [(-165, -140), (-125, -65), (-55, -20), (15, 35), (40, 180)],
        4: [(-165, -60), (-50, -22), (-24, -13), (5, 30), (30, 180)],
        5: [(-165, -95), (-78, -58), (-48, -42), (5, 30), (30, 165)],
        6: [(-135, -95), (-80, -57), (-6, 0), (8, 20), (20, 140), (155, 165)],
        7: [(-130, -55), (-10, 2), (-5, 40), (40, 140), (142, 145)],
        8: [(-125, -65), (-5, 40), (40, 135), (140, 145)],
        9: [(-125, -70), (-9, 3), (5, 18), (20, 28), (26, 45), (45, 130), (140, 142)],
        10: [(-124, -76), (-9, 0), (-6, 10), (15, 26), (26, 45), (45, 122), (126, 129), (132, 141)],
        11: [(-120, -80), (-115, -105), (-10, 32), (34, 60), (60, 122), (130, 135)],
        12: [(-115, -97), (-82, -80), (-15, 35), (35, 56), (56, 68), (68, 90), (90, 122)],
        13: [(-106, -97), (-85, -75), (-17, 37), (37, 60), (70, 88), (92, 110)],
        14: [(-105, -88), (-17, 40), (43, 52), (73, 82), (95, 110), (120, 125)],
        15: [(-92, -83), (-75, -60), (-17, 15), (15, 51), (75, 80), (98, 109), (122, 126)],
        16: [(-80, -52), (-13, 10), (10, 45), (80, 82), (98, 104), (115, 126)],
        17: [(-80, -50), (8, 42), (95, 105), (109, 119), (120, 125)],
        18: [(-81, -35), (9, 40), (100, 106), (110, 117), (120, 123), (131, 150)],
        19: [(-80, -35), (12, 40), (105, 115), (140, 150)],
        20: [(-77, -37), (13, 40), (47, 50), (130, 142)],
        21: [(-75, -39), (12, 37), (44, 50), (122, 146)],
        22: [(-70, -40), (14, 35), (44, 47), (114, 153)],
        23: [(-71, -48), (15, 32), (114, 153)],
        24: [(-72, -52), (18, 28), (115, 135), (138, 151)],
        25: [(-73, -57), (140, 150), (173, 178)],
        26: [(-74, -63), (145, 148), (167, 174)],
        27: [(-75, -66), (167, 169)],
        28: [(-75, -65)],
        30: [(-65, -55)],
        31: [(-75, -55), (40, 170)],
        32: [(-180, 180)], 33: [(-180, 180)], 34: [(-180, 180)], 35: [(-180, 180)],
    }
    m = np.zeros((36, 72), np.uint8)
    for r, spans in rows.items():
        for (a, b) in spans:
            c0 = int(np.floor((a + 180) / 5))
            c1 = int(np.ceil((b + 180) / 5))
            m[r, max(c0, 0):min(c1, 72)] = 255
    return m


class MapRenderer:
    def __init__(self, width=1080, height=1920, ctx=None):
        self.ctx = ctx or moderngl.create_standalone_context(backend='egl', require=430)
        self.prog = self.ctx.program(vertex_shader=VS, fragment_shader=FS)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '2f', 'in_pos')])
        self.size = None
        self.resize(width, height)
        em = earth_mask()
        self.t_earth = self.ctx.texture((72, 36), 1, np.ascontiguousarray(em).tobytes())
        self.t_earth.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.t_earth.repeat_y = False
        self.t_sat = [self.ctx.texture((4, 4), 3, np.zeros((4, 4, 3), np.uint8).tobytes()) for _ in range(2)]
        self.sat_half = [1.0, 1.0]
        self.sat_on = 0

    def resize(self, w, h):
        if self.size == (w, h):
            return
        if self.size is not None:
            for o in (self.col_fbo, self.col_tex, self.f_fbo, self.f_tex):
                o.release()
        self.size = (w, h)
        self.col_tex = self.ctx.texture((w, h), 4)
        self.col_fbo = self.ctx.framebuffer(color_attachments=[self.col_tex])
        self.f_tex = self.ctx.texture((w, h), 4, dtype='f4')
        self.f_fbo = self.ctx.framebuffer(color_attachments=[self.f_tex])

    def set_satellites(self, sats):
        """sats: [(half, img), (half, img)]: the sharp inner picture and the wide one, each (N, N, 3) uint8, row 0 =
        north, covering [-half, half] in x and y."""
        (h2, img2), (h1, img1) = sorted(sats, key=lambda s: s[0])
        for k, (half, img) in enumerate(((h1, img1), (h2, img2))):
            n = img.shape[0]
            t = self.ctx.texture((img.shape[1], n), 3, np.ascontiguousarray(img[::-1]).tobytes())
            t.build_mipmaps()
            t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
            t.anisotropy = 1.0
            self.t_sat[k] = t
            self.sat_half[k] = float(half)
        self.sat_on = 1

    def _draw(self, fbo, s, mode, t=0.0, cloud=0.0, earth=None, border=0.0, sat=True, clip=None):
        P = self.prog
        w, h = self.size
        P['u_res'] = (float(w), float(h))
        P['u_s'] = float(s)
        P['u_mode'] = int(mode)
        P['u_time'] = float(t)
        P['u_sun'] = tuple(float(v) for v in SUN)
        self.t_sat[0].use(0)
        P['u_sat'] = 0
        P['u_sat_half'] = float(self.sat_half[0])
        self.t_sat[1].use(2)
        P['u_sat2'] = 2
        P['u_sat2_half'] = float(self.sat_half[1])
        P['u_sat_on'] = int(self.sat_on if sat else 0)
        self.t_earth.use(1)
        P['u_earth_tex'] = 1
        if earth is None:
            P['u_earth_c'] = (0.0, 0.0, 1.0)
            P['u_earth_a'] = 0.0
        else:
            ex, ey, er, ea = earth
            P['u_earth_c'] = (float(ex), float(ey), float(er))
            P['u_earth_a'] = float(ea)
        P['u_cloud_a'] = float(cloud)
        P['u_border'] = float(border)
        P['u_clip'] = int(border > 0.0 if clip is None else clip)
        fbo.use()
        self.ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.finish()

    def render(self, L, t=0.0, cloud=0.0, earth=None, border=0.0, sat=True, clip=None):
        """The map of a view L metres wide, centred on spawn, north up: (h, w, 4) float in 0..1 (alpha 0 outside
        the world when clipped; by default it is whenever the border is drawn)."""
        w, h = self.size
        self._draw(self.col_fbo, L / w, 0, t, cloud, earth, border, sat, clip)
        img = np.frombuffer(self.col_fbo.read(components=4), np.uint8).reshape(h, w, 4)[::-1]
        return img.astype(np.float32) / 255.0

    def fields(self, half):
        """Height, water, biome and tree (trunk radius) for every block in [-half, half)^2 (row 0 = north)."""
        n = int(2 * half)
        self.resize(n, n)
        self._draw(self.f_fbo, 1.0, 1)
        f = np.frombuffer(self.f_fbo.read(components=4, dtype='f4'), np.float32).reshape(n, n, 4)[::-1]
        return f
