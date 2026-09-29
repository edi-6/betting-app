"""Named lighting/look presets for the film's scenes (see gfx.DEFAULT for what each key does)."""
import gfx
from gfx import sun_dir

SUN_AZ, SUN_EL = 172.0, 5.5           # low in the west, slightly north: behind the village seen from spawn
MOON_AZ, MOON_EL = 180.0, 6.0             # low in the west: through the west windows onto the east wall


def get(name, **kw):
    e = dict(PRESETS[name])
    e.update(kw)
    return gfx.env(**e)


PRESETS = {
    'sunset': dict(sky='sunset', light_dir=sun_dir(SUN_AZ, SUN_EL + 4.0), light_col=(3.4, 1.75, 0.85),
                   sun_disc_dir=sun_dir(SUN_AZ, SUN_EL), sun_size=0.011, sun_disc_col=(40.0, 22.0, 9.0),
                   sky_amb=(0.30, 0.33, 0.48), gnd_amb=(0.24, 0.18, 0.12), blk_col=(1.6, 1.08, 0.62),
                   aerial=0.0015, aerial_col=(0.95, 0.62, 0.42), fog=(0.0022, 16.0, 0.0, 110.0),
                   fog_sun=(0.5, 0.45, 0.4), fog_amb=(0.05, 0.05, 0.07), exposure=0.9, sat=1.02, contrast=1.06,
                   lift=(0.012, 0.006, 0.02), gain=(1.02, 1.0, 0.95), bloom=0.3, sky_mul=(0.72, 0.72, 0.72)),
    'dusk': dict(sky='dusk', light_dir=sun_dir(SUN_AZ, 8.0), light_col=(0.55, 0.34, 0.28),
                 sun_size=0.0, sky_amb=(0.10, 0.11, 0.20), gnd_amb=(0.05, 0.04, 0.05), blk_col=(1.6, 1.08, 0.62),
                 aerial=0.004, aerial_col=(0.12, 0.08, 0.12), fog=(0.006, 12.0, 0.0, 90.0), fog_sun=(0.6, 0.5, 0.6),
                 fog_amb=(0.02, 0.02, 0.04), exposure=1.0, sat=0.85, stars=0.3, moon_dir=sun_dir(MOON_AZ, 12.0),
                 moon_bright=1.2, lift=(0.01, 0.01, 0.03)),
    'night': dict(sky='night', light_dir=sun_dir(MOON_AZ, MOON_EL), light_col=(0.16, 0.20, 0.32),
                  sky_amb=(0.022, 0.030, 0.055), gnd_amb=(0.008, 0.010, 0.016), blk_col=(1.6, 1.08, 0.62),
                  moon_dir=sun_dir(MOON_AZ, MOON_EL), moon_bright=2.2, moon_size=0.03, stars=1.0,
                  aerial=0.005, aerial_col=(0.008, 0.010, 0.018), fog=(0.010, 10.0, 0.0, 70.0),
                  fog_sun=(0.12, 0.13, 0.16), fog_amb=(0.006, 0.008, 0.014), exposure=1.05, sat=0.7, contrast=1.08,
                  lift=(0.004, 0.006, 0.014), gain=(0.96, 1.0, 1.06), wind=0.35, bloom=0.4, bloom_thresh=1.4),
    'interior_night': dict(sky='night', light_dir=sun_dir(MOON_AZ, MOON_EL), light_col=(0.20, 0.25, 0.40),
                           sky_amb=(0.02, 0.028, 0.05), gnd_amb=(0.006, 0.008, 0.012), blk_col=(1.5, 1.0, 0.58),
                           moon_dir=sun_dir(MOON_AZ, MOON_EL), moon_bright=2.2, stars=1.0, aerial=0.0,
                           fog=(0.012, 6.0, 0.0, 30.0), fog_sun=(0.35, 0.35, 0.45), fog_amb=(0.004, 0.005, 0.008),
                           exposure=1.35, sat=0.8, contrast=1.08, wind=0.3, bloom=0.4, bloom_thresh=1.4),
    'underground': dict(sky='night', light_dir=sun_dir(MOON_AZ, MOON_EL), light_col=(0.0, 0.0, 0.0),
                        sky_amb=(0.0, 0.0, 0.0), gnd_amb=(0.0, 0.0, 0.0), blk_col=(1.6, 1.1, 0.66),
                        soul_col=(0.25, 0.9, 1.1), min_amb=(0.006, 0.006, 0.008), aerial=0.0,
                        fog=(0.018, 60.0, -40.0, 60.0), fog_amb=(0.0, 0.0, 0.0), exposure=1.15, sat=0.9,
                        contrast=1.08, lift=(0.004, 0.004, 0.006), wind=0.0, bloom=0.45, bloom_thresh=1.2),
    'chamber': dict(sky='night', light_dir=sun_dir(MOON_AZ, MOON_EL), light_col=(0.0, 0.0, 0.0),
                    sky_amb=(0.0, 0.0, 0.0), gnd_amb=(0.0, 0.0, 0.0), blk_col=(1.5, 1.0, 0.6),
                    soul_col=(0.22, 0.85, 1.05), min_amb=(0.0025, 0.0025, 0.0035), aerial=0.0,
                    fog=(0.02, 80.0, -96.0, 50.0), fog_amb=(0.0, 0.0, 0.0), exposure=1.2, sat=0.75, contrast=1.1,
                    wind=0.0, bloom=0.5, bloom_thresh=1.0),
    'wrong': dict(sky='wrong', light_dir=sun_dir(160.0, 14.0), light_col=(0.30, 0.42, 0.26),
                  sky_amb=(0.05, 0.09, 0.07), gnd_amb=(0.02, 0.03, 0.02), blk_col=(1.5, 1.0, 0.5),
                  moon_dir=sun_dir(160.0, 9.0), moon_bright=2.6, moon_size=0.24, stars=0.2,
                  aerial=0.006, aerial_col=(0.03, 0.06, 0.045), fog=(0.012, 14.0, 0.0, 90.0), fog_sun=(0.6, 0.7, 0.5),
                  fog_amb=(0.01, 0.02, 0.015), exposure=1.0, sat=0.8, contrast=1.1, lift=(0.0, 0.012, 0.008),
                  gain=(0.94, 1.03, 0.96), wind=0.8, bloom=0.4, bloom_thresh=1.3),
}
