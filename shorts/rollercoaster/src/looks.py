"""The light and look of each dimension (gfx.env keys; see gfx.DEFAULT).

OVERWORLD  golden hour: a low sun from the west-south-west, warm light, blue shade, a little haze and sun shafts.
NETHER     no sun: everything is lit by the lava (block light) through a thick red haze.
THE END    no sun: a dim, cold violet light, the pale end stone, the dark sky, the void fading to black below.
"""
import numpy as np

import gfx
from sky import sun_dir

SUN = sun_dir(196.0, 13.0)
SIFT_SUN = sun_dir(235.0, 30.0)

LOOKS = {
    'over': dict(sky='golden', light_dir=SUN, light_col=(3.3, 2.1, 1.15), sky_amb=(0.42, 0.48, 0.66),
                 gnd_amb=(0.32, 0.22, 0.14), blk_col=(2.6, 1.55, 0.78), sun_disc_dir=SUN, sun_size=0.028,
                 sun_disc_col=(40.0, 24.0, 10.0), exposure=0.85, fog=(0.0035, 40.0, 20.0, 260.0),
                 fog_sun=(1.2, 0.9, 0.6), fog_amb=(0.05, 0.06, 0.08), aerial=0.0016, aerial_col=(1.0, 0.72, 0.5),
                 max_dist=600.0, bloom=0.3, bloom_thresh=2.2, sat=1.08, contrast=1.05, vignette=0.22),
    'nether': dict(sky='nether', light_dir=sun_dir(30.0, 70.0), light_col=(0.0, 0.0, 0.0),
                   sky_amb=(0.0, 0.0, 0.0), gnd_amb=(0.0, 0.0, 0.0), blk_col=(2.2, 1.0, 0.36),
                   min_amb=(0.20, 0.085, 0.06), emit_gain=3.0, exposure=1.0,
                   fog=(0.010, 200.0, 0.0, 200.0), fog_sun=(0.0, 0.0, 0.0), fog_amb=(0.0, 0.0, 0.0),
                   fog_flat=(0.30, 0.085, 0.035), aerial=0.0, max_dist=320.0, bloom=0.45, bloom_thresh=1.8,
                   sat=1.1, contrast=1.06, vignette=0.26, wind=0.0),
    'end': dict(sky='end', sky_mul=(0.62, 0.55, 0.66), light_dir=sun_dir(40.0, 62.0), light_col=(0.78, 0.72, 0.66),
                sky_amb=(0.12, 0.11, 0.14), gnd_amb=(0.05, 0.045, 0.05), blk_col=(2.4, 1.5, 0.8),
                min_amb=(0.012, 0.008, 0.02), emit_gain=3.4, exposure=1.1, fog=(0.006, 30.0, 20.0, 300.0),
                fog_sun=(0.3, 0.25, 0.5), fog_amb=(0.03, 0.02, 0.05), aerial=0.0018, aerial_col=(0.10, 0.07, 0.16),
                max_dist=500.0, bloom=0.5, bloom_thresh=1.5, sat=1.05, contrast=1.08, vignette=0.28, wind=0.2,
                stars=0.0),
    'deep': dict(sky='deep', light_dir=sun_dir(30.0, 70.0), light_col=(0.0, 0.0, 0.0), sky_amb=(0.0, 0.0, 0.0),
                 gnd_amb=(0.0, 0.0, 0.0), blk_col=(2.0, 1.2, 0.6), soul_col=(0.9, 2.6, 3.0),
                 min_amb=(0.12, 0.20, 0.26), emit_gain=3.0, exposure=1.45, fog=(0.011, 200.0, 0.0, 160.0),
                 fog_sun=(0.0, 0.0, 0.0), fog_amb=(0.0, 0.0, 0.0), fog_flat=(0.035, 0.10, 0.13), aerial=0.0,
                 max_dist=260.0, bloom=0.55, bloom_thresh=1.4, sat=1.1, contrast=1.08, vignette=0.32, wind=0.0),
    'sift': dict(sky='sift', light_dir=SIFT_SUN, light_col=(2.4, 2.1, 2.05), sky_amb=(0.22, 0.46, 0.50),
                 gnd_amb=(0.34, 0.16, 0.20), blk_col=(2.4, 1.3, 0.9), sun_disc_dir=SIFT_SUN, sun_size=0.03,
                 sun_disc_col=(26.0, 22.0, 24.0), exposure=0.82, fog=(0.0020, 40.0, 20.0, 260.0),
                 fog_sun=(0.7, 0.55, 0.6), fog_amb=(0.03, 0.11, 0.12), aerial=0.0012, aerial_col=(0.25, 0.70, 0.72),
                 max_dist=600.0, bloom=0.35, bloom_thresh=2.0, sat=1.18, contrast=1.05, vignette=0.2, wind=0.5),
}


def get(world, **kw):
    e = gfx.env(**LOOKS[world])
    e.update(kw)
    return e
