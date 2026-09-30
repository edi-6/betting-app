"""The light and look of the flight (gfx.env keys; see gfx.DEFAULT).

THE SIFT AT GOLDEN HOUR: the sun low in the west-south-west (behind the player on the spire, ahead of them on the
climb between the coral towers), peach light, turquoise shade, pink light bounced off the meadow, a haze lying in
the valley that the sun's shafts cut through.
"""
import gfx
from sky import sun_dir

SUN = sun_dir(165.0, 12.0)

LOOKS = {
    'sift_gold': dict(sky='sift_gold', light_dir=SUN, light_col=(3.3, 2.15, 1.7), sky_amb=(0.30, 0.42, 0.50),
                      gnd_amb=(0.36, 0.18, 0.20), blk_col=(2.4, 1.3, 0.9), sun_disc_dir=SUN, sun_size=0.028,
                      sun_disc_col=(26.0, 17.0, 12.0), exposure=0.85, fog=(0.0015, 26.0, 44.0, 320.0),
                      fog_sun=(1.0, 0.72, 0.64), fog_amb=(0.03, 0.10, 0.12), fog_g=0.56, aerial=0.0006,
                      aerial_col=(0.35, 0.62, 0.70), max_dist=600.0, bloom=0.4, bloom_thresh=1.8, sat=1.10,
                      contrast=1.06, vignette=0.2, wind=0.8, hand_gain=0.95, fog_steps=28),
}


def get(name='sift_gold', **kw):
    e = gfx.env(**LOOKS[name])
    e.update(kw)
    return e
