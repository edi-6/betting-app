# The New Dimension at NIGHT (YouTube Shorts)

The twelfth video in the series: the same 15-second first-person elytra flight through the Sift as
[`../sift-flight`](../sift-flight), flown by night. Under a starry sky and a high moon the valley goes dark, and
everything that glows takes over. The stream, the lake and the two falls of molten-gold ichor burn in the dark, the
red sculk sparkles, pale motes drift like fireflies, and the firework show over the landing bursts pink, cyan and gold
against a black sky.

**The hook:** the first frame is the whole valley at night from the top of the spire, the gold stream glowing through
the dark meadow to a glowing lake. Then you step off and dive down it.

**Ready to upload:** [`release/sift-night.mp4`](release/sift-night.mp4), with an optional cover in
[`release/thumbnail.jpg`](release/thumbnail.jpg), titled *THE NEW DIMENSION / AT NIGHT*. The same frame without
text is in [`release/thumbnail_clean.jpg`](release/thumbnail_clean.jpg).

| | |
| --- | --- |
| Length | 15.0 s (900 frames), a seamless loop |
| Video | 1440x2560 (9:16), 60 fps, H.264 High@5.1, 30 Mbps 2-pass, closed GOP (30), BT.709 |
| Audio | AAC-LC 384 kbps, 48 kHz stereo, -14 LUFS, peak -1.2 dBFS (the flight's soundtrack) |

## Titles and hooks

* **Titles:** *Minecraft's New Dimension at NIGHT Is Insane* · *POV: Flying Through the Sift at Night* · *The Sift
  at Night (Minecraft's New Dimension)* · *Day or Night? Minecraft's New Dimension*
* **Hooks for the first second:** "This is the new dimension... at night" · "Wait for the lake" · "Rate this 1-10".
* **Pair it with the day version:** post them a day or two apart, or make a "day or night?" comparison. Put the
  first frames of the two videos side by side and ask viewers which one they'd pick.
* The timeline (the drop at 0:01.9, the fireworks at 0:05.6 and 0:11.25, the lake at 0:07, the arch at 0:11.5, the
  show at 0:12.7, the landing at 0:14) is the same as the day version's. See its README.

## Render it yourself

Everything comes from [`../sift-flight/src`](../sift-flight/src) with the night look switched on:

```bash
cd ../sift-flight
LOOK=sift_night bash render_segments.sh output/night        # resumable: one-second segments, then the encode
cd src && LOOK=sift_night python thumbnail.py ../../sift-night/release/thumbnail.jpg --t 0
```

The look is `sift_night` in `src/looks.py` and `src/sky.py`: the moon, the stars, faint moonlit clouds, and the
ichor's light carrying the scene.
