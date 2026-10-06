# MLG Water Bucket from the Build Limit (YouTube Shorts)

POV: you stand on a single block at **Y=320**, the build limit, with nothing in your hand but a water bucket. Far
below, the Sift's golden valley shows through gaps in Minecraft's clouds, with one small spire top as the target. You
step off, fall 220 blocks through the cloud layer while the world spins below you, and place the water at the last
instant. Splash. Then you scoop the water back up, the *Advancement Made!* toast slides in, and the view swings up
over the valley.

It runs **10 seconds**, all one first-person shot at **1440x2560 and 60 fps**. It is supersampled and motion-blurred,
with a heartbeat soundtrack that speeds up the whole way down.

**Ready to upload:** [`release/mlg-clutch.mp4`](release/mlg-clutch.mp4), with an optional cover in
[`release/thumbnail.jpg`](release/thumbnail.jpg). The cover is a frame from the fall, titled *MLG FROM THE BUILD
LIMIT*, with a yellow *CLUTCH?!* splash.

| | |
| --- | --- |
| Length | 10.0 s (600 frames) |
| Video | 1440x2560 (9:16), 60 fps, H.264 High@5.1, 30 Mbps 2-pass, closed GOP (30), BT.709 |
| Audio | AAC-LC 384 kbps, 48 kHz stereo, loudness-normalised to -14 LUFS, peak -1.2 dBFS |
| Rendering | 4 samples per pixel (rendered at 2880x5120), camera motion blur |

Everything is generated from code: there is no stock footage, no samples and no AI-generated media.

## Titles and hooks

* **Titles:** *MLG Water Bucket from the BUILD LIMIT* · *POV: You Clutch From Y=320* · *220 Blocks. One Bucket.* ·
  *Can You MLG From the Build Limit?* · *The Hardest MLG in Minecraft*
* **Hooks for the first second.** The first frame has no text except the Y counter, so your hook can go on top:
  * "POV: you're at the build limit with only a water bucket"
  * "Y=320. One bucket. Don't miss."
  * "Wait for the landing..."
* **Pinned comment:** *Would you have clutched it? Y=320 to Y=100, 220 blocks, zero hearts lost.* Comment bait
  that works well under this kind of Short: *Rate the clutch 1-10.*
* **Voice-over beats:**
  * 0:00.6, the step off: "here goes nothing..."
  * 0:03.7, through the clouds.
  * 0:05.2, the bucket: "NOW!"
  * 0:05.3, the splash.
  * 0:06.1, the scoop: "and we take the water back."
  * 0:06.4, the advancement.
* **Tags:** #minecraft #mlg #clutch #waterbucket #shorts

The toast says *Advancement Made! MLG Water Bucket*. That is the video's joke, not a real advancement in the game.

## Timeline

| Time | What happens |
| --- | --- |
| 0:00 | On one block at the build limit, looking down: the clouds, and the valley 220 blocks below. The counter reads **Y: 320** |
| 0:00.62 | The step off. Wind, a low choir and a riser start, and the heartbeat starts slowly |
| 0:03.7 | Through the cloud layer at Y=192. The valley opens up and the spire top grows in the middle of the view |
| 0:05.19 | **The bucket**, placed on the spire top, falling at over 200 km/h |
| 0:05.32 | **Splash.** Down into the water up to the eyes; the impact hits and a big A-major chord lands |
| 0:06.10 | The water is scooped back up |
| 0:06.35 | **Advancement Made! MLG Water Bucket** |
| 0:06.2-0:08.7 | The view swings up and round over the valley: the golden lake, the falls, the fossil |
| 0:10.0 | End |

## How it's made

It reuses [`../sift-flight`](../sift-flight)'s world and renderer: the Sift valley, golden-hour light, volumetric
haze, bloom and supersampled, motion-blurred frames. The short's own code is in
[`../sift-flight/src/mlg.py`](../sift-flight/src/mlg.py).

| | |
| --- | --- |
| The fall | A physical fall from Y=321.62 (the eye) that eases into a terminal speed of 62 blocks/s. The impact time is solved so the eye ends in the water on the spire top. The view looks nearly straight down and turns 21 degrees a second. The field of view widens with speed, and there is a shake from the wind and a jolt on landing |
| The clouds | Minecraft's own clouds: 12-block cells, 4 blocks thick, at Y=192, drifting east. The cloud cover is thinner over the valley and leaves the fall line clear, so the target shows through from the top |
| Beyond the valley | More of the dimension, laid out the way it looks from up there: 16-block tiles (one texel per block) of pink meadow with patches of teal sculk and mesas of siftslate, and larger tiles further out in the haze. This way the world doesn't end at the valley's edges |
| The clutch | The water bucket in the hand. The use swing places a water source block (with its true 0.88 height) on the spire top 0.13 s before impact. The view goes under the water, comes back up, and the bucket refills when the water is picked up |
| The HUD | The debug screen's Y coordinate counting down, and the advancement toast in the game's style, all in the pixel font |
| Sound | Built in `mlg.py` from the toolkit in `audio.py`. The wind follows the speed and cuts dead on the splash. Under the fall: a low choir, a riser, and a heartbeat that quickens. Then the bucket's pour, the splash and gloops, an impact, a supersaw and choir chord in A major, the scoop, the toast's whoosh and a bell arpeggio |

## Render it yourself

```bash
cd ../sift-flight/src
python mlg.py ../output/mlg --res 540 --ss 1 --stills 0,2,4,5.3,8      # quick stills
python mlg.py ../output/mlg                     # 1440x2560, 4 spp, in 1-second segments (resumes after a stop)
python mlg.py ../output/mlg --audio             # the soundtrack
python mlg.py ../output/mlg --cover 1.5 --res 1080 --ss 3              # the cover
```

A frame takes about 4 seconds on a 4-core CPU renderer, so the whole video takes about 45 minutes. Encode it with
`main.encode_youtube` as in [`../sift-flight`](../sift-flight).

## Upload tips

* Upload the MP4 as it is. It's 1440p, so YouTube keeps a sharper encode than it would for a 1080p upload.
* The first frame is already the hook, a straight drop from the build limit, so put your text on top.
* No flashing. The splash is a short blue tint.
