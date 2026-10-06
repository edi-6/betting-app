# How Big Is a Minecraft World? (YouTube Shorts)

A new format for the channel: no story, one shot. The camera starts on a single pixel of Steve's eye and pulls back
without a cut, past Steve, his camp, a chunk, a village and the render distance, out over thousands of kilometres of
generated world, past the Earth drawn to scale, to the world border. Then the twist: **the whole world is one pixel**,
the pupil of Steve's eye, and the video is back where it started. The loop is seamless, so it plays over and over,
and the second time round you know what's in his eye.

1. **The eye.** Six of Steve's pixels across. The counter reads **6.4 PIXELS**. The pupil, marked **1 PIXEL**,
   holds a tiny map (on a first watch it just looks odd).
2. **Steve.** He's lying in the grass staring at the sky, his dog curled up beside him and a campfire smoking by his
   head. **16 PIXELS = 1 BLOCK**, with one grass block outlined.
3. **The camp.** His house with a smoking chimney, a wheat farm, a crafting table, a furnace, a chest and a sheep
   pen. The game's chunk borders come up: **1 CHUNK, 16 x 16 BLOCKS**.
4. **The village.** A dirt path south to **A VILLAGE**: a well, ten houses, a church with a tower, a blacksmith with a
   lava pool, farms, villagers, an iron golem, and a bridge over the river.
5. **The render distance.** The world outside 12 chunks dims, as if it wasn't loaded: **ALL YOUR GAME LOADS**.
6. **The world.** Forests, plains, deserts, snowy mountains, then coastlines, oceans and continents. The walking
   time to cross the screen climbs from minutes to hours to days.
7. **The Earth to scale.** The whole planet, drawn over the map at its true size, is small.
8. **The Far Lands.** Where the old Java Edition's terrain used to break, 12,550,821 blocks out.
9. **The world border.** **60,000,000 BLOCKS** across, **7x BIGGER THAN EARTH** (its surface).
10. **The twist.** The counter flips to **1.0 PIXEL**: the world is the pupil of a giant Steve's eye. *THE WHOLE
    WORLD... IS ONE PIXEL OF STEVE'S EYE*. The face fills in round it, the camera settles on the eye, and it's the
    first frame again.

**Ready to upload:** [`release/zoom-out.mp4`](release/zoom-out.mp4). There's an optional cover image in
[`release/thumbnail.jpg`](release/thumbnail.jpg): the eye with the world in its pupil, titled *HOW BIG IS A
MINECRAFT WORLD?* and *THE WHOLE WORLD = 1 PIXEL*. The same image without text is in
[`release/thumbnail_clean.jpg`](release/thumbnail_clean.jpg).

| | |
| --- | --- |
| Length | 30.5 s, a seamless loop (the last frame runs straight into the first, and so does the sound) |
| Video | 1080x1920 (9:16), **60 fps**, H.264 High@4.2, 16 Mbps 2-pass, closed GOP (30), BT.709 |
| Audio | AAC-LC 384 kbps, 48 kHz stereo, loudness-normalised to -14 LUFS, true peak -2 dBFS |
| Rendering | 4 samples per pixel (rendered at 2160x3840), zoom motion blur |

Everything is generated from code: no stock footage, samples, fonts or AI-generated media. The world, the
Minecraft-style font, textures, characters, sounds and music are all drawn or synthesised in `src/`.

## Titles, hooks, pinned comment

* **Titles:** *How Big Is a Minecraft World?* · *The Minecraft World Is Bigger Than You Think* · *Minecraft World
  vs Earth (to Scale)* · *Zooming Out of a Minecraft World* · *Wait for the Last Pixel...*
* **Hooks for the first second.** The title is on screen the whole time, so a voice-over or caption can go straight
  in:
  * "This is one pixel. Now watch."
  * "How far does a Minecraft world go?"
  * "The Earth is in this video. To scale."
  * "Wait for the last pixel..."
* **Pinned comment:** *Did you notice what was in his eye at the start?* Or *How long would it take you to walk
  to the border?* (161 days at walking speed). Both send people back to the start.
* **Voice-over beats:**
  * 0:00 the pixel
  * 0:01.8 "one block"
  * 0:06.7 "one chunk"
  * 0:10.4 "a village"
  * 0:13.2 "this is all your game loads"
  * 0:16.5 ten thousand blocks
  * 0:23.1 "that's the Earth. To scale."
  * 0:24.6 the Far Lands
  * 0:25.7 "the world border"
  * 0:26.4 "...wait"
  * 0:26.6 "the whole world is one pixel"
* **Tags:** #minecraft #scale #zoom #minecraftfacts #shorts

## Timeline

| Time | View width | What's on screen |
| --- | --- | --- |
| 0:00 | 6.4 pixels | Steve's eye. The pupil (1 PIXEL) holds the whole world |
| 0:01.8 | 1 block | Steve's face, then all of him, lying in the grass |
| 0:04.5 | 4 blocks | His dog, the flowers, the campfire's smoke. 16 PIXELS = 1 BLOCK, one block outlined |
| 0:06.7 | 15 blocks | The camp, and the game's chunk borders: 1 CHUNK |
| 0:10.4 | 110 blocks | The path south to the village, ringed: A VILLAGE. The river, the forest |
| 0:11.0 | 150 blocks | The 3D world hands over to the map (seamlessly: the map shows a capture of the same 3D world) |
| 0:13.2 | 520 blocks | RENDER DISTANCE: everything outside 12 chunks dims. Clouds pass below |
| 0:14.1 to 0:20 | 1,000 to 1,000,000 blocks | Biomes, mountains, the first coasts and oceans, faster and faster |
| 0:23.1 | 13,500,000 blocks | EARTH (TO SCALE), drawn over the map |
| 0:24.6 | 28,000,000 blocks | THE FAR LANDS: a square at 12,550,821 blocks |
| 0:25.0 | 33,750,000 blocks | The world fills the screen's height. Beyond its edges: skin |
| 0:25.7 | 45,000,000 blocks | THE WORLD BORDER, 7x BIGGER THAN EARTH |
| 0:26.4 | 60,000,000 blocks | **1.0 PIXEL**, with a deep hit: the world is the pupil of a giant Steve's eye |
| 0:26.6 | 1.1 pixels | THE WHOLE WORLD... IS ONE PIXEL OF STEVE'S EYE |
| 0:29.3 | 3.5 pixels | The eye, 1 PIXEL outlined again |
| 0:30.5 | 6.4 pixels | The first frame (the video loops) |

## How it's made

| | |
| --- | --- |
| The world | `src/zoommap.py`: a world generator in a fragment shader, evaluated per pixel at any scale. Every view is centred on spawn, so its noise coordinates stay small and 32-bit floats work from a fraction of a block to 60,000 km. Continents and oceans, heights, temperature and humidity for 15 biomes, rivers, mountain ranges with snow, trees with their shadows, Minecraft's blocky clouds and their shadows, hill shading, the Earth to scale, the world border. Detail too small for a pixel is faded out and replaced by its average, so the map never shimmers |
| Spawn in 3D | `src/spawnworld.py`: 640 x 640 blocks round spawn, built from the same generator's height, river, biome and tree fields, so 3D and map are one world. On top: Steve's camp, the village (houses, church, blacksmith, farms, pens, lamp posts, villagers, an iron golem), the path and the bridge |
| The hand-over | `src/zscene.py`: the 3D lens narrows as the view widens (50 to 4 degrees), so by 150 blocks it's nearly orthographic. The map shows two orthographic "satellite" captures of the 3D world (a sharp one and a wide one), faded into the generated map. The 3D frame and the map at the hand-over differ by about 2% on average |
| Steve and the eye | `src/steve.py`: Steve as one voxel per skin pixel, lying with the pupil of his eye on the world's origin. The pupil's top face is exactly 1/16 of a block, and inside it the map is drawn at 960,000,000 times the scale: the world border is the edge of the pixel |
| The loop | `src/ztimeline.py`: the zoom's speed is a periodic spline over the scale, so speed and position match at the seam. 960,000,000 is exactly the world (60,000,000 blocks) over one pixel (1/16 block), so the last frame becomes the first |
| The HUD | `src/zhud.py`: the counter (pixels, then blocks, then the giant's pixels), the walking time, the milestones, and the outlines drawn on the world: the pixel, a block, the chunk borders, the village, the render distance, the Far Lands and the Earth |
| Sound | `src/zaudio.py`: soft felt piano and strings in D, the chords changing on the story's beats; an endless rising Shepard tone that climbs three octaves per loop in step with the zoom; chimes on each milestone; a deep hit for the pixel; the campfire and birds near Steve and high wind over the map. Everything is mixed round a circle of exactly the video's length, reverb and compressor included, so the loop has no seam |
| The renderer | `src/renderer.py`: the deferred renderer from the earlier videos, with orthographic cameras, per-frame shadow and ambient occlusion scales (from 1/16 of a block to hundreds), and a switchable colour grade |

## Render it yourself

```bash
pip install -r requirements.txt
# Linux also needs Mesa's EGL + llvmpipe, e.g.: apt install libegl1 libegl-mesa0 libgl1-mesa-dri
cd src
python main.py --out ../output                                   # delivered quality (about a second a frame), resumable
python main.py --out ../stills --frames 0,600,1200,1600 --no-audio  # a few frames as stills
python main.py --out ../output --encode-only                      # redo the soundtrack and the encode
python thumbnail.py ../output/thumbnail.jpg                       # the cover; --no-title for a clean one
python ztimeline.py                                               # the schedule: when each scale is on screen
python zaudio.py ../output/audio.wav                              # just the soundtrack
```

The first run builds the 3D world (about 10 s) and its satellite captures (about 15 s) into a cache. Frames go into
one-second segments, so a stopped render picks up where it left off.

## Upload tips

* Upload the MP4 as it is. It's 60 fps and already meets YouTube's recommended settings for 1080p. It's under a
  minute and 9:16, so YouTube treats it as a Short.
* It's built to loop: don't add an end card or a black frame, or the seam will show.
* The facts on screen: a block is 16 pixels; a chunk is 16 x 16 blocks; the default render distance is 12 chunks;
  the world border is about 30,000,000 blocks from spawn each way (60,000,000 across); the Far Lands began at 12,550,821
  blocks in Java Edition before Beta 1.8; the Earth is 12,742 km across, and the world's 3.6 billion km² is about 7
  times the Earth's surface (510 million km²); walking speed is 4.317 blocks a second. The world itself is
  generated for the video, not a real seed.
