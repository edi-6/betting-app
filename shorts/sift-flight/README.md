# POV: Your First Elytra Flight in Minecraft's New Dimension (YouTube Shorts)

The eleventh video in the series, after [`../rollercoaster`](../rollercoaster) and the nine before it. It is short on
purpose: **15 seconds**, one unbroken first-person flight on an elytra through **the Sift**, rendered at
**1440x2560, 60 fps**, supersampled and motion-blurred, with golden-hour light, volumetric haze and a soundtrack
written bar for bar to the flight. It loops without a seam: the last frame is the first.

1. **The edge.** You stand at the corner of a siftslate spire, 56 blocks up, a firework rocket in your hand. Below:
   the whole valley in the low sun, the pink Singer's Meadow, a stream of **molten gold ichor** winding away from the
   spire's foot to a golden lake, white trees with pale-blue crowns, the ribs of a colossal fossil, and a banded cliff
   with two falls of gold pouring off it. Wind, a pad holding its breath.
2. **The drop.** You step off. The elytra snaps open on the beat, and you dive straight down the stream. The beat
   drops as the dive peaks at over 200 km/h.
3. **The grove.** Pulling out low over the meadow, you thread the white trunks under the canopy, the trees whipping
   past on both sides.
4. **The fossil.** A firework on the downbeat: the boost fires you through the giant ribcage, rib after rib, past its
   half-buried skull.
5. **The lake.** Out over the golden ichor, skimming it, both falls roaring past on your left.
6. **The towers.** Between the coral towers, climbing into the sun, with a tall stone arch standing across the sky
   ahead.
7. **The arch.** The second firework on the peak of the music, and you shoot through the arch under its hanging
   vines.
8. **The landing.** Swinging round behind the spire, you look over at its top, and a firework show bursts over the
   valley behind it, pink, cyan and gold, on the beat. You touch down on the spire's top on the final chord, facing
   the valley again. The video starts over, and so does the flight.

**About the Sift:** it is the new dimension announced at Minecraft Live in September 2026 (see
[`../rollercoaster`](../rollercoaster), which rides into it). The turquoise sky, the pink and red land, siftslate,
healthy sculk, the Singer's Meadow, the Carapace and ichor follow what Mojang has shown. The valley, the golden lake
and the flight through it are this video's own take.

**Ready to upload:** [`release/sift-flight.mp4`](release/sift-flight.mp4), with an optional cover in
[`release/thumbnail.jpg`](release/thumbnail.jpg). The cover shows the dive over the gold stream, titled *POV: FIRST
ELYTRA FLIGHT / IN THE NEW DIMENSION* with a yellow *THE SIFT!* splash. The same frame without text is in
[`release/thumbnail_clean.jpg`](release/thumbnail_clean.jpg).

| | |
| --- | --- |
| Length | 15.0 s (900 frames), a seamless loop |
| Video | 1440x2560 (9:16), 60 fps, H.264 High@5.1, 30 Mbps 2-pass, closed GOP (30), BT.709 |
| Audio | AAC-LC 384 kbps, 48 kHz stereo, loudness-normalised to -14 LUFS, peak -1.2 dBFS |
| Rendering | 4 samples per pixel (rendered at 2880x5120), camera motion blur (180-degree shutter) |

Everything is generated from code: no stock footage, samples, fonts or AI-generated media. The textures, the item,
the world, the sky, the sounds and the music are all drawn or synthesised in `src/`.

## Titles and hooks

You add the titles and voice-over. Some ideas that fit:

* **Titles:** *POV: Your First Elytra Flight in Minecraft's NEW Dimension* · *Flying Through the Sift (Minecraft's
  New Dimension)* · *The New Dimension From the Sky* · *I Took an Elytra to Minecraft's New Dimension* · *15 Seconds
  in the Sift*
* **Hooks for the first second** (the first frame has no text or HUD on it, and the valley is all in view): "POV:
  you just got elytra in the new dimension" · "Wait for the lake..." · "This is what the new dimension looks like from
  the sky" · "Don't blink."
* **Voice-over beats:** 0:00.5 the step off ("here we go..."), 0:01.9 the drop, 0:05.6 the first firework ("BOOST"),
  0:05.7 the ribs ("is that a skeleton?!"), 0:07 the lake ("the lake is GOLD"), 0:11.25 the arch ("thread the
  needle"), 0:12.7 the fireworks, 0:14 the landing ("...one more time?").

## Timeline

The music is 8 bars at 128 bpm, one bar to each part of the flight.

| Time | Bar | What happens |
| --- | --- | --- |
| 0:00 | 1 | On the spire's corner, the valley below in the low sun. No text or HUD, so the hook can go on top |
| 0:00.47 | | The step off the edge |
| 0:00.94 | | The elytra snaps open, and the dive begins down the gold stream |
| 0:01.88 | 2 | **The drop** as the dive peaks at over 200 km/h |
| 0:02.5 | | Out of the dive, low into the grove of white trees |
| 0:05.63 | 4 | **Firework #1**: the boost, and through the fossil's ribs (0:05.6-0:06.4), past the skull |
| 0:06.5 | | The golden falls on the cliff ahead |
| 0:07.0 | 5 | Low over the golden lake, both falls roaring past on the left (0:07.4, 0:07.9) |
| 0:08.3 | 5 | Between the coral towers, climbing into the sun, the arch ahead |
| 0:11.25 | 7 | **Firework #2** on the peak of the music, and through the stone arch (0:11.5) |
| 0:12.0 | | Swinging round behind the spire, looking over at its top |
| 0:12.66 | | The firework show over the valley: pink, then cyan (0:13.13), then gold (0:13.59) |
| 0:14.03 | 8 | **Touchdown** on the spire's top on the final chord, facing the valley again |
| 0:15.0 | | The last frame runs straight into the first |

## How it's made

| | |
| --- | --- |
| The flight | `src/flight.py`: a centripetal Catmull-Rom through 31 control points, resampled by arc length. The timing is authored as a speed profile (monotone keys: standing, the fall, the dive's peak, the fireworks' boosts, the flare before landing) and integrated, so every beat lands where the music has it. The view banks into the turns from the lateral acceleration (roll capped like a head would), widens its field of view with speed, tips back only so far on a climb, turns to the spire's top on the approach, and breathes while standing. Every wobble has a whole number of cycles in the 15 seconds, and the path's ends are the same point, so the loop has no seam |
| The valley | `src/world_valley.py`: built round the flight and round the view from the spire. The meadow (pink grass on red sculk, blue and pink grass, flowers in drifts, patches of healthy sculk, boulders), the gold stream in its sunken channel, the grove (trunks well off the path, crowns meeting over it), the Carapace (sand, walls of blue stone, the fossil's spine, ribs and skull), the golden lake, the eastern cliff with its two falls, the coral towers, the stone arch across the climb, and ridges and mesas of banded siftslate all round: low in the west where the sun comes from, tall in the east. The flight's corridor is carved clear |
| The effects | `src/fx.py`: the firework rocket in the hand (kept in place on screen as the view widens, dipping on the use swing), the rocket's sparks and smoke streaming back, the firework show (rockets rising with sparkling trails, 340 stars a burst, slowing, falling, twinkling out and changing colour), pollen in the sun, pale motes over the meadow, spray at the foot of the falls, blubs |
| The renderer | `src/gfx.py`: the deferred OpenGL renderer from the earlier videos (shadows, SSAO, volumetric haze with sun shafts, bloom, tonemapping), with new work for this one: **camera motion blur** (a per-pixel reconstruction filter: velocities from depth and the camera at the shutter's opening and closing, the largest per tile, then a gather that lets fast foreground smear over the background, and never the held item), per-colour streaks, held-item lighting, the **molten-gold ichor** (slow warped swirls with bright veins, sampled on the block's pixel grid; streaming down the falls), and foliage in the wind |
| The look | `src/looks.py`, `src/sky.py`: the Sift at golden hour. A low sun in the west-north-west, peach light, turquoise shade, a haze in the valley, clouds lit pink and gold, and distant ranges round the horizon. The exposure stops down when the view turns into the sun |
| The loop | `src/director.py`: over the last 0.8 s, everything that moves with time (the ichor, the falls, the blubs, the motes, the last of the show) is faded into how it was in the first frame, while the rider stands still where they began |
| Sound | `src/audio.py` builds it all from the flight's cue sheet (`src/cues.py`, measured off the flight and the valley). **Music:** eight bars in E major. A held breath on the spire (a pad, a riser, the hook's pick-up), then the drop: four on the floor, an off-beat bass, supersaw chords pumping under the kick, a sixteenth-note pluck arpeggio through a ping-pong delay, and the lead's hook (E - B - C#m - A). An impact on each firework, a snare roll and riser into the peak, a choir at the top. It resolves to E as the feet touch down, and the pad holds round the loop. **Effects:** the wind with the speed (a breeze on the spire, a roar in the dive, the wings' flutter), the step and the elytra opening, the rockets' pop, hiss and crackle, a whoosh for every tree, rib and tower that goes past (timed, panned to its side, louder the closer it is), the arch, the falls' roar, the ichor's gloops, the show's whistles, booms and crackle, and the landing. The mix wraps round: what rings on past the end is heard under the start |

Shared with the earlier projects: the voxel engine (`voxel.py`, `blocks.py`, `textures.py`, `entities.py`), the Sift's
blocks and blub from the rollercoaster, and the renderer, all turned up for this one.

## Render it yourself

```bash
pip install -r requirements.txt
# Linux also needs Mesa's EGL + llvmpipe, e.g.: apt install libegl1 libegl-mesa0 libgl1-mesa-dri
cd src
python main.py --out ../output                            # delivered quality: 1440x2560, 4 spp; the audio; final.mp4
python main.py --out ../preview --res 540 --ss 1          # a quick small check with the sound (preview.mp4)
python main.py --out ../output --res 1080 --ss 3          # 1080x1920 instead (9 samples a pixel)
python main.py --out ../sound --cues-only                 # sound work: the cue sheet and the audio, no frames
python main.py --out ../output --encode-only              # redo the soundtrack and the encode from a render
python thumbnail.py ../output/thumbnail.jpg               # the cover; add --no-title for a clean one
python flight.py                                          # the flight's length, speeds and control-point times
python qa.py ../output/final.mp4                          # streams, frame checks and the loop seam
```

The first run bakes the sky and builds the valley into a cache (under a minute). A delivered frame takes about 3 to 4
seconds on a 4-core CPU renderer, so the whole video takes about 50 minutes.

## Make your own

* **The route:** the control points in `src/flight.py`. The valley is carved round the path and its features are
  placed along it (the grove, the ribs, the towers, the arch), so they follow when it moves.
* **The timing:** the speed keys `V_KEYS` in `src/flight.py`, and the firework times `FIREWORKS`.
* **The light:** `src/looks.py` and the `sift_gold` sky in `src/sky.py` (the sun's direction and height, the haze).

## Upload tips

* Upload the MP4 as-is. It is 1440p, so YouTube keeps a higher-quality encode of it than of a 1080p upload, and it
  looks sharper on phones too. It is under a minute and 9:16, so YouTube treats it as a Short.
* It loops seamlessly with YouTube's own looping of Shorts. The picture and the sound both run from the last frame
  straight into the first, so many viewers will go round more than once.
* The first frame has no text or HUD on it, so your hook can go on top. The valley and all the landmarks are in view.
* No flashing: the brightest moments are the firework bursts (a 0.1 s glow each, far off).
* Tag it for the new dimension: #minecraft #thesift #elytra #minecraftlive.
