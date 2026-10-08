# 1 vs 10 vs 100 vs 1,000 Block Tsunami vs Minecraft Village (YouTube Shorts)

A seaside Minecraft village is hit by four tsunamis, each ten times taller than the last:

* **1 block.** It washes over the fisherman's feet. He looks down. *Hmm.*
* **10 blocks.** It floods the village and breaks blocks out of the houses, but the houses stand.
* **100 blocks.** It wipes the village off the map.
* **1,000 blocks.** A kilometre-high wall blots out the sun... and you put down a **sponge**. The sponge drinks the
  whole sea.

The format does the hooking:

* **An escalation everybody can follow.** The four rounds sit across the top of the screen, so the viewer always
  knows how far there is to go. The village's health is ten Minecraft hearts that drain as it breaks up, and each
  round ends with a verdict (**SURVIVED** or **DESTROYED**).
* **The cold open shows the 1,000-block wall first.** Everyone stays to see what it does, and nobody guesses the
  sponge.
* **The twist is pure Minecraft.** A sponge soaks up the water around it and turns into a wet sponge, so "one sponge
  vs a kilometre-high tsunami" is the kind of joke players repeat in the comments. The fisherman's *hmm* opens and
  closes the video.

**Ready to upload:** [`release/tsunami.mp4`](release/tsunami.mp4). The cover is
[`release/thumbnail.jpg`](release/thumbnail.jpg): the 1,000-block wall towering over the village street, the
villagers and the iron golem staring up at it, titled *1 vs 1,000 / BLOCK TSUNAMI*. The same picture without text is
[`release/thumbnail_clean.jpg`](release/thumbnail_clean.jpg).

| | |
| --- | --- |
| Length | 31.9 s |
| Video | 1080x1920 (9:16), **60 fps**, H.264 High@4.2, 16 Mbps 2-pass, closed GOP (30), BT.709 |
| Audio | AAC-LC, 48 kHz stereo, loudness-normalised to -14 LUFS, true peak -1.4 dBFS |

Everything is made in code: the world, the water, the destruction, the characters, the music and the sound. There is
no footage, there are no samples, and there is no AI-generated media.

## Titles, hooks, pinned comment

* **Titles:** *1 vs 10 vs 100 vs 1,000 Block Tsunami vs Minecraft Village* · *Can a Minecraft Village Survive a
  1,000 Block Tsunami?* · *Minecraft Tsunamis: 1 Block to 1,000 Blocks* · *Wait for the 1,000 Block Tsunami...*.
  With the twist spoiled: *1 Sponge vs 1,000 Block Tsunami*.
* **Hooks for the first second.** The first frame is the wall over the village, so a line over it lands:
  * "Which tsunami destroys the village?"
  * "Wait for the 1,000 block one..."
  * "A Minecraft village vs a tsunami 10x bigger every round."
* **Pinned comment:** *The sponge did its job 🧽 Which round did you think the village would survive? 10,000 blocks
  next?* The question gets people commenting their guesses, and the second line turns requests into a series.
* **Description fact** (for anyone who asks how tall tsunamis really get): the tallest wave run-up ever measured was
  524 m, at Lituya Bay, Alaska, in 1958 ([Earth Magazine](https://www.earthmagazine.org/article/benchmarks-july-9-1958-megatsunami-drowns-lituya-bay-alaska/)).
  That's about half of round 4.
* **Tags:** #minecraft #tsunami #minecraftshorts #physics #satisfying #shorts

## Timeline

| Time | What happens |
| --- | --- |
| 0:00 | **Cold open.** The 1,000-block wall rises behind the village. The iron golem and the villagers stare up at it. *1 vs 10 vs 100 vs 1,000 / BLOCK TSUNAMI* |
| 0:01.5 | **1 BLOCK.** The fisherman stands with his back to the sea. A little wave breaks on the beach (0:02.6) and runs up round his feet. He looks down: *hmm.* **SURVIVED** (0:04.8) |
| 0:05.3 | **10 BLOCKS.** The wave rises out at sea and breaks on the beach (0:07.2). Seen from above the roofs, it slams into the first houses (0:08.1) and pours through the streets. 1,797 of the village's 8,022 blocks break, mostly glass, planks, crops and leaves, and the hearts drop to eight. **SURVIVED** (0:10.7) |
| 0:11.5 | **100 BLOCKS.** From the church tower, the horizon rises as the bell rings. The villagers run. From the street, the wall towers over the houses |
| 0:14.8 | Slow motion: the wall curls over the main street and comes down on the houses, and the village bursts into blocks. The hearts run out |
| 0:17.6 | The whole bay is under water, and the flood marches on into the hills. **DESTROYED** (0:18.5) |
| 0:19.7 | **1,000 BLOCKS.** A wall a kilometre high spans the horizon. The music falls away to a drone and a heartbeat under its roar. Its shadow falls over the village, and everyone in the street stares up |
| 0:23.5 | First person, on the beach. The sea has drained back from the sand ahead of the wall. You're holding... a sponge. You put it down (0:24.7). Silence |
| 0:25.2 | The sponge drinks: the sea drains away from it out to the horizon, the wall sinks, and the sponge turns into a wet sponge. The sea floor is left dry |
| 0:28.1 | The fisherman walks up to the wet sponge and looks at it. *Hmm.* **SURVIVED / WITH 1 SPONGE** (0:29.8), then he looks at you |
| 0:31.9 | End (and it loops back to the wall) |

## How it's made

| | |
| --- | --- |
| The coast | `src/world.py`: a 640 x 720 block map of a bay: a sandy beach, a flat meadow for the village, hills, rocky headlands, mountains behind, and the sea floor shelving to 48 blocks deep. Beyond it, a coarse far terrain runs out to 6 km with mountains and (for when the sea is gone) the sea floor's dunes. Meshed with numpy, about 1.1 million vertices |
| The village | `src/village.py`: 8,022 blocks: fourteen houses, a fisherman's hut, a dock on log piles, a smithy, a library with bookshelves, a stone church with a gold bell in its tower, a well, lamp posts, farms, hay bales and trees. `src/characters.py`: the villagers (six professions) and the iron golem as voxel models, one voxel per skin pixel, all pixel art drawn in code |
| The water | `src/swe.py`: a shallow-water solver. Finite volumes with HLL fluxes and the hydrostatic reconstruction of [Audusse et al. 2004](https://doi.org/10.1137/S1064827503431090), which keeps a lake at rest exactly still and handles wet and dry cells. Manning friction, foam carried with the flow, and a wavemaker on the sea edge that sends in each tsunami. The standing houses are part of the sea bed, so the water flows round them, and as they break up it pours through. Cells are 0.5 m, 1 m and 2 m for rounds 1 to 3 |
| The destruction | `src/sim.py`: every block feels the water pushing on its exposed sides (dynamic plus hydrostatic load) and breaks loose when that beats its material's strength, so glass goes first, then crops, planks, logs, cobblestone and stone bricks. Anything left hanging comes down with it. Loose blocks float or sink by density, drift with the current and tumble. The villagers run inland when they see the wave and get swept off their feet. The iron golem stands its ground |
| The 1,000-block wall | `src/wall.py`. At this height the shallow-water equations let the wave slump into a long ramp, and it never reaches the shore anyway, so this wall is drawn rather than simulated. It is a kilometre-high concave face on a 12 m grid with a ragged foot, foam pouring down it, a glowing lip and mist off the crest, coming in at 240 m/s. Ahead of it the sea drains back off the beach, as it does before a real tsunami ([Scripps, via AOL](https://www.aol.com/ominous-empty-harbors-why-water-133103362.html)). The sponge's drink is a dry circle racing outward from the sponge while everything outside it sinks |
| The picture | `src/renderer.py`: a deferred OpenGL renderer with cascaded shadows, SSAO, fog and bloom, and a golden-hour sky with clouds (`src/sky.py`, `src/clouds.py`). The water is drawn from the simulation's surface. It refracts and absorbs by thickness, reflects the sky with Fresnel and the sun with a glint, and the sun shines green through the thin crests. Ripples and foam streaks flow with the current. Each wave casts its shadow, and the 1,000-block wall's shadow and reflection are ray-marched over the water's heights. There are spray puffs, debris blocks and the first-person arm (`src/hand.py`) |
| The edit | `src/edit.py`: fourteen shots, each with its own camera moves and clock (real time, sped up, or slowed to 0.25x for the impacts), directed villagers (the fisherman, the crowd in the street) and camera shake. `src/hud.py`: the round tracker, the hearts, the round titles and the verdicts in a blocky pixel font drawn in code (`src/pixelfont.py`) |
| Sound | `src/audio.py` (toolkit in `src/sound.py`). The music is at 120 bpm and climbs with the rounds: a plucked tune for 1 block, drums and bass for 10, the church bell with strings and toms for 100, then a drone and a heartbeat under the 1,000-block wall's roar. It stops dead for the sponge and returns as a little victory tune. The effects are the sea, every wave's roar and crash, water rushing through the streets, and a crack or crunch for every block the simulation breaks (slowed down with the picture), plus the villager's *hmm*, the sponge going down and the sea being drunk |

## Render it yourself

```bash
pip install -r requirements.txt
cd src
python sim.py 1 2 3                                   # the three simulated rounds (2 to 5 minutes each)
python main.py --out ../preview --preview --sheet     # a contact sheet of every shot
python main.py --out ../output                        # render (1080x1920, 60 fps), sound, encode
python qa.py ../output                                # checks and a contact sheet
python thumbnail.py ../output/thumbnail.jpg           # the cover; --no-title for a clean one
```

The full render takes a few hours on 4 CPU cores with a software OpenGL driver. `--from`/`--to` with `--segments`
splits it across processes, and `--concat` joins the pieces.

## Sources

* Minecraft's sponge soaks up the water around it (up to 65 blocks of it) and becomes a wet sponge:
  [Minecraft.net, Block of the Week: Sponge](https://www.minecraft.net/en-us/article/block-week-sponge).
* The sea drawing back before a tsunami arrives:
  [Ominous empty harbors: why water recedes right before a tsunami](https://www.aol.com/ominous-empty-harbors-why-water-133103362.html).
* The tallest wave run-up on record, 524 m at Lituya Bay in 1958:
  [Earth Magazine](https://www.earthmagazine.org/article/benchmarks-july-9-1958-megatsunami-drowns-lituya-bay-alaska/).
* The solver's well-balanced scheme: E. Audusse, F. Bouchut, M.-O. Bristeau, R. Klein, B. Perthame, *A fast and
  stable well-balanced scheme with hydrostatic reconstruction for shallow water flows*, SIAM J. Sci. Comput. 25(6),
  2004, [doi:10.1137/S1064827503431090](https://doi.org/10.1137/S1064827503431090).
