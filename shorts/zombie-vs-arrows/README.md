# Giant Zombie vs 10,000 Arrows (YouTube Shorts)

The third video in the series, after [`../steve-vs-swords`](../steve-vs-swords) and
[`../creeper-vs-tnt`](../creeper-vs-tnt): a 32-block-tall voxel Zombie is hit by
**1 → 10 → 100 → 1,000 → 10,000 arrows**. Arrows fly real ballistic arcs, bury themselves in him and stay
stuck, so he turns into a pincushion. Every round ends with a red **X** (he survived) or a green **check**
(he's destroyed).

**Ready to upload:** [`release/zombie_vs_10000_arrows.mp4`](release/zombie_vs_10000_arrows.mp4), with an
optional cover image in [`release/thumbnail.jpg`](release/thumbnail.jpg).

| | |
| --- | --- |
| Length | 27.1 s (2 s teaser hook, then five rounds) |
| Video | 1080x1920 (9:16), 30 fps, H.264 High, 12 Mbps 2-pass, closed GOP (15), BT.709 |
| Audio | AAC-LC (384 kbps target), 48 kHz stereo, loudness-normalised to -14 LUFS, peak -1.2 dBFS |

Like the other two, everything is generated from code (no stock footage, samples or AI-generated media).
The only third-party asset is the **Montserrat** font (SIL Open Font License, see `assets/fonts/OFL.txt`).

## Timeline

| Time | Segment | Result |
| --- | --- | --- |
| 0:00 | Hook: the sky full of arrows, an arrow-cam dive at his head, a thousand arrows about to hit | |
| 0:02 | **1 Arrow**: arrow-cam in slow motion, a headshot between the eyes | X, loses half a heart |
| 0:06 | **10 Arrows**: a hovering stack fires one after another (head, arms, chest, legs) | X, 8.5 hearts left |
| 0:10 | **100 Arrows**: a side volley turns his right side into a pincushion | X, 5.5 hearts left |
| 0:14 | **1,000 Arrows**: a rain of arrows from the sky covers his whole front | X, 1.5 hearts left |
| 0:19 | **10,000 Arrows**: ten waves; the first lands in slow motion, then his arms and head are shot off and he is torn apart | check, dead |

The counts are exact: every round fires exactly the number of arrows its label says.

## What's new compared to the Creeper video

| Module | What it does |
| --- | --- |
| `src/zombie.py` | The voxel Zombie (106,496 voxels) with his arms stretched out in front: own pixel-art skin in the classic style (green skin, torn cyan shirt, blue trousers), dark red flesh and bones inside |
| `src/arrows.py` | The arrow sprite drawn in code and rendered game-style as two crossed quads, with a hand-built mip chain so thin shafts never fade out at a distance |
| `src/sim.py` | Arrow physics: exact ballistic flight, each arrow carves a thin channel and loses speed per voxel (bone costs more) until it stops and sticks, quivers after impact, falls out if the flesh around its tip is destroyed, rides along on pieces that break off. Misses sink into the ground at their impact angle. Scheduled "severs" shoot limbs and the head off as rigid pieces that tumble, settle flat and keep their arrows. Variable time step for slow motion |
| `src/timeline.py` | Rounds, formations (single arrow, a hovering stack, side volley, rain, ten waves timed by first contact), slow-motion schedules and cameras |
| `src/main.py` | Arrow-cam that rides an arrow, brakes short of the impact and swings round; Minecraft-style red damage flash; screen-door fade for arrows passing the lens |
| `src/renderer.py` | Instanced alpha-tested arrows with their own mip selection, motion streaks, damage tint |
| `src/audio.py` | Arrow sound design: bowstring twangs and distant volley releases, fly-by whistles, the roar of the swarm, flesh thunks, bone cracks, wet spray, ground thuds, hail beds for mass impacts, severed parts landing, slow-motion pitch drop with reverb and heartbeat, arrow-cam wind rush |

Shared with the earlier projects: the world and trees (`world.py`), volumetric sky (`clouds.py`, `sky.py`),
HUD (`hud.py`), deferred renderer core and the YouTube-spec encode in `main.py`. Dev helpers:
`check_cams.py` (tree/terrain clearance and how close the swarm gets to every fixed camera, measured with
the real simulation), `qa.py` (black/corrupt frames, flicker), `test_round.py` (contact sheets of any
moment, through the real pipeline) and `thumbnail.py`.

## Render it yourself

```bash
pip install -r requirements.txt
# Linux also needs Mesa's EGL + llvmpipe, e.g.: apt install libegl1 libegl-mesa0 libgl1-mesa-dri
cd src
python main.py --out ../output             # delivered quality (~9 min on 4 CPU cores)
python main.py --out ../preview --preview  # quick half-resolution check (~4 min)
python qa.py ../output/final.mp4 ../output/cues.json
python thumbnail.py ../output/thumbnail.png
```

The first run builds and caches the world mesh and the sky (a couple of minutes). Set `SVS_CACHE=/path`
to put that cache somewhere else (it can be shared with the other two projects).

## Customising

* Counts, formations, slow motion, cameras and when parts get shot off: `src/timeline.py`
  (`round_defs()` and `hook_shots()`).
* How deep arrows go and how much flesh they knock out: `RoundSim` arguments in `src/sim.py`
  (`voxel_cost`, `bone_cost`, `carve_radius`, `wound_radius`).
* The Zombie's colours and face: `src/zombie.py` (`PAL`, `make_zombie_skin`).

## Upload tips

* Upload the MP4 as-is. It already meets YouTube's recommended settings for 1080p, and at 27.1 s and 9:16
  it is automatically a Short.
* Title idea: **What happens if you shoot a Zombie with 10,000 arrows? 😳 #minecraft #shorts**
