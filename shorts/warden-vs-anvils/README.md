# Giant Warden vs 10,000 Anvils (YouTube Shorts)

The fourth video in the series, after [`../steve-vs-swords`](../steve-vs-swords),
[`../creeper-vs-tnt`](../creeper-vs-tnt) and [`../zombie-vs-arrows`](../zombie-vs-arrows): a 30-block-tall
voxel Warden is hit by **1 → 10 → 100 → 1,000 → 10,000 falling anvils**. The anvils crush dents into him, stack
up on his head and pour off into piles. His chest and tendrils glow with his heartbeat, which races as the
danger grows. In the last round he fights back with a sonic boom that blasts a hole through the storm, but
there are too many: they bury him under a pyramid of anvils and his heart stops.

**Ready to upload:** [`release/warden_vs_10000_anvils.mp4`](release/warden_vs_10000_anvils.mp4), with an
optional cover image in [`release/thumbnail.jpg`](release/thumbnail.jpg).

| | |
| --- | --- |
| Length | 27.5 s (2 s teaser hook, then five rounds) |
| Video | 1080x1920 (9:16), 30 fps, H.264 High, 12 Mbps 2-pass, closed GOP (15), BT.709 |
| Audio | AAC-LC (384 kbps target), 48 kHz stereo, loudness-normalised to -14 LUFS, peak -1.2 dBFS |

Like the others, everything is generated from code (no stock footage, samples or AI-generated media). The
only third-party asset is the **Montserrat** font (SIL Open Font License, see `assets/fonts/OFL.txt`).

## Timeline

| Time | Segment | Result |
| --- | --- | --- |
| 0:00 | Hook: 10,000 anvils hanging over him, an anvil-cam dive at his head, his sonic boom tearing into the cloud | |
| 0:02 | **1 Anvil**: anvil-cam in slow motion, it crushes into the top of his head | X, loses half a heart |
| 0:06 | **10 Anvils**: a row hanging over his head drops one after another | X, 8.5 hearts left |
| 0:10 | **100 Anvils**: a 10 x 10 slab lands flat on his head | X, 6 hearts left |
| 0:13.5 | **1,000 Anvils**: a 10 x 10 x 10 cube crushes his head and pours down into a pile | X, 1.5 hearts left |
| 0:19 | **10,000 Anvils**: he fires a sonic boom through the falling cloud, but the storm buries him under a pyramid of anvils and his heart stops | check, dead |

The counts are exact: every round drops exactly the number of anvils its label says.

## What's different from the earlier videos

| | |
| --- | --- |
| Look | Golden-hour lighting (low warm sun, violet-to-orange sky, warm-lit clouds, teal/orange grade) and a patch of glowing sculk around his feet |
| The Warden | `src/warden.py`: own pixel art in the Warden's style (271,596 voxels), with a ribcage full of glowing souls and two glowing tendrils; teal flesh with glowing soul specks and pale bones inside |
| Anvils | `src/anvil.py`: the classic anvil shape from four boxes, with iron textures in intact, chipped and damaged states; they fall straight down on a grid like the game's falling blocks |
| Physics | `src/sim.py`: anvils land on whatever is highest in their column (ground, anvils, the Warden, fallen pieces); on flesh they crush a dent that grows with impact speed; on steep stacks they tumble down to where they come to rest, so volleys pour into mounds; heavy stacks slowly crush their way down; a sonic boom blasts falling anvils out of the sky |
| Heartbeat | `src/main.py`: his glow pulses with his heartbeat, it speeds up with the danger, slows and stops when he dies, and the light goes out |
| Cameras | An anvil-cam that rides down with the anvil in round 1, brakes short of the hit and swings round to show it lodged in his head |
| Sound | `src/audio.py`: iron clunks with an inharmonic ring, crushes, thuds, avalanche rattles, falling whistles, the roar of the storm, his heartbeat and growl, the charge and blast of the sonic boom, the pile creaking as it settles |

Shared with the earlier projects: the world and trees (`world.py`), volumetric sky (`clouds.py`, `sky.py`),
HUD (`hud.py`), deferred renderer core and the YouTube-spec encode in `main.py`. Dev helpers: `check_cams.py`
(tree/terrain clearance, how close falling anvils get to every fixed camera and that the pile never swallows
the lens, measured with the real simulation), `qa.py` (black/corrupt frames, flicker), `test_round.py` (contact
sheets of any moment, through the real pipeline) and `thumbnail.py`.

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

The first run builds and caches the world mesh and the golden-hour sky (a couple of minutes). Set
`SVS_CACHE=/path` to put that cache somewhere else (the world mesh can be shared with the other projects).

## Customising

* Counts, formations, slow motion, cameras and when the sonic boom fires: `src/timeline.py` (`round_defs()`
  and `hook_shots()`).
* How deep anvils crush, how steep piles get, how fast stacks sink: `RoundSim` arguments in `src/sim.py`
  (`crush_k`, `crush_max`, `slide_steep`, `weight_crush`).
* The Warden's colours and face: `src/warden.py` (`PAL`, `make_warden_skin`); the time of day: `src/sky.py`.

## Upload tips

* Upload the MP4 as-is. It already meets YouTube's recommended settings for 1080p, and at 27.5 s and 9:16
  it is automatically a Short.
* Title idea: **What happens if you drop 10,000 anvils on a Warden? 😳 #minecraft #shorts**
