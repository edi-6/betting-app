# Giant Creeper vs TNT (YouTube Shorts)

The follow-up to [`../steve-vs-swords`](../steve-vs-swords): a 26-block-tall voxel Creeper is hit by
**1 → 10 → 100 → 1,000 → 10,000 primed TNT**. The TNT flies in flashing, blows craters out of him and the
ground, and every round ends with a red **X** (he survived) or a green **check** (he's destroyed).

**Ready to upload:** [`release/creeper_vs_10000_tnt.mp4`](release/creeper_vs_10000_tnt.mp4), with an
optional cover image in [`release/thumbnail.jpg`](release/thumbnail.jpg).

| | |
| --- | --- |
| Length | 28.7 s (2 s teaser hook, then five rounds) |
| Video | 1080x1920 (9:16), 30 fps, H.264 High, 12 Mbps 2-pass, closed GOP (15), BT.709 |
| Audio | AAC-LC (384 kbps target), 48 kHz stereo, loudness-normalised to -14 LUFS, peak -1.2 dBFS |

As with the Steve video, everything is generated from code (no stock footage, samples or AI-generated
media). The only third-party asset is the **Montserrat** font (SIL Open Font License, see
`assets/fonts/OFL.txt`).

## Timeline

| Time | Segment | Result |
| --- | --- | --- |
| 0:00 | Hook: three quick teaser shots of the incoming TNT | |
| 0:02 | **1 TNT**: blows a hole in his face | X, loses half a heart |
| 0:05.6 | **10 TNT**: two rows into his face | X, 8.5 hearts left |
| 0:09.5 | **100 TNT**: a 10x10 wall into his chest and face | X, 5.5 hearts left |
| 0:14.1 | **1,000 TNT**: a 10x10x10 cube into his upper side | X, 1.5 hearts left |
| 0:20.3 | **10,000 TNT**: a 25x25x16 block, total destruction and a crater | check, dead |

The counts are exact: every round spawns exactly the number of TNT blocks its label says.

## What's new compared to the Steve video

| Module | What it does |
| --- | --- |
| `src/creeper.py` | The voxel Creeper (81,920 voxels): mottled green skin in the classic style, red flesh inside and a grey gunpowder core |
| `src/tnt.py` | TNT block textures drawn in code (TNT label band, stick ends with a fuse, bottom) |
| `src/sim.py` | Explosion physics: TNT flies in with a flashing fuse, detonates on contact (or on its fuse after flying past), removes a ragged sphere of voxels (part of it pulverised, the rest thrown out of the crater), pushes nearby TNT, craters the ground and throws dirt clods; loose parts break off and shatter |
| `src/ground.py` | Destructible ground: a column height field of grass/dirt/stone blocks that explosions carve craters into |
| `src/vfx.py` | Fireball/smoke puffs, bright blast flashes and short-lived point lights; mass detonations are merged so 10,000 explosions stay renderable |
| `src/renderer.py` | Adds instanced flashing TNT, explosion point lights, soft depth-faded smoke, additive flashes and bloom |
| `src/audio.py` | TNT sound design: fuse hiss, layered explosions with distant variants and a rolling-thunder bed for mass detonations, flesh/gunpowder spray, dirt clods, trailer hits on the hook cuts, bus compression |

Shared with the Steve project: the world and trees (`world.py`), volumetric sky (`clouds.py`, `sky.py`),
HUD (`hud.py`), deferred renderer core, and the YouTube-spec encode in `main.py`. The dev helpers
`check_cams.py` (camera clearance, including never putting the lens inside a flying formation),
`qa.py` (black/corrupt frames and flicker, ignoring explosion flashes), `cam_test.py` and `test_round.py`
are here too.

## Render it yourself

```bash
pip install -r requirements.txt
# Linux also needs Mesa's EGL + llvmpipe, e.g.: apt install libegl1 libegl-mesa0 libgl1-mesa-dri
cd src
python main.py --out ../output             # delivered quality (~25 min on 4 CPU cores)
python main.py --out ../preview --preview  # quick half-resolution check (~8 min)
python qa.py ../output/final.mp4 ../output/cues.json
python thumbnail.py ../output/thumbnail.png
```

The first run builds and caches the world mesh and the sky (a couple of minutes). Set `SVS_CACHE=/path`
to put that cache somewhere else (it can be shared with the Steve project).

## Customising

* Counts, formations, damage and timing: `src/timeline.py` (`round_defs()` and `hook_shots()`).
* Explosion size, crater size, how much is pulverised: `RoundSim` arguments in `src/sim.py`.
* The Creeper's colours and face: `src/creeper.py` (`make_creeper_skin`).

## Upload tips

* Upload the MP4 as-is. It already meets YouTube's recommended settings for 1080p, and at 28.7 s and 9:16
  it is automatically a Short.
* Title idea: **What happens if you hit a Creeper with 10,000 TNT? 😳 #minecraft #shorts**
