# Giant Steve vs Swords (YouTube Shorts)

A fully procedural 9:16 Short in the style of the "giant blocky guy vs N weapons" physics videos. A
32-block-tall voxel giant is hit by **1 → 10 → 100 → 1,000 → 10,000 swords**. Each round shows a
10-heart health bar and a count label, then ends with a red **X** (he survived) or a green **check**
(he's destroyed).

**Ready to upload:** [`release/steve_vs_10000_swords.mp4`](release/steve_vs_10000_swords.mp4), with an
optional cover image in [`release/thumbnail.jpg`](release/thumbnail.jpg).

| | |
| --- | --- |
| Length | 29.0 s (2 s teaser hook, then five rounds) |
| Video | 1080x1920 (9:16), 30 fps, H.264 High, 12 Mbps 2-pass, closed GOP (15), BT.709 |
| Audio | AAC-LC (384 kbps target), 48 kHz stereo, loudness-normalised to -14 LUFS, peak -1.2 dBFS |

Everything is generated from code: textures, pixel art, sky and clouds, physics, sound effects and HUD. There
is no stock footage, no samples and no AI-generated media. The only third-party asset is the **Montserrat**
font (SIL Open Font License, see `assets/fonts/OFL.txt`).

## Timeline

| Time | Segment | Result |
| --- | --- | --- |
| 0:00 | Hook: three quick teaser shots of the incoming sword walls | |
| 0:02 | **1 Sword**: a diamond sword sticks in his chest | X, loses half a heart |
| 0:05 | **10 Swords** | X, 8.5 hearts left |
| 0:09 | **100 Swords**: tears through the torso | X, 5.5 hearts left |
| 0:14 | **1,000 Swords**: the whole side is shredded | X, 1.5 hearts left |
| 0:20 | **10,000 Swords**: ten waves of 1,000, total collapse | check, dead |

The counts are exact: every round spawns exactly the number of swords its label says.

## How it works

| Module | What it does |
| --- | --- |
| `src/textures.py` | Pixel art drawn in code: the Steve-style skin, six sword tiers, grass/dirt/stone/log/leaf blocks, hearts, X and check icons |
| `src/steve.py` | The voxel giant: every skin pixel is 4x4x4 voxels (106,496 in total). Outer faces carry the skin; cut faces show red flesh or bone |
| `src/world.py` | Grassy basin with terraced dirt/stone cliffs and ~2,200 blocky oak trees (chunked meshes) |
| `src/clouds.py`, `src/sky.py` | One-time bake of a volumetric cumulus sky panorama (Perlin-Worley noise, raymarched in GLSL) |
| `src/renderer.py` | Deferred OpenGL renderer that runs headless on the CPU (Mesa llvmpipe): soft sun shadows (PCSS), SSAO, sky light, metal sword reflections, aerial haze, ACES tonemapping, FXAA, supersampling |
| `src/sim.py` | Vectorised physics: swords carve voxels and lose energy, wounds crumble, debris sprays and piles up (with granular sliding), swords embed, fall, or land point-first, and disconnected parts fall as rigid chunks and shatter |
| `src/timeline.py` | The rounds (formations, camera moves, stamp timing) and the opening hook |
| `src/hud.py` | Hearts (half hearts, damage flash, low-health jiggle), count label, X / check stamps |
| `src/audio.py` | Synthesised sound design driven by simulation events: whooshes, impacts, crunches, debris patter, sword clatter, bone cracks, collapse boom, UI pops and stingers, wind bed |
| `src/main.py` | Runs everything and does the YouTube-spec encode |
| `src/thumbnail.py` | Renders `release/thumbnail.jpg` |
| `src/check_cams.py`, `src/qa.py`, `src/test_round.py` | Dev helpers: camera/tree clearance check, automated frame QA (black/corrupt frames, flicker), quick contact sheet of one round |

## Render it yourself

```bash
pip install -r requirements.txt
# Linux also needs Mesa's EGL + llvmpipe, e.g.: apt install libegl1 libegl-mesa0 libgl1-mesa-dri
cd src
python main.py --out ../output             # delivered quality (~30 min on 4 CPU cores)
python main.py --out ../preview --preview  # quick half-resolution check (~10 min)
python qa.py ../output/final.mp4 ../output/cues.json
```

The first run builds and caches the world mesh and the sky (a couple of minutes). Set `SVS_CACHE=/path`
to put that cache somewhere else. `python main.py --out ../output --encode-only` rebuilds only the sound
and the final encode from an existing render.

## Customising

* Counts, formations, damage and timing: `src/timeline.py` (`round_defs()` and `hook_shots()`).
* The character's colours: `src/textures.py` (`make_skin`).
* The look (exposure, saturation, fog): `Renderer.__init__` in `src/renderer.py`.

## Upload tips

* Upload the MP4 as-is. It already meets YouTube's recommended settings for 1080p, and at 29 s and 9:16 it
  is automatically a Short.
* The first frame already shows the giant and the incoming swords. If your YouTube app lets you upload a
  custom Shorts thumbnail, use `release/thumbnail.jpg` (it also works as a cover image on other platforms).
