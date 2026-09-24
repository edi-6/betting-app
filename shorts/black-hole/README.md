# A Black Hole Spawns in Minecraft (YouTube Shorts)

The sixth video in the series, after [`../steve-vs-swords`](../steve-vs-swords),
[`../creeper-vs-tnt`](../creeper-vs-tnt), [`../zombie-vs-arrows`](../zombie-vs-arrows),
[`../warden-vs-anvils`](../warden-vs-anvils) and [`../last-giant-standing`](../last-giant-standing). A tiny black
hole pops into existence in the arena and grows in four jumps: **SIZE 1, 10, 100, 1,000**. It rips the ground out
block by block, drags everything lying around into a glowing accretion disk, and eats the four giants one by one:
they lean into the pull, are torn off the ground, stretched like noodles (spaghettification) and swallowed. The
Creeper blows up at the edge of the hole, the Warden's sonic boom gets eaten, and a counter keeps score of every
block that goes in. Then it collapses into a point, and blows everything back out.

**Ready to upload:** [`release/black_hole.mp4`](release/black_hole.mp4), with an optional cover image in
[`release/thumbnail.jpg`](release/thumbnail.jpg).

| | |
| --- | --- |
| Length | 32.7 s (1.4 s flash-forward cold open, then SIZE 1, 10, 100, 1,000, the collapse and the explosion) |
| Video | 1080x1920 (9:16), 30 fps, H.264 High, 12 Mbps 2-pass, closed GOP (15), BT.709 |
| Audio | AAC-LC (384 kbps target), 48 kHz stereo, loudness-normalised to -14 LUFS, peak -1.2 dBFS |

Like the others, everything is generated from code (no stock footage, samples or AI-generated media). The only
third-party asset is the **Montserrat** font (SIL Open Font License, see `assets/fonts/OFL.txt`).

## Timeline

| Time | Segment | What happens |
| --- | --- | --- |
| 0:00 | Cold open | Flash-forward: Steve, torn off the ground, being dragged into the black hole. No text, so the hook can go on top |
| 0:01.4 | The arena | The four giants (Zombie, Creeper, Steve, Warden) on a calm afternoon |
| 0:02.4 | **SIZE 1** | A tiny black hole pops into existence in front of the Creeper and starts nibbling at the grass; the BLOCKS EATEN counter starts |
| 0:04.7 | **SIZE 10** | It grows: the ground rips open around it, the anvils, TNT and arrows lying around fly in (the TNT goes off in the disk) |
| 0:05.7 | The Creeper | Leans into the pull, peels, is torn off the ground, primes on the way in and blows up at the edge of the hole (slow motion). The blast is sucked back in. **Creeper eaten** |
| 0:10.4 | **SIZE 100** | A burst of blocks, the disk and the lensed ring appear |
| 0:11.2 | The Zombie | Its arms are torn off first (slow motion), then it is ripped off the ground and spirals in. **Zombie eaten** |
| 0:15.6 | Steve | Torn off the ground and spaghettified: drawn out into a noodle as it is pulled in (slow motion). **Steve eaten** |
| 0:20.1 | **SIZE 1,000** | The whole arena floor comes up; the counter races past 20,000 |
| 0:21.2 | The Warden | Fires a sonic boom at it; the black hole swallows the sonic boom. Then the Warden is dragged in too. **Warden eaten** |
| 0:26.1 | Collapse | With nothing left, it collapses into a point; the world goes dark and silent |
| 0:27.5 | The explosion | A white flash, shock waves, and everything it ate is blasted back out and rains down on the crater (49,865 blocks eaten) |
| 0:31.9 | ...or did it? | A tiny new black hole pops up in the crater: SIZE 1. The video loops back to the start |

## What's new in this one

| | |
| --- | --- |
| The black hole | `src/renderer.py`: a gravitational lens pass bends the finished frame around the hole (point-mass lens, black shadow, photon ring, the far side of the scene wrapped around it), and an accretion disk drawn in two halves: the half behind the hole is lensed into the arc over the top, the half in front stays in front. The sky and the sunlight dim as the hole grows; leaves bend towards it |
| The pull | `src/blackhole.py`: everything loose follows a velocity field around the hole: plain attraction far away, a Keplerian spiral in the (tilted) disk plane close in, heating up (red, orange) as it nears the horizon, gone once it crosses it |
| The ground | `src/ground.py`: the arena floor is a height field of real blocks that the hole peels off one at a time (grass, dirt, then stone) into a stepped funnel; every block torn out flies into the disk as a textured, tumbling block |
| The giants | Each giant is a rigid body over its own voxels with a pose the shader applies: it leans into the pull, is torn off the ground, spirals in on a scripted path, and is spaghettified: the half facing the hole is drawn out towards it and thinned, shed voxels stream ahead of it, and whatever reaches the horizon is gone. The Zombie's arms go first; the Creeper primes on the way in and goes off at the edge; the Warden fires a sonic boom at it |
| The story | `src/main.py`: an analysis pass runs the whole story first, so the BLOCKS EATEN counter, the SIZE banners and the EATEN stamps land on the right frames |
| HUD | `src/hud.py`: the counter with the four portraits (sucked away and crossed out when eaten), SIZE banners that slam in with a ring, EATEN stamps, vignette and flashes |
| Edit | `src/timeline.py`: 18 shots pinned to exact stretches of the story, slow motion on the big moments, cameras placed on the line each giant is pulled along (the giant above the hole in the tall frame), and a flash-forward cold open |
| Sound | `src/audio.py`: the hole's voice (a thin whine growing into a sub-bass roar), rushing air, blocks tearing, the hot disk sizzling, groans, rips, a rubbery stretch and a deep gulp for every giant, the creeper's hiss and blast, the sonic boom, the implosion, dead silence and the explosion; the score is a cathedral organ (one chord per size) over a ticking clock |

Shared with the earlier projects: the world and trees (`world.py`), the golden-hour sky (`clouds.py`, `sky.py`),
the characters (`models.py`) and the deferred renderer (`renderer.py`). Dev helpers: `check_cams.py` (every camera
clear of the trees and of the flying giants and blocks, measured with the real simulation), `qa.py`
(black/corrupt frames, flicker) and `thumbnail.py`.

## Render it yourself

```bash
pip install -r requirements.txt
# Linux also needs Mesa's EGL + llvmpipe, e.g.: apt install libegl1 libegl-mesa0 libgl1-mesa-dri
cd src
python main.py --out ../output             # delivered quality
python main.py --out ../preview --preview  # quick half-resolution check
python main.py --out ../stills --preview --stills --every 10 --fast --shots steve_spag,warden_in   # look development
python check_cams.py
python qa.py ../output/final.mp4 ../output/cues.json
python thumbnail.py ../output/thumbnail.png
```

The first run builds and caches the world mesh and the golden-hour sky (a couple of minutes). Set
`SVS_CACHE=/path` to put that cache somewhere else.

## Customising

* The story (when the hole appears and grows, how big, when each giant goes, the Creeper, the sonic boom, the
  collapse): `plan()` and the constants at the top of `src/timeline.py`.
* The edit (shots, cameras, slow motion, the cold open): `shots()` and `cold_open()` in `src/timeline.py`.
* The look of the hole (lens strength, disk size and brightness, how dark the sky gets): `bh_params()` and `mood()`
  in `src/main.py`.

## Upload tips

* Upload the MP4 as-is. It already meets YouTube's recommended settings for 1080p, and at under a minute and 9:16
  it is automatically a Short.
