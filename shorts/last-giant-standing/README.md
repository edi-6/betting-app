# Last Giant Standing (YouTube Shorts)

The fifth video in the series, after [`../steve-vs-swords`](../steve-vs-swords),
[`../creeper-vs-tnt`](../creeper-vs-tnt), [`../zombie-vs-arrows`](../zombie-vs-arrows) and
[`../warden-vs-anvils`](../warden-vs-anvils). This time all four giants from the earlier videos stand side by side
in one arena and fight to be the last one standing. Three rounds of escalating hazards (**1,000 arrows, 1,000
anvils, 10,000 TNT**) knock them out one by one, and damage carries over from round to round. A scoreboard tracks
every giant's hearts, so viewers pick a favourite in the first seconds and stay to see if it wins.

**Ready to upload:** [`release/last_giant_standing.mp4`](release/last_giant_standing.mp4), with an optional cover
image in [`release/thumbnail.jpg`](release/thumbnail.jpg).

| | |
| --- | --- |
| Length | 28.7 s (2 s flash-forward cold open, the line-up, three rounds, the winner) |
| Video | 1080x1920 (9:16), 30 fps, H.264 High, 12 Mbps 2-pass, closed GOP (15), BT.709 |
| Audio | AAC-LC (384 kbps target), 48 kHz stereo, loudness-normalised to -14 LUFS, peak -1.2 dBFS |

Like the others, everything is generated from code (no stock footage, samples or AI-generated media). The
only third-party asset is the **Montserrat** font (SIL Open Font License, see `assets/fonts/OFL.txt`).

## Timeline

| Time | Segment | What happens |
| --- | --- | --- |
| 0:00 | Cold open | Flash-forward to the finale: the Warden's sonic boom punches a hole through a wall of 10,000 TNT, then the TNT rains down on Steve. No text, so the hook can go on top |
| 0:02 | Line-up | Zombie, Creeper, Steve and Warden, each with a name slam; the scoreboard fills row by row |
| 0:04.7 | **Round 1: 1,000 arrows** | A volley loosed from the forest slams into all four. The Creeper starts to hiss and flash, then blows itself up, taking chunks out of the Zombie and Steve. **Creeper eliminated** |
| 0:12.1 | **Round 2: 1,000 anvils** | Three 10 x 10 blocks of anvils drop on the three who are left. They tear the Zombie's arms off, his knees give way and he topples face first into the pile. **Zombie eliminated** |
| 0:17.8 | **Round 3: 10,000 TNT** | A 50 x 10 x 20 wall of TNT hangs over the last two. The Warden fires two sonic booms that blast a hole through it and fling the TNT into the sky, where it goes off like fireworks; the rest buries Steve. **Steve eliminated** |
| 0:25.6 | Winner | The Warden stands alone on half a heart, missing his arms: **WINNER** |

The counts are exact: 1,000 arrows, 1,000 anvils (500 + 300 + 200) and 10,000 TNT, every one simulated.

## What's new in this one

| | |
| --- | --- |
| One arena | `src/arena.py`: one continuous simulation of all four giants and all three hazards (arrows from the Zombie video, anvils from the Warden video, TNT and craters from the Creeper video), generalised to several giants at their own positions. Damage carries over: arrows stay stuck, dents stay dented, the Creeper's blast takes bites out of its neighbours |
| The giants | `src/models.py`: one generic voxel giant (the four characters and skins from the earlier videos), with muted interiors (raw-beef Steve, rotten-olive Zombie, leafy Creeper) instead of the bright red of the earlier videos |
| Physics additions | A giant can topple over its stump like a felled tree (the Zombie); falling pieces knock anvils out of their way; anything resting on something that gets blown away falls again; the Creeper primes (flashing white and swelling) and explodes; sonic booms fling TNT skywards |
| The story | `src/main.py`: an analysis pass runs the whole battle first; hearts are derived from how much of each giant is destroyed, and the eliminations land on the right moments (the Creeper's blast, the Zombie hitting the ground, the last of Steve) |
| Scoreboard HUD | `src/hud.py`: portraits, names and 10 hearts per giant (they flash when hit and jiggle when low), round banners, fighter introductions, ELIMINATED stamps (portrait crossed out, name struck through) and a WINNER card with a crown |
| Edit | `src/timeline.py`: 17 shots with slow motion on the key moments, jump cuts over the quiet parts, and a flash-forward cold open made of the very same frames of the battle filmed from other angles |
| Sound | `src/audio.py`: everything from the earlier videos (bows, arrow thunks, anvils, TNT, explosions, heartbeat, sonic boom) plus trailer hits on the introductions and banners, an elimination stinger, a rewind out of the cold open, a dark drone that grows round by round and a victory chord |

Shared with the earlier projects: the world and trees (`world.py`, here with a wider clearing), volumetric
golden-hour sky (`clouds.py`, `sky.py`) and the deferred renderer (`renderer.py`, extended for several giants,
TNT and anvils together, arrows and the creeper flash). Dev helpers: `check_cams.py` (every camera clear of the
trees, and of falling anvils and TNT, measured with the real simulation), `qa.py` (black/corrupt frames,
flicker) and `thumbnail.py`.

## Render it yourself

```bash
pip install -r requirements.txt
# Linux also needs Mesa's EGL + llvmpipe, e.g.: apt install libegl1 libegl-mesa0 libgl1-mesa-dri
cd src
python main.py --out ../output             # delivered quality
python main.py --out ../preview --preview  # quick half-resolution check
python main.py --out ../stills --preview --stills --every 10 --shots R3_boom1,W_hero   # look development
python qa.py ../output/final.mp4 ../output/cues.json
python thumbnail.py ../output/thumbnail.png
```

The first run builds and caches the world mesh and the golden-hour sky (a couple of minutes). Set
`SVS_CACHE=/path` to put that cache somewhere else.

## Customising

* The battle (formations, counts, when the Creeper goes off, the severs and the Zombie's topple, the sonic booms):
  `plan()` in `src/timeline.py`.
* The edit (shots, cameras, slow motion, the cold open): `shots()` and `cold_open()` in `src/timeline.py`.
* How much damage each giant can take before it is out: `LETHAL` in `src/main.py`.
* The lineup and the giants: `LINEUP` and `SPECS` in `src/models.py`.

## Upload tips

* Upload the MP4 as-is. It already meets YouTube's recommended settings for 1080p, and at 28.7 s and 9:16
  it is automatically a Short.
* Title idea: **4 Minecraft Giants vs 12,000 Hazards… Only 1 Survives 😳 #minecraft #shorts**
