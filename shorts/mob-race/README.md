# Minecraft Mob Marble Race: Only ONE Survives the Lava (YouTube Shorts)

The eighth video in the series, after [`../steve-vs-swords`](../steve-vs-swords),
[`../creeper-vs-tnt`](../creeper-vs-tnt), [`../zombie-vs-arrows`](../zombie-vs-arrows),
[`../warden-vs-anvils`](../warden-vs-anvils), [`../last-giant-standing`](../last-giant-standing),
[`../black-hole`](../black-hole) and [`../hydraulic-press`](../hydraulic-press). This one is a new format: a
marble race. Eight Minecraft mobs, each a marble with its face on it, race down a marble run built on the face of
a cliff. The run has four rounds. Each one ends on a trapdoor over lava, and once enough marbles are across, the
trapdoor drops the last ones in. One marble is left at the end.

| Round | The course | Who's out |
| --- | --- | --- |
| 1. PLINKO | From the starting stalls down through a field of end rods | The trapdoor drops the last two. The Warden gets over it **0.07 s** after it starts to open (**SAVED!**), so only the Pig goes in |
| 2. SLIME & ICE | Zigzag down ice with slime walls to bounce off | Blaze and Villager; the Skeleton **just made it** |
| 3. PISTONS & TNT | Stone ramps, pistons punching out of the walls, TNT that a marble lights and that blows up | Creeper and Fox; the Warden is the **last one across** |
| THE FINAL | Warden, Enderman and Skeleton. The first across the golden trapdoor wins | Enderman and Skeleton. **The Warden wins** |

The race is a real physics simulation, not animation. Nothing is scripted except the choice of which race to
show. It is a comeback story: the Warden nearly goes out in round 1, scrapes through round 3 in last place, and
takes the final.

**Ready to upload:** [`release/mob_race.mp4`](release/mob_race.mp4), with an optional cover image in
[`release/thumbnail.jpg`](release/thumbnail.jpg) (the Pig dropping into the lava as the Warden just makes it,
titled *ONLY ONE SURVIVES*), or the same without the title in
[`release/thumbnail_clean.jpg`](release/thumbnail_clean.jpg) for your own text.

| | |
| --- | --- |
| Length | 39.6 s (a 2.5 s countdown, then the race; the winner's celebration from 0:33.9) |
| Video | 1080x1920 (9:16), 30 fps, H.264 High, 12 Mbps 2-pass, closed GOP (15), BT.709 |
| Audio | AAC-LC (384 kbps target), 48 kHz stereo, loudness-normalised to -14 LUFS, peak -1.2 dBFS |

As before, everything is generated from code: no stock footage, samples or AI-generated media. The only thing that
isn't made here is the font (Montserrat, SIL Open Font License, in `assets/fonts`).

## Timeline

| Time | What happens |
| --- | --- |
| 0:00 | The eight racers in their stalls, the camera panning along them. **3, 2, 1** on the beat, with no other text so a hook can go on top |
| 0:02.5 | **GO!** The stalls' floors drop, the beat drops. Round 1: **PLINKO** |
| 0:08.4 | Slow motion. The first trapdoor drops once six are across (0:09.2). The Warden gets over it as it goes: **SAVED! Warden by 0.07s** (0:09.4) |
| 0:10.8 | The Pig goes into the lava: **ELIMINATED** |
| 0:11.7 | The holding pen opens. Round 2: **SLIME & ICE** |
| 0:17.4 | The second trapdoor drops. The Skeleton **JUST MADE IT!** The Blaze and the Villager go in (0:18.1) |
| 0:19.2 | Round 3: **PISTONS & TNT**. A marble lights the TNT (0:21.2) and it goes off in slow motion (0:21.7), throwing the pack around |
| 0:27.2 | Slow motion. The Warden is the last one over the third trapdoor: **JUST MADE IT!** (0:27.8). The Creeper (0:29.6) and the Fox (0:30.5) go in |
| 0:31.4 | **THE FINAL**: Warden, Enderman and Skeleton. Winner takes all |
| 0:33.0 | Slow motion. The music cuts out to a heartbeat. The Warden crosses first and the golden trapdoor drops: **WINNER!** (0:33.9). Crown, confetti, fireworks, fanfare. The Enderman (0:35.3) and the Skeleton (0:35.8) go in |
| 0:36.5 | Close on the Warden on the golden podium until the end (0:39.6) |

The top of the screen carries a live broadcast graphic the whole way through: the round, how many are left, and
every racer's face in race order. Eliminated racers are greyed out and crossed, and the leader wears the crown. It
keeps viewers watching to see whether their mob makes it.

## How it's made

| | |
| --- | --- |
| The course | `src/course.py`: stalls, plinko pegs, ice ramps with slime walls, stone ramps with pistons and TNT, and at the end of each round a slope with a trapdoor in it over a lava basin, then a holding pen whose floor opens to start the next round with everyone together. In the final the pen is the winner's podium |
| The physics | `src/race.py`: pymunk (Chipmunk2D) in the plane of the course, one unit per block. Each material has its own friction and bounce: slime throws marbles back harder than they hit, ice is nearly frictionless. The trapdoors count who is across, then swing open. Pistons punch, TNT is lit by touch and blows 0.3 s later, and marbles in lava are out |
| Determinism | Every video frame advances a whole number of fixed 1/300 s physics steps, so the edit's slow motion and fast-forward never change the race. `src/timeline.py` holds the chosen race (seed and line-up) and the edit (speed keys) |
| Picking the race | `src/search.py` runs hundreds of races and keeps the ones with exactly one survivor, every trapdoor used, no jams, the right length and close calls. The race in the video is seed 111 |
| The marbles | `src/heads.py`, `src/props.py`: each mob's face as 8x8 pixel art, projected onto a sphere of small voxels, with glowing eyes where the mob has them. Marbles in lava sink, heat up to glowing orange and burn up, trailing embers and smoke |
| The scene | `src/scene.py`, `src/blocks.py`: a sunny cliff (grass on top, stone with ores and moss, deepslate further down) with a lava lake at the bottom. Every block is 16x16 pixel art drawn in code; the lava flows and lights its surroundings |
| The camera | `src/director.py`: frames every marble still in the round, plus whoever is going into the lava, and the trapdoor and pen when the round's end is near. It moves in and out (15 to 44 blocks away) to fit them, keeping everyone below the header and above the part of the screen that YouTube covers with the title. Smoothed so it glides; a pan along the stalls for the countdown, and a close-up of the winner at the end |
| The graphics | `src/hud.py`: the header and live standings strip, the countdown on the beat, round cards, **ELIMINATED** / **SAVED!** / **JUST MADE IT!** banners and the winner card. They are all timed from the recorded race |
| Effects | `src/effects.py`: lava splashes, embers and smoke, the TNT blast, piston dust, gold sparkles for a save, confetti and fireworks |
| Sound | `src/audio.py`: a 144 BPM chiptune dance track in A minor (square-wave arpeggios, a saw pad, an off-beat bass, four-on-the-floor with sidechain pumping). The countdown's numbers land on every second beat and the beat drops on GO. Each round builds the arrangement up; slow motion muffles it. The final strips it to a snare roll and a riser, cuts it dead for a heartbeat, then answers with a brass fanfare and a victory groove in C major. Sound effects follow the physics: glassy ticks off the pegs, clacks, knocks on wood and stone, clinks on ice, slime boings, the pack rolling, pistons, the TNT's fuse and blast, trapdoors, lava splashing, sizzling and bubbling, a rising blip for each marble counted through, the broadcast stings, fireworks. Race sounds drop in pitch in slow motion |

Shared with the earlier projects: the deferred renderer (`renderer.py`, `sky.py`), the explosion and dust effects
(`explosion_vfx.py`, `vfx.py`) and some textures (`textures.py`, `tnt.py`). Dev helpers: `qa.py` (black,
corrupt or flickering frames) and `thumbnail.py`.

## Render it yourself

```bash
pip install -r requirements.txt
# Linux also needs Mesa's EGL + llvmpipe, e.g.: apt install libegl1 libegl-mesa0 libgl1-mesa-dri
cd src
python main.py --out ../output             # delivered quality (~0.6-0.7 s a frame on a 4-core CPU renderer)
python main.py --out ../preview --preview  # quick half-resolution check
python main.py --out ../stills --preview --stills --every 30   # one still a second
python main.py --out ../sound --cues-only  # sound work: the cue sheet and the audio, no frames rendered
python qa.py ../output/final.mp4 ../output/cues.json
python thumbnail.py ../output/thumbnail.jpg   # the cover; add --no-title for a clean one
python timeline.py                         # where the speed keys land in video time
```

## Make another race

1. `python search.py 0 500` prints the races that qualify, best first, with their winners, close calls, line-ups
   and the race times of each trapdoor opening (`opens`) and TNT blast (`booms`).
2. Put the seed and line-up into `SEED` and `LINEUP` in `src/timeline.py`.
3. Move the slow-motion keys in `SPEED` to the new race: slow down about 0.4 s before a trapdoor you want to
   linger on opens, and about 0.4 s before the blast. Then `python timeline.py` shows the video times.
4. Render. The camera, graphics, banners and soundtrack all follow the new race by themselves.

The mobs (faces, colours, names) are in `HEADS` in `src/heads.py`. The course is `build()` in `src/course.py`; a
new course needs a new search. The track's tempo, chords and lead are `BPM`, `CHORDS` and `LEAD` in
`src/audio.py`.

## Upload tips

* Upload the MP4 as-is. It already meets YouTube's recommended settings for 1080p. It is under a minute and 9:16,
  so YouTube treats it as a Short automatically.
* The 2.5 s countdown has no text apart from the numbers, so it is free for a hook or a voice-over. On-screen
  text fits best in the lower third, below the numbers.
* Ask viewers to pick a mob before GO. Comments picking a racer keep people watching to the end, and a marble
  race video gets rewatched to follow a different racer.
