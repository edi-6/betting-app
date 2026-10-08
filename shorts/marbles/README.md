# 10,000 Marbles in Minecraft... Wait for the End (YouTube Shorts)

The follow-up to [`../dominoes`](../dominoes) and [`../dominoes-2`](../dominoes-2), the ones that got the views. It
keeps their formula: a huge physics chain in a Minecraft world, a counter, a fail, a save, a hidden picture and a
twist at the end. This time it's a **marble machine** standing in the same plains, and the picture is made by the
marbles themselves:

1. **The hook.** From above: a hopper heaped with 10,000 rainbow marbles and one big **golden ball** on top. The
   counter reads **0 / 10,000**, *GUESS THE PICTURE...*, and the marbles start to pour.
2. **The cascade.** They pour out of the hopper and rattle down a field of glowing end rods into a glass tank with a
   gold frame. They look random, but they pile up into a picture, bottom first: grass, flowers, two green feet...
3. **The jam.** The crater over the hopper's hole reaches the golden ball. It rolls in and plugs the hole with a
   clunk, and the music stops dead (*UH OH... IT'S STUCK!*). The last marbles land. Silence.
4. **The save.** *...A GOAT?* A goat trots in, stares at you and bleats, backs up, and charges at the camera
   screaming. It rams the pedestal so hard the whole machine shudders, the golden ball pops out, and the marbles
   gush back out (*THE GOAT SAVED IT!*).
5. **The picture.** With the music back at full strength, the picture rises: a sunset, a square sun, hills,
   trees... and a creeper.
6. **The last one.** The counter crawls to **9,999**. One marble drops on its own, every bounce a note climbing a
   scale. It lands at the top of the picture as a star: **10,000**. *IT'S A CREEPER!*
7. **The twist.** The picture's creeper starts to hiss and flash white like the real thing, the glass cracks
   (*RUN!!!*), and it blows up. 10,000 marbles fly out at you in slow motion and rain down over the meadow.

**For a second watch:** someone is standing in front of the sun in the picture, two white eyes on a dark figure. In
the last shot he's there again, tiny, on the edge of the cliff from [`../dominoes-2`](../dominoes-2), among the trees
to the left of the machine.

**Ready to upload:** [`release/marbles.mp4`](release/marbles.mp4). The optional cover is in
[`release/thumbnail.jpg`](release/thumbnail.jpg): the machine from the front, the picture three quarters done (the
sunset and something green standing in the meadow, no face yet) with marbles still raining in, titled *10,000
MARBLES / GUESS THE PICTURE*. The same frame without text is [`release/thumbnail_clean.jpg`](release/thumbnail_clean.jpg).

| | |
| --- | --- |
| Length | 37.4 s |
| Video | 1080x1920 (9:16), **60 fps**, H.264 High@4.2, 16 Mbps 2-pass, closed GOP (30), BT.709 |
| Audio | AAC-LC 384 kbps, 48 kHz stereo, loudness-normalised to -14 LUFS, true peak under -1 dBFS |
| Rendering | 4 samples per pixel (rendered at 2160x3840) |

Everything is made in code: no game footage, samples, fonts or AI-generated media. The physics of every one of the
10,000 marbles is simulated, and the counter counts the real marbles landing in the tank.

## Titles, hooks, pinned comment

* **Titles:** *10,000 Marbles in Minecraft... Wait for the End* · *10,000 Marbles Made a Picture...* · *Guess the
  Picture (10,000 Marbles)* · *The Goat Saved It* · *I Built a Marble Machine in Minecraft*
* **Hooks for the first second** (the opening is the hopper from above, so a line over it lands):
  * "10,000 marbles... guess what they make."
  * "Watch the picture."
  * "Wait for the goat."
  * "It took 10,000 marbles to make this."
* **Pinned comment:** *Did you spot who's standing in the sun?* Or *What did you think it was going to be?* The
  first sends people back to look; the second gets guesses.
* **Voice-over beats:**
  * 0:00 the pour
  * 0:08 the golden ball rolls ("uh oh")
  * 0:10 the jam
  * 0:11.6 the goat
  * 0:15.5 the ram
  * 0:17 the gush ("LET'S GO")
  * 0:25.3 the last marble
  * 0:29 "it's a creeper"
  * 0:31.6 the hiss
  * 0:33 the blast
* **Tags:** #minecraft #marbles #satisfying #creeper #goat #shorts

## Timeline

| Time | What happens |
| --- | --- |
| 0:00 | From above: the hopper heaped with rainbow marbles and the golden ball. **0 / 10,000**, *GUESS THE PICTURE...*. The flow starts and a crater opens over the hole |
| 0:00.8 | The camera dives down the front of the machine with the stream, through the end rods, to the tank |
| 0:02.8 | Close among the end rods: marbles clattering through |
| 0:05.0 | The bottom of the picture rising: grass, flowers |
| 0:07.3 | The hopper from above: the crater reaches the golden ball and it rolls in (*UH OH...*) |
| 0:10.0 | It plugs the hole: **clunk**, the music stops (*IT'S STUCK!*). The last marbles land, about 3,900 |
| 0:11.6 | A goat trots in, looks up at the machine, looks at you, bleats (*...A GOAT?*) |
| 0:13.8 | It backs up, lowers its head and charges at the camera, screaming |
| 0:15.0 | Slow motion from the side: it rams the pedestal. The machine shudders, the picture jumps |
| 0:16.0 | The hopper: the golden ball pops out over the back |
| 0:17.1 | The marbles gush out again (*THE GOAT SAVED IT!*), the music comes back big |
| 0:18.6 | The picture rising: the hills, the sun going down, a big green shape |
| 0:24.4 | The flow dies away; the counter crawls to **9,999** (*THE LAST ONE...*) |
| 0:25.3 | The last marble, followed all the way down: eight bounces, eight notes climbing |
| 0:28.9 | It lands at the top as a star: **10,000**. The camera pulls back: *IT'S A CREEPER!* |
| 0:31.6 | The creeper in the picture starts hissing and flashing white; the glass cracks (*RUN!!!*) |
| 0:33.0 | **It blows up.** Slow motion: the marbles and the glass come straight at you |
| 0:35.3 | The marbles rain down over the meadow. The empty frame smokes. Someone on the cliff |
| 0:37.4 | End (and it loops back to the full hopper) |

## How it's made

| | |
| --- | --- |
| The machine | `src/machine.py`: a wall of black concrete in the plains with a slot between the board and the glass exactly one marble deep, so the marbles move in a plane: 150 end rods in staggered rows, a tank 34 blocks wide, an iron hopper on top. `src/world.py` builds it out of blocks in the dominoes videos' world (pedestal of stone bricks, gold frame, iron, glowing end rods) |
| The physics | `src/sim.py`: pymunk (Chipmunk2D) in the plane of the slot. Marbles come out of the chute's throat at the rate the story wants (a ramp, a steady flow, nothing during the jam, a burst when it's freed, a trickle at the end); everything after that is physics. A single stream from the middle still fills the tank level: the marbles are bouncy, so the pile flattens itself. When the goat hits, every marble in the tank gets a kick. The last marble is dropped on its own onto the settled pile, from the best of hundreds of tries (lots of bounces, a couple of seconds, landing in plain view) |
| The picture | `src/picture.py`: a Minecraft sunset painted in code (a square sun, purple mountains, hills with oak trees, a meadow with flowers, a tiny figure in front of the sun) with a creeper ray-cast in three-quarter view from its voxel model. **Each marble is coloured by where it finally comes to rest**, so while they fall they look random, and they pile up into the picture |
| The hopper | `src/hopper.py`: only the surface you see is drawn, from a model of a bin draining (funnel flow): a crater opens over the hole and deepens, its sides at the angle of repose, marbles sliding down it in rings; the golden ball rolls in when the crater reaches it |
| Rendering | `src/renderer.py`: the dominoes engine (deferred shading, soft sun shadows, ambient occlusion, sky, bloom, 4x supersampling) plus marbles as ray-traced impostor spheres (true depth, shadows) shaded as glass (a sharp highlight, the sky in the rim, the sun focused through them into a glow of their own colour), polished gold for the ball, and the glass front (reflections, a glint, and cracks before the end) |
| The goat | `src/goat.py`: a voxel goat (horns, beard, shaggy fur) and its moves: trot, stop, look up, look at you, back up, head down, charge, ram, bounce off, shake its head |
| The blast | `src/explosion.py`, `src/fx.py`: every marble thrown out away from the creeper's chest and towards you, bouncing and rolling across the pedestal and the meadow; 700 glass shards; the fireball, smoke, flash and light; stone chips where the goat hits |
| The edit | `src/timeline.py`: 19 shots, each mapping its frames to story time (speed-ups during the fill, slow motion for the ram, the last marble and the blast) and to a camera that eases between keys or follows the last marble |
| HUD | `src/hud.py`: the counter in the game's font over an XP bar (gold at 10,000) and the captions |
| Sound | `src/audio.py` (toolkit in `src/sound.py`): the marbles' rattle driven by the physics (glass ticks off the end rods, clacks onto the pile, a shimmer for the densest moments), the golden ball's rumble and clunk, a tape-stop when the music dies, the goat's hooves, bleat, scream and ram, the pop and the gush, a bell for each bounce of the last marble, the level-up chime, the hiss, the blast, the glass and the marbles raining on grass. The music: a bouncy music-box tune in G over plucked piano chords, a sneaky pizzicato for the goat and a snare roll into its charge, the tune back with drums and strings, held strings and a heartbeat for the last marble, a big G major chord at 10,000 |

## Render it yourself

```bash
pip install -r requirements.txt
# Linux also needs Mesa's EGL + llvmpipe, e.g.: apt install libegl1 libegl-mesa0 libgl1-mesa-dri
cd src
python sim.py                                        # the physics (cached in ../cache, about 4 minutes)
python main.py --out ../output                       # delivered quality (about 2 s a frame on a 4-core CPU)
python main.py --out ../preview --preview --frames 100,900,1800    # quick half-resolution stills
python main.py --out ../output --encode-only         # redo the sound and the encode
python qa.py ../output                               # checks and a contact sheet
python thumbnail.py ../output/thumbnail.jpg          # the cover; --no-title for a clean one
```

## Upload tips

* Upload the MP4 as it is: 60 fps, 9:16, under a minute, so YouTube treats it as a Short.
* The creeper's flashing before the blast pulses at most about once a second on screen (it's in slow motion), and
  the blast is one flash. Still, "Flashing lights" in the description is good practice.
* The world is the dominoes videos' plains. The goat, the creeper and the figure are drawn for the video.
