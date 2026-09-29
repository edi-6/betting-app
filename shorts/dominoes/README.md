# 10,000 Dominoes in Minecraft... Wait for the End (YouTube Shorts)

The ninth video in the series, after [`../steve-vs-swords`](../steve-vs-swords),
[`../creeper-vs-tnt`](../creeper-vs-tnt), [`../zombie-vs-arrows`](../zombie-vs-arrows),
[`../warden-vs-anvils`](../warden-vs-anvils), [`../last-giant-standing`](../last-giant-standing),
[`../black-hole`](../black-hole), [`../hydraulic-press`](../hydraulic-press) and [`../mob-race`](../mob-race). A new
format again: a domino chain reaction through a Minecraft world, with a counter, a fail, a save and a hidden picture.

1. **The punch.** First person, the player's arm punches the first domino of a rainbow line.
2. **The run.** Rainbow dominoes race through a double spiral, split into a **red**, a **gold** and a **blue** line
   that race each other (blue wins a photo finish), join again, and cross the river on a bridge.
3. **The fail.** The run reaches the giant blank field... and its last domino lands **a block short**. The counter
   stops at **424 / 10,000**. Silence.
4. **The save.** A creeper walks in, turns to look at you, hisses, and blows up. The blast knocks the field's feeder
   line over, and the field starts to fall.
5. **The picture.** The white field falls in a wave and turns into a picture as it goes: a cyan shirt, a beard, a
   nose... and blank white eyes. It's **Herobrine**. The counter hits **10,000**, the eyes light up, the day goes
   dark, lightning strikes, and the counter glitches red.

Every domino is simulated, and the counter counts the real dominoes going down. The fail, the save and the
picture keep people watching to the end, and the ending is the kind people rewatch and comment on.

**Ready to upload:** [`release/dominoes.mp4`](release/dominoes.mp4), with an optional cover image in
[`release/thumbnail.jpg`](release/thumbnail.jpg) (the field half fallen with the trees of the plains behind it, the
face up to the nose, the eyes still hidden; titled *WAIT FOR THE END...*), or the same without the title in
[`release/thumbnail_clean.jpg`](release/thumbnail_clean.jpg).

| | |
| --- | --- |
| Length | 34.1 s |
| Video | 1080x1920 (9:16), 30 fps, H.264 High, 12 Mbps 2-pass, closed GOP (15), BT.709 |
| Audio | AAC-LC (384 kbps target), 48 kHz stereo, loudness-normalised to -14 LUFS, peak -1.2 dBFS |

Everything is generated from code: no stock footage, samples, fonts or AI-generated media. The Minecraft-style
font, textures, characters, sounds and music are all drawn or synthesised in `src/`.

## Timeline

| Time | What happens |
| --- | --- |
| 0:00 | First person: the arm, a rainbow line of dominoes running off into the plains, the race lines on the horizon. The counter reads **0 / 10,000**. No other text, so the hook can go on top |
| 0:00.6 | The punch. The first domino goes, and the camera lifts off and chases the wave |
| 0:02 | The double spiral from above: in clockwise, an S through the middle, out anticlockwise |
| 0:06.9 | The split: red, gold and blue lines race side by side |
| 0:10.3 | Photo finish in slow motion: **blue** gets there first and carries on |
| 0:12 | Over the river on the bridge, then on towards the field |
| 0:16.1 | The last domino falls in slow motion... and lands a block short of the field. **424 / 10,000**. The music stops |
| 0:17.5 | A creeper walks in, stops, and turns to look at the camera (0:19.6) |
| 0:20.0 | It hisses, swells and flashes |
| 0:21.5 | **BOOM** (slow motion): a crater, dirt and grass everywhere, and the field's feeder line goes over both ways |
| 0:22 | The field falls in a V-shaped wave, turning into a picture as it goes. The strings come in and the counter races |
| 0:30.1 | **10,000.** The counter turns gold. Straight down on the whole picture: it's **Herobrine** |
| 0:31.5 | His eyes start to glow and the day goes dark |
| 0:32.7 | Lightning strikes beside the field. The counter glitches red. It ends in the dark on his glowing eyes (0:34.1) |

## How it's made

| | |
| --- | --- |
| Where every domino stands | `src/layout.py`: the paths (a Hermite-curve start, an Archimedean double spiral with an S of two semicircles in the middle, three race lines that fan out at 35 degrees and come back in onto the merge domino, the tail over the bridge), dominoes every 0.6 blocks along them, the feeder line and the field grid. The field's size is chosen so there are exactly **10,000** |
| The physics | `src/dominoes.py`: pymunk (Chipmunk2D). Each line is a straight 2D chain however it curves on the ground (its dominoes are evenly spaced along it), set off by the one before it with the spin it would really get. All 72 field columns are the same chain, shifted in time. The wave runs at 11 blocks/s and fallen dominoes rest at 75.5 degrees on the next one, as the geometry says they should |
| The hidden picture | `src/picture.py`, `src/props.py`: one domino per pixel. Each field domino shows its colour on the face that ends up on top, and only once it tips, so the standing field is a blank white canvas from every angle |
| The world | `src/world.py`, `src/blocks.py`, `src/sky.py`, `src/clouds.py`: a plains biome (a flat meadow, a river with sandy banks, an oak bridge, oak and birch trees, hills and snowy mountains), every block drawn in code, under volumetric clouds. The water ripples and reflects the sky |
| The characters | `src/hand.py`: the player's arm in first person with the game's swing. `src/creeper.py`: the creeper from the earlier videos, walking with its legs in diagonal pairs, turning its head to the camera, swelling and flashing on its fuse |
| The blast | `src/effects.py`: fireball and smoke, dirt and grass blocks thrown out of a real crater, and the nearest dominoes flung as spinning rigid bodies that bounce and settle |
| The camera | `src/director.py`: one continuous move. POV, a chase behind the wave, a near top-down orbit of the spiral, behind the race, low at the finish, from the river bank, a close shot at the gap that turns to the creeper, a push-in on the fuse, the recoil, then a crane up over the field to look straight down on the picture |
| The edit | `src/timeline.py`: playback speed follows the story: real time, faster in the spiral, slow motion for the photo finish, the last domino, and the blast |
| The counter | `src/hud.py`, `src/pixelfont.py`: the fallen count in the game's pixel font over an XP-bar progress bar. It turns gold at 10,000, and red with a glitch at the end |
| Sound | `src/audio.py`: a soft felt piano in D major (each note synthesised: two slightly detuned strings with inharmonic partials, a hammer thump, a damper), strings for the race, five bars that stop unresolved exactly when the run does, a swell of strings and rising arpeggios over the field resolving on D major at 10,000, then a low beating drone, a reversed swell and thunder. Every domino clicks as it's hit and again as it settles, so the field's thousands of clicks become a rattle. Also the punch, footsteps on grass, the fuse, the blast, the 10,000 chime, wind, birds and the river |

Shared with the earlier projects: the deferred renderer (`renderer.py`; this one adds dominoes painted per
instance, glowing eyes and water), the explosion and dust effects (`explosion_vfx.py`, `vfx.py`), textures
(`textures.py`) and the creeper's skin. Dev helpers: `qa.py` (black, corrupt or flickering frames) and
`thumbnail.py`.

## Render it yourself

```bash
pip install -r requirements.txt
# Linux also needs Mesa's EGL + llvmpipe, e.g.: apt install libegl1 libegl-mesa0 libgl1-mesa-dri
cd src
python main.py --out ../output             # delivered quality (about a second a frame on a 4-core CPU renderer)
python main.py --out ../preview --preview  # quick half-resolution check
python main.py --out ../stills --preview --stills --every 30   # one still a second
python main.py --out ../sound --cues-only  # sound work: the cue sheet and the audio, no frames rendered
python qa.py ../output/final.mp4 ../output/cues.json
python thumbnail.py ../output/thumbnail.jpg   # the cover; add --no-title for a clean one
python layout.py                           # counts and lengths of every line (always 10,000 in total)
```

The first run builds the world (about 40 s) and bakes the clouds (about 80 s) into a cache.

## Make your own

* **A different picture:** `picture()` in `src/picture.py` returns any 132 x 72 image (row 0 = the top). The
  reveal follows automatically, and the pixels in its `eyes` mask are the ones that glow at the end.
* **The run:** the spiral, the race lines and the tail are defined at the top of `src/layout.py`. Change them and
  the physics, the counter and the camera follow. The field's height adjusts to keep the total at 10,000.
* **The story's timing:** the creeper's walk, turn and fuse, and the speed ramps, are in `src/timeline.py`.

## Upload tips

* Upload the MP4 as-is. It already meets YouTube's recommended settings for 1080p. It is under a minute and 9:16,
  so YouTube treats it as a Short automatically.
* The first half second, before the punch, is clean for a hook or a voice-over. The counter already sets up the
  challenge (0 / 10,000).
* The creeper's fuse flashes no more than three times a second, to stay clear of photosensitivity limits.
