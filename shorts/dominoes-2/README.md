# 100,000 Dominoes in Minecraft... Part 2 (YouTube Shorts)

The sequel to [`../dominoes`](../dominoes), the one that got the views. It keeps that video's format: a domino
chain reaction through a Minecraft world, a counter, a race, a fail, a save and a hidden picture. Part 2 makes it
bigger, faster and creepier:

1. **The punch.** First person: the player punches the first domino of a rainbow line. At the end of the line stands
   a domino the size of a building, and the counter reads **0 / 100,000**.
2. **The growth.** The line runs into ten dominoes, each **1.4 times bigger** than the one before. Each knocks
   over the next, bigger and bigger, until the 29-block **giant** tips over. *PICK A COLOR!*
3. **The slam.** The camera is on the ground beside where it lands, looking up as it comes down. Its slam (in slow
   motion, with a cloud of dust) sets off a four-colour race: **red, gold, green and blue**.
4. **The finish.** Red takes the shortest way and gets there first, but its last domino **falls a block short** of
   the finish domino (*RED FELL SHORT!*). Gold and blue arrive together in slow motion: **gold wins** by a hundredth
   of a second.
5. **The fail.** Over the river on the bridge, on towards the field of almost 100,000 white dominoes, and the run
   **stops a block short** again (*541 / 100,000*). Silence. A storm comes in and the sky goes dark. It starts to
   rain, and there's a heartbeat.
6. **The save.** **Lightning** strikes right where it stopped, then again and again down the line, and the whole
   field starts to fall.
7. **The message.** The camera swings up around the field onto the top of a cliff, and the falling field spells
   out, word by word in blood red: **LOOK... BEHIND... YOU**. The O's are Herobrine's white eyes. The counter hits
   **100,000** and the words start to glow.
8. **The turn.** You turn around. **He's right there.** The picture glitches to black.

**Herobrine is in the video three more times before the end**, small, for people to find on a second watch:

* beside the giant in the first frame
* in the open by the race
* on the edge of the cliff in the storm, exactly where you end up standing

The ending sends people back to the start to look for him, and Shorts loop on their own.

**Ready to upload:** [`release/dominoes-2.mp4`](release/dominoes-2.mp4), with an optional cover image in
[`release/thumbnail.jpg`](release/thumbnail.jpg). The cover shows the glowing LOOK BEHIND YOU from the cliff,
titled *100,000 DOMINOES / PART 2*. The same image without the title is in
[`release/thumbnail_clean.jpg`](release/thumbnail_clean.jpg).

| | |
| --- | --- |
| Length | 38.4 s |
| Video | 1080x1920 (9:16), **60 fps**, H.264 High@4.2, 16 Mbps 2-pass, closed GOP (30), BT.709 |
| Audio | AAC-LC 384 kbps, 48 kHz stereo, loudness-normalised to -14 LUFS, peak -1.2 dBFS |
| Rendering | 4 samples per pixel (rendered at 2160x3840) |

Everything is generated from code: no stock footage, samples, fonts or AI-generated media. The Minecraft-style
font, textures, characters, sounds and music are all drawn or synthesised in `src/`. Every one of the 100,000
dominoes is simulated, and the counter counts the real dominoes going down.

## Titles, hooks, pinned comment

* **Titles:** *100,000 Dominoes in Minecraft... Part 2* · *100,000 Dominoes (Don't Look Behind You)* · *The
  Dominoes Spelled Out a Message...* · *I Set Up 100,000 Dominoes in Minecraft* · *Wait for the End... Part 2*
* **Hooks for the first second.** The first frame has only the counter on it, so your hook can go on top:
  * "Part 2. This time: 100,000 dominoes."
  * "Pick a color before the giant falls"
  * "Wait for the end..."
  * "Every domino is 40% bigger than the last"
* **Pinned comment:** *Did you spot him before the end? He's in it 3 more times.* Or *Which color did you pick?*
  Both make people comment and rewatch.
* **Voice-over beats:**
  * 0:00.5 the punch
  * 0:04 the growth ("each one bigger than the last...")
  * 0:08.5 the giant tips
  * 0:11.4 the slam
  * 0:15.8 red falls short ("NO WAY")
  * 0:16.1 gold wins
  * 0:21 the run stops ("not again...")
  * 0:24 lightning ("WHAT?!")
  * 0:28.6 "wait... it's spelling something"
  * 0:35.3 100,000
  * 0:36.3 the turn
* **Tags:** #minecraft #dominoes #herobrine #satisfying #shorts

## Timeline

| Time | What happens |
| --- | --- |
| 0:00 | First person: the arm, a rainbow line running off to the growth and the giant. The counter reads **0 / 100,000**. Herobrine stands beside the giant |
| 0:00.55 | The punch. The camera lifts off and chases the wave |
| 0:04.0 | The growth, from the side: ten dominoes, each 1.4 times bigger, the camera pulling back as they grow |
| 0:08.5 | The giant (29 blocks tall) tips over. On the ground beside where it will land, looking up; *PICK A COLOR!* |
| 0:11.4 | **The slam**, in slow motion: dust, a shake, and the four race lines set off |
| 0:11.6 | The race from behind: red, gold, green, blue. Herobrine stands in the open on the left |
| 0:15.8 | The finish, in slow motion: red gets there first and falls a block short (*RED FELL SHORT!*) |
| 0:16.1 | Gold and blue together: **gold wins** by 0.015 s (*GOLD WINS!*) |
| 0:17 | Over the river on the bridge, then on towards the field |
| 0:21.0 | The last domino lands a block short of the field. **541 / 100,000**. The music stops |
| 0:21.5 | Silence, wind and a heartbeat. The storm comes in, rain. The camera tilts up to the cliff, where Herobrine stands on the edge |
| 0:24.0 | **Lightning** right at the gap, then down the line on both sides (0:24.7, 0:25.4): the field starts to fall |
| 0:24.5 | Up and around the field in a long arc, onto the cliff |
| 0:28.6 | From the cliff's edge: the field falls towards you and the words appear: LOOK, BEHIND, YOU |
| 0:35.3 | **100,000.** The counter turns gold and the words glow red, the O's white like eyes |
| 0:36.3 | You turn around. Herobrine is right there. The counter glitches red |
| 0:37.9 | The picture tears and cuts to black (0:38.4) |

## How it's made

| | |
| --- | --- |
| Where every domino stands | `src/layout.py`: the run (an S-bend), the growth (each domino 1.4 times the size of the last, spaced so the gap in front of each is 0.45 of its height), the four race lines (each its own wiggle, coming in onto one merge domino; red's line ends just short of it), the tail over the bridge, the feeder in five segments, and the field: 257 x 386 dominoes. The start of the run is chosen so the total is exactly **100,000** |
| The physics | `src/dominoes.py`: pymunk (Chipmunk2D). Each line is a straight 2D chain however it curves on the ground, and the growth is one chain of dominoes of different sizes (and masses). The giant's landing sets off the race. The photo finish is real: gold and blue have the same number of dominoes, and the giant comes down a hair twisted. All 257 field columns are the same chain, shifted in time, under stronger gravity so the 230-block field falls in time |
| The message | `src/picture.py`: anamorphic, like 3D street art. The words are laid out on the screen of the last shot and projected back onto the field, so they read straight from the cliff and stretch away from anywhere else. The field falls towards the cliff, so the words appear in reading order. As each field domino tips, all its faces take their picture colour |
| The world | `src/world.py`, `src/blocks.py`: the plains from part 1 made bigger, with a river and bridge, and north of the field a 72-block cliff with a craggy face (deepslate at its foot, granite, andesite, ore, moss under a lip of dirt) and a grassy top |
| Herobrine | `src/herobrine.py`: Steve's skin with blank white eyes, one voxel per pixel. The eyes are the renderer's glowing material, so they shine in the dark |
| The storm | `src/storm.py`, `src/effects.py`: the day darkening, rain (streaks around the camera), blocky lightning bolts with flickers and lights, the clods and smoke where they hit, a scorch mark, and the giant's dust cloud |
| The camera | `src/director.py`: one continuous move. POV, the chase, the growth from the side, low beside the giant's landing, behind the race, above the finish, the bridge, the gap, the tilt up to the cliff, the arc around the field onto the cliff's edge, the turn |
| The edit | `src/timeline.py`: playback speed follows the story: fast for the small dominoes, slow motion for the slam, the finish, the last domino and the strike, fast while the field falls, real time for the end |
| The HUD | `src/hud.py`: the counter in the game's font over an XP bar (gold at 100,000, red and glitching at the turn); the race's four colours with *PICK A COLOR!*, red crossed out, and the winner |
| Sound | `src/audio.py`: part 1's felt piano in D for the run, with strings coming in as the dominoes grow; a held breath as the giant tips, a creak and the slam; a driving race theme held through the slow-motion finish; red's "wah wah", the chord as gold wins, the piano cut off where the run stops. Then wind, rain, a heartbeat and thunder, and under the falling field the piano gone dark (D minor), building to a wide D minor chord at 100,000. A drone, a whoosh as you turn, a dissonant sting, a glitch. Every domino clicks as it's hit and as it settles, and the growth's knocks get deeper with each size |

## Render it yourself

```bash
pip install -r requirements.txt
# Linux also needs Mesa's EGL + llvmpipe, e.g.: apt install libegl1 libegl-mesa0 libgl1-mesa-dri
cd src
python main.py --out ../output                         # delivered quality (about a second a frame on a 4-core CPU)
python main.py --out ../preview --preview --stills --every 20 --no-audio   # quick half-resolution stills
python main.py --out ../stills --frames 600,1500,2140 --no-audio          # a few frames at full quality
python main.py --out ../output --cues-only             # sound work: the cue sheet and the audio, no frames
python main.py --out ../output --encode-only           # redo the soundtrack and the encode from a render
python thumbnail.py ../output/thumbnail.jpg            # the cover; --no-title for a clean one
python stills.py ../views                              # fixed views of the world (dev)
python layout.py                                       # counts and lengths of every line (always 100,000)
```

The first run builds the world (about 2 minutes) and the sky (about 40 s) into a cache, and simulates the chains
(about 30 s, also cached). Frames go into one-second segments, so a stopped render picks up where it left off.

## Upload tips

* Upload the MP4 as it is. It's 60 fps and already meets YouTube's recommended settings for 1080p. It's under a
  minute and 9:16, so YouTube treats it as a Short.
* Mention part 1 ("Part 2 of the 10,000 dominoes") so people who saw it recognise the format and so the two get
  recommended together.
* Lightning flashes no more than three times a second, and the brightest is under 0.2 s, to stay clear of
  photosensitivity limits.
