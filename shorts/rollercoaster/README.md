# Minecraft Rollercoaster Through All 3 Dimensions (YouTube Shorts)

The tenth video in the series, after [`../steve-vs-swords`](../steve-vs-swords),
[`../creeper-vs-tnt`](../creeper-vs-tnt), [`../zombie-vs-arrows`](../zombie-vs-arrows),
[`../warden-vs-anvils`](../warden-vs-anvils), [`../last-giant-standing`](../last-giant-standing),
[`../black-hole`](../black-hole), [`../hydraulic-press`](../hydraulic-press), [`../mob-race`](../mob-race) and
[`../dominoes`](../dominoes). A new format again, and the first at **60 fps**: a first-person minecart ride that goes
through the Overworld, the Nether and the End in 37 seconds, and loops back to its start without a seam.

1. **The drop.** It opens at the top of the lift hill with the cart tipping over, 97 blocks above a badlands canyon.
   The beat drops with the cart. It pulls out just over the river at 180 km/h and races the canyon's bends, through a
   waterfall, under a natural arch, over an airtime hill, and into a ruined nether portal.
2. **The Nether.** Out onto a cliff ledge high over a lava sea, and down. A ghast floating ahead cries, turns, and
   fires, and its fireball blows up the track just ahead. The cart can't stop. It flies the gap **in slow motion**,
   over the span collapsing into the lava, as a second ghast fires straight at it. The fireball passes just over the
   rider's head, and the cart slams back onto the rails in a spray of sparks. Then comes a fortress bridge lined with
   fire, and a dive into the End portal at its end.
3. **The End.** Off the obsidian platform over the void, into the ring of obsidian pillars with the crystals burning
   on top. The **ender dragon** swoops low over the cart and roars. A banked spiral out over the void, and back to
   the exit portal, where the dragon has landed and the crystals' beams are healing it. It roars as the cart dives in
   under its head.
4. **The loop.** A white flash, and the cart is back on the lift hill in the Overworld, clicking up to the top. The
   last frame runs straight into the first, and the music's build-up lands on the opening drop, so on YouTube's
   loop it never ends.

Everything moves by physics: the cart's speed comes from the track's heights (gravity, friction, drag, powered
rails, the lift chain), and the jump is a real ballistic arc at the cart's speed. The rider's head turns into the
curves, looks down the drops, glances at the ghast and the dragon, and dips on the landing.

**Ready to upload:** [`release/rollercoaster.mp4`](release/rollercoaster.mp4), with an optional cover image in
[`release/thumbnail.jpg`](release/thumbnail.jpg): three bands cut on the diagonal, one per dimension (the canyon
drop, the jump over the lava with the ghast firing, the dragon perched with the crystal beams), titled
*MINECRAFT ROLLERCOASTER / ALL 3 DIMENSIONS*. The same image without the title is in
[`release/thumbnail_clean.jpg`](release/thumbnail_clean.jpg).

| | |
| --- | --- |
| Length | 37.0 s |
| Video | 1080x1920 (9:16), 60 fps, H.264 High, 16 Mbps 2-pass, closed GOP (30), BT.709 |
| Audio | AAC-LC (384 kbps target), 48 kHz stereo, loudness-normalised to -14 LUFS, peak -1.2 dBFS |

Everything is generated from code: no stock footage, samples, fonts or AI-generated media. The Minecraft-style
font, textures, mobs, sounds and music are all drawn or synthesised in `src/`.

## Titles and hooks

You add the titles and voice-over. Here are some ideas that fit the video:

* **Titles:** *Minecraft Rollercoaster Through All 3 Dimensions* · *POV: The Minecraft Rollercoaster That Goes to
  the End* · *I Rode a Rollercoaster Through Every Minecraft Dimension* · *Minecraft Rollercoaster... But a Ghast
  Breaks the Track* · *This Minecraft Rollercoaster Never Ends*
* **Hooks for the first second** (the first frame has no text or HUD on it): "This rollercoaster goes through all 3
  dimensions..." · "Keep your arms inside the minecart." · "Wait for the ghast..." · "Rate this ride 1 to 10."
* **Voice-over beats:** 0:14 a ghast cries ("oh no..."), 0:16.8 the track explodes ("THE TRACK!"), 0:18-0:21 the slow
  motion ("we're not gonna make it"), 0:22 the landing ("WE MADE IT"), 0:28 the dragon swoops ("DRAGON!"), 0:37 the
  loop ("...wait, again?").

## Timeline

| Time | What happens |
| --- | --- |
| 0:00 | Top of the lift hill, the cart tipping over the crest: the canyon opens up 97 blocks below. The beat drops with it. No HUD yet, so the hook can go on top |
| 0:01 | The drop: straight down the canyon wall, the speed on the action bar climbing past 150 km/h |
| 0:05 | The pull-out just over the river at 180 km/h, and the canyon's long bends |
| 0:08 | Through the waterfall pouring off the rim (the sound goes under water for a moment) |
| 0:09 | Under the arch, over an airtime hill, onto a powered stretch and straight into the ruined portal |
| 0:10.7 | **The Nether** (the portal's swirl). *Advancement Made! We Need to Go Deeper* |
| 0:11 | Off the ledge and down to the lava sea, then low over it between pillars, glowstone and lava falls |
| 0:14 | A ghast drifting ahead of the cart cries. At 0:15.6 it turns to the track and fires |
| 0:16.8 | **BOOM.** The span ahead explodes, and its rails, deck and pillars fall into the lava |
| 0:17.8 | **Slow motion:** the cart flies off the broken end, over the collapsing span. A second ghast fires straight at it, and the fireball passes over the rider's head (0:20) |
| 0:21.9 | Back at full speed as the wheels hit the rails: a jolt and a spray of sparks. The music slams back in |
| 0:23 | The fortress bridge, with fire burning on its posts, and a dive into the End portal at its end (0:24.6) |
| 0:24.6 | **The End.** *Advancement Made! The End?* The obsidian platform, the void, the pillars and their crystals |
| 0:28.3 | The ender dragon swoops low over the cart and roars |
| 0:29 | A banked spiral out over the void and back round the pillars |
| 0:32.3 | The dragon lands beside the exit portal, and the crystals' beams heal it. It roars as the cart dives in (0:33.6) |
| 0:33.6 | A white flash: back on the lift hill in the Overworld, clicking up. The last frame (0:37.0) runs into the first |

## How it's made

| | |
| --- | --- |
| The track and its physics | `src/track.py`: a "turtle" builder steers the track segment by segment (length, turn, pitch), and integrates the cart's speed along it: gravity, rolling friction, air drag, powered-rail boosters and the lift chain. The jump is a ballistic segment flown at the cart's real speed. The rails bank into the curves as the speed and curvature ask, on rotation-minimising frames |
| The ride | `src/paths.py`: the three layouts. `src/ride.py`: the timeline (which world, which stretch of which track, the slow motion), with each world built once and cached |
| The rider | `src/rider.py`: the eye in the cart. The head looks ahead along the track (farther at speed, down the big drops), keeps part of its own sense of up against the banking, widens its field of view with speed like the game's sprint, shakes with speed and g-force, glances at the ghast and the dragon, and dips when the cart lands |
| The worlds | `src/world_over.py`: a badlands canyon with terracotta strata, a river, a waterfall off the rim, a natural arch, trees, cacti and a ruined portal. `src/world_nether.py`: a lava sea under a netherrack ceiling, with pillars, glowstone, lava falls, basalt, a crimson forest, a fortress bridge and an End portal (all its eyes in). `src/world_end.py`: the main island in the void, the ring of obsidian pillars (two caged), the exit portal's fountain with the dragon egg, and the obsidian platform. The track clears its own way through all of them, and supports hold it up |
| The mobs | `src/mobs.py`: the ghast (with its shooting face), the ender dragon (a neck and tail that follow the body, scalloped wings, a jaw that drops to roar, glowing eyes), endermen and end crystals, all built from boxes and textured like the game's models |
| The story and the effects | `src/fx.py`: the ghasts and their fireballs, the blast, the span collapsing piece by piece into the lava with splashes, embers off the lava, the landing's sparks, the fire on the bridge, the dragon's flight (circling, the swoop, the landing), the crystal beams, the portals' surfaces, and the transitions (the nether portal's swirl, the End portal's starfield, the white flash) |
| The renderer | `src/gfx.py`: a deferred OpenGL renderer with shadows, SSAO, a volumetric haze, bloom and tonemapping. It adds flowing, glowing lava and falling water to the engine from the earlier videos. `src/looks.py`: the light of each dimension (golden hour, a lava-lit red haze, the End's cold violet) |
| The HUD | `src/hud.py`, `src/pixelfont.py`: the speed on the action bar in the game's font, and the advancement toasts |
| Sound | `src/audio.py` builds everything from the ride's cue sheet (`src/cues.py`). **Music:** one piece in D. The Overworld has a heroic supersaw theme at 135 bpm that pumps with the kick. The Nether is half-time drum and bass with a distorted reese bass, dropping to a heartbeat, a drone and a choir during the slow motion, and slamming back in on the landing. The End is choir, bells, taiko and a brass hit on the dragon's swoop. The lift hill has a snare roll and a riser that resolve on the loop. **Effects:** the wheels' roar and the rail joints' clack (silent in the air), wind with the square of the speed, the chain ratchet, the waterfall, lava bubbling and the falls' roar, the portals, the ghasts' cries, the fireballs (panned and Doppler-shifted), the blast, the landing's clang and screech, the fire, the dragon's wings and roars, and the crystals' hum |

Shared with the earlier projects: the voxel engine (`voxel.py`, `blocks.py`, `textures.py`, `sky.py`,
`entities.py`) and the renderer from the horror film, turned vertical.

## Render it yourself

```bash
pip install -r requirements.txt
# Linux also needs Mesa's EGL + llvmpipe, e.g.: apt install libegl1 libegl-mesa0 libgl1-mesa-dri
cd src
python main.py --out ../output                      # delivered quality: 2218 frames, then the soundtrack and final.mp4
python main.py --out ../preview --preview --fps 12  # quick half-resolution check with the sound (preview.mp4)
python main.py --out ../sound --cues-only           # sound work: the cue sheet and the audio, no frames rendered
python main.py --out ../output --encode-only        # redo the soundtrack and the encode from an existing render
python thumbnail.py ../output/thumbnail.jpg         # the cover; add --no-title for a clean one
python paths.py                                     # lengths, speeds and marks of the three tracks
```

The first run builds the three worlds (about a minute each) into a cache. A frame takes about a second to render
on a 4-core CPU renderer.

## Make your own

* **The track:** each layout in `src/paths.py` is a list of segments. Change them, and the speeds, the banking, the
  worlds (they are carved round the track), the story's timing and the music's bars all follow.
* **The slow motion:** the `slow` list of each segment in `src/ride.py`.
* **The look of a dimension:** `src/looks.py`.

## Upload tips

* Upload the MP4 as-is. It meets YouTube's recommended settings for 1080p60, and it is under a minute and 9:16, so
  YouTube treats it as a Short.
* The video loops seamlessly, which works with YouTube's own looping of Shorts: many viewers will go round twice
  before they notice.
* The first frame has no text or HUD on it, so your hook can go on top. The speed readout fades in as the cart picks
  up speed.
* The flashes stay within photosensitivity limits: the blast's flash lasts 0.12 s, and the white flash at the exit
  portal fades over about a second.
