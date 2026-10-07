# I Dug Straight Down (X-Ray View) (YouTube Shorts)

Minecraft's oldest rule is "never dig straight down". This short breaks it, in cross-section: a slice of a Minecraft
world seen from the side, like an x-ray, with Steve digging a one-block shaft from the grass to the diamonds. The
vertical format *is* the depth. Viewers can always see what's under him before he does: a cave, a mineshaft, a
geode, an Ancient City with a Warden in it, five diamonds… and a lava lake one block below the last one.

1. **The hook.** The whole slice, sky to bedrock, with **YOU ARE HERE** at the top and **DIAMONDS** near the bottom
   (just above something glowing orange). It zooms in on Steve and he starts digging.
2. **A cave.** He breaks into it and falls: **OUCH**, hearts lost. A **ZOMBIE** shuffles over and stares down the
   hole after him.
3. **An abandoned mineshaft**, with cobwebs, rails, a spawner and cave spiders scuttling over.
4. **Y = 0: deepslate.** The rock goes dark and the swings get heavier.
5. **An amethyst geode.** He drops through the crystals.
6. **The Ancient City.** Every swing sends a vibration to the sculk sensors. The shrieker screams, the screen
   pulses dark, and **THE WARDEN** digs itself out of the floor, roars, and comes for the wall next to the shaft.
7. **Diamonds.** Five of them, the counter dinging up to **x5**.
8. **"wait..."** The music stops. A heartbeat. He mines the block under his feet, slowly. Under it is the lava.
9. **NOOOOO.** The diamonds burn with him. **You died! Score: 5. NEVER DIG STRAIGHT DOWN.** Respawn, and the video
   is back at the start.

**Ready to upload:** [`release/dig-down.mp4`](release/dig-down.mp4). The optional cover is in
[`release/thumbnail.jpg`](release/thumbnail.jpg): a close-up of Steve raising his pickaxe over his last block, the
lava glowing under it, *I DUG STRAIGHT DOWN … BIG MISTAKE*. The same image without text is [`release/thumbnail_clean.jpg`](release/thumbnail_clean.jpg).

| | |
| --- | --- |
| Length | 38 s; it ends on the Respawn button and starts again at the spawn |
| Video | 1080x1920 (9:16), **60 fps**, H.264 High@4.2, 14 Mbps 2-pass, closed GOP (30), BT.709 |
| Audio | AAC-LC 384 kbps, 48 kHz stereo, loudness-normalised to -14 LUFS, true peak -2 dBFS |
| Look | Pixel art at 16 pixels a block, scaled up crisp (96 px a block); smooth lighting from the sky, torches, lava and soul lanterns |

Everything is made in code: no game footage, samples, fonts or AI-generated media. Every block texture, the
characters, the Minecraft-style font, the sounds and the music are drawn or synthesised in `src/`.

## Titles, hooks, pinned comment

* **Titles:** *I Dug Straight Down in Minecraft (X-Ray View)* · *Why You Never Dig Straight Down* · *Digging
  Straight Down to Diamonds…* · *Minecraft but You Can See Underground* · *He Was ONE Block Away*
* **Hooks for the first second** (the opening shows the whole slice, so a line over it lands):
  * "Every Minecraft player knows this rule…"
  * "He found diamonds. Then he mined one more block."
  * "Watch what's under him."
  * "Never dig straight down. Here's why."
* **Pinned comment:** *Did you see the lava before he did?* Or *What would you have done at the last block?* Both
  start arguments ("place a block!", "water bucket!"), which is what the algorithm likes.
* **Tags:** #minecraft #xray #digstraightdown #diamonds #warden #shorts

## Timeline

| Time | Y | What happens |
| --- | --- | --- |
| 0:00 | | The whole slice from the sky to bedrock: YOU ARE HERE, DIAMONDS. Zoom in to Steve |
| 0:01.4 | 64 | He starts digging |
| 0:04.7 | 55 | A CAVE! He breaks through its roof and falls 8 blocks: OUCH, 2.5 hearts gone |
| 0:06.3 | 47 | ZOMBIE!! It shuffles over to the edge of the hole |
| 0:08 to 0:11 | 40 to 22 | Fast digging through stone, coal and iron; he places torches as he goes |
| 0:11.7 | 18 | An abandoned mineshaft: cobwebs, rails, a spawner, cave spiders |
| 0:15.1 | 0 | Y = 0: DEEPSLATE. Darker rock, slower swings |
| 0:19.9 | -20 | An amethyst geode: he falls through the crystals (another 1.5 hearts) |
| 0:23.5 | -38 | The Ancient City. Each swing sends a vibration to a sculk sensor |
| 0:25.2 | -43 | The shrieker screams; the darkness effect pulses |
| 0:26.1 | -47 | THE WARDEN digs out of the floor; it roars at 0:27.8 |
| 0:27.8 | -53 | DIAMONDS!! Five, one after another; the Warden walks to the wall beside the shaft |
| 0:31.7 | -56 | "wait..." The music stops; a heartbeat; the last block takes 1.5 s to mine |
| 0:33.2 | | The block breaks. He falls into the lava: NOOOOO |
| 0:35.3 | | You died! Score: 5. NEVER DIG STRAIGHT DOWN |
| 0:37.7 | | Respawn (and the video starts again) |

## How it's made

| | |
| --- | --- |
| The world | `src/world.py`: a cross-section 91 blocks wide from y = 80 to bedrock at -64: grass and dirt, stone with granite, diorite, andesite and gravel, deepslate and tuff below 0, ores in their real depth bands, and the set pieces along the shaft (a cave, a dungeon, a mineshaft, a geode, an Ancient City with sensors, a shrieker and soul lanterns, the diamond vein, the lava lake) |
| Pixel art | `src/art.py`: every texture (16x16 per block, ores with chunky nuggets, glowing redstone and diamonds, animated lava), sprites (torches, cobwebs, rails, crystals, lanterns, sculk sensors, the shrieker), the ten mining cracks, and the characters in profile |
| The renderer | `src/render.py`: the slice drawn at its own pixel size and scaled up nearest-neighbour, so it stays crisp. Behind air you see the rock's back wall, darker; edges facing air are bevelled. Light floods through the air like the game's (sky light down the shaft, warm light from torches and lava, cold light from soul lanterns), smoothed between blocks, with an x-ray floor so the rock always shows. Glowing texels and a bloom for lava, ores and crystals |
| The story | `src/story.py`: the dig, block by block, scripted from the world itself: how long each block takes depends on its depth, falls into the cave and the geode cost hearts, torches go up as it gets dark, the city hears every swing. The camera keeps Steve in the upper part of the frame so what's below him shows |
| Characters | `src/chars.py`: Steve as a rig (head, body, arms, legs) that swings the pickaxe, falls, lands, cheers and burns; the zombie, cave spiders, the Warden rising out of the floor with its ribs glowing to its heartbeat; dropped items flying into his pocket, debris from every block, vibrations, the shriek |
| HUD | `src/hud.py`: the Y level (like the F3 screen), hearts that shake when low, the diamond counter, captions, and the game's death screen with the Respawn button |
| Sound | `src/sfx.py`: a sound for every swing by material (grass, stone, deepslate, amethyst), crumbling breaks and item pops, falls and hurts, the zombie, spiders, sculk clicks, the shrieker, the Warden's heartbeat and roar, diamond chimes, the lava; a soft piano and strings in the game's style that stop dead on the last block |

## Render it yourself

```bash
pip install -r requirements.txt
cd src
python main.py --out ../output                              # the video + soundtrack (about 0.3 s a frame)
python main.py --out ../stills --frames 0,600,1700 --no-audio  # a few frames as stills
python main.py --out ../output --encode-only                 # redo the soundtrack and the encode
python thumbnail.py ../output/thumbnail.jpg                  # the cover; --no-title for a clean one
```

## Upload tips

* Upload the MP4 as it is: 60 fps, 9:16, under a minute, so YouTube treats it as a Short.
* The game facts on screen: deepslate starts at Y = 0; diamonds are most common near the bottom of the world; the
  Warden is summoned by a sculk shrieker after the sensors hear you; lava lakes sit near the bottom of the world.
  The world itself is drawn for the video, not a real seed.
