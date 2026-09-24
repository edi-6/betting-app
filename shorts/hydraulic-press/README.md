# Hydraulic Press vs Minecraft: Can It Crush Bedrock? (YouTube Shorts)

The seventh video in the series, after [`../steve-vs-swords`](../steve-vs-swords),
[`../creeper-vs-tnt`](../creeper-vs-tnt), [`../zombie-vs-arrows`](../zombie-vs-arrows),
[`../warden-vs-anvils`](../warden-vs-anvils), [`../last-giant-standing`](../last-giant-standing) and
[`../black-hole`](../black-hole). This one has a new concept and look: the hydraulic-press format, done in
Minecraft. Nine blocks go under the press one after another, in hotbar order. Each one gives way differently:

| Block | What happens | Pressure |
| --- | --- | --- |
| Grass Block | Squashes and crumbles into dirt | 6 t |
| Glass | Shatters instantly (slow motion) | 2 t |
| Melon | Bulges and bursts, juice everywhere | 9 t |
| Slime Block | Squashes flat, then throws the ram back up (**BOING!**), then gets splatted | 20 t |
| TNT | Primes: flashes white and swells like in the game, then blows up (slow motion) | 15 t |
| Chest | Bursts and spills its loot: diamonds, gold, iron, emeralds, an apple | 25 t |
| Block of Diamond | Cracks through Minecraft's destroy stages, then shatters and pops out diamonds | 400 t |
| Obsidian | Grinds and sparks, purple particles, and finally breaks | 2,500 t |
| **Bedrock** | Doesn't give. The pressure climbs to 100,000 t, the hoses burst and spray oil, the alarm goes, and the **press** blows apart. The bedrock is untouched: **UNBREAKABLE** | 100,000 t |

The video opens on a flash-forward of the bedrock straining at full pressure, then rewinds to the grass block.
From the first frame the hotbar shows bedrock waiting in the last slot, which keeps people watching to the end.

**Ready to upload:** [`release/hydraulic_press.mp4`](release/hydraulic_press.mp4), with an optional cover image in
[`release/thumbnail.jpg`](release/thumbnail.jpg).

| | |
| --- | --- |
| Length | 32.2 s (1.5 s flash-forward cold open and a 0.4 s rewind, then the nine blocks) |
| Video | 1080x1920 (9:16), 30 fps, H.264 High, 12 Mbps 2-pass, closed GOP (15), BT.709 |
| Audio | AAC-LC (384 kbps target), 48 kHz stereo, loudness-normalised to -14 LUFS, peak -1.2 dBFS |

As before, everything is generated from code: no stock footage, samples, fonts or AI-generated media. The
Minecraft-style font, textures, icons and sounds are all drawn or synthesised in `src/`.

## Timeline

| Time | Segment | What happens |
| --- | --- | --- |
| 0:00 | Cold open | Bedrock under the ram, sparks flying, the pressure racing from 7,000 to 50,000 t, the hoses bursting, the press's own health bar draining. No text, so the hook can go on top |
| 0:01.5 | Rewind | VHS rewind back to the start (the music stops like a tape) |
| 0:01.9 | Grass Block | The press in its studio, then straight in. **CRUSHED!** on the first drop of the beat (0:03.0) |
| 0:03.7 | Glass | **SHATTERED!** (0:04.6), shards flying at the lens in slow motion |
| 0:06.2 | Melon | **SPLAT!** (0:07.4) |
| 0:08.1 | Slime Block | Squashed flat, then **BOING!** (0:09.7) it throws the ram back up; **SQUASHED!** (0:10.6) |
| 0:11.3 | TNT | Primed (0:12.5): flashing white, swelling. **BOOM!** (0:13.4), slow motion |
| 0:14.5 | Chest | **JACKPOT!** (0:15.9): the loot flies out |
| 0:16.8 | Block of Diamond | Cracks stage by stage, **CRUSHED!** at 400 t (0:18.7) |
| 0:19.8 | Obsidian | Sparks and purple particles, **CRUSHED!** at 2,500 t (0:22.4) |
| 0:22.8 | Bedrock | The beat drops out. Contact (0:23.6), the pressure climbs, the Hydraulic Press gets its own boss bar and it drains |
| 0:25.4 | The hoses burst | Oil sprays, the redstone lamps flash red, the alarm goes; the pressure keeps climbing to 100,000 t |
| 0:27.0 | The press explodes | On the bar line: the drop. Slow motion, the beat comes back in |
| 0:30.0 | Survivor | The smoke clears: the bedrock hasn't moved. **UNBREAKABLE** (0:30.2), then *Challenge Complete! Unbreakable* (0:30.7) until the end (0:32.2) |

Every block gives way exactly on the beat (eighth notes at 130 BPM) and the press breaks on a bar line.
`src/timeline.py` nudges when each block goes on until that is true, and the camera cuts snap to the beat as well.

## What's new in this one

| | |
| --- | --- |
| The press | `src/scene.py`: a dark studio (deepslate floor, stone brick walls, redstone lamps) and a hydraulic press (base, platen, columns, beam, chrome cylinder, hoses); the ram and its rod are voxels so they can break |
| The blocks | `src/pixelart.py`, `src/items.py`: all nine blocks drawn as 16x16 pixel art and built from 16³ voxels (one per texture pixel). Glass keeps only its frame and glints, so you can see through it. The chest is hollow and full of loot items (extruded 16x16 sprites, the way the game draws items) |
| The crush | `src/press.py`: each material has its own script (how far it squashes and bulges, when it gives, how hard the pieces fly) and effects (dust, juice, slime, sparkles, purple particles, sparks, oil). Broken blocks become rigid voxel chunks and crumbs that collide with the platen, the press, the floor and the moving ram, which squeezes them out sideways. Diamond and obsidian crack through the ten destroy stages first |
| Minecraft UI | `src/pixelfont.py`, `src/hud.py`: a from-scratch pixel font with the game's drop shadow; the **boss bar** (the block under test is the boss; its bar is its integrity), the pressure in tons shown as the **XP level** over the XP bar (log scale), the **hotbar** with isometric block icons (crushed blocks crossed out, the survivor gets the enchantment glint), **/title**-style result titles, and the **advancement toast** |
| Look | `src/renderer.py`: depth of field for a macro product-shot look (two passes: circle of confusion, then a golden-angle gather), coloured point lights (redstone lamps, a front fill, the red alarm), squash-and-bulge deformation of the block under the ram |
| Sound | `src/audio.py`: drift phonk at 130 BPM in C# minor (distorted 808s, claps, metallic hats and the pitched 808 cowbell riff), with a breakdown when the bedrock goes on and a drop when the press blows. Sound effects: the hotbar click, place sounds, the pump's whine and oil hiss as the ram moves, metal groaning, one break sound per material, the TNT fuse, item pops, sparks grinding, a hose bursting, the alarm, the explosion and shrapnel, and the challenge fanfare. Slow motion lowers the pitch and muffles the music |

Shared with the earlier projects: the deferred renderer (`renderer.py`), the explosion and dust effects
(`explosion_vfx.py`, `vfx.py`), the grass and TNT textures (`textures.py`, `tnt.py`). Dev helpers: `check_cams.py`
(a contact sheet of every shot with the HUD), `qa.py` (black, corrupt or flickering frames) and `thumbnail.py`.

## Render it yourself

```bash
pip install -r requirements.txt
# Linux also needs Mesa's EGL + llvmpipe, e.g.: apt install libegl1 libegl-mesa0 libgl1-mesa-dri
cd src
python main.py --out ../output             # delivered quality (~1.7 s a frame on a CPU renderer)
python main.py --out ../preview --preview  # quick half-resolution check
python main.py --out ../stills --stills --every 12 --shots cold,diamond,survivor   # look development
python main.py --out ../sound --cues-only  # sound work: the cue sheet and the audio, no frames rendered
python check_cams.py ../cams.png
python qa.py ../output/final.mp4 ../output/cues.json
python thumbnail.py ../output/thumbnail.jpg
python timeline.py                         # when everything happens, in sim and video time, and on which beat
```

## Customising

* The line-up and its order: `KINDS` and `T0_NOMINAL` in `src/timeline.py`, and the blocks' art in
  `make_items()` in `src/pixelart.py`.
* How each material behaves (how much it squashes, when it gives, the pressure it takes, how the pieces fly):
  `MATS` in `src/press.py`.
* The edit (cameras, slow motion, the cold open): `shots()`, `SLOW` and `COLD` in `src/timeline.py`.
* The titles and bar colours: `TITLES` and `BOSS_COL` in `src/main.py`.
* The track (tempo, riff, 808 line): `BPM` in `src/timeline.py`, `RIFF`, `BASS` and `Beat` in `src/audio.py`.

## Upload tips

* Upload the MP4 as-is. It already meets YouTube's recommended settings for 1080p. It is under a minute and 9:16,
  so YouTube treats it as a Short automatically.
* The first 1.5 s (the bedrock under pressure) is left clean for a hook line or a voice-over.
