# THE PLAYER WHO NEVER LOGGED OUT: production plan

A 12:46 cinematic Minecraft-style horror ARG, generated entirely from code, like the Shorts in `../../shorts`.
It's 16:9 at 1920x1080 and 24 fps. The protagonist's lines are burned-in subtitles. A timed script and SRT file are
included so a real voice can be recorded over them.

## Lore (never stated in the film, but every clue is consistent with it)

* A world needs two players: one **plays** it and one **watches**.
* NOAH_404 was a player. On day 19 he found out what the world wants, and he became the watcher. He is now
  "underneath" the world.
* The world copies everything a day ahead. The copied houses are dated with the day each player arrived:
  * PLAYER_1: 2018-06-14 (Noah)
  * PLAYER_2: 2020-11-03
  * PLAYER_3: 2025-07-01
  * PLAYER_4: 2026-04-04, which is *tomorrow* (the protagonist)
* In-story today is 2026-04-03. Tomorrow is 4/04.
* The figure with the protagonist's skin is the world's copy of him. When he turns around, the copy takes over the
  playing and he becomes the watcher ("You stayed."). The final card, PLAYER 2 IS ONLINE, points at the viewer.
* Recurring numbers: 404 and 04:04 (timestamps), 19 (days and houses), 2 (players).

## Timeline (as built; each act is a `src/seq_*.py` module)

| # | Time | Sequence | Beats |
| - | ---- | -------- | ----- |
| 0 | 0:00–0:35 | HOOK | Black and silence. Join and leave messages (big centred text, then a fast server log with dates 2018→2026 and 1-frame hidden lines). "PLAYER JOINED THE GAME / NOAH_404", then "... / YOU". Silence. The desktop of an old PC, and a launcher showing NOAH_404's skin, then his own. The world select screen: NOAH_FINAL, "Last played: 1 day ago". Loading. TITLE |
| 1 | 0:35–2:03 | THE PERFECT WORLD | Sunset spawn and "YOU joined the game". Wide beauty shots with villagers, animals, smoke and fog; a tiny figure on the forest edge. POV walk. A house with the furnace still burning. The chest holds a RULES book, one rule per page (5 pages). "You already broke Rule 4." Hard cut |
| 2 | 2:03–3:26 | THE FIRST LIE | Villagers are name-tagged (Tom, Mira, Elias...) and one is NOAH. Interact, and he walks off and is followed (third person shows the protagonist's skin). The map house has a wall of maps (BEFORE, ABANDONED, BURNED, TODAY). A held map shows a red mark at his position, then a second one behind him. A slow turn: nothing. The villager is gone |
| 3 | 3:26–4:59 | THE FOOTSTEPS | Night. Inside the house, the rule 2 memory. Steps outside circle the house; the windows show nothing (a torch moves by itself far away). Silence. Steps inside, on the floor, then above (there's no upstairs). A second shadow in the moonlight on the floor. The ceiling block cracks and breaks: nothing. A sign falls: "GOOD. YOU DIDN'T TURN AROUND." Hold |
| 4 | 4:59–6:39 | THE UNDERGROUND ROOM | A clock past midnight ("Rule one..."). A trapdoor under the carpet leads down a long torch-lit stair. Reveal: a cavern with dozens of copies of the house, date signs 2018→tomorrow. Inside the TODAY copy it's exact, down to the broken ceiling block and the sign. The wall reads YOU WILL SLEEP HERE TONIGHT. Second bed: "This bed is occupied" |
| 5 | 6:39–8:02 | THE RECORDINGS | A row of lecterns, DAY 1 to DAY 19. Fragments: days 1–4 readable, then a fast flip through 5–17 (pause to read). DAY 18 is signed "by YOU". DAY 19: "I found out what it wants." with a missing page. Chests; a torn page: "It doesn't want to kill us." ... "It wants another player." Then "NOAH_404 joined the game" |
| 6 | 8:02–9:18 | THE SECOND PLAYER | Chat: "don't move", "it can see your screen" ("How is he messaging me?"; the Tab list shows only YOU), "I'm not in your world." ... "I'm underneath it." A pit and a ladder into darkness, with a held torch. A vast chamber with a still player at the centre (Noah's skin from behind). "don't look at him". It turns slowly, and it's the protagonist's skin. FREEZE |
| 7 | 9:18–10:43 | THE WORLD CHANGES | Unfreeze on the surface: a wrong green sky, an enormous moon, moved houses, dead spruces, every villager named NOAH. Paths loop back. Signs: LEFT LEFT LEFT, then YOU'RE GOING THE WRONG WAY. A giant crater at spawn with a house at the bottom: his real room. The monitor shows THIS VIDEO (recursive) |
| 8 | 10:43–11:28 | THE REVEAL | The PLAYER 2 book (5 pages). "It's not predicting me. It's recording me. Before I do it." Footsteps right behind him. DON'T TURN AROUND. He turns: cut to black |
| 9 | 11:28–12:17 | THE MOST DISTURBING MOMENT | Silence, then "That's not me." A third-person camera moving by itself; the keystroke overlay shows S and ESC held while he walks forward. The door opens on his real room, where another him sits at the PC, and the screen shows the video player at the current time. It turns to the lens: "You stayed." |
| 10 | 12:17–12:46 | FINAL ARG HOOK | The desktop, normal, with no Minecraft. A new folder NOAH_FINAL holds PLAYER_1..4.mp4 (dates match the houses; PLAYER_4 is dated tomorrow and runs exactly this long). The cursor hovers PLAYER_1. Toasts: "NOAH_404 joined the game", "NOAH_404: he's watching too". Hard black; after 4 s, PLAYER 2 IS ONLINE |

## Hidden clues (for rewatchers)

* The hook log's timestamps are all 04:04:xx, and its dates match the house signs. There are 1-frame lines:
  "<NOAH_404> is this recording?" and "??? joined the game".
* On the world select screen, the folder name is "NOAH_FINAL (2)", so this is a copy.
* The rules book is by NOAH_404; DAY 18 is signed "by YOU".
* The clock in the house reads midnight at sunset, and the furnace is still lit.
* The maps: BEFORE shows only the house. BURNED shows the crater at spawn. TODAY carries a tiny date. Each map's
  corner date matches a house sign.
* A figure on the forest edge in the opening wide shot; a torch moving by itself at night; a second shadow in the
  moonlight; a player on the path ahead in the changed world who is gone behind a tree.
* The underground copy dated today has the broken ceiling block and the fallen sign; one house is signed
  NOAH_404 instead of a date.
* The Tab list shows only YOU while NOAH_404 is typing.
* In the changed world, every villager is NOAH.
* The monitor's timecode matches the video's own time; the final folder's PLAYER_4.mp4 runs exactly as long as
  the film.
* 1-frame inserts (the chamber figure, a DAY 20 page, the NOAH_FINAL folder, HE STAYED).
* The rules-page lullaby comes back reversed under the finale.

## Engine (src/)

| Module | What |
| ------ | ---- |
| `blocks.py` | block registry: ids, per-face textures, shapes (cube / model / cross / none), opacity, light emission; the non-cube models (torch, lantern, bed, chest, door, fence, pane, sign, lectern, stairs, carpet, item frame, crops...) |
| `textures.py` | 16x16 procedural block textures, emission masks and item icons |
| `voxel.py` | the world grid (uint16 ids + per-voxel state), editing helpers, the vectorised mesher (culled faces, merged runs, 16³ sections), light propagation into a light volume |
| `worldgen.py` | terrain, the village and its houses, the stair and the cavern of copies, the chamber, the changed world with the crater and his real house, the finale |
| `maps.py`, `decals.py` | the maps on the wall (and in his hands), sign text, item frames, painted words and posters |
| `skins.py`, `entities.py` | procedural 64x64 skins; box models (player, villager, animals), poses, walking, first-person arms and held items |
| `gfx.py`, `scene.py`, `sky.py`, `clouds_gl.py`, `looks.py` | the deferred GL 4.3 renderer on llvmpipe (G-buffer, sun/moon shadows with a per-shot penumbra, light volume, SSAO, volumetric fog, particles, depth of field, bloom, grade, FXAA), baked sky panoramas and the lighting presets |
| `screen.py` | the in-game monitor: a glowing quad textured every frame with a video player |
| `ui.py`, `pixelfont.py`, `post.py` | the game's interface (HUD, chat, tab list, books, chests, world select, loading), the old desktop (launcher, explorer, notifications), keystrokes, subtitles; grain, VHS, tearing, freeze, fades |
| `anim.py`, `common.py` | keyframes, paths, the first-person camera, walkers; shared props, villagers, name tags, scene helpers |
| `seq_*.py`, `story.py` | the eleven acts as shot lists (cameras, actors, props, overlays, subtitles, sound cues) and the film that joins them |
| `film.py` | the timeline; `render` (resumable per-shot 3D intermediates), `compose` (interface, subtitles and film look, twice: with and without subtitles), `probe`, `cues`, `srt` |
| `audio.py` | the soundtrack, synthesised from the cue sheet |
| `deliver.py`, `thumbnail.py`, `webpage.py` | final files, GitHub parts, the web stream, thumbnails, the release page |

## Making it

```
cd src
python film.py render            # 3D intermediates into cache/frames (about 1.5 h at 1080p on 4 CPU cores)
python film.py cues && python audio.py
python film.py compose           # output/film_noaudio.mp4 and output/film_clean_noaudio.mp4
python film.py srt               # release/voiceover.srt
python thumbnail.py              # release/thumb_*.jpg
python deliver.py git            # release/film/: the films in parts under 95 MB, with join scripts
```
Add `--preview` to `render` / `compose` for a half-resolution pass, and `python film.py probe DIR --shots a,b` for
composed stills of chosen shots.

## Status

Done: every act, the soundtrack, the release kit (`release/`).
