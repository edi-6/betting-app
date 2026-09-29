# THE PLAYER WHO NEVER LOGGED OUT: production plan

A 13–14 minute cinematic Minecraft-style horror ARG, generated entirely from code, like the Shorts in `../../shorts`.
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

## Timeline (targets; each sequence is a module in `src/story/`)

| # | Time | Sequence | Beats |
| - | ---- | -------- | ----- |
| 0 | 0:00–0:32 | HOOK | Black and silence. Join and leave messages (big centred text, then a fast server log with dates 2018→2026 and 1-frame hidden lines). "PLAYER JOINED THE GAME / NOAH_404", then "... / YOU". Silence. The desktop of an old PC, and a launcher showing NOAH_404's skin, then his own. The world select screen: NOAH_FINAL, "Last played: 1 day ago". Loading. TITLE |
| 1 | 0:32–2:00 | THE PERFECT WORLD | Sunset spawn and "YOU joined the game". Wide beauty shots with villagers, animals, smoke and fog; a tiny figure on the forest edge. POV walk. A house with the furnace still burning. The chest holds a RULES book, one rule per page (5 pages). "You already broke Rule 4." Hard cut |
| 2 | 2:00–3:25 | THE FIRST LIE | Villagers are name-tagged (Tom, Mira, Elias...) and one is NOAH. Interact, and he walks off and is followed (third person shows the protagonist's skin). The map house has a wall of maps (BEFORE, ABANDONED, BURNED, TODAY). A held map shows a red mark at his position, then a second one behind him. A slow turn: nothing. The villager is gone |
| 3 | 3:25–5:00 | THE FOOTSTEPS | Night. Inside the house, the rule 2 memory. Steps outside circle the house; the windows show nothing (a torch moves by itself far away). Silence. Steps inside, on the floor, then above (there's no upstairs). A second shadow in the moonlight on the floor. The ceiling block cracks and breaks: nothing. A sign falls: "GOOD. YOU DIDN'T TURN AROUND." Hold |
| 4 | 5:00–6:35 | THE UNDERGROUND ROOM | A clock past midnight ("Rule one..."). A trapdoor under the carpet leads down a long torch-lit stair. Reveal: a cavern with dozens of copies of the house, date signs 2018→tomorrow. Inside the TODAY copy it's exact, down to the broken ceiling block and the sign. The wall reads YOU WILL SLEEP HERE TONIGHT. Second bed: "This bed is occupied" |
| 5 | 6:35–8:05 | THE RECORDINGS | A row of lecterns, DAY 1 to DAY 19. Fragments: days 1–4 readable, then a fast flip through 5–17 (pause to read). DAY 18 is signed "by YOU". DAY 19: "I found out what it wants." with a missing page. Chests; a torn page: "It doesn't want to kill us." ... "It wants another player." Then "NOAH_404 joined the game" |
| 6 | 8:05–9:35 | THE SECOND PLAYER | Chat: "don't move", "it can see your screen" ("How is he messaging me?"; the Tab list shows only YOU), "I'm not in your world." ... "I'm underneath it." A pit and a ladder into darkness, with a held torch. A vast chamber with a still player at the centre (Noah's skin from behind). "don't look at him". It turns slowly, and it's the protagonist's skin. FREEZE |
| 7 | 9:35–11:05 | THE WORLD CHANGES | Unfreeze on the surface: a wrong green sky, an enormous moon, moved houses, dead spruces, every villager named NOAH. Paths loop back. Signs: LEFT LEFT LEFT, then YOU'RE GOING THE WRONG WAY. A giant crater at spawn with a house at the bottom: his real room. The monitor shows THIS VIDEO (recursive) |
| 8 | 11:05–12:15 | THE REVEAL | The PLAYER 2 book (5 pages). "It's not predicting me. It's recording me. Before I do it." Footsteps right behind him. DON'T TURN AROUND. He turns: cut to black |
| 9 | 12:15–13:25 | THE MOST DISTURBING MOMENT | Silence, then "That's not me." A third-person camera moving by itself; the keystroke overlay shows S and ESC held while he walks forward. The door opens on his real room, where another him sits at the PC, and the screen shows the video player at the current time. It turns to the lens: "You stayed." |
| 10 | 13:25–13:50 | FINAL ARG HOOK | The desktop, normal, with no Minecraft. A new folder NOAH_FINAL holds PLAYER_1..4.mp4 (dates match the houses; PLAYER_4 is dated tomorrow and runs exactly this long). The cursor hovers PLAYER_1. Toasts: "NOAH_404 joined the game", "NOAH_404: he's watching too". Hard black; after 4 s, PLAYER 2 IS ONLINE |

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
| `blocks.py` | block registry: ids, per-face textures, shape (cube / model / cross / none), opacity, light emission |
| `textures.py` | 16x16 procedural block textures + emission masks, item icons, GUI pieces |
| `models.py` | non-cube block models (torch, lantern, bed, chest, door, fence, pane, sign, lectern, stairs, slab, carpet, item frame, crops, flowers...) |
| `voxel.py` | World grid (uint16 ids + per-voxel state), editing helpers, vectorised mesher (culled faces, run merged, 16³ sections), light propagation -> RG8 light volume |
| `worldgen.py` | terrain, village, houses, underground, chamber, changed world, crater + real room |
| `entities.py` | box models with 64x64 skins (player, villager, cow, sheep, pig, chicken), poses, walk cycles |
| `skins.py` | procedural skins |
| `renderer.py` | deferred GL 4.3 on llvmpipe: G-buffer (albedo, normal+mat, extra), sun/moon shadow cascades, light volume, emission, SSAO, volumetric fog with shafts and torch halos, sky presets, particles, bloom, grade, FXAA |
| `sky.py` | panoramas (sunset, night, wrong sky) with baked clouds |
| `ui/` | fonts, Minecraft GUI (chat, book, chest, HUD, tab list, world select, loading), desktop (launcher, explorer, toasts), keystrokes, subtitles, video player |
| `post.py` | grain, VHS, tearing, freeze, fades |
| `story/` | sequences: actor tracks, cameras, props, overlays, sound cues |
| `render.py` | resumable per-shot rendering to intermediates; preview mode |
| `compose.py` | 3D intermediates + UI + post -> final frames -> encode with audio |
| `audio/` | synthesised ambience, foley, UI sounds, music, drones; cue-driven mix |

## Status

See the task list; this file is updated as sequences land.
