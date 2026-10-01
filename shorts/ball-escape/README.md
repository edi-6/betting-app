# Can the Ball Escape All 30 Rings? (YouTube Shorts)

Something new for the channel: one of Shorts' most-watched formats, the **ball escape**. A glowing ball bounces
inside 30 rotating neon rings, each with a gap. Slip through a gap and the ring shatters. **Every bounce plays the next
note of Beethoven's *Für Elise*** (public domain), and every shattered ring rings out. The counter at the top ticks down
from 30 until the last ring breaks in a burst: **ESCAPED!**

**Why it works:** viewers lock on at the first frame (30 rings, one ball, a counter at 30) and stay to see whether it
gets out. The long stretches where it can't find a gap build tension, and then several rings break at once as the
gaps line up (around 0:15, 0:21 and the 0:45 finale). The music is the reward.

**Ready to upload:** [`release/ball-escape.mp4`](release/ball-escape.mp4) (1080x1920, 60 fps, about 48 s, H.264,
AAC at -14 LUFS), and a cover in [`release/thumbnail.jpg`](release/thumbnail.jpg): the opening frame titled *CAN IT
ESCAPE / ALL 30 RINGS*.

## Titles and hooks

* **Titles:** *Can the Ball Escape All 30 Rings?* · *This Ball Plays Für Elise While Escaping 30 Rings* · *Wait for
  the Ending...* · *Satisfying Ball Escape (with Music)*
* **Hooks for the first second:** "Can it escape all 30?" · "Wait for the last ring" · "Comment how many rings you
  think it breaks".
* **Pinned comment:** "Which ring did you think it would get stuck on?" Comments and rewatches are what push this format.

## How it's made

`src/make.py` does it all. It simulates the ball (gravity, bounces off the rings with a little spin from the moving
wall, its speed kept lively) and the rings (each turning its own way, the gaps widening slowly). It picks the seed
whose escape lands near 45 seconds, then draws every frame (neon rings with glow, shards, the ball's trail, the
counter in the Minecraft-style font) and builds the sound from the simulation. Each bounce is a plucked note and a
bell, panned with the ball, each break is a crack and a chime, and the escape is a hit, a chord and a choir over a soft
pad. The synths come from [`../sift-flight/src/audio.py`](../sift-flight/src/audio.py).

```bash
cd src && python make.py ../output      # about 15 minutes on a 4-core CPU
```
