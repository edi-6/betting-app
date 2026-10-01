# Which Color Wins? (YouTube Shorts)

A new format for the channel: a **color war**. Red, blue, green and yellow each start with a quarter of the board and
two balls. A ball bounces through enemy tiles and flips them to its own color, so territory swings back and forth.
Live percentage bars and a 45-second countdown run the whole time, and at 0:00 the color holding the most of the
board wins.

**Why it works:** viewers pick a color in the first second and watch to the end to see if it wins. Then they comment,
and they rewatch to see where it turned. The battle used here was chosen from 40 simulated ones for the most lead
changes and the closest finish. Every flipped tile plays a note (its column picks the pitch) over a beat, the last 10
seconds turn the clock red, and there's a snare roll into the result.

**Ready to upload:** [`release/color-war.mp4`](release/color-war.mp4) (1080x1920, 60 fps, about 48 s, -14 LUFS),
cover [`release/thumbnail.jpg`](release/thumbnail.jpg) (the opening board, *PICK A COLOR*).

* **Titles:** *Which Color Wins?* · *Pick a Color Before It Starts* · *This Color War Came Down to the Last Second*
* **Hooks:** "Pick a color NOW" · "Comment your color before it ends" · "Don't skip the last 5 seconds".
* **Pinned comment:** "Which color did you pick? 🔴🔵🟢🟡"

Made by `src/make.py` (`cd src && python make.py ../output`): the simulation, the frames, the sound (synths from
[`../sift-flight/src/audio.py`](../sift-flight/src/audio.py)) and the encode, in a few minutes on a CPU.
