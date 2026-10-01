# Find the Odd Block (YouTube Shorts)

A new format for the channel, still Minecraft: a **spot-the-odd-one-out challenge**. Five grids of Minecraft blocks,
each with one different block hidden in it, 5 seconds per level with a countdown, and then the reveal: everything
dims, a red ring closes on the odd block, and *FOUND IT?* The grids grow and the blocks get more alike every level
(the pairs are picked by measuring how alike their textures are), so level 5 is genuinely hard. It ends on *HOW MANY
DID YOU GET? COMMENT YOUR SCORE*.

**Why it works:** viewers play along from the first frame. Everyone wants to beat the timer, they rewatch the levels
they missed, and "comment your score" turns into thousands of comments.

**Ready to upload:** [`release/odd-block.mp4`](release/odd-block.mp4) (1080x1920, 60 fps, about 37 s, -14 LUFS:
a light beat, a tick each second, faster in the last two, a ding on every reveal), cover
[`release/thumbnail.jpg`](release/thumbnail.jpg) (level 1, *FIND THE ODD BLOCK*).

* **Titles:** *Find the Odd Block in 5 Seconds* · *Only Real Minecraft Players Find All 5* · *Can You Beat Level 5?*
* **Hook:** "You have 5 seconds" or "Pause if you need to... cheater."
* **Pinned comment:** "Comment your score /5 👇"

Made by `src/make.py` (`cd src && python make.py ../output`), about a minute on a CPU. It uses the block textures from
[`../sift-flight/src/textures.py`](../sift-flight/src/textures.py).
