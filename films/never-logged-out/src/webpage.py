"""The release page (a private web page): the film to watch and download, his lines for a voice-over, the publishing
kit and the clue guide, built from the release folder's documents.

    python webpage.py DIR        # DIR/index.html (the film's pieces and thumbnails go next to it)
"""
import html
import json
import os
import re
import sys

import film as FM

REL = os.path.join(FM.ROOT, 'release')


def inline(md):
    """Escape, then **bold** and `code`."""
    s = html.escape(md)
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'`(.+?)`', r'<code>\1</code>', s)
    return s


def srt_lines():
    txt = open(os.path.join(REL, 'voiceover.srt')).read()
    return txt


def voiceover():
    """[(heading, mood, [(time, line, note)])] from VOICEOVER.md."""
    acts = []
    for ln in open(os.path.join(REL, 'VOICEOVER.md')).read().splitlines():
        m = re.match(r'### (\d+:\d\d) (.+?): (.+)$', ln)
        if m:
            acts.append((m.group(1), m.group(2), m.group(3), []))
            continue
        m = re.match(r'\| (\d+:\d\d\.\d) \| (.+?) \|(.*?)\|$', ln)
        if m and acts:
            acts[-1][3].append((m.group(1), m.group(2).strip(), m.group(3).strip()))
    return acts


def clues():
    txt = open(os.path.join(REL, 'ARG_CLUES.md')).read()
    lore = txt.split("## What's \"really\" going on")[1].split('## Where everything is hidden')[0]
    rows = re.findall(r'^\| ([\d:–/ +]+?) \| (.+?) \|$', txt, re.M)
    return lore, rows


def lore_html(md):
    out = []
    inlist = False
    table = []
    for ln in md.strip().splitlines():
        s = ln.strip()
        if s.startswith('|'):
            if '---' not in s:
                table.append([c.strip() for c in s.strip('|').split('|')])
            continue
        if table:
            out.append('<table class="mini"><tbody>' + ''.join(
                '<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>' for r in table[1:]) + '</tbody></table>')
            table = []
        if s.startswith('- '):
            if not inlist:
                out.append('<ul>')
                inlist = True
            out.append(f'<li>{inline(s[2:])}</li>')
        elif s:
            if inlist:
                out.append('</ul>')
                inlist = False
            out.append(f'<p>{inline(s)}</p>')
    if inlist:
        out.append('</ul>')
    return '\n'.join(out)


def youtube():
    txt = open(os.path.join(REL, 'YOUTUBE.md')).read()
    blocks = re.findall(r'```\n(.*?)```', txt, re.S)
    alts = re.findall(r'^- (.+)$', txt.split('Alternatives')[1].split('## Thumbnails')[0], re.M)
    follow = re.findall(r'^- \*\*(.+?)\*\* (.+)$', txt.split('## Follow-ups')[1].split('## Tags')[0], re.M)
    tags = txt.split('## Tags')[1].strip().replace('\n', ' ')
    return dict(description=blocks[0].strip(), chapters=blocks[1].strip(), pinned=blocks[2].strip(), alts=alts,
                follow=follow, tags=tags)


PAGE = r'''<title>The Player Who Never Logged Out</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=JetBrains+Mono:wght@400;500&family=Pixelify+Sans:wght@500;700&display=swap">
<style>
/* Layout: one dark column, the screening room at 04:04. The film first, lit like the moonlit window of act 3;
   everything else (his lines, the publishing kit, the clues) waits below in three tabs. */
:root {
  color-scheme: dark;
  --night: #0b0e14;
  --wall: #121722;
  --line: #242b39;
  --paper: #e8e3d6;
  --ash: #8e97a8;
  --moon: #b3c4e4;
  --chat: #ffff55;
  --warn: #e0806a;
  --display: "Pixelify Sans", "Courier New", ui-monospace, monospace;
  --body: "Atkinson Hyperlegible", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
  --s-1: 0.8125rem; --s0: 1rem; --s1: 1.25rem; --s2: 1.6rem;
}
html { background: var(--night); }
body { background: var(--night); color: var(--paper); font: 400 var(--s0)/1.6 var(--body); margin: 0; }
.wrap { max-width: 1000px; margin: 0 auto; padding-inline: 16px; padding-block: 28px 64px; display: grid;
  grid-template-columns: minmax(0, 1fr); gap: 28px; }
.wrap > * { min-width: 0; }
a { color: var(--moon); }
code { font-family: var(--mono); font-size: 0.9em; color: var(--moon); }
:focus-visible { outline: 2px solid var(--moon); outline-offset: 2px; }

header { display: grid; gap: 10px; }
.log { font: 400 var(--s-1)/1.4 var(--mono); color: var(--chat); margin: 0; overflow-wrap: anywhere; }
h1 { font: 700 clamp(2rem, 6.4vw, 4rem)/1.02 var(--display); letter-spacing: 0.02em; margin: 0; text-wrap: balance; }
.meta { margin: 0; color: var(--ash); font-size: var(--s-1); font-variant-numeric: tabular-nums; }

.screen { position: relative; aspect-ratio: 16 / 9; max-width: 100%; background: #000; border: 1px solid var(--line);
  box-shadow: 0 0 0 1px #000, 0 24px 80px -20px rgba(179, 196, 228, 0.28); }
.screen video, .screen img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; background: #000; }
.watch { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); display: inline-flex; align-items: center;
  gap: 12px; padding: 14px 22px; background: rgba(11, 14, 20, 0.78); color: var(--paper); border: 1px solid var(--moon);
  font: 700 var(--s1) var(--display); letter-spacing: 0.04em; cursor: pointer; }
.watch::before { content: ""; border-left: 14px solid var(--paper); border-top: 9px solid transparent; border-bottom: 9px solid transparent; }
.watch:hover { background: rgba(11, 14, 20, 0.92); }

.bar { display: flex; flex-wrap: wrap; gap: 12px 18px; align-items: center; }
.btn { font: 700 var(--s0) var(--body); padding: 10px 18px; border: 1px solid var(--line); background: var(--wall);
  color: var(--paper); cursor: pointer; }
.btn:hover { border-color: var(--moon); }
.btn.primary { background: var(--paper); color: var(--night); border-color: var(--paper); }
.btn.primary:hover { background: #fff; }
.btn.small { font-size: var(--s-1); padding: 6px 12px; }
.btn[disabled] { opacity: 0.5; cursor: default; }
.status { color: var(--ash); font-size: var(--s-1); font-variant-numeric: tabular-nums; min-width: 0; }
progress { width: 180px; height: 6px; accent-color: var(--moon); }
.note { margin: 0; color: var(--ash); font-size: var(--s-1); max-width: 70ch; }

.tabs { display: flex; flex-wrap: wrap; gap: 4px; border-bottom: 1px solid var(--line); }
.tabs button { font: 500 var(--s0) var(--display); letter-spacing: 0.03em; background: none; color: var(--ash);
  border: 0; border-bottom: 2px solid transparent; padding: 10px 14px; cursor: pointer; }
.tabs button[aria-selected="true"] { color: var(--paper); border-bottom-color: var(--moon); }
section[role="tabpanel"] { display: grid; gap: 22px; }
h2 { font: 700 var(--s2) var(--display); margin: 0; letter-spacing: 0.02em; }
h3 { font: 500 var(--s1) var(--display); margin: 0; letter-spacing: 0.02em; }
.lead { margin: 0; max-width: 68ch; }
.act { display: grid; gap: 8px; }
.act .mood { color: var(--ash); font-size: var(--s-1); margin: 0; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.95rem; }
td, th { text-align: left; vertical-align: top; padding: 7px 12px 7px 0; border-top: 1px solid var(--line); }
th { font: 500 var(--s-1) var(--mono); color: var(--ash); text-transform: uppercase; letter-spacing: 0.06em; }
td.t { font: 500 var(--s-1)/1.9 var(--mono); color: var(--moon); white-space: nowrap; width: 5.5rem; font-variant-numeric: tabular-nums; }
td.n { color: var(--ash); font-size: var(--s-1); min-width: 12rem; }
td.line { min-width: 14rem; }
table.mini td { border: 0; padding: 2px 16px 2px 0; font-variant-numeric: tabular-nums; }

.kit { display: grid; gap: 18px; }
.block { display: grid; gap: 8px; min-width: 0; }
.block header { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
pre { margin: 0; padding: 14px 16px; background: var(--wall); border: 1px solid var(--line); font: 400 var(--s-1)/1.6 var(--mono);
  white-space: pre-wrap; overflow-wrap: anywhere; color: var(--paper); }
.thumbs { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
figure { margin: 0; display: grid; gap: 8px; min-width: 0; }
figure img { width: 100%; aspect-ratio: 16 / 9; object-fit: cover; border: 1px solid var(--line); background: #000; }
figcaption { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; font-size: var(--s-1); color: var(--ash); }
ul { margin: 0; padding-left: 1.2em; display: grid; gap: 6px; }
.gate { display: grid; gap: 12px; justify-items: start; padding: 20px; border: 1px dashed var(--line); }
.gate p { margin: 0; max-width: 64ch; }
.spoil { color: var(--warn); }
@media (max-width: 560px) {
  .watch { font-size: var(--s0); padding: 10px 16px; }
  /* his lines: the note goes under the line instead of a third column */
  .act table, .act tbody { display: block; }
  .act tr { display: grid; grid-template-columns: 4.2rem minmax(0, 1fr); }
  .act td.t { width: auto; }
  .act td.line, .act td.n { min-width: 0; }
  .act td.n { grid-column: 2; border-top: 0; padding-top: 0; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
  <header>
    <p class="log">[2026-04-03 04:04:04] [Server thread/INFO]: YOU joined the game</p>
    <h1>THE PLAYER WHO NEVER LOGGED OUT</h1>
    <p class="meta">A Minecraft horror ARG · 12:46 · 1920 × 1080 · 24 fps · stereo, −16 LUFS · 72 lines for a voice-over</p>
  </header>

  <div class="screen" id="screen">
    <img id="poster" src="poster.jpg" alt="Moonlight from a window on a wall, and two shadows in it">
    <video id="player" controls playsinline preload="none" hidden></video>
    <button class="watch" id="watch" type="button">Watch here</button>
  </div>

  <div class="bar">
    <button class="btn primary" id="dl" type="button">Download the film, no subtitles (MP4)</button>
    <progress id="prog" max="1" value="0" hidden></progress>
    <span class="status" id="status" role="status"></span>
  </div>
  <p class="note">This is the clean film, without his subtitles, ready for your voice-over. While you watch here,
    his lines show as captions; turn them off with the player's CC button. The same lines are in the voice-over script
    below, and in an SRT file for YouTube captions. The version with his lines burned in is in the repository under
    <code>films/never-logged-out/release/film/</code>, split into parts, with a script to join them.</p>

  <div class="tabs" role="tablist">
    <button role="tab" id="tab-vo" aria-controls="vo" aria-selected="true" type="button">Voice-over script</button>
    <button role="tab" id="tab-publish" aria-controls="publish" aria-selected="false" type="button">Publishing kit</button>
    <button role="tab" id="tab-clues" aria-controls="clues" aria-selected="false" type="button">Clue guide</button>
  </div>

  <section role="tabpanel" id="vo" aria-labelledby="tab-vo">
    <p class="lead">His 72 lines, timed to the film. He is a real person playing alone at 4 AM, not a narrator: close to
      the microphone, quiet, never screaming. From the footsteps on, most lines are nearly whispered. The mix already
      dips the ambience and music about 3 dB under each line.</p>
    <div class="bar">
      <button class="btn small" id="copy-srt" type="button">Copy the SRT captions</button>
      <button class="btn small" id="save-srt" type="button">Save the SRT as text</button>
      <span class="status" id="srt-status" role="status"></span>
    </div>
    __VO__
    <p class="note">“You stayed.” at 12:13 is not his voice. It appears in the chat, typed by the other him. Leave it
      unvoiced; the silence after it is the point.</p>
    <pre id="srt" hidden>__SRT__</pre>
  </section>

  <section role="tabpanel" id="publish" aria-labelledby="tab-publish" hidden>
    <div class="kit">
      <div class="block">
        <h2>Title</h2>
        <p class="lead"><strong>THE PLAYER WHO NEVER LOGGED OUT</strong>. For YouTube's Test &amp; compare:</p>
        <ul>__ALTS__</ul>
      </div>
      <div class="block">
        <h2>Thumbnails</h2>
        <div class="thumbs">
          <figure><img src="thumb_a.jpg" alt="Two shadows on a moonlit wall, text: 2 shadows. 1 player."><figcaption><span>A · two shadows, the quietest option</span><button class="btn small" data-save="thumb_a.jpg" type="button">Save</button></figcaption></figure>
          <figure><img src="thumb_b.jpg" alt="The figure with his face under blue light, text: Don't look at him"><figcaption><span>B · the turn</span><button class="btn small" data-save="thumb_b.jpg" type="button">Save</button></figcaption></figure>
          <figure><img src="thumb_c.jpg" alt="The cavern street of identical houses, text: 24 copies of my house"><figcaption><span>C · the copies</span><button class="btn small" data-save="thumb_c.jpg" type="button">Save</button></figcaption></figure>
        </div>
      </div>
      <div class="block">
        <header><h2>Description</h2><button class="btn small" data-copy="desc" type="button">Copy</button></header>
        <pre id="desc">__DESC__</pre>
        <p class="note">Keep it short and explain nothing. Leave the chapters out for the first days: they spoil the pacing.</p>
      </div>
      <div class="block">
        <header><h2>Pinned comment</h2><button class="btn small" data-copy="pinned" type="button">Copy</button></header>
        <pre id="pinned">__PINNED__</pre>
        <p class="note">A nudge towards two frame-perfect clues: the odd line in the server log, and the one frame where
          his blurred account name reads PLAYER_2.</p>
      </div>
      <div class="block">
        <header><h2>Tags</h2><button class="btn small" data-copy="tags" type="button">Copy</button></header>
        <pre id="tags">__TAGS__</pre>
      </div>
      <div class="block">
        <header><h2>Chapters, for later</h2><button class="btn small" data-copy="chapters" type="button">Copy</button></header>
        <pre id="chapters">__CHAPTERS__</pre>
      </div>
      <div class="block">
        <h2>Keeping the ARG going</h2>
        <ul>__FOLLOW__</ul>
      </div>
      <span class="status" id="kit-status" role="status"></span>
    </div>
  </section>

  <section role="tabpanel" id="clues" aria-labelledby="tab-clues" hidden>
    <div class="gate" id="gate">
      <p><strong class="spoil">Spoilers.</strong> This is what the film never says out loud, and where every clue is
        hidden, with times. It's for you, not for the description.</p>
      <button class="btn" id="reveal" type="button">Show the clue guide</button>
    </div>
    <div id="guide" hidden>
      <div class="act">__LORE__</div>
      <h2>Where everything is hidden</h2>
      <div class="scroll"><table><thead><tr><th>Time</th><th>Clue</th></tr></thead><tbody>__CLUES__</tbody></table></div>
    </div>
  </section>
</div>

<script src="https://cdn.jsdelivr.net/npm/hls.js@1.6.15/dist/hls.min.js"></script>
<script>
(function () {
  var $ = function (s) { return document.querySelector(s); };
  var manifest = null;
  var status = $('#status');
  function mb(b) { return Math.round(b / 1048576); }

  fetch('manifest.json').then(function (r) { return r.ok ? r.json() : null; }).then(function (m) {
    manifest = m;
    if (m) $('#dl').textContent = 'Download the film, no subtitles (MP4, ' + mb(m.bytes) + ' MB)';
  }).catch(function () { manifest = null; });

  // his lines as captions over the clean film, from the SRT on this page, placed where the film's own subtitles sit
  function addCaptions(v) {
    if (v.textTracks.length || typeof VTTCue === 'undefined') return;
    var tr = v.addTextTrack('subtitles', 'His lines', 'en');
    function sec(h, mi, s, ms) { return +h * 3600 + +mi * 60 + +s + +ms / 1000; }
    $('#srt').textContent.trim().split(/\n\s*\n/).forEach(function (block) {
      var ls = block.split('\n');
      var m = /(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)/.exec(ls[1] || '');
      if (!m) return;
      var cue = new VTTCue(sec(m[1], m[2], m[3], m[4]), sec(m[5], m[6], m[7], m[8]), ls.slice(2).join('\n'));
      cue.snapToLines = false; cue.line = 78;
      tr.addCue(cue);
    });
    tr.mode = 'showing';
  }

  // --- watching: the film's HLS playlist (playlist.txt) and its pieces are published next to this page
  function cannotStream() {
    status.textContent = "This browser can't stream the film here. Use Download instead.";
    $('#player').hidden = true; $('#poster').hidden = false; $('#watch').hidden = false;
  }
  $('#watch').addEventListener('click', function () {
    if (!manifest) { status.textContent = 'The film is still being uploaded. Try again in a minute.'; return; }
    var v = $('#player');
    var url = 'playlist.txt';
    $('#watch').hidden = true; $('#poster').hidden = true; v.hidden = false;
    addCaptions(v);
    if (window.Hls && window.Hls.isSupported()) {
      var hls = new window.Hls({ maxBufferLength: 60 });
      hls.on(window.Hls.Events.ERROR, function (e, d) { if (d && d.fatal) { hls.destroy(); cannotStream(); } });
      hls.loadSource(url);
      hls.attachMedia(v);
      hls.on(window.Hls.Events.MANIFEST_PARSED, function () { v.play().catch(function () {}); });
    } else if (v.canPlayType('application/vnd.apple.mpegurl')) {
      v.src = url;
      v.addEventListener('error', cannotStream, { once: true });
      v.play().catch(function () {});
    } else {
      cannotStream();
    }
  });

  // --- saving files (the viewer confirms every save)
  async function save(filename, data, out) {
    var dl = window.claude ? await window.claude.use('downloads') : null;
    if (!dl) { out.textContent = "Saving files isn't available in this view."; return false; }
    try {
      await dl.save({ filename: filename, data: data });
      out.textContent = 'Saved ' + filename + '.';
      return true;
    } catch (e) {
      var code = e && e.code;
      out.textContent = code === 'declined' ? 'Save cancelled.' :
        code === 'too_large' ? 'Too large for this app. Save it from a computer, or join the parts in the repository.' :
        code === 'rate_limited' ? 'A save is already waiting for your answer.' :
        'Could not save the file here.';
      return false;
    }
  }

  $('#dl').addEventListener('click', async function () {
    if (!manifest) { status.textContent = 'The film is still being uploaded. Try again in a minute.'; return; }
    var btn = $('#dl'), prog = $('#prog');
    btn.disabled = true; prog.hidden = false;
    var files = [manifest.init].concat(manifest.segments.map(function (s) { return s.file; }));
    var parts = [], got = 0;
    try {
      for (var i = 0; i < files.length; i++) {
        var r = await fetch(files[i]);
        if (!r.ok) throw new Error(files[i]);
        var b = await r.blob();
        parts.push(b); got += b.size;
        prog.value = got / manifest.bytes;
        status.textContent = 'Fetching the film: ' + mb(got) + ' of ' + mb(manifest.bytes) + ' MB';
      }
      status.textContent = 'Ready. Confirm the save.';
      await save(manifest.filename, new Blob(parts, { type: 'video/mp4' }), status);
    } catch (err) {
      status.textContent = 'A piece of the film did not load. Try again.';
    } finally {
      btn.disabled = false; prog.hidden = true;
    }
  });

  document.querySelectorAll('[data-save]').forEach(function (b) {
    b.addEventListener('click', async function () {
      var f = b.getAttribute('data-save');
      try { var blob = await (await fetch(f)).blob(); await save(f, blob, $('#kit-status')); }
      catch (e) { $('#kit-status').textContent = 'Could not load ' + f + '.'; }
    });
  });

  // --- copying
  function copy(el, out) {
    function selectIt() {
      if (el.hidden) el.hidden = false;
      var r = document.createRange(); r.selectNodeContents(el);
      var sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
      out.textContent = "Copying is blocked here. The text is selected: copy it with your keyboard or menu.";
    }
    try {
      navigator.clipboard.writeText(el.textContent).then(function () { out.textContent = 'Copied.'; }, selectIt);
    } catch (e) { selectIt(); }
  }
  document.querySelectorAll('[data-copy]').forEach(function (b) {
    b.addEventListener('click', function () { copy($('#' + b.getAttribute('data-copy')), $('#kit-status')); });
  });
  $('#copy-srt').addEventListener('click', function () { copy($('#srt'), $('#srt-status')); });
  $('#save-srt').addEventListener('click', function () {
    save('voiceover.srt.txt', $('#srt').textContent, $('#srt-status')).then(function (ok) {
      if (ok) $('#srt-status').textContent = 'Saved. Rename it to voiceover.srt for caption uploads.';
    });
  });

  // --- tabs (the chosen one is remembered on this device)
  var tabs = Array.prototype.slice.call(document.querySelectorAll('[role="tab"]'));
  function show(id) {
    tabs.forEach(function (t) {
      var on = t.getAttribute('aria-controls') === id;
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
    });
    try { localStorage.setItem('nlo-tab', id); } catch (e) {}
  }
  tabs.forEach(function (t) { t.addEventListener('click', function () { show(t.getAttribute('aria-controls')); }); });
  var start = (location.hash || '').slice(1);
  if (!document.getElementById(start) || !tabs.some(function (t) { return t.getAttribute('aria-controls') === start; })) {
    try { start = localStorage.getItem('nlo-tab') || 'vo'; } catch (e) { start = 'vo'; }
  }
  if (start !== 'clues') show(start === 'publish' ? 'publish' : 'vo'); else show('clues');
  $('#reveal').addEventListener('click', function () { $('#gate').hidden = true; $('#guide').hidden = false; });
})();
</script>
'''


def build(dst):
    acts = voiceover()
    vo = []
    for (tm, title, mood, rows) in acts:
        body = ''.join(f'<tr><td class="t">{html.escape(t)}</td><td class="line">{inline(l)}</td>'
                       f'<td class="n">{inline(n)}</td></tr>' for (t, l, n) in rows)
        vo.append(f'<div class="act"><h3>{html.escape(tm)} · {html.escape(title)}</h3>'
                  f'<p class="mood">{inline(mood)}</p><div class="scroll"><table><tbody>{body}</tbody></table></div></div>')
    lore, rows = clues()
    y = youtube()
    page = PAGE
    rep = {
        '__VO__': '\n'.join(vo),
        '__SRT__': html.escape(srt_lines()),
        '__ALTS__': ''.join(f'<li>{inline(a)}</li>' for a in y['alts']),
        '__DESC__': html.escape(y['description']),
        '__PINNED__': html.escape(y['pinned']),
        '__TAGS__': html.escape(y['tags']),
        '__CHAPTERS__': html.escape(y['chapters']),
        '__FOLLOW__': ''.join(f'<li><strong>{inline(a)}</strong> {inline(b)}</li>' for (a, b) in y['follow']),
        '__LORE__': lore_html(lore),
        '__CLUES__': ''.join(f'<tr><td class="t">{html.escape(a)}</td><td>{inline(b)}</td></tr>' for (a, b) in rows),
    }
    for k, v in rep.items():
        assert k in page, k
        page = page.replace(k, v)
    os.makedirs(dst, exist_ok=True)
    path = os.path.join(dst, 'index.html')
    with open(path, 'w') as fh:
        fh.write(page)
    print(path, len(page) // 1024, 'KB;', sum(len(a[3]) for a in acts), 'lines,', len(rows), 'clues')
    return path


if __name__ == '__main__':
    build(sys.argv[1])
