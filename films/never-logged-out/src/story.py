"""The whole film, act by act (see PLAN.md for the beats)."""
import film as FM

_CTX = None


def film(ctx=None):
    import seq_hook
    import seq_perfect
    shots = []
    shots += seq_hook.shots()
    c = ctx if ctx is not None else _lazy_ctx()
    shots += seq_perfect.shots(c)
    try:
        import seq_lie
        shots += seq_lie.shots(c)
        import seq_footsteps
        shots += seq_footsteps.shots(c)
        import seq_under
        shots += seq_under.shots(c)
        import seq_records
        shots += seq_records.shots(c)
        import seq_second
        shots += seq_second.shots(c)
        import seq_changed
        shots += seq_changed.shots(c)
        import seq_reveal
        shots += seq_reveal.shots(c)
        import seq_finale
        shots += seq_finale.shots(c)
        import seq_end
        shots += seq_end.shots(c)
    except ImportError as e:
        print('[story] (not all acts written yet:', e, ')')
    f = FM.Film(shots)
    for (t, name, kw) in f.cues:
        if name == 'chat':
            f.chat_log.append((t, kw['text'], tuple(kw.get('color', (255, 255, 255)))))
    f.chat_log.sort(key=lambda c: c[0])
    return f


class _World:
    """Worlds without GL, for building the story (points, heights, props need only the voxels)."""


def _lazy_ctx():
    global _CTX
    if _CTX is None:
        _CTX = FM.Ctx(need_gl=False)
    return _CTX
