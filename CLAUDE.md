# Working in this repo

## Git

**Push to `main`.** Do not create or push to a `claude/*` branch — the owner asked for
work to land on `main` directly. If your session is configured with a different default
branch, `main` still wins.

```bash
git push origin main
```

## Before you push

```bash
npm run check     # typecheck + lint + 270-odd tests; all three must pass
```

For anything touching layout, navigation or a native module, also confirm both
platforms still bundle — this catches dependency and resolution breakage that the tests
cannot see:

```bash
npx expo export --platform ios --platform android --output-dir /tmp/betledger-export
```

## Architecture rules

- **`src/domain/` is pure TypeScript.** No React, no React Native imports. It is the
  money maths and it is tested in isolation — keep it that way.
- **Settlement is a multiplier, not a branch.** Every leg maps to the fraction of the
  stake it returns, so singles, parlays, Asian handicaps, voids and free bets all fall
  out of one expression. Do not add per-bet-type special cases; extend the multiplier.
- **Persisted data goes through the sanitisers** in `src/domain/validation.ts`. A
  corrupt or partial payload must degrade to the rows we understand, never crash.
  Adding a persisted field means updating the sanitiser and `SCHEMA_VERSION`.

## Liquid Glass

- Glass belongs on **chrome that floats over scrolling content** — the tab bar, sheets,
  the action button. Cards and charts stay opaque; glass over a flat background has
  nothing to refract.
- Always go through `GlassSurface`. It checks `isGlassEffectAPIAvailable()` *and*
  `isLiquidGlassAvailable()` separately, wraps both in try/catch, and honours the
  system "reduce transparency" setting. All three guards are load-bearing: some iOS 26
  betas ship the design without the API, and `requireNativeModule` throws outright in a
  runtime that does not bundle the module.

## Demo data

`createDemoData()` is seeded and pinned. It drives first-launch impressions and every
store screenshot, so tests assert it stays representative — modestly profitable, a
sub-50% win rate, a real drawdown, positive CLV. If you change the generator, re-check
those tests rather than relaxing them.

## The web target

`npm run web` exists for previewing and screenshots without a device. iOS and Android
are what the app ships to. Web-only breakage is still worth fixing — it has caught real
iOS bugs — but never trade native correctness for it.
