# BetLedger

A sports betting tracker, bankroll manager and analytics app for **iOS and Android**, built with
Expo and React Native.

BetLedger is a record-keeping and analysis tool. It does not place bets, supply tips, or predict
results, and every byte of data stays on the device.

---

## What it does

### Track
- **Singles and parlays** with any number of legs, each carrying its own sport, league, event,
  market, selection, price and result.
- **Every real-world settlement case**: won, lost, push, void, Asian-handicap half-win and
  half-loss, free/bonus bets, and early cash-outs.
- **Closing odds** per leg, which unlocks closing line value analysis.
- Bookmaker, tags, 1–5 confidence rating, free-text notes and the exact time the bet was struck.
- One-tap grading from the bet detail screen, or per-leg grading for parlays.

### Analyse
- **Overview** — cumulative profit curve (scrub it with your finger), ROI, turnover, win rate,
  stake-weighted average odds, results donut, monthly profit bars, and a calibration chart showing
  whether the prices you take actually win as often as they imply.
- **Breakdown** — profit and ROI grouped by sport, league, market, bookmaker, tag, odds range, bet
  type, leg count, day of week or confidence rating, with a minimum-sample filter so a single lucky
  bet doesn't masquerade as an edge.
- **CLV** — how often you beat the closing line, your stake-weighted and median CLV, and the
  individual bets where you got the best and worst of the market.
- **Risk** — maximum drawdown and how long it lasted, volatility, daily swing, a 95% confidence
  interval for your true ROI, and a t-statistic and p-value answering "is this edge real, or is it
  variance?"

### Manage the bankroll
- Balance, available funds and money currently at risk, all derived rather than typed in.
- Deposits, withdrawals and manual adjustments, with a balance-over-time chart.
- **Responsible-gambling guard rails**: daily stake cap, bets-per-day cap, weekly and monthly loss
  limits, and a maximum stake as a share of bankroll. Limits warn — they never silently block you —
  and the dashboard surfaces a breach the moment it happens.

### Calculate
Seven tools under the **Tools** tab:

| Tool | What it answers |
| --- | --- |
| Odds converter | The same price in decimal, American and fractional, plus implied and break-even probability |
| EV & Kelly | Expected value, edge, fair odds, and a fractional-Kelly stake for your bankroll |
| No-vig | The bookmaker's true probabilities — multiplicative, additive, power or Shin |
| Parlay | Combined odds, returns and implied probability for any number of legs |
| Arbitrage | Whether a set of prices is an arb, and how to split the stakes |
| Hedge | The opposing stake that equalises both outcomes |
| Cash out | Whether an offer is fair, and the margin the bookmaker is keeping |

### Own your data
- Export bets and transactions as **CSV**, or take a full **JSON backup**.
- Import CSV from a spreadsheet — a hand-written file with just `placed_at, odds, stake` works, and
  parlay detail survives a full export/import round-trip.
- Imports are **idempotent**: rows replace existing bets with the same id, so re-importing the same
  file never duplicates anything.
- Demo data generator (a realistic, deterministic six-month history) to explore the app before
  committing your own records.

### Details that matter
- **iOS 26 Liquid Glass** on the floating chrome — the tab bar, bottom sheets and the
  action button are real glass that refracts the content scrolling behind them, with
  Apple's interactive press response on the action button.
- Dark and light themes that follow the system setting, or can be pinned.
- Decimal, American or fractional odds throughout — one setting, applied everywhere.
- 19 currencies, formatted without relying on the device's ICU build.
- Optional haptics, safe-area aware layouts, and accessibility labels on every control.

---

## Getting started

You need [Node.js](https://nodejs.org) (the LTS build) and git. Nothing else — no
Xcode, no Android Studio.

```bash
git clone https://github.com/edi-6/betting-app.git
cd betting-app
git checkout claude/sports-betting-tracker-app-f5zk3f
npm install        # a minute or two, only needed once
npx expo start     # prints a QR code
```

Install **Expo Go** on your phone, then scan the QR code — with the Camera app on
iPhone, or from inside Expo Go on Android. The phone and the computer must be on the
same Wi-Fi; if yours blocks device-to-device traffic, use `npx expo start --tunnel`
instead, which routes through Expo's servers and works from anywhere.

On first launch you get an empty ledger. Tap **Load demo data** to fill it with a
sample six-month history. Edits to the source reload on the phone as you save.

Other ways to run it:

```bash
npm run ios        # iOS simulator (macOS only)
npm run android    # Android emulator or device
npm run web        # runs in a browser — handy for a quick look without a device
```

The web target is a preview convenience, not a shipping platform: it is useful for
screenshots and for checking layout without a simulator, but iOS and Android are what
the app is built and tested for.

### Quality gates

```bash
npm run typecheck  # tsc --noEmit, strict mode
npm run lint       # eslint
npm test           # jest
npm run check      # all three
```

### Building for the stores

Native binaries are produced with [EAS Build](https://docs.expo.dev/build/introduction/), so no
Xcode or Android Studio setup is required locally:

```bash
npm install -g eas-cli
eas login
eas build --profile preview    --platform all   # internal testing (.apk / ad-hoc .ipa)
eas build --profile production --platform all   # store-ready .aab / .ipa
eas submit --platform all
```

Bundle identifiers are set in `app.json` (`com.edidagoat.betledger` on both platforms). Store
listing copy, review notes and the content-rating answers live in [`store/LISTING.md`](./store/LISTING.md).

---

## Architecture

```
App.tsx                  Entry point
src/
  AppRoot.tsx            Provider stack (safe area → data → theme → navigation)
  domain/                Pure TypeScript: no React, no React Native
    types.ts             The data model
    odds.ts              Format conversion, implied probability, vig removal
    settlement.ts        Payout multipliers — the heart of the money maths
    analytics.ts         Summaries, breakdowns, time series, CLV, risk, calibration
    bankroll.ts          Balances and responsible-gambling limits
    calculators.ts       EV, Kelly, arbitrage, hedge, cash-out, parlay
    filters.ts           Search, filter, sort, facets
    csv.ts               RFC 4180 encode/decode, CSV and JSON import/export
    validation.ts        Sanitisers that make corrupt input impossible to crash on
    format.ts dates.ts   Locale-independent formatting
    catalog.ts demo.ts   Reference data and the demo dataset
  data/                  AsyncStorage persistence, migrations, file sharing/picking
  store/AppStore.tsx     Reducer + context, with debounced persistence
  theme/                 Design tokens and the theme provider
  components/            Reusable UI, including SVG charts
  screens/               Dashboard, Bets, Bet form, Bet detail, Analytics, Bankroll, Tools, Settings
  navigation/            Typed bottom tabs + native stack
```

### Liquid Glass, and what happens without it

`GlassSurface` picks the best material the device offers and degrades cleanly:

| Capability | Rendered with | When |
| --- | --- | --- |
| `liquid` | `GlassView` (`expo-glass-effect`) | iOS 26+, built against the iOS 26 SDK |
| `blur` | `BlurView` (`expo-blur`) | older iOS, Android 12+, and web via `backdrop-filter` |
| `solid` | an opaque fill | reduced transparency, or the user switches it off |

Three rules keep it honest:

1. **`isGlassEffectAPIAvailable()` is checked separately from `isLiquidGlassAvailable()`.**
   Some iOS 26 betas ship the Liquid Glass *design* without the API behind it, and
   touching `GlassView` on those builds crashes the app.
2. **The system "reduce transparency" setting wins.** Apple requires glass to collapse to
   a solid surface when it is on, so the effect is a progressive enhancement and never a
   legibility risk. The detection is feature-detected, not platform-guessed — the API
   does not exist on web at all.
3. **Only floating chrome gets glass.** Cards, charts and stat tiles stay opaque. Glass
   over a flat background has nothing to refract and just reads as a muddy rectangle.

Settings → Display → Glass effects turns it off, and its caption names the material
actually in use so the switch is never a silent no-op.

Liquid Glass needs the app compiled against the iOS 26 SDK. EAS Build's current image
does this automatically; the app opts in simply by *not* setting
`UIDesignRequiresCompatibility` in `app.json`.

### Why settlement is a multiplier

Rather than branching on bet type, every leg maps to the fraction of the stake it returns:

| Result | Multiplier |
| --- | --- |
| Won | `odds` |
| Half won | `(1 + odds) / 2` |
| Push / void | `1` |
| Half lost | `0.5` |
| Lost | `0` |

`returns = stake × Π legMultiplier`. Singles, parlays, Asian handicaps and voided legs then all fall
out of the same expression, and a free bet is simply `stake × max(multiplier − 1, 0)`.

### Storage

Everything lives in a single AsyncStorage key (`betledger/app-data/v1`), written 150 ms after the
last change. Reads run through a sanitiser, so a corrupt or partial payload degrades to "the rows we
could understand" instead of a crash loop. A forward-only migration runner is wired up for future
schema changes.

---

## Tests

256 tests across 15 suites:

- **Domain** — odds conversion round-trips, all four de-vig methods, every settlement case,
  analytics against hand-computed figures, Kelly against the textbook formula, arbitrage stake
  splits, CSV round-trips, filters, and sanitisers fed deliberate junk.
- **Data** — persistence round-trips, corrupt-payload recovery, migrations, and file export/import
  against an in-memory filesystem.
- **Store** — every reducer action, including idempotent imports and cash-out reversal.
- **UI** — the real app rendered end to end: every tab and analytics view, all seven calculators,
  adding a bet, building a parlay, grading legs, cashing out and undoing it, editing, filtering,
  changing currency and odds format, adding a deposit, and limit warnings.

Both platforms are verified to bundle with `npx expo export --platform ios --platform android`.

---

## Responsible gambling

Betting should stay entertainment. If it stops being fun, support is free and confidential:

- **UK** — BeGambleAware, 0808 8020 133, <https://www.begambleaware.org/>
- **US** — National Problem Gambling Helpline, 1-800-522-4700
- **AU** — Gambling Help Online, 1800 858 858

---

## Licence

MIT — see [LICENSE](./LICENSE).
