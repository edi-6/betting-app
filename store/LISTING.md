# Store listing — BetLedger

Copy/paste material for App Store Connect and Google Play Console.
Bundle identifier: **`com.edidagoat.betledger`** (both platforms).

Character limits are noted where the stores enforce them; every field below is
already within its limit.

---

## Names and subtitles

| Field | Value | Limit |
| --- | --- | --- |
| App name (both stores) | `BetLedger` | 30 |
| Apple subtitle | `Bet tracker & bankroll stats` | 30 |
| Play short description | `Track every bet, manage your bankroll, and find out where your edge really is.` | 80 |

## Apple keywords (100 characters, comma-separated, no spaces)

```
bet,tracker,betting,bankroll,sports,wager,parlay,odds,roi,clv,journal,stats,kelly,ev,record
```

Don't repeat words already in the app name or subtitle — Apple indexes those anyway.

---

## Full description

> Works as-is for Google Play (4000 char limit) and Apple (4000 char limit).

```
BetLedger is a private betting journal and analytics tool. Log the bets you place,
and it tells you what is actually working — by sport, by market, by bookmaker, and
by the prices you take.

It does not place bets, sell picks, or predict results. Everything stays on your phone.

TRACK EVERY KIND OF BET
• Singles and parlays with any number of legs
• Won, lost, push, void, Asian handicap half-wins and half-losses
• Free bets and bonus stakes, handled correctly
• Early cash-outs, with one tap to undo
• Bookmaker, tags, confidence rating and notes on every slip
• Closing odds per leg, so you can measure yourself against the market

SEE WHERE YOUR EDGE IS
• Profit curve you can scrub with your finger
• ROI, turnover, win rate and stake-weighted average odds
• Break down profit by sport, league, market, bookmaker, tag, odds range, bet type,
  day of week or confidence — with a minimum sample filter so one lucky bet doesn't
  look like a system
• Calibration chart: do the prices you take win as often as they imply?
• Monthly profit bars to see whether you're trending the right way

CLOSING LINE VALUE
Beating the closing price is the best early signal that your edge is real — long
before profit becomes statistically meaningful. BetLedger shows how often you beat
the close, your average and median CLV, and the individual bets where you got the
best and worst of the market.

KNOW IF IT'S SKILL OR LUCK
• Maximum drawdown, and how long you spent underwater
• Volatility and typical daily swing
• A 95% confidence interval for your true ROI
• A p-value answering the only question that matters: could this just be variance?

BANKROLL MANAGEMENT
• Balance, available funds and money currently at risk, all calculated for you
• Deposits, withdrawals and adjustments, with a balance-over-time chart
• Optional limits: daily stake cap, bets per day, weekly and monthly loss limits,
  and a maximum stake as a share of your bankroll

BUILT-IN CALCULATORS
• Odds converter — decimal, American, fractional, implied probability
• Expected value and fractional Kelly staking
• No-vig fair odds — multiplicative, additive, power and Shin methods
• Parlay, arbitrage and hedge calculators
• Cash-out checker: is the offer fair, or is the book keeping a slice?

YOUR DATA IS YOURS
• Export bets and transactions to CSV, or take a full JSON backup
• Import from a spreadsheet — re-importing the same file never duplicates anything
• No account, no sign-up, no servers, no tracking, no ads

DETAILS
Built for iOS 26: the tab bar, sheets and action button are real Liquid Glass that
refracts the content moving behind them, and fall back gracefully on older devices.
Dark and light themes. Decimal, American or fractional odds throughout. 19
currencies. Works fully offline.

PLEASE GAMBLE RESPONSIBLY
BetLedger is a record-keeping tool for people who already bet. It is designed to
show you the truth about your results, including when that truth is uncomfortable.
If betting stops being fun, take a break — free, confidential support is available
in every region.
```

---

## App Review notes (Apple) / Reviewer comments (Google)

Paste this into the review notes field. It heads off the most likely question.

```
BetLedger is an offline record-keeping and statistics app for people who bet on
sport. It is a journal, not a gambling product.

The app does NOT:
- accept, place, broker or facilitate any wager
- handle real money, payments, deposits or withdrawals of any kind
- connect to any bookmaker, sportsbook, exchange or betting API
- offer, sell or display tips, picks, predictions or recommended bets
- contain any simulated gambling, casino or game-of-chance mechanic
- contain advertising, or any link to a gambling operator

The app DOES:
- let the user manually type in bets they already placed elsewhere
- calculate statistics from those entries (profit, ROI, win rate, closing line value)
- provide general arithmetic calculators (odds format conversion, expected value)
- store everything locally on the device; there is no account, server or network call

"Deposits" and "withdrawals" in the Bankroll tab are manual bookkeeping entries the
user types in to reconcile their own records. No payment processing occurs anywhere
in the app.

The only outbound link in the app opens BeGambleAware, a problem-gambling support
service, from the Settings screen.

NO LOGIN REQUIRED. To see the app fully populated, open it and tap "Load demo data"
on the first screen (or Settings > Your data > Load demo data). This fills the app
with a sample six-month history so every analytics screen has data to show.
```

---

## Content rating answers

Answer honestly; these are the ones that matter.

**Apple age rating questionnaire**
- *Contests* → None
- *Simulated Gambling (frequency/intensity)* → **None** — there is no simulated gambling; the app records real bets placed elsewhere
- *Gambling and Contests* (the "does your app contain gambling" toggle) → **No**
- *Unrestricted Web Access* → No
- Expect a final rating of **17+** or **12+** depending on how Apple weighs the subject matter. Do not contest it.

**Google Play content rating (IARC) questionnaire**
- Category: **Reference, News, or Educational** (or *Utility*) — not "Game"
- *Does the app allow users to gamble with real money?* → **No**
- *Does the app simulate gambling?* → **No**
- *Does the app provide information about or promote gambling?* → **Yes** — it is a tool for people who already gamble. This drives the rating up but answering "no" risks removal later.
- Expect **Mature 17+** in some regions.

**Play Data safety form**
- Data collected: **None**
- Data shared: **None**
- Data encrypted in transit: N/A (no data leaves the device)
- Users can request deletion: **Yes** — Settings → Your data → Erase all data

**Apple privacy "nutrition label"**
- **Data Not Collected.** The app has no analytics, no network calls and no account.

---

## Privacy policy

Both stores require a reachable privacy policy URL, even for an app that collects
nothing. Host `store/PRIVACY.md` (rendered) anywhere public — GitHub Pages works:

```
https://edi-6.github.io/betting-app/privacy
```

---

## Screenshots

Required sizes:

| Store | Required | Notes |
| --- | --- | --- |
| Apple | 6.9" iPhone (1320×2868) | Apple scales these down for smaller devices |
| Apple | 13" iPad (2064×2752) | Only if you keep `supportsTablet: true` |
| Google Play | 2–8 phone shots, min 1080px on the short side | Plus a 1024×500 feature graphic |

Suggested order — the story is "log it, then learn from it":

1. **Dashboard** — bankroll, profit curve, KPI tiles. Caption: *Your whole ledger at a glance*
2. **Analytics → Breakdown** — profit by sport. Caption: *Find out which sports actually pay*
3. **Analytics → CLV** — closing line value. Caption: *Are you beating the closing line?*
4. **Analytics → Risk** — drawdown and p-value. Caption: *Skill, or just variance?*
5. **Bet form** — payout preview with EV and Kelly. Caption: *Log a bet in seconds*
6. **Tools** — odds converter or EV calculator. Caption: *Seven calculators built in*
7. **Bankroll** — limits with progress bars. Caption: *Set your own guard rails*

Capture them on a simulator with demo data loaded:

```bash
npx expo start --ios   # then Settings > Your data > Load demo data
# Simulator > File > Save Screen  (⌘S)
```

Use an iPhone 16 Pro Max simulator for the 6.9" size and a 13" iPad simulator for
the tablet shots.

---

## Pre-submission checklist

- [ ] Bundle ID `com.edidagoat.betledger` is registered in both consoles
- [ ] Privacy policy URL is live and reachable
- [ ] Screenshots captured at the required sizes
- [ ] `npm run check` passes
- [ ] `eas build --profile production --platform all` succeeds
- [ ] Installed the production build on a real device and added, graded and exported a bet
- [ ] Support email address set in both consoles
- [ ] Apple: export compliance answered (`ITSAppUsesNonExemptEncryption: false` is already in `app.json`)
- [ ] Google: first `.aab` uploaded manually through Play Console (required before `eas submit` works)
- [ ] Google: if this is a new personal developer account, the 12-tester / 14-day closed test is running
