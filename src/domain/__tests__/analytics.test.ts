import {
  bankrollSeries,
  breakdown,
  byBetType,
  byBookmaker,
  byOddsRange,
  bySport,
  byTag,
  calibration,
  clvStats,
  computeStreaks,
  gradeBets,
  normalCdf,
  profitSeries,
  riskStats,
  settledChronologically,
  summarize,
} from '../analytics';
import { makeBet, makeLeg, makeTransaction, settledSingle } from './factories';

describe('summarize', () => {
  it('reports zeroes for an empty ledger', () => {
    const summary = summarize([]);
    expect(summary.totalBets).toBe(0);
    expect(summary.profit).toBe(0);
    expect(summary.roi).toBe(0);
    expect(summary.winRate).toBe(0);
    expect(summary.currentStreak).toEqual({ type: 'none', count: 0 });
  });

  it('adds up profit, ROI and win rate', () => {
    const bets = [
      settledSingle(2, 'won', { placedAt: '2026-03-01T12:00:00.000Z' }),
      settledSingle(2, 'lost', { placedAt: '2026-03-02T12:00:00.000Z' }),
      settledSingle(3, 'won', { placedAt: '2026-03-03T12:00:00.000Z' }),
    ];
    const summary = summarize(bets);

    expect(summary.settledBets).toBe(3);
    expect(summary.turnover).toBe(300);
    expect(summary.profit).toBeCloseTo(200, 10);
    expect(summary.roi).toBeCloseTo(200 / 300, 10);
    expect(summary.wins).toBe(2);
    expect(summary.losses).toBe(1);
    expect(summary.winRate).toBeCloseTo(2 / 3, 10);
    expect(summary.averageStake).toBe(100);
    expect(summary.averageOdds).toBeCloseTo(7 / 3, 10);
    expect(summary.biggestWin).toBeCloseTo(200, 10);
    expect(summary.biggestLoss).toBeCloseTo(-100, 10);
  });

  it('excludes pending bets from ROI but counts them as exposure', () => {
    const bets = [settledSingle(2, 'won'), makeBet({ stake: 50 })];
    const summary = summarize(bets);
    expect(summary.settledBets).toBe(1);
    expect(summary.pendingBets).toBe(1);
    expect(summary.pendingStake).toBe(50);
    expect(summary.turnover).toBe(100);
  });

  it('excludes voids from the win rate', () => {
    const bets = [settledSingle(2, 'won'), settledSingle(2, 'void'), settledSingle(2, 'lost')];
    const summary = summarize(bets);
    expect(summary.neutrals).toBe(1);
    expect(summary.winRate).toBeCloseTo(0.5, 10);
  });

  it('counts a profitable cash-out as a win', () => {
    const summary = summarize([makeBet({ cashOutReturn: 150 })]);
    expect(summary.wins).toBe(1);
    expect(summary.profit).toBeCloseTo(50, 10);
  });
});

describe('computeStreaks', () => {
  const graded = (statuses: ('won' | 'lost' | 'void')[]) =>
    settledChronologically(
      gradeBets(
        statuses.map((status, index) =>
          settledSingle(2, status, {
            placedAt: `2026-03-0${index + 1}T12:00:00.000Z`,
            settledAt: `2026-03-0${index + 1}T20:00:00.000Z`,
          }),
        ),
      ),
    );

  it('tracks the current run', () => {
    const streaks = computeStreaks(graded(['lost', 'won', 'won', 'won']));
    expect(streaks.current).toEqual({ type: 'win', count: 3 });
    expect(streaks.longestWin).toBe(3);
    expect(streaks.longestLoss).toBe(1);
  });

  it('ignores voids without breaking a run', () => {
    const streaks = computeStreaks(graded(['won', 'void', 'won']));
    expect(streaks.current).toEqual({ type: 'win', count: 2 });
  });

  it('tracks losing runs', () => {
    const streaks = computeStreaks(graded(['won', 'lost', 'lost']));
    expect(streaks.current).toEqual({ type: 'loss', count: 2 });
    expect(streaks.longestLoss).toBe(2);
  });
});

describe('breakdown', () => {
  const bets = [
    settledSingle(2, 'won', {
      legs: [makeLeg({ sport: 'Football', odds: 2, status: 'won' })],
      bookmaker: 'Pinnacle',
      tags: ['model'],
    }),
    settledSingle(2, 'lost', {
      legs: [makeLeg({ sport: 'Basketball', odds: 2, status: 'lost' })],
      bookmaker: 'Bet365',
      tags: ['hunch'],
    }),
    settledSingle(4, 'won', {
      legs: [makeLeg({ sport: 'Football', odds: 4, status: 'won' })],
      bookmaker: 'Pinnacle',
      tags: ['model', 'live'],
    }),
  ];

  it('groups by sport, sorted by profit', () => {
    const rows = breakdown(bets, bySport);
    expect(rows).toHaveLength(2);
    expect(rows[0]?.label).toBe('Football');
    expect(rows[0]?.bets).toBe(2);
    expect(rows[0]?.profit).toBeCloseTo(400, 10);
    expect(rows[0]?.roi).toBeCloseTo(2, 10);
    expect(rows[1]?.label).toBe('Basketball');
    expect(rows[1]?.profit).toBeCloseTo(-100, 10);
  });

  it('groups by bookmaker', () => {
    const rows = breakdown(bets, byBookmaker);
    expect(rows.map((row) => row.key)).toEqual(['Pinnacle', 'Bet365']);
  });

  it('counts a bet once per tag', () => {
    const rows = breakdown(bets, byTag);
    const model = rows.find((row) => row.key === 'model');
    const live = rows.find((row) => row.key === 'live');
    expect(model?.bets).toBe(2);
    expect(live?.bets).toBe(1);
  });

  it('buckets odds ranges', () => {
    const rows = breakdown(bets, byOddsRange);
    expect(rows.map((row) => row.key).sort()).toEqual(['2.00 – 2.99', '3.00 – 4.99']);
  });

  it('separates singles from parlays', () => {
    const withParlay = [
      ...bets,
      makeBet({
        legs: [makeLeg({ odds: 2, status: 'won' }), makeLeg({ odds: 2, status: 'won' })],
        settledAt: '2026-03-04T12:00:00.000Z',
      }),
    ];
    const rows = breakdown(withParlay, byBetType);
    expect(rows.find((row) => row.key === 'Parlay')?.bets).toBe(1);
    expect(rows.find((row) => row.key === 'Single')?.bets).toBe(3);
  });

  it('honours the minimum sample size', () => {
    expect(breakdown(bets, bySport, { minBets: 2 })).toHaveLength(1);
  });

  it('ignores pending bets', () => {
    expect(breakdown([makeBet()], bySport)).toHaveLength(0);
  });
});

describe('profitSeries', () => {
  it('accumulates day by day', () => {
    const bets = [
      settledSingle(2, 'won', {
        placedAt: '2026-03-01T12:00:00.000Z',
        settledAt: '2026-03-01T20:00:00.000Z',
      }),
      settledSingle(2, 'lost', {
        placedAt: '2026-03-02T12:00:00.000Z',
        settledAt: '2026-03-02T20:00:00.000Z',
      }),
      settledSingle(3, 'won', {
        placedAt: '2026-03-02T13:00:00.000Z',
        settledAt: '2026-03-02T21:00:00.000Z',
      }),
    ];
    const series = profitSeries(bets);
    expect(series).toHaveLength(2);
    expect(series[0]?.profit).toBeCloseTo(100, 10);
    expect(series[0]?.cumulative).toBeCloseTo(100, 10);
    expect(series[1]?.bets).toBe(2);
    expect(series[1]?.profit).toBeCloseTo(100, 10);
    expect(series[1]?.cumulative).toBeCloseTo(200, 10);
  });

  it('is empty without settled bets', () => {
    expect(profitSeries([makeBet()])).toEqual([]);
  });
});

describe('bankrollSeries', () => {
  it('combines deposits, withdrawals and profit', () => {
    const transactions = [
      makeTransaction({ type: 'deposit', amount: 500, date: '2026-03-01T09:00:00.000Z' }),
      makeTransaction({ type: 'withdrawal', amount: 200, date: '2026-03-03T09:00:00.000Z' }),
    ];
    const bets = [
      settledSingle(2, 'won', {
        placedAt: '2026-03-02T12:00:00.000Z',
        settledAt: '2026-03-02T20:00:00.000Z',
      }),
    ];
    const series = bankrollSeries(bets, transactions, 1000);

    expect(series).toHaveLength(3);
    expect(series[0]?.balance).toBeCloseTo(1500, 10);
    expect(series[1]?.balance).toBeCloseTo(1600, 10);
    expect(series[2]?.balance).toBeCloseTo(1400, 10);
    expect(series[2]?.profit).toBeCloseTo(100, 10);
  });
});

describe('clvStats', () => {
  it('measures how often the price beat the close', () => {
    const bets = [
      settledSingle(2.1, 'won', { legs: [makeLeg({ odds: 2.1, closingOdds: 2, status: 'won' })] }),
      settledSingle(1.9, 'lost', { legs: [makeLeg({ odds: 1.9, closingOdds: 2, status: 'lost' })] }),
    ];
    const stats = clvStats(bets);
    expect(stats.tracked).toBe(2);
    expect(stats.beat).toBe(1);
    expect(stats.beatRate).toBeCloseTo(0.5, 10);
    expect(stats.averageClv).toBeCloseTo((0.05 - 0.05) / 2, 6);
  });

  it('skips bets without a closing price', () => {
    expect(clvStats([settledSingle(2, 'won')]).tracked).toBe(0);
  });

  it('weights CLV by stake', () => {
    const bets = [
      settledSingle(2.2, 'won', {
        stake: 300,
        legs: [makeLeg({ odds: 2.2, closingOdds: 2, status: 'won' })],
      }),
      settledSingle(1.9, 'lost', {
        stake: 100,
        legs: [makeLeg({ odds: 1.9, closingOdds: 2, status: 'lost' })],
      }),
    ];
    const stats = clvStats(bets);
    expect(stats.averageClv).toBeCloseTo((0.1 * 300 + -0.05 * 100) / 400, 8);
    expect(stats.clvEdge).toBeCloseTo(0.1 * 300 + -0.05 * 100, 8);
  });
});

describe('riskStats', () => {
  it('finds the worst peak-to-trough fall', () => {
    const bets = [
      settledSingle(3, 'won', {
        placedAt: '2026-03-01T12:00:00.000Z',
        settledAt: '2026-03-01T20:00:00.000Z',
      }),
      settledSingle(2, 'lost', {
        placedAt: '2026-03-02T12:00:00.000Z',
        settledAt: '2026-03-02T20:00:00.000Z',
      }),
      settledSingle(2, 'lost', {
        placedAt: '2026-03-03T12:00:00.000Z',
        settledAt: '2026-03-03T20:00:00.000Z',
      }),
    ];
    const stats = riskStats(bets);
    expect(stats.maxDrawdown).toBeCloseTo(200, 10);
    expect(stats.longestDrawdownDays).toBe(2);
    expect(stats.volatility).toBeGreaterThan(0);
  });

  it('reports the drawdown as a share of peak bankroll when capital is known', () => {
    const bets = [
      settledSingle(3, 'won', {
        placedAt: '2026-03-01T12:00:00.000Z',
        settledAt: '2026-03-01T20:00:00.000Z',
      }),
      settledSingle(2, 'lost', {
        placedAt: '2026-03-02T12:00:00.000Z',
        settledAt: '2026-03-02T20:00:00.000Z',
      }),
    ];
    // Peak profit is +200 on a 1,000 base, so the high-water mark is 1,200.
    const stats = riskStats(bets, 1000);
    expect(stats.maxDrawdown).toBeCloseTo(100, 10);
    expect(stats.maxDrawdownPercent).toBeCloseTo(100 / 1200, 10);
  });

  it('withholds the percentage when no capital base is known', () => {
    // Measured against peak profit alone a drawdown can exceed 100%, which is
    // meaningless — so the figure is omitted rather than shown as nonsense.
    const bets = [
      settledSingle(3, 'won', {
        placedAt: '2026-03-01T12:00:00.000Z',
        settledAt: '2026-03-01T20:00:00.000Z',
      }),
      settledSingle(2, 'lost', {
        placedAt: '2026-03-02T12:00:00.000Z',
        settledAt: '2026-03-02T20:00:00.000Z',
      }),
      settledSingle(2, 'lost', {
        placedAt: '2026-03-03T12:00:00.000Z',
        settledAt: '2026-03-03T20:00:00.000Z',
      }),
    ];
    const stats = riskStats(bets);
    expect(stats.maxDrawdown).toBeCloseTo(200, 10);
    expect(stats.maxDrawdownPercent).toBeUndefined();
  });

  it('handles a single bet without dividing by zero', () => {
    const stats = riskStats([settledSingle(2, 'won')], 1000);
    expect(Number.isFinite(stats.volatility)).toBe(true);
    expect(stats.volatility).toBe(0);
    expect(stats.tStatistic).toBe(0);
    expect(stats.pValue).toBeCloseTo(1, 6);
  });
});

describe('normalCdf', () => {
  it('matches known values', () => {
    expect(normalCdf(0)).toBeCloseTo(0.5, 6);
    expect(normalCdf(1.96)).toBeCloseTo(0.975, 3);
    expect(normalCdf(-1.96)).toBeCloseTo(0.025, 3);
    expect(normalCdf(3)).toBeCloseTo(0.99865, 4);
  });
});

describe('calibration', () => {
  it('compares implied probability with the realised win rate', () => {
    const bets = [
      settledSingle(2, 'won'),
      settledSingle(2, 'lost'),
      settledSingle(2, 'won'),
      settledSingle(2, 'won'),
    ];
    const rows = calibration(bets);
    const band = rows.find((row) => row.label === '40–60%');
    expect(band?.bets).toBe(4);
    expect(band?.expected).toBeCloseTo(0.5, 10);
    expect(band?.actual).toBeCloseTo(0.75, 10);
  });

  it('always returns every band', () => {
    expect(calibration([])).toHaveLength(5);
  });
});
