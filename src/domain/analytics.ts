import { monthKey, dayKey, startOfDay, WEEKDAY_ABBREVIATIONS } from './dates';
import { impliedProbability } from './odds';
import {
  amountAtRisk,
  betType,
  combinedClosingOdds,
  combinedOdds,
  isLoss,
  isNeutral,
  isWin,
  settleBet,
} from './settlement';
import type { Bet, BetStatus, Settlement, Transaction } from './types';

/* -------------------------------------------------------------------------- */
/* Shared shapes                                                               */
/* -------------------------------------------------------------------------- */

export interface GradedBet {
  bet: Bet;
  settlement: Settlement;
  status: BetStatus;
  odds: number;
  /** Date the profit hit the bankroll (settlement date, or placement date if unknown). */
  effectiveDate: string;
}

export interface PerformanceSummary {
  totalBets: number;
  settledBets: number;
  pendingBets: number;
  wins: number;
  losses: number;
  neutrals: number;
  /** Total staked on settled bets — the denominator for ROI. */
  turnover: number;
  /** Stake tied up in bets that have not been graded yet. */
  pendingStake: number;
  totalReturns: number;
  profit: number;
  /** profit ÷ turnover. */
  roi: number;
  /** wins ÷ (wins + losses). Pushes and voids are excluded. */
  winRate: number;
  averageStake: number;
  /** Stake-weighted average decimal odds across settled bets. */
  averageOdds: number;
  biggestWin: number;
  biggestLoss: number;
  currentStreak: Streak;
  longestWinStreak: number;
  longestLossStreak: number;
}

export interface Streak {
  type: 'win' | 'loss' | 'none';
  count: number;
}

export interface BreakdownRow {
  key: string;
  label: string;
  bets: number;
  wins: number;
  losses: number;
  staked: number;
  profit: number;
  roi: number;
  winRate: number;
  averageOdds: number;
}

export interface SeriesPoint {
  /** "YYYY-MM-DD" local day key. */
  key: string;
  timestamp: number;
  profit: number;
  cumulative: number;
  bets: number;
  staked: number;
}

/* -------------------------------------------------------------------------- */
/* Grading                                                                     */
/* -------------------------------------------------------------------------- */

export function gradeBet(bet: Bet): GradedBet {
  const settlement = settleBet(bet);
  return {
    bet,
    settlement,
    status: settlement.status,
    odds: combinedOdds(bet),
    effectiveDate: bet.settledAt ?? bet.placedAt,
  };
}

export function gradeBets(bets: Bet[]): GradedBet[] {
  return bets.map(gradeBet);
}

/** Settled bets, oldest first, ordered by the date their profit landed. */
export function settledChronologically(graded: GradedBet[]): GradedBet[] {
  return graded
    .filter((g) => g.settlement.isSettled)
    .slice()
    .sort((a, b) => new Date(a.effectiveDate).getTime() - new Date(b.effectiveDate).getTime());
}

/* -------------------------------------------------------------------------- */
/* Summary                                                                     */
/* -------------------------------------------------------------------------- */

export function summarize(bets: Bet[]): PerformanceSummary {
  const graded = gradeBets(bets);
  const settled = settledChronologically(graded);
  const pending = graded.filter((g) => !g.settlement.isSettled);

  let turnover = 0;
  let totalReturns = 0;
  let profit = 0;
  let wins = 0;
  let losses = 0;
  let neutrals = 0;
  let biggestWin = 0;
  let biggestLoss = 0;
  let weightedOdds = 0;

  for (const g of settled) {
    turnover += g.bet.stake;
    totalReturns += g.settlement.returns;
    profit += g.settlement.profit;
    weightedOdds += g.odds * g.bet.stake;

    if (isWin(g.status) || (g.status === 'cashed_out' && g.settlement.profit > 0)) {
      wins += 1;
    } else if (isLoss(g.status) || (g.status === 'cashed_out' && g.settlement.profit < 0)) {
      losses += 1;
    } else {
      neutrals += 1;
    }

    biggestWin = Math.max(biggestWin, g.settlement.profit);
    biggestLoss = Math.min(biggestLoss, g.settlement.profit);
  }

  const pendingStake = pending.reduce((sum, g) => sum + amountAtRisk(g.bet), 0);
  const decided = wins + losses;
  const streakStats = computeStreaks(settled);

  return {
    totalBets: bets.length,
    settledBets: settled.length,
    pendingBets: pending.length,
    wins,
    losses,
    neutrals,
    turnover,
    pendingStake,
    totalReturns,
    profit,
    roi: turnover > 0 ? profit / turnover : 0,
    winRate: decided > 0 ? wins / decided : 0,
    averageStake: settled.length > 0 ? turnover / settled.length : 0,
    averageOdds: turnover > 0 ? weightedOdds / turnover : 0,
    biggestWin,
    biggestLoss,
    currentStreak: streakStats.current,
    longestWinStreak: streakStats.longestWin,
    longestLossStreak: streakStats.longestLoss,
  };
}

export function computeStreaks(settledOldestFirst: GradedBet[]): {
  current: Streak;
  longestWin: number;
  longestLoss: number;
} {
  let longestWin = 0;
  let longestLoss = 0;
  let runType: 'win' | 'loss' | 'none' = 'none';
  let runCount = 0;

  for (const g of settledOldestFirst) {
    const profit = g.settlement.profit;
    const outcome: 'win' | 'loss' | 'none' =
      isNeutral(g.status) || profit === 0 ? 'none' : profit > 0 ? 'win' : 'loss';

    if (outcome === 'none') {
      continue; // Pushes and voids neither extend nor break a streak.
    }
    if (outcome === runType) {
      runCount += 1;
    } else {
      runType = outcome;
      runCount = 1;
    }
    if (runType === 'win') {
      longestWin = Math.max(longestWin, runCount);
    } else {
      longestLoss = Math.max(longestLoss, runCount);
    }
  }

  return {
    current: runCount === 0 ? { type: 'none', count: 0 } : { type: runType, count: runCount },
    longestWin,
    longestLoss,
  };
}

/* -------------------------------------------------------------------------- */
/* Breakdowns                                                                  */
/* -------------------------------------------------------------------------- */

/** Selector returns one or more bucket keys per bet; return `[]` to skip the bet. */
export type BucketSelector = (graded: GradedBet) => string[];

export function breakdown(
  bets: Bet[],
  selector: BucketSelector,
  options: { labels?: Record<string, string>; minBets?: number } = {},
): BreakdownRow[] {
  const graded = settledChronologically(gradeBets(bets));
  const buckets = new Map<string, BreakdownRow & { weightedOdds: number }>();

  for (const g of graded) {
    for (const key of selector(g)) {
      let row = buckets.get(key);
      if (!row) {
        row = {
          key,
          label: options.labels?.[key] ?? key,
          bets: 0,
          wins: 0,
          losses: 0,
          staked: 0,
          profit: 0,
          roi: 0,
          winRate: 0,
          averageOdds: 0,
          weightedOdds: 0,
        };
        buckets.set(key, row);
      }
      row.bets += 1;
      row.staked += g.bet.stake;
      row.profit += g.settlement.profit;
      row.weightedOdds += g.odds * g.bet.stake;
      if (g.settlement.profit > 0) row.wins += 1;
      else if (g.settlement.profit < 0) row.losses += 1;
    }
  }

  const minBets = options.minBets ?? 1;
  return Array.from(buckets.values())
    .filter((row) => row.bets >= minBets)
    .map(({ weightedOdds, ...row }) => ({
      ...row,
      roi: row.staked > 0 ? row.profit / row.staked : 0,
      winRate: row.wins + row.losses > 0 ? row.wins / (row.wins + row.losses) : 0,
      averageOdds: row.staked > 0 ? weightedOdds / row.staked : 0,
    }))
    .sort((a, b) => b.profit - a.profit);
}

export const bySport: BucketSelector = (g) =>
  unique(g.bet.legs.map((leg) => leg.sport || 'Unspecified'));

export const byLeague: BucketSelector = (g) =>
  unique(g.bet.legs.map((leg) => leg.league || 'Unspecified'));

export const byMarket: BucketSelector = (g) =>
  unique(g.bet.legs.map((leg) => leg.market || 'Unspecified'));

export const byBookmaker: BucketSelector = (g) => [g.bet.bookmaker || 'Unspecified'];

export const byTag: BucketSelector = (g) => (g.bet.tags.length > 0 ? unique(g.bet.tags) : []);

export const byBetType: BucketSelector = (g) => [betType(g.bet) === 'parlay' ? 'Parlay' : 'Single'];

export const byWeekday: BucketSelector = (g) => [
  WEEKDAY_ABBREVIATIONS[new Date(g.bet.placedAt).getDay()] ?? 'Unknown',
];

export const byMonth: BucketSelector = (g) => [monthKey(g.effectiveDate)];

export const byConfidence: BucketSelector = (g) =>
  g.bet.confidence ? [`${g.bet.confidence}★`] : [];

export const ODDS_BUCKETS: { label: string; min: number; max: number }[] = [
  { label: '< 1.50', min: 1, max: 1.5 },
  { label: '1.50 – 1.99', min: 1.5, max: 2 },
  { label: '2.00 – 2.99', min: 2, max: 3 },
  { label: '3.00 – 4.99', min: 3, max: 5 },
  { label: '5.00 – 9.99', min: 5, max: 10 },
  { label: '10.00+', min: 10, max: Infinity },
];

export const byOddsRange: BucketSelector = (g) => {
  const bucket = ODDS_BUCKETS.find((b) => g.odds >= b.min && g.odds < b.max);
  return [bucket?.label ?? 'Unknown'];
};

export const byLegCount: BucketSelector = (g) => {
  const count = g.bet.legs.length;
  if (count <= 1) return ['1 leg'];
  if (count >= 6) return ['6+ legs'];
  return [`${count} legs`];
};

function unique(values: string[]): string[] {
  return Array.from(new Set(values));
}

/* -------------------------------------------------------------------------- */
/* Time series                                                                 */
/* -------------------------------------------------------------------------- */

/** Daily profit plus a running total, oldest first. Only settled bets contribute. */
export function profitSeries(bets: Bet[]): SeriesPoint[] {
  const settled = settledChronologically(gradeBets(bets));
  const byDay = new Map<string, { profit: number; bets: number; staked: number; ts: number }>();

  for (const g of settled) {
    const key = dayKey(g.effectiveDate);
    const entry = byDay.get(key) ?? {
      profit: 0,
      bets: 0,
      staked: 0,
      ts: startOfDay(g.effectiveDate).getTime(),
    };
    entry.profit += g.settlement.profit;
    entry.bets += 1;
    entry.staked += g.bet.stake;
    byDay.set(key, entry);
  }

  let cumulative = 0;
  return Array.from(byDay.entries())
    .sort((a, b) => a[1].ts - b[1].ts)
    .map(([key, entry]) => {
      cumulative += entry.profit;
      return {
        key,
        timestamp: entry.ts,
        profit: entry.profit,
        cumulative,
        bets: entry.bets,
        staked: entry.staked,
      };
    });
}

export interface BankrollPoint {
  key: string;
  timestamp: number;
  balance: number;
  /** Net deposits up to this point. */
  invested: number;
  profit: number;
}

/**
 * Bankroll over time: starting balance, plus deposits/withdrawals, plus realised profit.
 */
export function bankrollSeries(
  bets: Bet[],
  transactions: Transaction[],
  startingBankroll: number,
): BankrollPoint[] {
  type Event = { ts: number; cash: number; profit: number };
  const events: Event[] = [];

  for (const t of transactions) {
    events.push({ ts: startOfDay(t.date).getTime(), cash: signedTransactionAmount(t), profit: 0 });
  }
  for (const g of settledChronologically(gradeBets(bets))) {
    events.push({
      ts: startOfDay(g.effectiveDate).getTime(),
      cash: 0,
      profit: g.settlement.profit,
    });
  }

  events.sort((a, b) => a.ts - b.ts);

  const points: BankrollPoint[] = [];
  let invested = startingBankroll;
  let profit = 0;

  for (const event of events) {
    invested += event.cash;
    profit += event.profit;
    const key = dayKey(event.ts);
    const last = points[points.length - 1];
    if (last && last.key === key) {
      last.invested = invested;
      last.profit = profit;
      last.balance = invested + profit;
    } else {
      points.push({ key, timestamp: event.ts, invested, profit, balance: invested + profit });
    }
  }

  return points;
}

export function signedTransactionAmount(transaction: Transaction): number {
  switch (transaction.type) {
    case 'deposit':
      return Math.abs(transaction.amount);
    case 'withdrawal':
      return -Math.abs(transaction.amount);
    case 'adjustment':
    default:
      return transaction.amount;
  }
}

/* -------------------------------------------------------------------------- */
/* Closing line value                                                          */
/* -------------------------------------------------------------------------- */

export interface ClvStats {
  /** Bets with a closing price on every leg. */
  tracked: number;
  /** Bets whose price beat the close. */
  beat: number;
  beatRate: number;
  /** Stake-weighted mean of (odds ÷ closing odds − 1). */
  averageClv: number;
  /** Simple mean of the same figure. */
  medianClv: number;
  /** Money-weighted CLV edge expressed in account currency. */
  clvEdge: number;
}

export function clvStats(bets: Bet[]): ClvStats {
  const samples: { clv: number; stake: number }[] = [];

  for (const bet of bets) {
    const closing = combinedClosingOdds(bet);
    const taken = combinedOdds(bet);
    if (closing === undefined || !Number.isFinite(closing) || closing <= 1) {
      continue;
    }
    samples.push({ clv: taken / closing - 1, stake: bet.stake });
  }

  if (samples.length === 0) {
    return { tracked: 0, beat: 0, beatRate: 0, averageClv: 0, medianClv: 0, clvEdge: 0 };
  }

  const totalStake = samples.reduce((sum, s) => sum + s.stake, 0);
  const beat = samples.filter((s) => s.clv > 0).length;
  const weighted = samples.reduce((sum, s) => sum + s.clv * s.stake, 0);
  const sorted = samples.map((s) => s.clv).sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median =
    sorted.length % 2 === 0
      ? ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2
      : (sorted[mid] ?? 0);

  return {
    tracked: samples.length,
    beat,
    beatRate: beat / samples.length,
    averageClv: totalStake > 0 ? weighted / totalStake : 0,
    medianClv: median,
    clvEdge: weighted,
  };
}

/* -------------------------------------------------------------------------- */
/* Risk & significance                                                         */
/* -------------------------------------------------------------------------- */

export interface RiskStats {
  /** Largest peak-to-trough fall of the cumulative profit curve, as a positive number. */
  maxDrawdown: number;
  /** Same figure relative to the peak, 0–1. 0 when the peak was never positive. */
  maxDrawdownPercent: number;
  /** Days spent below the previous peak during the worst drawdown. */
  longestDrawdownDays: number;
  /** Standard deviation of per-bet returns measured in units of stake. */
  volatility: number;
  /** t-statistic for "is this ROI distinguishable from zero?". */
  tStatistic: number;
  /** Two-sided p-value for the t-statistic, via a normal approximation. */
  pValue: number;
  /** 95% confidence interval for the true ROI, in units of stake. */
  roiConfidenceInterval: [number, number];
  /** Standard deviation of the bankroll curve's daily change. */
  averageDailySwing: number;
}

export function riskStats(bets: Bet[]): RiskStats {
  const settled = settledChronologically(gradeBets(bets));
  const series = profitSeries(bets);

  let peak = 0;
  let peakTimestamp = series[0]?.timestamp ?? 0;
  let maxDrawdown = 0;
  let maxDrawdownPercent = 0;
  let longestDrawdownDays = 0;

  for (const point of series) {
    if (point.cumulative > peak) {
      peak = point.cumulative;
      peakTimestamp = point.timestamp;
    }
    const drawdown = peak - point.cumulative;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
      maxDrawdownPercent = peak > 0 ? drawdown / peak : 0;
    }
    if (drawdown > 0) {
      const days = Math.round((point.timestamp - peakTimestamp) / 86400000);
      longestDrawdownDays = Math.max(longestDrawdownDays, days);
    }
  }

  const returns = settled
    .filter((g) => g.bet.stake > 0)
    .map((g) => g.settlement.profit / g.bet.stake);
  const n = returns.length;
  const mean = n > 0 ? returns.reduce((a, b) => a + b, 0) / n : 0;
  const variance =
    n > 1 ? returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / (n - 1) : 0;
  const volatility = Math.sqrt(variance);
  const standardError = n > 0 ? volatility / Math.sqrt(n) : 0;
  const tStatistic = standardError > 0 ? mean / standardError : 0;

  const dailyChanges = series.map((point) => point.profit);
  const dailyMean =
    dailyChanges.length > 0 ? dailyChanges.reduce((a, b) => a + b, 0) / dailyChanges.length : 0;
  const dailyVariance =
    dailyChanges.length > 1
      ? dailyChanges.reduce((sum, v) => sum + (v - dailyMean) ** 2, 0) / (dailyChanges.length - 1)
      : 0;

  return {
    maxDrawdown,
    maxDrawdownPercent,
    longestDrawdownDays,
    volatility,
    tStatistic,
    pValue: twoSidedPValue(tStatistic),
    roiConfidenceInterval: [mean - 1.96 * standardError, mean + 1.96 * standardError],
    averageDailySwing: Math.sqrt(dailyVariance),
  };
}

/** Abramowitz & Stegun 7.1.26 approximation of the standard normal CDF. */
export function normalCdf(z: number): number {
  const sign = z < 0 ? -1 : 1;
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-x * x);
  return 0.5 * (1 + sign * y);
}

export function twoSidedPValue(z: number): number {
  if (!Number.isFinite(z)) {
    return 1;
  }
  return Math.min(1, Math.max(0, 2 * (1 - normalCdf(Math.abs(z)))));
}

/* -------------------------------------------------------------------------- */
/* Calibration                                                                 */
/* -------------------------------------------------------------------------- */

export interface CalibrationRow {
  label: string;
  /** Mid-point of the implied-probability band. */
  expected: number;
  actual: number;
  bets: number;
}

const CALIBRATION_BANDS: { label: string; min: number; max: number }[] = [
  { label: '0–20%', min: 0, max: 0.2 },
  { label: '20–40%', min: 0.2, max: 0.4 },
  { label: '40–60%', min: 0.4, max: 0.6 },
  { label: '60–80%', min: 0.6, max: 0.8 },
  { label: '80–100%', min: 0.8, max: 1.01 },
];

/**
 * Do bets priced at ~40% actually win ~40% of the time? A well-calibrated bettor
 * tracks the diagonal; systematic gaps point at a market or price bias.
 */
export function calibration(bets: Bet[]): CalibrationRow[] {
  const settled = settledChronologically(gradeBets(bets)).filter(
    (g) => !isNeutral(g.status) && g.status !== 'cashed_out',
  );

  return CALIBRATION_BANDS.map((band) => {
    const inBand = settled.filter((g) => {
      const p = impliedProbability(g.odds);
      return p >= band.min && p < band.max;
    });
    const wins = inBand.filter((g) => g.settlement.profit > 0).length;
    const expected =
      inBand.length > 0
        ? inBand.reduce((sum, g) => sum + impliedProbability(g.odds), 0) / inBand.length
        : (band.min + Math.min(band.max, 1)) / 2;
    return {
      label: band.label,
      expected,
      actual: inBand.length > 0 ? wins / inBand.length : 0,
      bets: inBand.length,
    };
  });
}
