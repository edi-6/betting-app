import { gradeBets, settledChronologically, signedTransactionAmount } from './analytics';
import { dayKey, startOfMonth, startOfWeek } from './dates';
import { amountAtRisk, settleBet } from './settlement';
import type { Bet, ResponsibleGamblingLimits, Transaction } from './types';

export interface BankrollState {
  /** The balance the user started tracking with. */
  startingBankroll: number;
  deposits: number;
  withdrawals: number;
  adjustments: number;
  /** deposits − withdrawals + adjustments. */
  netDeposits: number;
  /** Profit from graded bets. */
  realizedProfit: number;
  /** startingBankroll + netDeposits + realizedProfit. */
  balance: number;
  /** Stake locked up in ungraded bets. */
  atRisk: number;
  /** balance − atRisk: what the user could withdraw today. */
  available: number;
  /** realizedProfit ÷ (startingBankroll + deposits): return on money put in. */
  returnOnCapital: number;
}

export function computeBankroll(
  bets: Bet[],
  transactions: Transaction[],
  startingBankroll: number,
): BankrollState {
  let deposits = 0;
  let withdrawals = 0;
  let adjustments = 0;

  for (const transaction of transactions) {
    const signed = signedTransactionAmount(transaction);
    if (transaction.type === 'deposit') deposits += signed;
    else if (transaction.type === 'withdrawal') withdrawals += Math.abs(signed);
    else adjustments += signed;
  }

  const graded = gradeBets(bets);
  const realizedProfit = settledChronologically(graded).reduce(
    (sum, g) => sum + g.settlement.profit,
    0,
  );
  const atRisk = graded
    .filter((g) => !g.settlement.isSettled)
    .reduce((sum, g) => sum + amountAtRisk(g.bet), 0);

  const netDeposits = deposits - withdrawals + adjustments;
  const balance = startingBankroll + netDeposits + realizedProfit;
  const capital = startingBankroll + deposits;

  return {
    startingBankroll,
    deposits,
    withdrawals,
    adjustments,
    netDeposits,
    realizedProfit,
    balance,
    atRisk,
    available: balance - atRisk,
    returnOnCapital: capital > 0 ? realizedProfit / capital : 0,
  };
}

/* -------------------------------------------------------------------------- */
/* Responsible gambling guard rails                                            */
/* -------------------------------------------------------------------------- */

export type LimitSeverity = 'ok' | 'approaching' | 'exceeded';

export interface LimitStatus {
  key: keyof ResponsibleGamblingLimits;
  label: string;
  description: string;
  used: number;
  limit: number;
  /** used ÷ limit, clamped to [0, 2]. */
  ratio: number;
  severity: LimitSeverity;
  /** True when the figure is money rather than a count or a percentage. */
  isCurrency: boolean;
}

function severityFor(ratio: number): LimitSeverity {
  if (ratio >= 1) return 'exceeded';
  if (ratio >= 0.8) return 'approaching';
  return 'ok';
}

/** Total staked on bets placed on the given local day. */
export function stakedOnDay(bets: Bet[], day: Date): number {
  const key = dayKey(day);
  return bets
    .filter((bet) => dayKey(bet.placedAt) === key)
    .reduce((sum, bet) => sum + amountAtRisk(bet), 0);
}

export function betsPlacedOnDay(bets: Bet[], day: Date): number {
  const key = dayKey(day);
  return bets.filter((bet) => dayKey(bet.placedAt) === key).length;
}

/** Net loss (positive number) over bets settled on or after `since`. */
export function netLossSince(bets: Bet[], since: Date): number {
  const cutoff = since.getTime();
  const profit = bets
    .filter((bet) => {
      const settlement = settleBet(bet);
      if (!settlement.isSettled) return false;
      return new Date(bet.settledAt ?? bet.placedAt).getTime() >= cutoff;
    })
    .reduce((sum, bet) => sum + settleBet(bet).profit, 0);
  return profit < 0 ? -profit : 0;
}

export function evaluateLimits(
  bets: Bet[],
  limits: ResponsibleGamblingLimits,
  bankrollBalance: number,
  now: Date = new Date(),
): LimitStatus[] {
  const statuses: LimitStatus[] = [];

  if (limits.dailyStakeLimit && limits.dailyStakeLimit > 0) {
    const used = stakedOnDay(bets, now);
    const ratio = clampRatio(used / limits.dailyStakeLimit);
    statuses.push({
      key: 'dailyStakeLimit',
      label: 'Daily stake',
      description: 'Total risked on bets placed today',
      used,
      limit: limits.dailyStakeLimit,
      ratio,
      severity: severityFor(ratio),
      isCurrency: true,
    });
  }

  if (limits.dailyBetCountLimit && limits.dailyBetCountLimit > 0) {
    const used = betsPlacedOnDay(bets, now);
    const ratio = clampRatio(used / limits.dailyBetCountLimit);
    statuses.push({
      key: 'dailyBetCountLimit',
      label: 'Bets today',
      description: 'Number of slips placed today',
      used,
      limit: limits.dailyBetCountLimit,
      ratio,
      severity: severityFor(ratio),
      isCurrency: false,
    });
  }

  if (limits.weeklyLossLimit && limits.weeklyLossLimit > 0) {
    const used = netLossSince(bets, startOfWeek(now));
    const ratio = clampRatio(used / limits.weeklyLossLimit);
    statuses.push({
      key: 'weeklyLossLimit',
      label: 'Weekly loss',
      description: 'Net loss since Monday',
      used,
      limit: limits.weeklyLossLimit,
      ratio,
      severity: severityFor(ratio),
      isCurrency: true,
    });
  }

  if (limits.monthlyLossLimit && limits.monthlyLossLimit > 0) {
    const used = netLossSince(bets, startOfMonth(now));
    const ratio = clampRatio(used / limits.monthlyLossLimit);
    statuses.push({
      key: 'monthlyLossLimit',
      label: 'Monthly loss',
      description: 'Net loss this calendar month',
      used,
      limit: limits.monthlyLossLimit,
      ratio,
      severity: severityFor(ratio),
      isCurrency: true,
    });
  }

  if (limits.maxStakePercent && limits.maxStakePercent > 0 && bankrollBalance > 0) {
    const biggestToday = bets
      .filter((bet) => dayKey(bet.placedAt) === dayKey(now))
      .reduce((max, bet) => Math.max(max, bet.stake), 0);
    const used = (biggestToday / bankrollBalance) * 100;
    const ratio = clampRatio(used / limits.maxStakePercent);
    statuses.push({
      key: 'maxStakePercent',
      label: 'Max stake size',
      description: 'Largest stake today as a share of bankroll',
      used,
      limit: limits.maxStakePercent,
      ratio,
      severity: severityFor(ratio),
      isCurrency: false,
    });
  }

  return statuses;
}

export interface StakeWarning {
  severity: 'warning' | 'block';
  message: string;
}

/**
 * Warnings shown while the user is filling in the bet form — before the money is risked.
 */
export function warningsForStake(
  stake: number,
  bets: Bet[],
  limits: ResponsibleGamblingLimits,
  bankrollBalance: number,
  now: Date = new Date(),
): StakeWarning[] {
  const warnings: StakeWarning[] = [];
  if (!Number.isFinite(stake) || stake <= 0) {
    return warnings;
  }

  if (limits.maxStakePercent && bankrollBalance > 0) {
    const percent = (stake / bankrollBalance) * 100;
    if (percent > limits.maxStakePercent) {
      warnings.push({
        severity: 'warning',
        message: `This stake is ${percent.toFixed(1)}% of your bankroll — your limit is ${limits.maxStakePercent}%.`,
      });
    }
  }

  if (limits.dailyStakeLimit) {
    const projected = stakedOnDay(bets, now) + stake;
    if (projected > limits.dailyStakeLimit) {
      warnings.push({
        severity: 'warning',
        message: `This would take today's stakes to ${projected.toFixed(2)}, over your ${limits.dailyStakeLimit.toFixed(2)} daily limit.`,
      });
    }
  }

  if (limits.dailyBetCountLimit) {
    const projected = betsPlacedOnDay(bets, now) + 1;
    if (projected > limits.dailyBetCountLimit) {
      warnings.push({
        severity: 'warning',
        message: `That's bet ${projected} today — your daily limit is ${limits.dailyBetCountLimit}.`,
      });
    }
  }

  if (limits.weeklyLossLimit) {
    const loss = netLossSince(bets, startOfWeek(now));
    if (loss >= limits.weeklyLossLimit) {
      warnings.push({
        severity: 'warning',
        message: `You've already hit your weekly loss limit. Consider stopping for the week.`,
      });
    }
  }

  if (bankrollBalance > 0 && stake > bankrollBalance) {
    warnings.push({
      severity: 'block',
      message: 'This stake is larger than your available bankroll.',
    });
  }

  return warnings;
}

function clampRatio(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(Math.max(value, 0), 2);
}
