import { combineOdds } from './odds';
import type { Bet, BetStatus, BetType, Leg, LegStatus, Settlement } from './types';

/** Floating-point tolerance used when comparing payout multipliers against 1. */
const EPSILON = 1e-9;

/**
 * The fraction of the stake returned by a single leg.
 *
 * Modelling settlement as a multiplier makes singles, parlays and Asian-handicap
 * half-wins/half-losses fall out of the same formula:
 *
 *   returns = stake × Π legMultiplier(leg)
 */
export function legMultiplier(status: LegStatus, decimalOdds: number): number | undefined {
  switch (status) {
    case 'pending':
      return undefined;
    case 'won':
      return decimalOdds;
    case 'half_won':
      return (1 + decimalOdds) / 2;
    case 'push':
    case 'void':
      return 1;
    case 'half_lost':
      return 0.5;
    case 'lost':
      return 0;
    default:
      return undefined;
  }
}

export function betType(bet: Pick<Bet, 'legs'>): BetType {
  return bet.legs.length > 1 ? 'parlay' : 'single';
}

/**
 * Combined decimal odds for the slip as it was struck (every leg at full price).
 * Voided legs are excluded because the bookmaker reprices the slip without them.
 */
export function combinedOdds(bet: Pick<Bet, 'legs'>): number {
  const live = bet.legs.filter((leg) => leg.status !== 'void' && leg.status !== 'push');
  if (live.length === 0) {
    return 1;
  }
  return combineOdds(live.map((leg) => leg.odds));
}

/** Combined odds ignoring settlement — used for previews on the bet form. */
export function nominalOdds(legs: Pick<Leg, 'odds'>[]): number {
  return combineOdds(legs.map((leg) => leg.odds));
}

/** Combined closing odds, when every leg has a closing price recorded. */
export function combinedClosingOdds(bet: Pick<Bet, 'legs'>): number | undefined {
  const live = bet.legs.filter((leg) => leg.status !== 'void' && leg.status !== 'push');
  if (live.length === 0 || live.some((leg) => leg.closingOdds === undefined)) {
    return undefined;
  }
  return combineOdds(live.map((leg) => leg.closingOdds as number));
}

/** The payout multiplier once every leg has been graded. */
export function settlementMultiplier(legs: Pick<Leg, 'status' | 'odds'>[]): number | undefined {
  if (legs.length === 0) {
    return undefined;
  }
  let product = 1;
  for (const leg of legs) {
    const multiplier = legMultiplier(leg.status, leg.odds);
    if (multiplier === undefined) {
      return undefined;
    }
    product *= multiplier;
  }
  return product;
}

/** The slip's status, derived from its legs (a cash-out always wins). */
export function deriveBetStatus(bet: Pick<Bet, 'legs' | 'cashOutReturn'>): BetStatus {
  if (bet.cashOutReturn !== undefined) {
    return 'cashed_out';
  }
  const { legs } = bet;
  if (legs.length === 0) {
    return 'pending';
  }
  if (legs.some((leg) => leg.status === 'pending')) {
    return 'pending';
  }
  if (legs.some((leg) => leg.status === 'lost')) {
    return 'lost';
  }

  const multiplier = settlementMultiplier(legs) ?? 1;
  const hasPartial = legs.some((leg) => leg.status === 'half_won' || leg.status === 'half_lost');

  if (multiplier > 1 + EPSILON) {
    return hasPartial ? 'half_won' : 'won';
  }
  if (multiplier < 1 - EPSILON) {
    return 'half_lost';
  }
  if (legs.every((leg) => leg.status === 'void')) {
    return 'void';
  }
  return 'push';
}

/**
 * Full money outcome of a bet.
 *
 * Free bets (bonus stakes) keep the winnings but not the stake, and cost nothing when
 * they lose — so their profit is `stake × max(multiplier − 1, 0)`.
 */
export function settleBet(bet: Bet): Settlement {
  const { stake, isFreeBet, cashOutReturn } = bet;

  if (cashOutReturn !== undefined) {
    const returns = cashOutReturn;
    return {
      status: 'cashed_out',
      multiplier: stake > 0 ? returns / stake : undefined,
      returns,
      profit: isFreeBet ? returns : returns - stake,
      isSettled: true,
    };
  }

  const multiplier = settlementMultiplier(bet.legs);
  if (multiplier === undefined) {
    return { status: 'pending', multiplier: undefined, returns: 0, profit: 0, isSettled: false };
  }

  const returns = isFreeBet ? stake * Math.max(multiplier - 1, 0) : stake * multiplier;
  const profit = isFreeBet ? returns : returns - stake;

  return {
    status: deriveBetStatus(bet),
    multiplier,
    returns,
    profit,
    isSettled: true,
  };
}

/** Cash that would come back if every remaining leg wins. */
export function potentialReturns(bet: Pick<Bet, 'legs' | 'stake' | 'isFreeBet'>): number {
  const odds = nominalOdds(bet.legs);
  return bet.isFreeBet ? bet.stake * (odds - 1) : bet.stake * odds;
}

/** Profit if every remaining leg wins. */
export function potentialProfit(bet: Pick<Bet, 'legs' | 'stake' | 'isFreeBet'>): number {
  const odds = nominalOdds(bet.legs);
  return bet.stake * (odds - 1);
}

/** Money genuinely exposed. Free bets risk nothing. */
export function amountAtRisk(bet: Pick<Bet, 'stake' | 'isFreeBet'>): number {
  return bet.isFreeBet ? 0 : bet.stake;
}

export function isPending(bet: Bet): boolean {
  return deriveBetStatus(bet) === 'pending';
}

export function isSettled(bet: Bet): boolean {
  return !isPending(bet);
}

/** Won / lost for win-rate purposes. Pushes, voids and pending slips are excluded. */
export function isWin(status: BetStatus): boolean {
  return status === 'won' || status === 'half_won';
}

export function isLoss(status: BetStatus): boolean {
  return status === 'lost' || status === 'half_lost';
}

export function isNeutral(status: BetStatus): boolean {
  return status === 'push' || status === 'void';
}

export const BET_STATUS_LABELS: Record<BetStatus, string> = {
  pending: 'Pending',
  won: 'Won',
  lost: 'Lost',
  push: 'Push',
  void: 'Void',
  half_won: 'Half won',
  half_lost: 'Half lost',
  cashed_out: 'Cashed out',
};

export const LEG_STATUS_LABELS: Record<LegStatus, string> = {
  pending: 'Pending',
  won: 'Won',
  lost: 'Lost',
  push: 'Push',
  void: 'Void',
  half_won: 'Half won',
  half_lost: 'Half lost',
};

export const LEG_STATUSES: LegStatus[] = [
  'pending',
  'won',
  'lost',
  'half_won',
  'half_lost',
  'push',
  'void',
];
