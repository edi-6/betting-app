import { combineOdds, impliedProbability, isValidDecimalOdds } from './odds';

/* -------------------------------------------------------------------------- */
/* Expected value                                                              */
/* -------------------------------------------------------------------------- */

export interface ExpectedValueResult {
  /** Expected profit in account currency for the given stake. */
  expectedValue: number;
  /** Expected profit as a share of the stake (the theoretical ROI). */
  expectedRoi: number;
  /** Win probability needed just to break even at these odds. */
  breakEvenProbability: number;
  /** trueProbability − breakEvenProbability. Positive means a positive-EV bet. */
  edge: number;
  /** Fair decimal odds implied by the true probability. */
  fairOdds: number;
}

export function expectedValue(
  decimalOdds: number,
  trueProbability: number,
  stake: number,
): ExpectedValueResult {
  const invalid =
    !isValidDecimalOdds(decimalOdds) ||
    !Number.isFinite(trueProbability) ||
    trueProbability <= 0 ||
    trueProbability >= 1;

  if (invalid) {
    return {
      expectedValue: NaN,
      expectedRoi: NaN,
      breakEvenProbability: NaN,
      edge: NaN,
      fairOdds: NaN,
    };
  }

  const profitIfWon = decimalOdds - 1;
  const expectedRoi = trueProbability * profitIfWon - (1 - trueProbability);
  const breakEven = impliedProbability(decimalOdds);

  return {
    expectedValue: expectedRoi * stake,
    expectedRoi,
    breakEvenProbability: breakEven,
    edge: trueProbability - breakEven,
    fairOdds: 1 / trueProbability,
  };
}

/* -------------------------------------------------------------------------- */
/* Kelly staking                                                               */
/* -------------------------------------------------------------------------- */

export interface KellyResult {
  /** Full-Kelly stake as a fraction of bankroll (0 when the bet has no edge). */
  fullKellyFraction: number;
  /** fullKellyFraction × the user's Kelly multiplier. */
  stakeFraction: number;
  /** The recommended stake in account currency. */
  stake: number;
  edge: number;
  expectedValue: number;
  /** True when the bet has no edge and the correct stake is zero. */
  noEdge: boolean;
}

/**
 * Kelly criterion: f* = (b·p − q) ÷ b, where b = decimalOdds − 1.
 *
 * `fraction` lets the user bet a multiple of full Kelly — quarter-Kelly (0.25) is the
 * usual recommendation because it keeps most of the growth with far less variance.
 */
export function kellyStake(
  bankroll: number,
  decimalOdds: number,
  trueProbability: number,
  fraction = 0.25,
): KellyResult {
  const empty: KellyResult = {
    fullKellyFraction: 0,
    stakeFraction: 0,
    stake: 0,
    edge: 0,
    expectedValue: 0,
    noEdge: true,
  };

  if (
    !isValidDecimalOdds(decimalOdds) ||
    !Number.isFinite(trueProbability) ||
    trueProbability <= 0 ||
    trueProbability >= 1 ||
    !Number.isFinite(bankroll) ||
    bankroll <= 0
  ) {
    return empty;
  }

  const b = decimalOdds - 1;
  const q = 1 - trueProbability;
  const fullKelly = (b * trueProbability - q) / b;
  const ev = expectedValue(decimalOdds, trueProbability, bankroll);

  if (fullKelly <= 0) {
    return { ...empty, edge: ev.edge, expectedValue: 0 };
  }

  const clampedFraction = Math.min(Math.max(fraction, 0), 1);
  const stakeFraction = Math.min(fullKelly * clampedFraction, 1);

  return {
    fullKellyFraction: fullKelly,
    stakeFraction,
    stake: bankroll * stakeFraction,
    edge: ev.edge,
    expectedValue: expectedValue(decimalOdds, trueProbability, bankroll * stakeFraction)
      .expectedValue,
    noEdge: false,
  };
}

/* -------------------------------------------------------------------------- */
/* Arbitrage                                                                   */
/* -------------------------------------------------------------------------- */

export interface ArbitrageResult {
  isArbitrage: boolean;
  /** Σ 1/odds. Below 1 means a guaranteed profit exists. */
  totalImplied: number;
  /** Guaranteed return as a share of the total outlay. */
  profitPercent: number;
  profit: number;
  /** How to split `totalStake` across the outcomes. */
  stakes: number[];
  /** The identical return produced by each outcome. */
  guaranteedReturn: number;
}

export function arbitrage(decimalOdds: number[], totalStake: number): ArbitrageResult {
  const empty: ArbitrageResult = {
    isArbitrage: false,
    totalImplied: NaN,
    profitPercent: NaN,
    profit: NaN,
    stakes: [],
    guaranteedReturn: NaN,
  };

  if (decimalOdds.length < 2 || decimalOdds.some((o) => !isValidDecimalOdds(o))) {
    return empty;
  }
  if (!Number.isFinite(totalStake) || totalStake <= 0) {
    return empty;
  }

  const totalImplied = decimalOdds.reduce((sum, odds) => sum + 1 / odds, 0);
  const stakes = decimalOdds.map((odds) => (totalStake * (1 / odds)) / totalImplied);
  const guaranteedReturn = totalStake / totalImplied;

  return {
    isArbitrage: totalImplied < 1,
    totalImplied,
    profitPercent: 1 / totalImplied - 1,
    profit: guaranteedReturn - totalStake,
    stakes,
    guaranteedReturn,
  };
}

/* -------------------------------------------------------------------------- */
/* Hedging & cash out                                                          */
/* -------------------------------------------------------------------------- */

export interface HedgeResult {
  /** Stake needed on the opposite side to lock in the same result either way. */
  hedgeStake: number;
  /** Profit locked in, net of both stakes. */
  guaranteedProfit: number;
  /** Return if the original bet wins (after paying the hedge stake). */
  profitIfOriginalWins: number;
  /** Return if the hedge wins (after losing the original stake). */
  profitIfHedgeWins: number;
}

/**
 * Work out the stake on the opposing side that equalises both outcomes.
 *
 * Solving `originalStake·originalOdds − hedgeStake = hedgeStake·hedgeOdds − originalStake`
 * gives `hedgeStake = originalStake · originalOdds ÷ hedgeOdds`.
 */
export function hedge(
  originalStake: number,
  originalOdds: number,
  hedgeOdds: number,
): HedgeResult {
  if (
    !isValidDecimalOdds(originalOdds) ||
    !isValidDecimalOdds(hedgeOdds) ||
    !Number.isFinite(originalStake) ||
    originalStake <= 0
  ) {
    return {
      hedgeStake: NaN,
      guaranteedProfit: NaN,
      profitIfOriginalWins: NaN,
      profitIfHedgeWins: NaN,
    };
  }

  const hedgeStake = (originalStake * originalOdds) / hedgeOdds;
  const profitIfOriginalWins = originalStake * originalOdds - originalStake - hedgeStake;
  const profitIfHedgeWins = hedgeStake * hedgeOdds - hedgeStake - originalStake;

  return {
    hedgeStake,
    guaranteedProfit: Math.min(profitIfOriginalWins, profitIfHedgeWins),
    profitIfOriginalWins,
    profitIfHedgeWins,
  };
}

export interface CashOutAssessment {
  /** Risk-neutral value of the open position at the current price. */
  fairValue: number;
  offer: number;
  /** offer − fairValue. Negative means the bookmaker is short-changing you. */
  difference: number;
  /** The margin the bookmaker keeps on the cash-out, as a share of fair value. */
  marginPercent: number;
  recommendation: 'take' | 'hold' | 'unknown';
}

/**
 * A cash-out offer is fair when it equals `stake × originalOdds ÷ currentOdds` —
 * the payout multiplied by the market's current probability of the bet landing.
 */
export function assessCashOut(
  stake: number,
  originalOdds: number,
  currentOdds: number,
  offer: number,
): CashOutAssessment {
  if (
    !isValidDecimalOdds(originalOdds) ||
    !isValidDecimalOdds(currentOdds) ||
    !Number.isFinite(stake) ||
    stake <= 0 ||
    !Number.isFinite(offer)
  ) {
    return {
      fairValue: NaN,
      offer,
      difference: NaN,
      marginPercent: NaN,
      recommendation: 'unknown',
    };
  }

  const fairValue = (stake * originalOdds) / currentOdds;
  const difference = offer - fairValue;

  return {
    fairValue,
    offer,
    difference,
    marginPercent: fairValue > 0 ? -difference / fairValue : NaN,
    recommendation: difference >= 0 ? 'take' : 'hold',
  };
}

/* -------------------------------------------------------------------------- */
/* Parlay                                                                      */
/* -------------------------------------------------------------------------- */

export interface ParlayResult {
  combinedOdds: number;
  returns: number;
  profit: number;
  impliedProbability: number;
}

export function parlay(decimalOdds: number[], stake: number): ParlayResult {
  const valid = decimalOdds.filter(isValidDecimalOdds);
  if (valid.length === 0 || !Number.isFinite(stake) || stake <= 0) {
    return { combinedOdds: NaN, returns: NaN, profit: NaN, impliedProbability: NaN };
  }
  const odds = combineOdds(valid);
  return {
    combinedOdds: odds,
    returns: stake * odds,
    profit: stake * (odds - 1),
    impliedProbability: 1 / odds,
  };
}

/** The win rate you need at these odds simply to break even. */
export function breakEvenWinRate(decimalOdds: number): number {
  return impliedProbability(decimalOdds);
}

/** Units of profit per unit staked if you win at these odds `winRate` of the time. */
export function expectedRoiAtWinRate(decimalOdds: number, winRate: number): number {
  if (!isValidDecimalOdds(decimalOdds) || !Number.isFinite(winRate)) {
    return NaN;
  }
  return winRate * (decimalOdds - 1) - (1 - winRate);
}
