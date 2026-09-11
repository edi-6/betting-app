/**
 * Core domain model for BetLedger.
 *
 * Everything in `src/domain` is pure TypeScript with no React Native imports so it can be
 * unit-tested in isolation and reused by any presentation layer.
 */

/** How odds are displayed. Internally odds are ALWAYS stored as decimal (European) odds. */
export type OddsFormat = 'decimal' | 'american' | 'fractional';

/** Outcome of a single selection inside a bet. */
export type LegStatus =
  | 'pending'
  | 'won'
  | 'lost'
  | 'push'
  | 'void'
  | 'half_won'
  | 'half_lost';

/** Outcome of the whole bet slip. Derived from its legs unless the bet was cashed out. */
export type BetStatus = LegStatus | 'cashed_out';

export type BetType = 'single' | 'parlay';

/** A single selection. A `single` bet has exactly one leg; a `parlay` has two or more. */
export interface Leg {
  id: string;
  /** e.g. "Football", "Basketball" */
  sport: string;
  /** e.g. "NBA", "Premier League" */
  league: string;
  /** e.g. "Lakers vs Celtics" */
  event: string;
  /** e.g. "Moneyline", "Total Points", "Asian Handicap" */
  market: string;
  /** e.g. "Lakers", "Over 214.5" */
  selection: string;
  /** Decimal odds, > 1. */
  odds: number;
  /** Decimal closing odds, used for CLV. Undefined when not recorded. */
  closingOdds?: number;
  status: LegStatus;
  /** ISO-8601 timestamp of kick-off / tip-off. */
  startsAt?: string;
}

export interface Bet {
  id: string;
  /** ISO-8601 timestamp for when the bet was struck. Drives all time-series analytics. */
  placedAt: string;
  /** ISO-8601 timestamp for when the bet was graded. */
  settledAt?: string;
  createdAt: string;
  updatedAt: string;
  /** Amount risked in the account currency. Always > 0. */
  stake: number;
  legs: Leg[];
  bookmaker: string;
  tags: string[];
  notes?: string;
  /** A bonus / free bet: the stake is not returned on a win and is not lost on a loss. */
  isFreeBet: boolean;
  /**
   * Total amount returned by the bookmaker when the slip was cashed out early.
   * When set, it overrides leg-based settlement.
   */
  cashOutReturn?: number;
  /** Subjective 1–5 confidence rating, used for self-calibration analytics. */
  confidence?: number;
}

export type TransactionType = 'deposit' | 'withdrawal' | 'adjustment';

export interface Transaction {
  id: string;
  type: TransactionType;
  /** Always a positive magnitude; `type` carries the sign. `adjustment` may be negative. */
  amount: number;
  date: string;
  bookmaker?: string;
  note?: string;
  createdAt: string;
}

export type ThemeMode = 'system' | 'light' | 'dark';

/** Opt-in guard rails. All values are disabled when undefined. */
export interface ResponsibleGamblingLimits {
  /** Maximum total stake allowed in a single calendar day. */
  dailyStakeLimit?: number;
  /** Maximum net loss tolerated in a calendar week before the app warns. */
  weeklyLossLimit?: number;
  /** Maximum net loss tolerated in a calendar month before the app warns. */
  monthlyLossLimit?: number;
  /** Maximum stake as a percentage (0–100) of the current bankroll. */
  maxStakePercent?: number;
  /** Maximum number of bets placed per day. */
  dailyBetCountLimit?: number;
}

export interface Settings {
  currency: string;
  oddsFormat: OddsFormat;
  themeMode: ThemeMode;
  /** Pre-filled stake on the bet form. */
  defaultStake: number;
  defaultBookmaker: string;
  /** Starting bankroll recorded when the user first set the app up. */
  startingBankroll: number;
  limits: ResponsibleGamblingLimits;
  /** Kelly fraction (0–1) used by the staking calculator, e.g. 0.25 for quarter-Kelly. */
  kellyFraction: number;
  hapticsEnabled: boolean;
  /** Set once the user has acknowledged the responsible-gambling notice. */
  disclaimerAcceptedAt?: string;
}

export interface AppData {
  /** Schema version, used by the migration runner. */
  version: number;
  bets: Bet[];
  transactions: Transaction[];
  settings: Settings;
}

/** The money outcome of one bet. */
export interface Settlement {
  status: BetStatus;
  /** Product of the effective per-leg multipliers. `undefined` while the slip is pending. */
  multiplier?: number;
  /** Cash handed back by the bookmaker (stake included for non-free bets). */
  returns: number;
  /** returns − stake for cash bets; returns for free bets. */
  profit: number;
  /** True once every leg has been graded (or the bet was cashed out). */
  isSettled: boolean;
}
