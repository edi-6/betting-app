import { createId } from '../ids';
import type { Bet, Leg, LegStatus, Transaction } from '../types';

export function makeLeg(overrides: Partial<Leg> = {}): Leg {
  return {
    id: overrides.id ?? createId('leg'),
    sport: 'Football',
    league: 'Premier League',
    event: 'Arsenal vs Liverpool',
    market: '1X2',
    selection: 'Arsenal',
    odds: 2,
    status: 'pending',
    ...overrides,
  };
}

export function makeBet(overrides: Partial<Bet> = {}): Bet {
  const placedAt = overrides.placedAt ?? '2026-03-01T12:00:00.000Z';
  return {
    id: overrides.id ?? createId('bet'),
    placedAt,
    settledAt: overrides.settledAt,
    createdAt: placedAt,
    updatedAt: placedAt,
    stake: 100,
    legs: overrides.legs ?? [makeLeg()],
    bookmaker: 'Pinnacle',
    tags: [],
    isFreeBet: false,
    ...overrides,
  };
}

/** A settled single at the given odds. */
export function settledSingle(
  odds: number,
  status: LegStatus,
  overrides: Partial<Bet> = {},
): Bet {
  const placedAt = overrides.placedAt ?? '2026-03-01T12:00:00.000Z';
  return makeBet({
    placedAt,
    settledAt: overrides.settledAt ?? placedAt,
    legs: [makeLeg({ odds, status })],
    ...overrides,
  });
}

export function makeTransaction(overrides: Partial<Transaction> = {}): Transaction {
  const date = overrides.date ?? '2026-01-01T12:00:00.000Z';
  return {
    id: overrides.id ?? createId('txn'),
    type: 'deposit',
    amount: 500,
    date,
    createdAt: date,
    ...overrides,
  };
}
