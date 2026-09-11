import {
  betsPlacedOnDay,
  computeBankroll,
  evaluateLimits,
  netLossSince,
  stakedOnDay,
  warningsForStake,
} from '../bankroll';
import { makeBet, makeTransaction, settledSingle } from './factories';

describe('computeBankroll', () => {
  it('starts at the starting bankroll', () => {
    const state = computeBankroll([], [], 1000);
    expect(state.balance).toBe(1000);
    expect(state.available).toBe(1000);
    expect(state.netDeposits).toBe(0);
    expect(state.returnOnCapital).toBe(0);
  });

  it('adds deposits and subtracts withdrawals', () => {
    const state = computeBankroll(
      [],
      [
        makeTransaction({ type: 'deposit', amount: 500 }),
        makeTransaction({ type: 'withdrawal', amount: 200 }),
        makeTransaction({ type: 'adjustment', amount: -50 }),
      ],
      1000,
    );
    expect(state.deposits).toBe(500);
    expect(state.withdrawals).toBe(200);
    expect(state.adjustments).toBe(-50);
    expect(state.netDeposits).toBe(250);
    expect(state.balance).toBe(1250);
  });

  it('folds realised profit into the balance', () => {
    const state = computeBankroll([settledSingle(2, 'won')], [], 1000);
    expect(state.realizedProfit).toBeCloseTo(100, 10);
    expect(state.balance).toBeCloseTo(1100, 10);
    expect(state.returnOnCapital).toBeCloseTo(0.1, 10);
  });

  it('ring-fences money riding on open bets', () => {
    const state = computeBankroll([makeBet({ stake: 250 })], [], 1000);
    expect(state.atRisk).toBe(250);
    expect(state.balance).toBe(1000);
    expect(state.available).toBe(750);
  });

  it('does not count free bets as money at risk', () => {
    const state = computeBankroll([makeBet({ stake: 250, isFreeBet: true })], [], 1000);
    expect(state.atRisk).toBe(0);
    expect(state.available).toBe(1000);
  });
});

describe('daily aggregates', () => {
  const today = new Date('2026-03-10T15:00:00.000Z');

  it('sums stakes placed on a given day', () => {
    const bets = [
      makeBet({ stake: 50, placedAt: today.toISOString() }),
      makeBet({ stake: 75, placedAt: today.toISOString() }),
      makeBet({ stake: 500, placedAt: '2026-03-01T12:00:00.000Z' }),
    ];
    expect(stakedOnDay(bets, today)).toBe(125);
    expect(betsPlacedOnDay(bets, today)).toBe(2);
  });

  it('reports net losses only', () => {
    const losing = [
      settledSingle(2, 'lost', {
        placedAt: '2026-03-09T12:00:00.000Z',
        settledAt: '2026-03-09T20:00:00.000Z',
      }),
    ];
    expect(netLossSince(losing, new Date('2026-03-01T00:00:00.000Z'))).toBe(100);

    const winning = [
      settledSingle(2, 'won', {
        placedAt: '2026-03-09T12:00:00.000Z',
        settledAt: '2026-03-09T20:00:00.000Z',
      }),
    ];
    expect(netLossSince(winning, new Date('2026-03-01T00:00:00.000Z'))).toBe(0);
  });
});

describe('evaluateLimits', () => {
  const now = new Date('2026-03-10T15:00:00.000Z');

  it('returns nothing when no limits are set', () => {
    expect(evaluateLimits([], {}, 1000, now)).toEqual([]);
  });

  it('flags an exceeded daily stake limit', () => {
    const bets = [makeBet({ stake: 300, placedAt: now.toISOString() })];
    const statuses = evaluateLimits(bets, { dailyStakeLimit: 200 }, 1000, now);
    expect(statuses).toHaveLength(1);
    expect(statuses[0]?.severity).toBe('exceeded');
    expect(statuses[0]?.used).toBe(300);
    expect(statuses[0]?.ratio).toBeCloseTo(1.5, 10);
  });

  it('warns as a limit is approached', () => {
    const bets = [makeBet({ stake: 170, placedAt: now.toISOString() })];
    const statuses = evaluateLimits(bets, { dailyStakeLimit: 200 }, 1000, now);
    expect(statuses[0]?.severity).toBe('approaching');
  });

  it('reports the largest stake as a share of bankroll', () => {
    const bets = [makeBet({ stake: 100, placedAt: now.toISOString() })];
    const statuses = evaluateLimits(bets, { maxStakePercent: 5 }, 1000, now);
    expect(statuses[0]?.used).toBeCloseTo(10, 10);
    expect(statuses[0]?.severity).toBe('exceeded');
  });
});

describe('warningsForStake', () => {
  const now = new Date('2026-03-10T15:00:00.000Z');

  it('is quiet for a sensible stake', () => {
    expect(warningsForStake(20, [], { maxStakePercent: 5 }, 1000, now)).toEqual([]);
  });

  it('warns about an oversized stake', () => {
    const warnings = warningsForStake(200, [], { maxStakePercent: 5 }, 1000, now);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]?.severity).toBe('warning');
    expect(warnings[0]?.message).toContain('20.0%');
  });

  it('blocks a stake larger than the bankroll', () => {
    const warnings = warningsForStake(2000, [], {}, 1000, now);
    expect(warnings.some((warning) => warning.severity === 'block')).toBe(true);
  });

  it('projects the daily stake limit', () => {
    const bets = [makeBet({ stake: 150, placedAt: now.toISOString() })];
    const warnings = warningsForStake(100, bets, { dailyStakeLimit: 200 }, 1000, now);
    expect(warnings.some((warning) => warning.message.includes('daily limit'))).toBe(true);
  });

  it('ignores an empty stake', () => {
    expect(warningsForStake(0, [], { maxStakePercent: 1 }, 1000, now)).toEqual([]);
  });
});
