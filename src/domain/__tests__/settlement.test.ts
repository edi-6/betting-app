import {
  BET_STATUS_LABELS,
  amountAtRisk,
  betType,
  combinedClosingOdds,
  combinedOdds,
  deriveBetStatus,
  isLoss,
  isNeutral,
  isWin,
  legMultiplier,
  potentialProfit,
  potentialReturns,
  settleBet,
  settlementMultiplier,
} from '../settlement';
import type { LegStatus } from '../types';
import { makeBet, makeLeg, settledSingle } from './factories';

describe('legMultiplier', () => {
  it('maps every outcome to a payout multiplier', () => {
    expect(legMultiplier('won', 2.5)).toBe(2.5);
    expect(legMultiplier('lost', 2.5)).toBe(0);
    expect(legMultiplier('push', 2.5)).toBe(1);
    expect(legMultiplier('void', 2.5)).toBe(1);
    expect(legMultiplier('half_won', 3)).toBe(2);
    expect(legMultiplier('half_lost', 3)).toBe(0.5);
    expect(legMultiplier('pending', 2.5)).toBeUndefined();
  });
});

describe('settleBet — singles', () => {
  it('pays out a winner', () => {
    const settlement = settleBet(settledSingle(2.5, 'won'));
    expect(settlement.status).toBe('won');
    expect(settlement.returns).toBeCloseTo(250, 10);
    expect(settlement.profit).toBeCloseTo(150, 10);
    expect(settlement.isSettled).toBe(true);
  });

  it('loses the stake on a loser', () => {
    const settlement = settleBet(settledSingle(2.5, 'lost'));
    expect(settlement.status).toBe('lost');
    expect(settlement.returns).toBe(0);
    expect(settlement.profit).toBe(-100);
  });

  it('returns the stake on a push and a void', () => {
    expect(settleBet(settledSingle(2.5, 'push')).profit).toBe(0);
    expect(settleBet(settledSingle(2.5, 'push')).status).toBe('push');
    expect(settleBet(settledSingle(2.5, 'void')).status).toBe('void');
    expect(settleBet(settledSingle(2.5, 'void')).returns).toBe(100);
  });

  it('halves the win on an Asian handicap half-win', () => {
    const settlement = settleBet(settledSingle(2, 'half_won'));
    expect(settlement.status).toBe('half_won');
    expect(settlement.returns).toBeCloseTo(150, 10);
    expect(settlement.profit).toBeCloseTo(50, 10);
  });

  it('halves the loss on a half-loss', () => {
    const settlement = settleBet(settledSingle(2, 'half_lost'));
    expect(settlement.status).toBe('half_lost');
    expect(settlement.returns).toBeCloseTo(50, 10);
    expect(settlement.profit).toBeCloseTo(-50, 10);
  });

  it('stays pending until the leg is graded', () => {
    const settlement = settleBet(makeBet());
    expect(settlement.isSettled).toBe(false);
    expect(settlement.status).toBe('pending');
    expect(settlement.profit).toBe(0);
    expect(settlement.multiplier).toBeUndefined();
  });
});

describe('settleBet — free bets', () => {
  it('keeps the winnings but not the stake', () => {
    const bet = settledSingle(3, 'won', { isFreeBet: true });
    const settlement = settleBet(bet);
    expect(settlement.returns).toBeCloseTo(200, 10);
    expect(settlement.profit).toBeCloseTo(200, 10);
  });

  it('costs nothing when it loses', () => {
    const settlement = settleBet(settledSingle(3, 'lost', { isFreeBet: true }));
    expect(settlement.profit).toBe(0);
    expect(settlement.returns).toBe(0);
  });

  it('is a no-op when voided', () => {
    expect(settleBet(settledSingle(3, 'void', { isFreeBet: true })).profit).toBe(0);
  });

  it('risks nothing', () => {
    expect(amountAtRisk({ stake: 100, isFreeBet: true })).toBe(0);
    expect(amountAtRisk({ stake: 100, isFreeBet: false })).toBe(100);
  });
});

describe('settleBet — parlays', () => {
  const parlay = (statuses: LegStatus[], odds = [2, 3]) =>
    makeBet({
      legs: statuses.map((status, index) =>
        makeLeg({ id: `leg${index}`, status, odds: odds[index] ?? 2 }),
      ),
      settledAt: '2026-03-02T12:00:00.000Z',
    });

  it('multiplies winning legs', () => {
    const settlement = settleBet(parlay(['won', 'won']));
    expect(settlement.multiplier).toBeCloseTo(6, 10);
    expect(settlement.returns).toBeCloseTo(600, 10);
    expect(settlement.profit).toBeCloseTo(500, 10);
    expect(settlement.status).toBe('won');
  });

  it('is lost as soon as one leg loses', () => {
    const settlement = settleBet(parlay(['won', 'lost']));
    expect(settlement.status).toBe('lost');
    expect(settlement.profit).toBe(-100);
  });

  it('reprices around a voided leg', () => {
    const settlement = settleBet(parlay(['won', 'void']));
    expect(settlement.multiplier).toBeCloseTo(2, 10);
    expect(settlement.profit).toBeCloseTo(100, 10);
    expect(combinedOdds(parlay(['won', 'void']))).toBeCloseTo(2, 10);
  });

  it('handles a half-lost leg inside a winning parlay', () => {
    const settlement = settleBet(parlay(['won', 'half_lost']));
    expect(settlement.multiplier).toBeCloseTo(1, 10);
    expect(settlement.profit).toBeCloseTo(0, 10);
    expect(settlement.status).toBe('push');
  });

  it('flags a partial win when a half-won leg carries the slip', () => {
    const settlement = settleBet(parlay(['won', 'half_won']));
    expect(settlement.multiplier).toBeCloseTo(4, 10);
    expect(settlement.status).toBe('half_won');
  });

  it('stays pending while any leg is ungraded', () => {
    expect(settleBet(parlay(['won', 'pending'])).isSettled).toBe(false);
    expect(settlementMultiplier([{ status: 'won', odds: 2 }, { status: 'pending', odds: 2 }]))
      .toBeUndefined();
  });

  it('reports the right bet type', () => {
    expect(betType(parlay(['won', 'won']))).toBe('parlay');
    expect(betType(makeBet())).toBe('single');
  });
});

describe('settleBet — cash out', () => {
  it('overrides leg settlement', () => {
    const bet = makeBet({ cashOutReturn: 140 });
    const settlement = settleBet(bet);
    expect(settlement.status).toBe('cashed_out');
    expect(settlement.returns).toBe(140);
    expect(settlement.profit).toBeCloseTo(40, 10);
    expect(settlement.isSettled).toBe(true);
  });

  it('can book a loss', () => {
    expect(settleBet(makeBet({ cashOutReturn: 60 })).profit).toBeCloseTo(-40, 10);
  });

  it('keeps the whole cash-out for a free bet', () => {
    expect(settleBet(makeBet({ cashOutReturn: 60, isFreeBet: true })).profit).toBe(60);
  });
});

describe('derived helpers', () => {
  it('classifies statuses', () => {
    expect(isWin('won')).toBe(true);
    expect(isWin('half_won')).toBe(true);
    expect(isLoss('lost')).toBe(true);
    expect(isLoss('half_lost')).toBe(true);
    expect(isNeutral('void')).toBe(true);
    expect(isNeutral('push')).toBe(true);
    expect(isWin('push')).toBe(false);
  });

  it('labels every status', () => {
    expect(BET_STATUS_LABELS.cashed_out).toBe('Cashed out');
    expect(Object.keys(BET_STATUS_LABELS)).toHaveLength(8);
  });

  it('treats an empty slip as pending', () => {
    expect(deriveBetStatus({ legs: [], cashOutReturn: undefined })).toBe('pending');
  });

  it('projects potential returns', () => {
    const bet = makeBet({ legs: [makeLeg({ odds: 2 }), makeLeg({ odds: 3 })] });
    expect(potentialReturns(bet)).toBeCloseTo(600, 10);
    expect(potentialProfit(bet)).toBeCloseTo(500, 10);
    expect(potentialReturns({ ...bet, isFreeBet: true })).toBeCloseTo(500, 10);
  });

  it('only reports combined closing odds when every leg has one', () => {
    const withClosing = makeBet({
      legs: [makeLeg({ odds: 2, closingOdds: 1.9 }), makeLeg({ odds: 3, closingOdds: 2.8 })],
    });
    expect(combinedClosingOdds(withClosing)).toBeCloseTo(5.32, 10);

    const partial = makeBet({
      legs: [makeLeg({ odds: 2, closingOdds: 1.9 }), makeLeg({ odds: 3 })],
    });
    expect(combinedClosingOdds(partial)).toBeUndefined();
  });
});
