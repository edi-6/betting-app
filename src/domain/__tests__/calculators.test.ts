import {
  arbitrage,
  assessCashOut,
  breakEvenWinRate,
  expectedRoiAtWinRate,
  expectedValue,
  hedge,
  kellyStake,
  parlay,
} from '../calculators';

describe('expectedValue', () => {
  it('prices a positive-EV bet', () => {
    const result = expectedValue(2.1, 0.55, 100);
    expect(result.expectedRoi).toBeCloseTo(0.55 * 1.1 - 0.45, 10);
    expect(result.expectedValue).toBeCloseTo(15.5, 8);
    expect(result.breakEvenProbability).toBeCloseTo(1 / 2.1, 10);
    expect(result.edge).toBeCloseTo(0.55 - 1 / 2.1, 10);
    expect(result.fairOdds).toBeCloseTo(1 / 0.55, 10);
  });

  it('prices a negative-EV bet', () => {
    expect(expectedValue(1.8, 0.5, 100).expectedValue).toBeCloseTo(-10, 8);
  });

  it('is exactly zero at the break-even probability', () => {
    expect(expectedValue(2.5, 0.4, 100).expectedValue).toBeCloseTo(0, 10);
  });

  it('rejects impossible inputs', () => {
    expect(Number.isNaN(expectedValue(1, 0.5, 100).expectedValue)).toBe(true);
    expect(Number.isNaN(expectedValue(2, 0, 100).expectedValue)).toBe(true);
    expect(Number.isNaN(expectedValue(2, 1, 100).expectedValue)).toBe(true);
  });
});

describe('kellyStake', () => {
  it('matches the textbook full-Kelly fraction', () => {
    // b = 1, p = 0.6 → f* = (1 × 0.6 − 0.4) ÷ 1 = 0.2
    const result = kellyStake(1000, 2, 0.6, 1);
    expect(result.fullKellyFraction).toBeCloseTo(0.2, 10);
    expect(result.stake).toBeCloseTo(200, 8);
    expect(result.noEdge).toBe(false);
  });

  it('scales by the chosen Kelly fraction', () => {
    expect(kellyStake(1000, 2, 0.6, 0.25).stake).toBeCloseTo(50, 8);
    expect(kellyStake(1000, 2, 0.6, 0.5).stake).toBeCloseTo(100, 8);
  });

  it('stakes nothing without an edge', () => {
    const result = kellyStake(1000, 2, 0.45, 1);
    expect(result.stake).toBe(0);
    expect(result.noEdge).toBe(true);
    expect(result.edge).toBeLessThan(0);
  });

  it('guards against bad inputs', () => {
    expect(kellyStake(0, 2, 0.6).stake).toBe(0);
    expect(kellyStake(1000, 1, 0.6).stake).toBe(0);
    expect(kellyStake(1000, 2, 1.2).stake).toBe(0);
  });

  it('never recommends more than the whole bankroll', () => {
    expect(kellyStake(1000, 1.01, 0.999, 1).stakeFraction).toBeLessThanOrEqual(1);
  });
});

describe('arbitrage', () => {
  it('detects a genuine arb and splits the stakes', () => {
    const result = arbitrage([2.1, 2.1], 1000);
    expect(result.isArbitrage).toBe(true);
    expect(result.totalImplied).toBeCloseTo(2 / 2.1, 10);
    expect(result.stakes[0]).toBeCloseTo(500, 8);
    expect(result.stakes[1]).toBeCloseTo(500, 8);
    expect(result.guaranteedReturn).toBeCloseTo(1050, 8);
    expect(result.profit).toBeCloseTo(50, 8);
    expect(result.profitPercent).toBeCloseTo(0.05, 10);
  });

  it('produces identical returns on every outcome', () => {
    const odds = [2.4, 3.2, 4.1];
    const result = arbitrage(odds, 500);
    const returns = result.stakes.map((stake, index) => stake * (odds[index] as number));
    returns.forEach((value) => expect(value).toBeCloseTo(result.guaranteedReturn, 8));
  });

  it('rejects a book with a margin', () => {
    const result = arbitrage([1.9, 1.9], 1000);
    expect(result.isArbitrage).toBe(false);
    expect(result.profit).toBeLessThan(0);
  });

  it('needs at least two valid prices', () => {
    expect(arbitrage([2], 100).stakes).toEqual([]);
    expect(arbitrage([2, 1], 100).stakes).toEqual([]);
    expect(arbitrage([2, 3], 0).stakes).toEqual([]);
  });
});

describe('hedge', () => {
  it('equalises both outcomes', () => {
    const result = hedge(100, 3, 1.8);
    expect(result.hedgeStake).toBeCloseTo(300 / 1.8, 8);
    expect(result.profitIfOriginalWins).toBeCloseTo(result.profitIfHedgeWins, 8);
    expect(result.guaranteedProfit).toBeCloseTo(result.profitIfOriginalWins, 8);
  });

  it('locks in a profit when the price has shortened', () => {
    expect(hedge(100, 5, 1.5).guaranteedProfit).toBeGreaterThan(0);
  });

  it('locks in a loss when both sides carry a margin', () => {
    // 1/1.5 + 1/2.0 = 1.167 — there is no free money here.
    const result = hedge(100, 1.5, 2);
    expect(result.hedgeStake).toBeCloseTo(75, 8);
    expect(result.guaranteedProfit).toBeCloseTo(-25, 8);
  });

  it('rejects bad inputs', () => {
    expect(Number.isNaN(hedge(0, 2, 2).hedgeStake)).toBe(true);
    expect(Number.isNaN(hedge(100, 1, 2).hedgeStake)).toBe(true);
  });
});

describe('assessCashOut', () => {
  it('values an open position at the current price', () => {
    // A 100 @ 4.00 bet now trading at 2.00 is worth 100 × 4 ÷ 2 = 200.
    const result = assessCashOut(100, 4, 2, 180);
    expect(result.fairValue).toBeCloseTo(200, 8);
    expect(result.difference).toBeCloseTo(-20, 8);
    expect(result.marginPercent).toBeCloseTo(0.1, 8);
    expect(result.recommendation).toBe('hold');
  });

  it('recommends taking a generous offer', () => {
    expect(assessCashOut(100, 4, 2, 220).recommendation).toBe('take');
  });

  it('rejects bad inputs', () => {
    expect(assessCashOut(100, 1, 2, 100).recommendation).toBe('unknown');
  });
});

describe('parlay', () => {
  it('multiplies the legs', () => {
    const result = parlay([2, 3, 1.5], 50);
    expect(result.combinedOdds).toBeCloseTo(9, 10);
    expect(result.returns).toBeCloseTo(450, 8);
    expect(result.profit).toBeCloseTo(400, 8);
    expect(result.impliedProbability).toBeCloseTo(1 / 9, 10);
  });

  it('ignores invalid legs', () => {
    expect(parlay([2, 0.5], 50).combinedOdds).toBeCloseTo(2, 10);
    expect(Number.isNaN(parlay([], 50).combinedOdds)).toBe(true);
  });
});

describe('break-even helpers', () => {
  it('reports the required win rate', () => {
    expect(breakEvenWinRate(2)).toBeCloseTo(0.5, 10);
    expect(breakEvenWinRate(1.91)).toBeCloseTo(0.5236, 4);
  });

  it('projects ROI from a win rate', () => {
    expect(expectedRoiAtWinRate(2, 0.55)).toBeCloseTo(0.1, 10);
    expect(expectedRoiAtWinRate(2, 0.5)).toBeCloseTo(0, 10);
  });
});
