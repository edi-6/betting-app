import { createDemoData } from '../demo';
import { settleBet } from '../settlement';
import { clvStats, riskStats, summarize } from '../analytics';
import {
  hasErrors,
  sanitizeAppData,
  sanitizeBet,
  sanitizeLeg,
  sanitizeSettings,
  sanitizeTransaction,
  validateBet,
} from '../validation';
import { makeBet, makeLeg } from './factories';

describe('sanitizeLeg', () => {
  it('accepts a well-formed leg', () => {
    const leg = sanitizeLeg({ odds: 2.5, sport: 'Football', status: 'won' });
    expect(leg?.odds).toBe(2.5);
    expect(leg?.status).toBe('won');
    expect(leg?.id).toBeTruthy();
  });

  it('coerces numeric strings', () => {
    expect(sanitizeLeg({ odds: '2.5' })?.odds).toBe(2.5);
  });

  it('rejects invalid odds', () => {
    expect(sanitizeLeg({ odds: 0.5 })).toBeNull();
    expect(sanitizeLeg({ odds: 'abc' })).toBeNull();
    expect(sanitizeLeg(null)).toBeNull();
  });

  it('falls back to pending for an unknown status', () => {
    expect(sanitizeLeg({ odds: 2, status: 'exploded' })?.status).toBe('pending');
  });

  it('drops an invalid closing price', () => {
    expect(sanitizeLeg({ odds: 2, closingOdds: 0.4 })?.closingOdds).toBeUndefined();
    expect(sanitizeLeg({ odds: 2, closingOdds: 1.9 })?.closingOdds).toBe(1.9);
  });
});

describe('sanitizeBet', () => {
  it('rejects a bet with no valid legs or stake', () => {
    expect(sanitizeBet({ stake: 10, legs: [] })).toBeNull();
    expect(sanitizeBet({ stake: 0, legs: [{ odds: 2 }] })).toBeNull();
    expect(sanitizeBet({ stake: -5, legs: [{ odds: 2 }] })).toBeNull();
  });

  it('keeps only string tags', () => {
    const bet = sanitizeBet({ stake: 10, legs: [{ odds: 2 }], tags: ['a', 5, null, 'b'] });
    expect(bet?.tags).toEqual(['a', 'b']);
  });

  it('clamps confidence to 1–5', () => {
    expect(sanitizeBet({ stake: 10, legs: [{ odds: 2 }], confidence: 9 })?.confidence).toBe(5);
    expect(sanitizeBet({ stake: 10, legs: [{ odds: 2 }], confidence: 0 })?.confidence).toBe(1);
    expect(sanitizeBet({ stake: 10, legs: [{ odds: 2 }] })?.confidence).toBeUndefined();
  });

  it('reads boolean-ish free bet flags', () => {
    expect(sanitizeBet({ stake: 10, legs: [{ odds: 2 }], isFreeBet: 'true' })?.isFreeBet).toBe(true);
    expect(sanitizeBet({ stake: 10, legs: [{ odds: 2 }], isFreeBet: 'no' })?.isFreeBet).toBe(false);
  });

  it('falls back to now for an unreadable date', () => {
    const bet = sanitizeBet({ stake: 10, legs: [{ odds: 2 }], placedAt: 'never' });
    expect(Number.isNaN(new Date(bet?.placedAt ?? '').getTime())).toBe(false);
  });
});

describe('sanitizeTransaction', () => {
  it('normalises the sign by type', () => {
    expect(sanitizeTransaction({ type: 'withdrawal', amount: -200 })?.amount).toBe(200);
    expect(sanitizeTransaction({ type: 'adjustment', amount: -50 })?.amount).toBe(-50);
  });

  it('rejects a non-numeric amount', () => {
    expect(sanitizeTransaction({ type: 'deposit', amount: 'lots' })).toBeNull();
  });
});

describe('sanitizeSettings', () => {
  it('fills in defaults', () => {
    const settings = sanitizeSettings(null);
    expect(settings.currency).toBe('USD');
    expect(settings.oddsFormat).toBe('decimal');
    expect(settings.limits).toEqual({
      dailyStakeLimit: undefined,
      weeklyLossLimit: undefined,
      monthlyLossLimit: undefined,
      maxStakePercent: undefined,
      dailyBetCountLimit: undefined,
    });
  });

  it('rejects unknown enum values', () => {
    expect(sanitizeSettings({ oddsFormat: 'martian' }).oddsFormat).toBe('decimal');
    expect(sanitizeSettings({ themeMode: 'neon' }).themeMode).toBe('system');
  });

  it('clamps the Kelly fraction', () => {
    expect(sanitizeSettings({ kellyFraction: 5 }).kellyFraction).toBe(1);
    expect(sanitizeSettings({ kellyFraction: -1 }).kellyFraction).toBe(0);
  });

  it('drops non-positive limits', () => {
    expect(sanitizeSettings({ limits: { dailyStakeLimit: 0 } }).limits.dailyStakeLimit)
      .toBeUndefined();
    expect(sanitizeSettings({ limits: { dailyStakeLimit: 100 } }).limits.dailyStakeLimit).toBe(100);
  });
});

describe('sanitizeAppData', () => {
  it('survives complete junk', () => {
    const data = sanitizeAppData('nope');
    expect(data.bets).toEqual([]);
    expect(data.transactions).toEqual([]);
    expect(data.settings.currency).toBe('USD');
  });

  it('drops unusable rows but keeps the rest', () => {
    const data = sanitizeAppData({
      bets: [{ stake: 10, legs: [{ odds: 2 }] }, { stake: 'x' }, null],
      transactions: [{ type: 'deposit', amount: 100 }, 'junk'],
    });
    expect(data.bets).toHaveLength(1);
    expect(data.transactions).toHaveLength(1);
  });
});

describe('validateBet', () => {
  it('passes a valid bet', () => {
    expect(hasErrors(validateBet(makeBet()))).toBe(false);
  });

  it('catches a missing stake', () => {
    const errors = validateBet(makeBet({ stake: 0 }));
    expect(errors.stake).toBeDefined();
    expect(hasErrors(errors)).toBe(true);
  });

  it('catches an unnamed selection', () => {
    const bet = makeBet({ legs: [makeLeg({ id: 'x', event: '', selection: '' })] });
    expect(validateBet(bet).legErrors.x).toBeDefined();
  });

  it('catches a negative cash out', () => {
    expect(validateBet(makeBet({ cashOutReturn: -1 })).general).toBeDefined();
  });
});

describe('demo data', () => {
  const data = createDemoData(new Date('2026-09-11T12:00:00.000Z'));

  it('produces a full, valid dataset', () => {
    expect(data.bets.length).toBeGreaterThan(100);
    expect(data.transactions.length).toBeGreaterThan(0);
    data.bets.forEach((bet) => {
      expect(bet.stake).toBeGreaterThan(0);
      expect(bet.legs.length).toBeGreaterThan(0);
      expect(Number.isFinite(settleBet(bet).profit)).toBe(true);
    });
  });

  it('is deterministic for a given seed', () => {
    const a = createDemoData(new Date('2026-09-11T12:00:00.000Z'));
    const b = createDemoData(new Date('2026-09-11T12:00:00.000Z'));
    expect(summarize(a.bets).profit).toBeCloseTo(summarize(b.bets).profit, 8);
    expect(a.bets.length).toBe(b.bets.length);
  });

  it('survives a sanitise round-trip unchanged', () => {
    const restored = sanitizeAppData(JSON.parse(JSON.stringify(data)));
    expect(restored.bets).toHaveLength(data.bets.length);
    expect(summarize(restored.bets).profit).toBeCloseTo(summarize(data.bets).profit, 8);
  });

  it('leaves some bets open and settles the rest', () => {
    const summary = summarize(data.bets);
    expect(summary.settledBets).toBeGreaterThan(0);
    expect(summary.settledBets).toBeLessThanOrEqual(data.bets.length);
  });

  it('is representative rather than a jackpot', () => {
    // The demo drives first-launch impressions and every store screenshot, so it
    // should look like a competent bettor's ledger: modestly ahead, under 50% of
    // bets landing, and with a real drawdown on the way.
    const summary = summarize(data.bets);
    const risk = riskStats(data.bets, data.settings.startingBankroll + 750);

    expect(summary.roi).toBeGreaterThan(0.02);
    expect(summary.roi).toBeLessThan(0.2);
    expect(summary.winRate).toBeLessThan(0.5);
    expect(risk.maxDrawdown).toBeGreaterThan(0);
    expect(risk.maxDrawdownPercent ?? 0).toBeLessThan(0.45);
    // Beating the close is the story the CLV screen tells; the demo should show it.
    expect(clvStats(data.bets).averageClv).toBeGreaterThan(0);
  });

  it('keeps parlays to a believable number of legs', () => {
    const maxLegs = Math.max(...data.bets.map((bet) => bet.legs.length));
    expect(maxLegs).toBeLessThanOrEqual(3);
  });
});
