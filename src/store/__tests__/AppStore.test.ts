import { createEmptyData } from '../../domain/defaults';
import { settleBet } from '../../domain/settlement';
import type { Bet } from '../../domain/types';
import { makeBet, makeLeg, makeTransaction } from '../../domain/__tests__/factories';
import { appReducer, type AppState } from '../AppStore';

function stateWith(bets: Bet[] = []): AppState {
  return { data: { ...createEmptyData(), bets }, hydrated: true };
}

describe('appReducer', () => {
  it('hydrates', () => {
    const data = { ...createEmptyData(), bets: [makeBet()] };
    const next = appReducer({ data: createEmptyData(), hydrated: false }, { type: 'hydrate', data });
    expect(next.hydrated).toBe(true);
    expect(next.data.bets).toHaveLength(1);
  });

  it('adds a bet at the top of the list', () => {
    const first = makeBet({ id: 'a' });
    const second = makeBet({ id: 'b' });
    const next = appReducer(stateWith([first]), { type: 'addBet', bet: second });
    expect(next.data.bets.map((bet) => bet.id)).toEqual(['b', 'a']);
  });

  it('stamps settledAt when a new bet arrives already graded', () => {
    const bet = makeBet({ legs: [makeLeg({ status: 'won' })] });
    const next = appReducer(stateWith(), { type: 'addBet', bet });
    expect(next.data.bets[0]?.settledAt).toBeTruthy();
  });

  it('leaves settledAt empty for a pending bet', () => {
    const next = appReducer(stateWith(), { type: 'addBet', bet: makeBet() });
    expect(next.data.bets[0]?.settledAt).toBeUndefined();
  });

  it('updates a bet in place', () => {
    const bet = makeBet({ id: 'a', stake: 100 });
    const next = appReducer(stateWith([bet]), {
      type: 'updateBet',
      bet: { ...bet, stake: 250 },
    });
    expect(next.data.bets[0]?.stake).toBe(250);
    expect(next.data.bets).toHaveLength(1);
  });

  it('deletes a bet', () => {
    const next = appReducer(stateWith([makeBet({ id: 'a' }), makeBet({ id: 'b' })]), {
      type: 'deleteBet',
      id: 'a',
    });
    expect(next.data.bets.map((bet) => bet.id)).toEqual(['b']);
  });

  it('settles legs and stamps the settlement date', () => {
    const bet = makeBet({ id: 'a', legs: [makeLeg({ id: 'l1', odds: 2 })] });
    const next = appReducer(stateWith([bet]), {
      type: 'setLegStatuses',
      id: 'a',
      statuses: { l1: 'won' },
    });
    const updated = next.data.bets[0]!;
    expect(updated.legs[0]?.status).toBe('won');
    expect(updated.settledAt).toBeTruthy();
    expect(settleBet(updated).profit).toBeCloseTo(100, 8);
  });

  it('keeps a parlay pending until every leg is graded', () => {
    const bet = makeBet({
      id: 'a',
      legs: [makeLeg({ id: 'l1' }), makeLeg({ id: 'l2' })],
    });
    const next = appReducer(stateWith([bet]), {
      type: 'setLegStatuses',
      id: 'a',
      statuses: { l1: 'won' },
    });
    expect(next.data.bets[0]?.settledAt).toBeUndefined();
  });

  it('clears a cash-out when legs are graded again', () => {
    const bet = makeBet({ id: 'a', cashOutReturn: 120, legs: [makeLeg({ id: 'l1' })] });
    const next = appReducer(stateWith([bet]), {
      type: 'setLegStatuses',
      id: 'a',
      statuses: { l1: 'lost' },
    });
    expect(next.data.bets[0]?.cashOutReturn).toBeUndefined();
    expect(settleBet(next.data.bets[0]!).profit).toBe(-100);
  });

  it('records and reverses a cash-out', () => {
    const bet = makeBet({ id: 'a' });
    const cashedOut = appReducer(stateWith([bet]), { type: 'cashOut', id: 'a', amount: 140 });
    expect(cashedOut.data.bets[0]?.cashOutReturn).toBe(140);
    expect(cashedOut.data.bets[0]?.settledAt).toBeTruthy();

    const reverted = appReducer(cashedOut, { type: 'clearCashOut', id: 'a' });
    expect(reverted.data.bets[0]?.cashOutReturn).toBeUndefined();
    expect(reverted.data.bets[0]?.settledAt).toBeUndefined();
  });

  it('manages transactions', () => {
    const transaction = makeTransaction({ id: 't1' });
    let state = appReducer(stateWith(), { type: 'addTransaction', transaction });
    expect(state.data.transactions).toHaveLength(1);

    state = appReducer(state, {
      type: 'updateTransaction',
      transaction: { ...transaction, amount: 999 },
    });
    expect(state.data.transactions[0]?.amount).toBe(999);

    state = appReducer(state, { type: 'deleteTransaction', id: 't1' });
    expect(state.data.transactions).toEqual([]);
  });

  it('merges settings without dropping untouched limits', () => {
    let state = appReducer(stateWith(), {
      type: 'updateSettings',
      settings: { limits: { dailyStakeLimit: 100 } },
    });
    state = appReducer(state, { type: 'updateSettings', settings: { currency: 'EUR' } });

    expect(state.data.settings.currency).toBe('EUR');
    expect(state.data.settings.limits.dailyStakeLimit).toBe(100);
  });

  it('merges imported bets idempotently', () => {
    const existing = makeBet({ id: 'a', stake: 100 });
    const state = stateWith([existing]);

    const once = appReducer(state, {
      type: 'mergeBets',
      bets: [{ ...existing, stake: 250 }, makeBet({ id: 'b' })],
    });
    expect(once.data.bets).toHaveLength(2);
    expect(once.data.bets.find((bet) => bet.id === 'a')?.stake).toBe(250);

    const twice = appReducer(once, {
      type: 'mergeBets',
      bets: [{ ...existing, stake: 250 }, makeBet({ id: 'b' })],
    });
    expect(twice.data.bets).toHaveLength(2);
  });

  it('merges imported transactions idempotently', () => {
    const transaction = makeTransaction({ id: 't1' });
    const once = appReducer(stateWith(), {
      type: 'mergeTransactions',
      transactions: [transaction],
    });
    const twice = appReducer(once, { type: 'mergeTransactions', transactions: [transaction] });
    expect(twice.data.transactions).toHaveLength(1);
  });

  it('replaces and resets the dataset', () => {
    const replaced = appReducer(stateWith([makeBet()]), {
      type: 'replaceData',
      data: { ...createEmptyData(), bets: [makeBet(), makeBet()] },
    });
    expect(replaced.data.bets).toHaveLength(2);

    const reset = appReducer(replaced, { type: 'reset' });
    expect(reset.data.bets).toEqual([]);
    expect(reset.data.transactions).toEqual([]);
  });

  it('ignores actions for unknown ids', () => {
    const state = stateWith([makeBet({ id: 'a' })]);
    expect(appReducer(state, { type: 'deleteBet', id: 'zzz' }).data.bets).toHaveLength(1);
    expect(appReducer(state, { type: 'cashOut', id: 'zzz', amount: 10 }).data.bets[0]?.cashOutReturn)
      .toBeUndefined();
  });
});
