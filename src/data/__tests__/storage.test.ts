import { summarize } from '../../domain/analytics';
import { createEmptyData, SCHEMA_VERSION } from '../../domain/defaults';
import { createDemoData } from '../../domain/demo';
import {
  clearAppData,
  createMemoryStore,
  loadAppData,
  migrate,
  saveAppData,
  STORAGE_KEY,
} from '../storage';

describe('storage round-trip', () => {
  it('returns null before anything has been saved', async () => {
    await expect(loadAppData(createMemoryStore())).resolves.toBeNull();
  });

  it('saves and reloads the dataset', async () => {
    const store = createMemoryStore();
    const data = createDemoData(new Date('2026-09-11T12:00:00.000Z'));

    await saveAppData(data, store);
    const loaded = await loadAppData(store);

    expect(loaded?.bets).toHaveLength(data.bets.length);
    expect(loaded?.settings.currency).toBe(data.settings.currency);
    expect(summarize(loaded?.bets ?? []).profit).toBeCloseTo(summarize(data.bets).profit, 6);
  });

  it('stamps the current schema version', async () => {
    const store = createMemoryStore();
    await saveAppData({ ...createEmptyData(), version: 0 }, store);
    const raw = await store.getItem(STORAGE_KEY);
    expect(JSON.parse(raw ?? '{}').version).toBe(SCHEMA_VERSION);
  });

  it('quarantines a corrupt payload instead of crashing', async () => {
    const store = createMemoryStore({ [STORAGE_KEY]: '{ not json' });
    await expect(loadAppData(store)).resolves.toBeNull();
  });

  it('repairs a partially valid payload', async () => {
    const store = createMemoryStore({
      [STORAGE_KEY]: JSON.stringify({
        version: 1,
        bets: [{ stake: 10, legs: [{ odds: 2 }] }, { nonsense: true }],
        transactions: 'not an array',
        settings: { currency: 'GBP', oddsFormat: 'bogus' },
      }),
    });

    const loaded = await loadAppData(store);
    expect(loaded?.bets).toHaveLength(1);
    expect(loaded?.transactions).toEqual([]);
    expect(loaded?.settings.currency).toBe('GBP');
    expect(loaded?.settings.oddsFormat).toBe('decimal');
  });

  it('clears everything', async () => {
    const store = createMemoryStore();
    await saveAppData(createEmptyData(), store);
    await clearAppData(store);
    await expect(loadAppData(store)).resolves.toBeNull();
  });
});

describe('migrate', () => {
  it('brings old data up to the current version', () => {
    expect(migrate({ ...createEmptyData(), version: 0 }).version).toBe(SCHEMA_VERSION);
  });

  it('is a no-op for current data', () => {
    const data = createEmptyData();
    expect(migrate(data)).toEqual(data);
  });
});
