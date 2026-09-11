import type { AppData, Settings } from './types';

/** Bumped whenever the persisted shape changes; see `src/data/migrations.ts`. */
export const SCHEMA_VERSION = 1;

export const DEFAULT_SETTINGS: Settings = {
  currency: 'USD',
  oddsFormat: 'decimal',
  themeMode: 'system',
  defaultStake: 25,
  defaultBookmaker: '',
  startingBankroll: 0,
  limits: {},
  kellyFraction: 0.25,
  hapticsEnabled: true,
};

export function createEmptyData(): AppData {
  return {
    version: SCHEMA_VERSION,
    bets: [],
    transactions: [],
    settings: { ...DEFAULT_SETTINGS, limits: {} },
  };
}
