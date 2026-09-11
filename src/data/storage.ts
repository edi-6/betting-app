import AsyncStorage from '@react-native-async-storage/async-storage';

import { SCHEMA_VERSION } from '../domain/defaults';
import type { AppData } from '../domain/types';
import { sanitizeAppData } from '../domain/validation';

export const STORAGE_KEY = 'betledger/app-data/v1';

/** Minimal key/value contract so tests (and future backends) can swap the storage out. */
export interface KeyValueStore {
  getItem(key: string): Promise<string | null>;
  setItem(key: string, value: string): Promise<void>;
  removeItem(key: string): Promise<void>;
}

export const asyncStorageStore: KeyValueStore = {
  getItem: (key) => AsyncStorage.getItem(key),
  setItem: (key, value) => AsyncStorage.setItem(key, value),
  removeItem: (key) => AsyncStorage.removeItem(key),
};

/** In-memory store used by tests and previews. */
export function createMemoryStore(initial: Record<string, string> = {}): KeyValueStore {
  const map = new Map<string, string>(Object.entries(initial));
  return {
    async getItem(key) {
      return map.get(key) ?? null;
    },
    async setItem(key, value) {
      map.set(key, value);
    },
    async removeItem(key) {
      map.delete(key);
    },
  };
}

/**
 * Forward-only migrations. Each entry upgrades data from version `n` to `n + 1`.
 * Adding a new persisted field only needs a bump of `SCHEMA_VERSION` plus an entry here.
 */
export const migrations: Record<number, (data: AppData) => AppData> = {};

export function migrate(data: AppData): AppData {
  let current = data;
  let version = Number.isFinite(current.version) ? current.version : 0;

  while (version < SCHEMA_VERSION) {
    const step = migrations[version];
    current = step ? step(current) : current;
    version += 1;
  }

  return { ...current, version: SCHEMA_VERSION };
}

export async function loadAppData(store: KeyValueStore = asyncStorageStore): Promise<AppData | null> {
  const raw = await store.getItem(STORAGE_KEY);
  if (raw === null) {
    return null;
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    return migrate(sanitizeAppData(parsed));
  } catch {
    // Corrupt payload: keep a copy for support, then start clean rather than crash-looping.
    await store.setItem(`${STORAGE_KEY}/corrupt-${Date.now()}`, raw);
    return null;
  }
}

export async function saveAppData(
  data: AppData,
  store: KeyValueStore = asyncStorageStore,
): Promise<void> {
  await store.setItem(STORAGE_KEY, JSON.stringify({ ...data, version: SCHEMA_VERSION }));
}

export async function clearAppData(store: KeyValueStore = asyncStorageStore): Promise<void> {
  await store.removeItem(STORAGE_KEY);
}
