import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
} from 'react';

import { clearAppData, loadAppData, saveAppData, type KeyValueStore } from '../data/storage';
import { createDemoData } from '../domain/demo';
import { createEmptyData } from '../domain/defaults';
import { createId } from '../domain/ids';
import type { AppData, Bet, LegStatus, Settings, Transaction } from '../domain/types';

/* -------------------------------------------------------------------------- */
/* Reducer                                                                     */
/* -------------------------------------------------------------------------- */

export interface AppState {
  data: AppData;
  /** False until the persisted data has been read back from storage. */
  hydrated: boolean;
}

export type AppAction =
  | { type: 'hydrate'; data: AppData }
  | { type: 'addBet'; bet: Bet }
  | { type: 'updateBet'; bet: Bet }
  | { type: 'deleteBet'; id: string }
  | { type: 'setLegStatuses'; id: string; statuses: Record<string, LegStatus> }
  | { type: 'cashOut'; id: string; amount: number }
  | { type: 'clearCashOut'; id: string }
  | { type: 'addTransaction'; transaction: Transaction }
  | { type: 'updateTransaction'; transaction: Transaction }
  | { type: 'deleteTransaction'; id: string }
  | { type: 'updateSettings'; settings: Partial<Settings> }
  | { type: 'mergeBets'; bets: Bet[] }
  | { type: 'mergeTransactions'; transactions: Transaction[] }
  | { type: 'replaceData'; data: AppData }
  | { type: 'reset' };

const now = () => new Date().toISOString();

/** Whether every leg has been graded — used to stamp `settledAt`. */
function allLegsGraded(bet: Bet): boolean {
  return bet.legs.length > 0 && bet.legs.every((leg) => leg.status !== 'pending');
}

function touch(bet: Bet): Bet {
  const graded = allLegsGraded(bet) || bet.cashOutReturn !== undefined;
  return {
    ...bet,
    updatedAt: now(),
    settledAt: graded ? (bet.settledAt ?? now()) : undefined,
  };
}

export function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'hydrate':
      return { data: action.data, hydrated: true };

    case 'addBet':
      return {
        ...state,
        data: { ...state.data, bets: [touch(action.bet), ...state.data.bets] },
      };

    case 'updateBet':
      return {
        ...state,
        data: {
          ...state.data,
          bets: state.data.bets.map((bet) => (bet.id === action.bet.id ? touch(action.bet) : bet)),
        },
      };

    case 'deleteBet':
      return {
        ...state,
        data: { ...state.data, bets: state.data.bets.filter((bet) => bet.id !== action.id) },
      };

    case 'setLegStatuses':
      return {
        ...state,
        data: {
          ...state.data,
          bets: state.data.bets.map((bet) => {
            if (bet.id !== action.id) return bet;
            const legs = bet.legs.map((leg) =>
              action.statuses[leg.id] ? { ...leg, status: action.statuses[leg.id] as LegStatus } : leg,
            );
            const next: Bet = { ...bet, legs, cashOutReturn: undefined };
            const graded = allLegsGraded(next);
            return {
              ...next,
              updatedAt: now(),
              settledAt: graded ? (bet.settledAt ?? now()) : undefined,
            };
          }),
        },
      };

    case 'cashOut':
      return {
        ...state,
        data: {
          ...state.data,
          bets: state.data.bets.map((bet) =>
            bet.id === action.id
              ? {
                  ...bet,
                  cashOutReturn: action.amount,
                  settledAt: bet.settledAt ?? now(),
                  updatedAt: now(),
                }
              : bet,
          ),
        },
      };

    case 'clearCashOut':
      return {
        ...state,
        data: {
          ...state.data,
          bets: state.data.bets.map((bet) => {
            if (bet.id !== action.id) return bet;
            const next: Bet = { ...bet, cashOutReturn: undefined, updatedAt: now() };
            return { ...next, settledAt: allLegsGraded(next) ? bet.settledAt : undefined };
          }),
        },
      };

    case 'addTransaction':
      return {
        ...state,
        data: {
          ...state.data,
          transactions: [action.transaction, ...state.data.transactions],
        },
      };

    case 'updateTransaction':
      return {
        ...state,
        data: {
          ...state.data,
          transactions: state.data.transactions.map((transaction) =>
            transaction.id === action.transaction.id ? action.transaction : transaction,
          ),
        },
      };

    case 'deleteTransaction':
      return {
        ...state,
        data: {
          ...state.data,
          transactions: state.data.transactions.filter(
            (transaction) => transaction.id !== action.id,
          ),
        },
      };

    case 'updateSettings':
      return {
        ...state,
        data: {
          ...state.data,
          settings: {
            ...state.data.settings,
            ...action.settings,
            limits: { ...state.data.settings.limits, ...(action.settings.limits ?? {}) },
          },
        },
      };

    case 'mergeBets': {
      // Imported ids replace existing rows so re-importing the same file is idempotent.
      const incoming = new Map(action.bets.map((bet) => [bet.id, bet]));
      const merged = state.data.bets.map((bet) => incoming.get(bet.id) ?? bet);
      const existingIds = new Set(state.data.bets.map((bet) => bet.id));
      const added = action.bets.filter((bet) => !existingIds.has(bet.id));
      return {
        ...state,
        data: {
          ...state.data,
          bets: [...added, ...merged].sort(
            (a, b) => new Date(b.placedAt).getTime() - new Date(a.placedAt).getTime(),
          ),
        },
      };
    }

    case 'mergeTransactions': {
      const incoming = new Map(action.transactions.map((t) => [t.id, t]));
      const merged = state.data.transactions.map((t) => incoming.get(t.id) ?? t);
      const existingIds = new Set(state.data.transactions.map((t) => t.id));
      const added = action.transactions.filter((t) => !existingIds.has(t.id));
      return {
        ...state,
        data: {
          ...state.data,
          transactions: [...added, ...merged].sort(
            (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime(),
          ),
        },
      };
    }

    case 'replaceData':
      return { ...state, data: action.data };

    case 'reset':
      return { ...state, data: createEmptyData() };

    default:
      return state;
  }
}

/* -------------------------------------------------------------------------- */
/* Context                                                                     */
/* -------------------------------------------------------------------------- */

export interface AppContextValue extends AppState {
  bets: Bet[];
  transactions: Transaction[];
  settings: Settings;
  dispatch: React.Dispatch<AppAction>;
  addBet: (bet: Bet) => void;
  updateBet: (bet: Bet) => void;
  deleteBet: (id: string) => void;
  setLegStatuses: (id: string, statuses: Record<string, LegStatus>) => void;
  settleAllLegs: (id: string, status: LegStatus) => void;
  cashOut: (id: string, amount: number) => void;
  clearCashOut: (id: string) => void;
  addTransaction: (input: Omit<Transaction, 'id' | 'createdAt'>) => void;
  updateTransaction: (transaction: Transaction) => void;
  deleteTransaction: (id: string) => void;
  updateSettings: (settings: Partial<Settings>) => void;
  importBets: (bets: Bet[]) => void;
  importTransactions: (transactions: Transaction[]) => void;
  replaceData: (data: AppData) => void;
  loadDemoData: () => void;
  resetAll: () => Promise<void>;
}

const AppContext = createContext<AppContextValue | null>(null);

export interface AppProviderProps {
  children: React.ReactNode;
  /** Injected in tests. Defaults to AsyncStorage. */
  store?: KeyValueStore;
  /** Skip disk hydration and start from this data (used in tests and Storybook-style previews). */
  initialData?: AppData;
}

export function AppProvider({ children, store, initialData }: AppProviderProps) {
  const [state, dispatch] = useReducer(appReducer, {
    data: initialData ?? createEmptyData(),
    hydrated: initialData !== undefined,
  });

  const storeRef = useRef(store);
  useEffect(() => {
    storeRef.current = store;
  }, [store]);

  // Hydrate once on mount.
  useEffect(() => {
    if (initialData !== undefined) {
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const loaded = await loadAppData(storeRef.current);
        if (!cancelled) {
          dispatch({ type: 'hydrate', data: loaded ?? createEmptyData() });
        }
      } catch {
        if (!cancelled) {
          dispatch({ type: 'hydrate', data: createEmptyData() });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persist on every change once hydrated.
  useEffect(() => {
    if (!state.hydrated) {
      return;
    }
    let cancelled = false;
    const handle = setTimeout(() => {
      if (!cancelled) {
        void saveAppData(state.data, storeRef.current).catch(() => undefined);
      }
    }, 150);
    return () => {
      cancelled = true;
      clearTimeout(handle);
    };
  }, [state.data, state.hydrated]);

  const addTransaction = useCallback((input: Omit<Transaction, 'id' | 'createdAt'>) => {
    dispatch({
      type: 'addTransaction',
      transaction: { ...input, id: createId('txn'), createdAt: new Date().toISOString() },
    });
  }, []);

  const resetAll = useCallback(async () => {
    dispatch({ type: 'reset' });
    await clearAppData(storeRef.current).catch(() => undefined);
  }, []);

  const value = useMemo<AppContextValue>(() => {
    return {
      ...state,
      bets: state.data.bets,
      transactions: state.data.transactions,
      settings: state.data.settings,
      dispatch,
      addBet: (bet) => dispatch({ type: 'addBet', bet }),
      updateBet: (bet) => dispatch({ type: 'updateBet', bet }),
      deleteBet: (id) => dispatch({ type: 'deleteBet', id }),
      setLegStatuses: (id, statuses) => dispatch({ type: 'setLegStatuses', id, statuses }),
      settleAllLegs: (id, status) => {
        const bet = state.data.bets.find((candidate) => candidate.id === id);
        if (!bet) return;
        const statuses: Record<string, LegStatus> = {};
        for (const leg of bet.legs) {
          statuses[leg.id] = status;
        }
        dispatch({ type: 'setLegStatuses', id, statuses });
      },
      cashOut: (id, amount) => dispatch({ type: 'cashOut', id, amount }),
      clearCashOut: (id) => dispatch({ type: 'clearCashOut', id }),
      addTransaction,
      updateTransaction: (transaction) => dispatch({ type: 'updateTransaction', transaction }),
      deleteTransaction: (id) => dispatch({ type: 'deleteTransaction', id }),
      updateSettings: (settings) => dispatch({ type: 'updateSettings', settings }),
      importBets: (bets) => dispatch({ type: 'mergeBets', bets }),
      importTransactions: (transactions) => dispatch({ type: 'mergeTransactions', transactions }),
      replaceData: (data) => dispatch({ type: 'replaceData', data }),
      loadDemoData: () => dispatch({ type: 'replaceData', data: createDemoData() }),
      resetAll,
    };
  }, [state, addTransaction, resetAll]);

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp(): AppContextValue {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used inside an <AppProvider>.');
  }
  return context;
}

export function useSettings(): Settings {
  return useApp().settings;
}

export function useBets(): Bet[] {
  return useApp().bets;
}
