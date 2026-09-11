import { useCallback, useMemo } from 'react';

import {
  formatCompactCurrency,
  formatCurrency,
  formatSignedCurrency,
} from '../domain/format';
import { formatOdds, oddsPlaceholder, parseOdds } from '../domain/odds';
import { useSettings } from '../store/AppStore';

/**
 * Formatting bound to the user's currency and odds-format preferences, so screens
 * never have to thread `settings` through every call.
 */
export function useFormatters() {
  const settings = useSettings();
  const { currency, oddsFormat } = settings;

  const money = useCallback(
    (value: number, decimals = 2) => formatCurrency(value, currency, decimals),
    [currency],
  );
  const signedMoney = useCallback(
    (value: number, decimals = 2) => formatSignedCurrency(value, currency, decimals),
    [currency],
  );
  const compactMoney = useCallback(
    (value: number) => formatCompactCurrency(value, currency),
    [currency],
  );
  const odds = useCallback((decimal: number) => formatOdds(decimal, oddsFormat), [oddsFormat]);
  const readOdds = useCallback((input: string) => parseOdds(input, oddsFormat), [oddsFormat]);

  return useMemo(
    () => ({
      currency,
      oddsFormat,
      money,
      signedMoney,
      compactMoney,
      odds,
      readOdds,
      oddsHint: oddsPlaceholder(oddsFormat),
    }),
    [currency, oddsFormat, money, signedMoney, compactMoney, odds, readOdds],
  );
}
