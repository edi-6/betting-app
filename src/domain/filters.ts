import { addDays, startOfDay, startOfYear } from './dates';
import { gradeBet } from './analytics';
import { betType, combinedOdds, settleBet } from './settlement';
import type { Bet, BetStatus, BetType } from './types';

export type DateRangePreset = 'all' | '7d' | '30d' | '90d' | 'ytd' | '1y' | 'custom';

export const DATE_RANGE_PRESETS: { key: DateRangePreset; label: string }[] = [
  { key: '7d', label: '7D' },
  { key: '30d', label: '30D' },
  { key: '90d', label: '90D' },
  { key: 'ytd', label: 'YTD' },
  { key: '1y', label: '1Y' },
  { key: 'all', label: 'All' },
];

export interface BetFilter {
  query: string;
  statuses: BetStatus[];
  sports: string[];
  leagues: string[];
  bookmakers: string[];
  tags: string[];
  betType: BetType | 'all';
  range: DateRangePreset;
  /** Only used when `range` is 'custom'. */
  from?: string;
  to?: string;
  minOdds?: number;
  maxOdds?: number;
  minStake?: number;
  maxStake?: number;
}

export const EMPTY_FILTER: BetFilter = {
  query: '',
  statuses: [],
  sports: [],
  leagues: [],
  bookmakers: [],
  tags: [],
  betType: 'all',
  range: 'all',
};

export function resolveRange(
  filter: Pick<BetFilter, 'range' | 'from' | 'to'>,
  now: Date = new Date(),
): { from?: Date; to?: Date } {
  switch (filter.range) {
    case '7d':
      return { from: startOfDay(addDays(now, -6)) };
    case '30d':
      return { from: startOfDay(addDays(now, -29)) };
    case '90d':
      return { from: startOfDay(addDays(now, -89)) };
    case 'ytd':
      return { from: startOfYear(now) };
    case '1y':
      return { from: startOfDay(addDays(now, -364)) };
    case 'custom':
      return {
        from: filter.from ? startOfDay(filter.from) : undefined,
        to: filter.to ? new Date(new Date(filter.to).setHours(23, 59, 59, 999)) : undefined,
      };
    case 'all':
    default:
      return {};
  }
}

function matchesQuery(bet: Bet, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (needle.length === 0) {
    return true;
  }
  const haystack = [
    bet.bookmaker,
    bet.notes ?? '',
    ...bet.tags,
    ...bet.legs.flatMap((leg) => [leg.sport, leg.league, leg.event, leg.market, leg.selection]),
  ]
    .join(' ')
    .toLowerCase();
  return needle
    .split(/\s+/)
    .every((token) => haystack.includes(token));
}

export function applyFilter(bets: Bet[], filter: BetFilter, now: Date = new Date()): Bet[] {
  const { from, to } = resolveRange(filter, now);

  return bets.filter((bet) => {
    const placed = new Date(bet.placedAt).getTime();
    if (from && placed < from.getTime()) return false;
    if (to && placed > to.getTime()) return false;

    if (filter.statuses.length > 0) {
      const status = settleBet(bet).status;
      if (!filter.statuses.includes(status)) return false;
    }

    if (filter.betType !== 'all' && betType(bet) !== filter.betType) return false;

    if (
      filter.bookmakers.length > 0 &&
      !filter.bookmakers.some((b) => b.toLowerCase() === bet.bookmaker.toLowerCase())
    ) {
      return false;
    }

    if (
      filter.sports.length > 0 &&
      !bet.legs.some((leg) => filter.sports.includes(leg.sport))
    ) {
      return false;
    }

    if (
      filter.leagues.length > 0 &&
      !bet.legs.some((leg) => filter.leagues.includes(leg.league))
    ) {
      return false;
    }

    if (filter.tags.length > 0 && !bet.tags.some((tag) => filter.tags.includes(tag))) {
      return false;
    }

    const odds = combinedOdds(bet);
    if (filter.minOdds !== undefined && odds < filter.minOdds) return false;
    if (filter.maxOdds !== undefined && odds > filter.maxOdds) return false;
    if (filter.minStake !== undefined && bet.stake < filter.minStake) return false;
    if (filter.maxStake !== undefined && bet.stake > filter.maxStake) return false;

    return matchesQuery(bet, filter.query);
  });
}

export function countActiveFilters(filter: BetFilter): number {
  let count = 0;
  if (filter.query.trim().length > 0) count += 1;
  if (filter.statuses.length > 0) count += 1;
  if (filter.sports.length > 0) count += 1;
  if (filter.leagues.length > 0) count += 1;
  if (filter.bookmakers.length > 0) count += 1;
  if (filter.tags.length > 0) count += 1;
  if (filter.betType !== 'all') count += 1;
  if (filter.range !== 'all') count += 1;
  if (filter.minOdds !== undefined || filter.maxOdds !== undefined) count += 1;
  if (filter.minStake !== undefined || filter.maxStake !== undefined) count += 1;
  return count;
}

export type SortKey =
  | 'date_desc'
  | 'date_asc'
  | 'stake_desc'
  | 'stake_asc'
  | 'profit_desc'
  | 'profit_asc'
  | 'odds_desc'
  | 'odds_asc';

export const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: 'date_desc', label: 'Newest first' },
  { key: 'date_asc', label: 'Oldest first' },
  { key: 'stake_desc', label: 'Biggest stake' },
  { key: 'profit_desc', label: 'Best result' },
  { key: 'profit_asc', label: 'Worst result' },
  { key: 'odds_desc', label: 'Longest odds' },
  { key: 'odds_asc', label: 'Shortest odds' },
];

export function sortBets(bets: Bet[], key: SortKey): Bet[] {
  const copy = bets.slice();
  const time = (bet: Bet) => new Date(bet.placedAt).getTime();
  const profit = (bet: Bet) => gradeBet(bet).settlement.profit;

  switch (key) {
    case 'date_asc':
      return copy.sort((a, b) => time(a) - time(b));
    case 'stake_desc':
      return copy.sort((a, b) => b.stake - a.stake);
    case 'stake_asc':
      return copy.sort((a, b) => a.stake - b.stake);
    case 'profit_desc':
      return copy.sort((a, b) => profit(b) - profit(a));
    case 'profit_asc':
      return copy.sort((a, b) => profit(a) - profit(b));
    case 'odds_desc':
      return copy.sort((a, b) => combinedOdds(b) - combinedOdds(a));
    case 'odds_asc':
      return copy.sort((a, b) => combinedOdds(a) - combinedOdds(b));
    case 'date_desc':
    default:
      return copy.sort((a, b) => time(b) - time(a));
  }
}

/** Distinct values across the dataset, used to populate the filter sheet. */
export function facetsFor(bets: Bet[]): {
  sports: string[];
  leagues: string[];
  bookmakers: string[];
  tags: string[];
} {
  const sports = new Set<string>();
  const leagues = new Set<string>();
  const bookmakers = new Set<string>();
  const tags = new Set<string>();

  for (const bet of bets) {
    if (bet.bookmaker) bookmakers.add(bet.bookmaker);
    bet.tags.forEach((tag) => tags.add(tag));
    for (const leg of bet.legs) {
      if (leg.sport) sports.add(leg.sport);
      if (leg.league) leagues.add(leg.league);
    }
  }

  const sorted = (set: Set<string>) => Array.from(set).sort((a, b) => a.localeCompare(b));
  return {
    sports: sorted(sports),
    leagues: sorted(leagues),
    bookmakers: sorted(bookmakers),
    tags: sorted(tags),
  };
}
