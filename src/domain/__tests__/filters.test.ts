import {
  EMPTY_FILTER,
  applyFilter,
  countActiveFilters,
  facetsFor,
  resolveRange,
  sortBets,
} from '../filters';
import { makeBet, makeLeg, settledSingle } from './factories';

const now = new Date('2026-03-15T12:00:00.000Z');

const bets = [
  makeBet({
    id: 'a',
    placedAt: '2026-03-14T12:00:00.000Z',
    stake: 50,
    bookmaker: 'Pinnacle',
    tags: ['model'],
    legs: [makeLeg({ sport: 'Football', league: 'Premier League', odds: 2, status: 'won' })],
    settledAt: '2026-03-14T20:00:00.000Z',
  }),
  makeBet({
    id: 'b',
    placedAt: '2026-02-01T12:00:00.000Z',
    stake: 200,
    bookmaker: 'Bet365',
    tags: ['hunch'],
    legs: [makeLeg({ sport: 'Basketball', league: 'NBA', odds: 4, status: 'lost' })],
    settledAt: '2026-02-01T20:00:00.000Z',
  }),
  makeBet({
    id: 'c',
    placedAt: '2026-03-15T09:00:00.000Z',
    stake: 25,
    bookmaker: 'Pinnacle',
    tags: [],
    legs: [
      makeLeg({ sport: 'Tennis', league: 'ATP', odds: 1.5, event: 'Alcaraz vs Sinner' }),
      makeLeg({ sport: 'Football', league: 'La Liga', odds: 1.8 }),
    ],
  }),
];

describe('applyFilter', () => {
  it('returns everything by default', () => {
    expect(applyFilter(bets, EMPTY_FILTER, now)).toHaveLength(3);
  });

  it('filters by date range', () => {
    const result = applyFilter(bets, { ...EMPTY_FILTER, range: '7d' }, now);
    expect(result.map((bet) => bet.id).sort()).toEqual(['a', 'c']);
  });

  it('filters by status', () => {
    expect(
      applyFilter(bets, { ...EMPTY_FILTER, statuses: ['pending'] }, now).map((bet) => bet.id),
    ).toEqual(['c']);
    expect(
      applyFilter(bets, { ...EMPTY_FILTER, statuses: ['won', 'lost'] }, now).map((bet) => bet.id),
    ).toEqual(['a', 'b']);
  });

  it('filters by sport and league', () => {
    expect(
      applyFilter(bets, { ...EMPTY_FILTER, sports: ['Football'] }, now).map((bet) => bet.id),
    ).toEqual(['a', 'c']);
    expect(
      applyFilter(bets, { ...EMPTY_FILTER, leagues: ['NBA'] }, now).map((bet) => bet.id),
    ).toEqual(['b']);
  });

  it('filters by bookmaker, case-insensitively', () => {
    expect(
      applyFilter(bets, { ...EMPTY_FILTER, bookmakers: ['pinnacle'] }, now).map((bet) => bet.id),
    ).toEqual(['a', 'c']);
  });

  it('filters by tag and bet type', () => {
    expect(applyFilter(bets, { ...EMPTY_FILTER, tags: ['model'] }, now).map((b) => b.id)).toEqual([
      'a',
    ]);
    expect(
      applyFilter(bets, { ...EMPTY_FILTER, betType: 'parlay' }, now).map((b) => b.id),
    ).toEqual(['c']);
  });

  it('filters by odds and stake ranges', () => {
    expect(applyFilter(bets, { ...EMPTY_FILTER, minOdds: 2.5 }, now).map((b) => b.id)).toEqual([
      'b',
      'c',
    ]);
    expect(applyFilter(bets, { ...EMPTY_FILTER, maxStake: 100 }, now).map((b) => b.id)).toEqual([
      'a',
      'c',
    ]);
  });

  it('searches across every text field', () => {
    expect(applyFilter(bets, { ...EMPTY_FILTER, query: 'alcaraz' }, now).map((b) => b.id)).toEqual([
      'c',
    ]);
    expect(applyFilter(bets, { ...EMPTY_FILTER, query: 'nba' }, now).map((b) => b.id)).toEqual([
      'b',
    ]);
    expect(applyFilter(bets, { ...EMPTY_FILTER, query: 'zzz' }, now)).toHaveLength(0);
  });

  it('requires every search token to match', () => {
    expect(
      applyFilter(bets, { ...EMPTY_FILTER, query: 'football premier' }, now).map((b) => b.id),
    ).toEqual(['a']);
  });

  it('supports an explicit custom range', () => {
    const result = applyFilter(
      bets,
      { ...EMPTY_FILTER, range: 'custom', from: '2026-02-01', to: '2026-02-28' },
      now,
    );
    expect(result.map((bet) => bet.id)).toEqual(['b']);
  });
});

describe('resolveRange', () => {
  it('maps presets to a start date', () => {
    expect(resolveRange({ range: 'all' }, now)).toEqual({});
    expect(resolveRange({ range: '7d' }, now).from?.getDate()).toBe(
      new Date('2026-03-09T12:00:00.000Z').getDate(),
    );
    expect(resolveRange({ range: 'ytd' }, now).from?.getMonth()).toBe(0);
  });
});

describe('countActiveFilters', () => {
  it('counts each active dimension once', () => {
    expect(countActiveFilters(EMPTY_FILTER)).toBe(0);
    expect(countActiveFilters({ ...EMPTY_FILTER, query: 'x', range: '7d' })).toBe(2);
    expect(countActiveFilters({ ...EMPTY_FILTER, minOdds: 2, maxOdds: 5 })).toBe(1);
  });
});

describe('sortBets', () => {
  const graded = [
    settledSingle(2, 'won', { id: 'w', stake: 100, placedAt: '2026-03-01T12:00:00.000Z' }),
    settledSingle(5, 'lost', { id: 'l', stake: 300, placedAt: '2026-03-05T12:00:00.000Z' }),
  ];

  it('sorts by date', () => {
    expect(sortBets(graded, 'date_desc').map((b) => b.id)).toEqual(['l', 'w']);
    expect(sortBets(graded, 'date_asc').map((b) => b.id)).toEqual(['w', 'l']);
  });

  it('sorts by stake, profit and odds', () => {
    expect(sortBets(graded, 'stake_desc')[0]?.id).toBe('l');
    expect(sortBets(graded, 'stake_asc')[0]?.id).toBe('w');
    expect(sortBets(graded, 'profit_desc')[0]?.id).toBe('w');
    expect(sortBets(graded, 'profit_asc')[0]?.id).toBe('l');
    expect(sortBets(graded, 'odds_desc')[0]?.id).toBe('l');
    expect(sortBets(graded, 'odds_asc')[0]?.id).toBe('w');
  });

  it('does not mutate the input', () => {
    const input = graded.slice();
    sortBets(input, 'profit_desc');
    expect(input.map((b) => b.id)).toEqual(['w', 'l']);
  });
});

describe('facetsFor', () => {
  it('collects the distinct values, sorted', () => {
    const facets = facetsFor(bets);
    expect(facets.sports).toEqual(['Basketball', 'Football', 'Tennis']);
    expect(facets.bookmakers).toEqual(['Bet365', 'Pinnacle']);
    expect(facets.tags).toEqual(['hunch', 'model']);
    expect(facets.leagues).toContain('Premier League');
  });
});
