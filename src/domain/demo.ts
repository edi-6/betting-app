import { BOOKMAKERS, SPORTS } from './catalog';
import { createEmptyData } from './defaults';
import { addDays } from './dates';
import { createId } from './ids';
import { impliedProbability } from './odds';
import type { AppData, Bet, Leg, LegStatus, Transaction } from './types';

/** Deterministic PRNG (mulberry32) so the demo dataset is reproducible. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const TEAMS: Record<string, string[]> = {
  Football: [
    'Arsenal',
    'Liverpool',
    'Real Madrid',
    'Barcelona',
    'Bayern Munich',
    'Inter',
    'Napoli',
    'PSG',
    'Ajax',
    'Benfica',
    'Man City',
    'Atletico Madrid',
  ],
  Basketball: [
    'Lakers',
    'Celtics',
    'Nuggets',
    'Bucks',
    'Warriors',
    'Heat',
    'Knicks',
    'Suns',
    'Thunder',
    'Mavericks',
  ],
  'American Football': [
    'Chiefs',
    'Eagles',
    '49ers',
    'Ravens',
    'Bills',
    'Cowboys',
    'Lions',
    'Dolphins',
  ],
  Tennis: ['Alcaraz', 'Sinner', 'Djokovic', 'Zverev', 'Rune', 'Medvedev', 'Swiatek', 'Sabalenka'],
  'Ice Hockey': ['Bruins', 'Oilers', 'Panthers', 'Rangers', 'Avalanche', 'Stars'],
  Baseball: ['Dodgers', 'Yankees', 'Braves', 'Astros', 'Orioles', 'Phillies'],
  MMA: ['Makhachev', 'Pereira', 'Topuria', 'Edwards', 'Volkanovski', 'Adesanya'],
  Esports: ['G2', 'Faze', 'T1', 'Navi', 'Fnatic', 'Vitality'],
};

const TAG_POOL = ['value', 'model', 'live', 'tipster', 'hunch', 'promo', 'steam', 'longshot'];

function pick<T>(random: () => number, items: T[]): T {
  const index = Math.min(items.length - 1, Math.floor(random() * items.length));
  return items[index] as T;
}

function roundOdds(value: number): number {
  return Math.round(value * 100) / 100;
}

/**
 * Seed chosen so the sample history is representative rather than a jackpot:
 * roughly +8% ROI over 180 bets, a 21% drawdown along the way, a sub-50% win rate
 * and positive closing line value. Changing it will change every demo screenshot.
 */
export const DEMO_SEED = 5;

/**
 * A realistic six-month history: ~180 bets across several sports and books, with a
 * positive edge on modelled picks and a negative one on hunches — enough to make
 * every analytics screen meaningful on first launch.
 */
export function createDemoData(now: Date = new Date(), seed = DEMO_SEED): AppData {
  const random = createRandom(seed);
  const data = createEmptyData();
  const bets: Bet[] = [];
  const transactions: Transaction[] = [];

  const startingBankroll = 1000;
  data.settings = {
    ...data.settings,
    currency: 'USD',
    defaultStake: 25,
    defaultBookmaker: 'Pinnacle',
    startingBankroll,
    kellyFraction: 0.25,
    limits: { dailyStakeLimit: 250, monthlyLossLimit: 600, maxStakePercent: 5 },
    disclaimerAcceptedAt: addDays(now, -182).toISOString(),
  };

  transactions.push({
    id: createId('txn'),
    type: 'deposit',
    amount: 500,
    date: addDays(now, -120).toISOString(),
    bookmaker: 'Pinnacle',
    note: 'Top-up',
    createdAt: addDays(now, -120).toISOString(),
  });
  transactions.push({
    id: createId('txn'),
    type: 'withdrawal',
    amount: 300,
    date: addDays(now, -45).toISOString(),
    bookmaker: 'Bet365',
    note: 'Profit taking',
    createdAt: addDays(now, -45).toISOString(),
  });
  transactions.push({
    id: createId('txn'),
    type: 'deposit',
    amount: 250,
    date: addDays(now, -20).toISOString(),
    bookmaker: 'DraftKings',
    note: 'Top-up',
    createdAt: addDays(now, -20).toISOString(),
  });

  const sportPool = SPORTS.filter((sport) => TEAMS[sport.name]);
  const bookPool = BOOKMAKERS.slice(0, 8);

  for (let i = 0; i < 180; i += 1) {
    const daysAgo = Math.floor(random() * 182);
    const placedAt = addDays(now, -daysAgo);
    placedAt.setHours(10 + Math.floor(random() * 11), Math.floor(random() * 60), 0, 0);

    const isParlay = random() < 0.18;
    // Two or three legs. Four-leg longshots make the whole dataset swing on a
    // single result, which is not what a real ledger looks like.
    const legCount = isParlay ? 2 + Math.floor(random() * 2) : 1;
    const legs: Leg[] = [];
    const tags: string[] = [];

    if (random() < 0.7) {
      tags.push(pick(random, TAG_POOL));
    }
    const isModelled = tags.includes('model') || tags.includes('value');
    const isHunch = tags.includes('hunch') || tags.includes('longshot');

    for (let l = 0; l < legCount; l += 1) {
      const sport = pick(random, sportPool);
      const teams = TEAMS[sport.name] ?? ['Home', 'Away'];
      const home = pick(random, teams);
      let away = pick(random, teams);
      while (away === home && teams.length > 1) {
        away = pick(random, teams);
      }

      // Odds concentrated around the 1.6–3.0 band most bettors live in.
      const odds = roundOdds(1.4 + Math.pow(random(), 2) * 4.5);
      // Closing line: modelled picks tend to shorten (positive CLV), hunches drift out.
      const clvDrift = isModelled ? -0.03 : isHunch ? 0.03 : -0.004;
      const closingOdds = roundOdds(
        Math.max(1.05, odds * (1 + clvDrift + (random() - 0.5) * 0.06)),
      );

      legs.push({
        id: createId('leg'),
        sport: sport.name,
        league: pick(random, sport.leagues.length > 0 ? sport.leagues : ['Main']),
        event: `${home} vs ${away}`,
        market: pick(random, sport.markets),
        selection: random() < 0.5 ? home : away,
        odds,
        closingOdds: random() < 0.85 ? closingOdds : undefined,
        status: 'pending',
        startsAt: placedAt.toISOString(),
      });
    }

    const combined = legs.reduce((product, leg) => product * leg.odds, 1);
    const stake = roundOdds(10 + Math.floor(random() * 8) * 5);

    // Settle everything except the most recent couple of days.
    const isPending = daysAgo <= 2 && random() < 0.8;
    let settledAt: string | undefined;

    if (!isPending) {
      const settleDate = addDays(placedAt, random() < 0.8 ? 0 : 1);
      settleDate.setHours(22, 30, 0, 0);
      settledAt = settleDate.toISOString();

      for (const leg of legs) {
        // Modelled picks beat the implied price; hunches underperform it. The blend
        // leaves the demo bettor modestly ahead, which is what a tracker is for.
        const edge = isModelled ? 0.05 : isHunch ? -0.035 : 0.006;
        const winProbability = Math.min(0.95, Math.max(0.02, impliedProbability(leg.odds) + edge));
        const roll = random();
        let status: LegStatus = roll < winProbability ? 'won' : 'lost';
        if (random() < 0.03) {
          status = 'void';
        } else if (random() < 0.02) {
          status = roll < winProbability ? 'half_won' : 'half_lost';
        }
        leg.status = status;
      }
    }

    const bet: Bet = {
      id: createId('bet'),
      placedAt: placedAt.toISOString(),
      settledAt,
      createdAt: placedAt.toISOString(),
      updatedAt: (settledAt ?? placedAt.toISOString()) as string,
      stake,
      legs,
      bookmaker: pick(random, bookPool),
      tags,
      notes: random() < 0.15 ? 'Line moved after team news.' : undefined,
      isFreeBet: random() < 0.04,
      confidence: random() < 0.6 ? 1 + Math.floor(random() * 5) : undefined,
    };

    // A handful of early cash-outs, priced near their fair value.
    if (!isPending && random() < 0.04) {
      bet.cashOutReturn = roundOdds(stake * (0.5 + random() * (combined - 0.4)));
      bet.legs = bet.legs.map((leg) => ({ ...leg, status: 'pending' as LegStatus }));
    }

    bets.push(bet);
  }

  bets.sort((a, b) => new Date(b.placedAt).getTime() - new Date(a.placedAt).getTime());

  data.bets = bets;
  data.transactions = transactions;
  return data;
}
