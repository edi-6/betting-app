/**
 * Reference data used to pre-populate pickers. Everything here is only a suggestion —
 * users can type any sport, league, market or bookmaker they like.
 */

export interface SportDefinition {
  name: string;
  /** Ionicons glyph name. */
  icon: string;
  leagues: string[];
  markets: string[];
}

const COMMON_MARKETS = ['Moneyline', 'Handicap', 'Total (Over/Under)', 'Draw No Bet', 'Live'];

export const SPORTS: SportDefinition[] = [
  {
    name: 'Football',
    icon: 'football-outline',
    leagues: [
      'Premier League',
      'La Liga',
      'Serie A',
      'Bundesliga',
      'Ligue 1',
      'Champions League',
      'Europa League',
      'MLS',
      'Eredivisie',
      'Primeira Liga',
      'Championship',
      'World Cup',
    ],
    markets: [
      '1X2',
      'Double Chance',
      'Asian Handicap',
      'Over/Under Goals',
      'Both Teams To Score',
      'Correct Score',
      'Anytime Goalscorer',
      'Corners',
      'Cards',
      'Draw No Bet',
    ],
  },
  {
    name: 'Basketball',
    icon: 'basketball-outline',
    leagues: ['NBA', 'EuroLeague', 'NCAAB', 'WNBA', 'ACB', 'BBL'],
    markets: [
      'Moneyline',
      'Spread',
      'Total Points',
      'Player Points',
      'Player Rebounds',
      'Player Assists',
      'First Half Spread',
      'Quarter Moneyline',
    ],
  },
  {
    name: 'American Football',
    icon: 'american-football-outline',
    leagues: ['NFL', 'NCAAF', 'CFL'],
    markets: [
      'Moneyline',
      'Spread',
      'Total Points',
      'Player Passing Yards',
      'Player Rushing Yards',
      'Player Receptions',
      'Anytime Touchdown',
      'First Half Spread',
    ],
  },
  {
    name: 'Tennis',
    icon: 'tennisball-outline',
    leagues: ['ATP', 'WTA', 'Grand Slam', 'Challenger', 'ITF'],
    markets: [
      'Match Winner',
      'Set Betting',
      'Games Handicap',
      'Total Games',
      'First Set Winner',
      'Total Sets',
    ],
  },
  {
    name: 'Ice Hockey',
    icon: 'snow-outline',
    leagues: ['NHL', 'KHL', 'SHL', 'Liiga', 'DEL'],
    markets: ['Moneyline', 'Puck Line', 'Total Goals', 'Anytime Goalscorer', 'Period Betting'],
  },
  {
    name: 'Baseball',
    icon: 'baseball-outline',
    leagues: ['MLB', 'NPB', 'KBO'],
    markets: ['Moneyline', 'Run Line', 'Total Runs', 'First 5 Innings', 'Player Strikeouts'],
  },
  {
    name: 'MMA',
    icon: 'hand-left-outline',
    leagues: ['UFC', 'Bellator', 'PFL', 'ONE'],
    markets: ['Moneyline', 'Method of Victory', 'Round Betting', 'Total Rounds', 'Fight To Go The Distance'],
  },
  {
    name: 'Boxing',
    icon: 'fitness-outline',
    leagues: ['Heavyweight', 'Middleweight', 'Welterweight', 'Lightweight'],
    markets: ['Moneyline', 'Method of Victory', 'Round Betting', 'Total Rounds'],
  },
  {
    name: 'Golf',
    icon: 'golf-outline',
    leagues: ['PGA Tour', 'DP World Tour', 'Majors', 'LIV'],
    markets: ['Outright Winner', 'Top 5 Finish', 'Top 10 Finish', 'Tournament Matchups', '3-Ball'],
  },
  {
    name: 'Cricket',
    icon: 'ellipse-outline',
    leagues: ['IPL', 'Test', 'ODI', 'T20 World Cup', 'BBL'],
    markets: ['Match Winner', 'Top Batsman', 'Total Runs', 'Method of Dismissal'],
  },
  {
    name: 'Motorsport',
    icon: 'car-sport-outline',
    leagues: ['Formula 1', 'MotoGP', 'NASCAR', 'IndyCar'],
    markets: ['Race Winner', 'Podium Finish', 'Head to Head', 'Fastest Lap', 'Points Finish'],
  },
  {
    name: 'Esports',
    icon: 'game-controller-outline',
    leagues: ['LoL', 'CS2', 'Dota 2', 'Valorant', 'Rocket League'],
    markets: ['Match Winner', 'Map Handicap', 'Total Maps', 'Correct Score'],
  },
  {
    name: 'Horse Racing',
    icon: 'ribbon-outline',
    leagues: ['Flat', 'National Hunt', 'Handicap'],
    markets: ['Win', 'Each Way', 'Place', 'Forecast'],
  },
  {
    name: 'Darts',
    icon: 'locate-outline',
    leagues: ['PDC World Championship', 'Premier League Darts', 'World Matchplay'],
    markets: ['Match Winner', 'Set Handicap', 'Total 180s', 'Correct Score'],
  },
  {
    name: 'Rugby',
    icon: 'american-football-outline',
    leagues: ['Six Nations', 'Premiership', 'Super Rugby', 'NRL'],
    markets: ['Match Winner', 'Handicap', 'Total Points', 'First Try Scorer'],
  },
  {
    name: 'Other',
    icon: 'ellipse-outline',
    leagues: [],
    markets: COMMON_MARKETS,
  },
];

export const SPORT_NAMES = SPORTS.map((sport) => sport.name);

export function sportDefinition(name: string): SportDefinition | undefined {
  return SPORTS.find((sport) => sport.name.toLowerCase() === name.toLowerCase());
}

export function iconForSport(name: string): string {
  return sportDefinition(name)?.icon ?? 'ellipse-outline';
}

export function leaguesForSport(name: string): string[] {
  return sportDefinition(name)?.leagues ?? [];
}

export function marketsForSport(name: string): string[] {
  return sportDefinition(name)?.markets ?? COMMON_MARKETS;
}

export const BOOKMAKERS = [
  'Bet365',
  'Pinnacle',
  'DraftKings',
  'FanDuel',
  'BetMGM',
  'Caesars',
  'William Hill',
  'Betfair',
  'Unibet',
  'Bwin',
  'Betway',
  'Ladbrokes',
  'Paddy Power',
  'Sky Bet',
  'PointsBet',
  'ESPN Bet',
  'Fanatics',
  'Stake',
  'Betano',
  '1xBet',
  'Coral',
  'BetRivers',
  'Novibet',
  'Superbet',
  'Other',
];

export const SUGGESTED_TAGS = [
  'value',
  'model',
  'live',
  'tipster',
  'hunch',
  'arb',
  'promo',
  'middle',
  'hedge',
  'system',
  'steam',
  'longshot',
];
