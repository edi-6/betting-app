import { DEFAULT_SETTINGS, SCHEMA_VERSION } from './defaults';
import { createId } from './ids';
import { isValidDecimalOdds } from './odds';
import type {
  AppData,
  Bet,
  Leg,
  LegStatus,
  OddsFormat,
  ResponsibleGamblingLimits,
  Settings,
  ThemeMode,
  Transaction,
  TransactionType,
} from './types';

const LEG_STATUS_VALUES: LegStatus[] = [
  'pending',
  'won',
  'lost',
  'push',
  'void',
  'half_won',
  'half_lost',
];

const TRANSACTION_TYPES: TransactionType[] = ['deposit', 'withdrawal', 'adjustment'];
const ODDS_FORMATS: OddsFormat[] = ['decimal', 'american', 'fractional'];
const THEME_MODES: ThemeMode[] = ['system', 'light', 'dark'];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function asNumber(value: unknown, fallback: number): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

function asOptionalNumber(value: unknown): number | undefined {
  if (value === undefined || value === null || value === '') return undefined;
  const parsed = asNumber(value, NaN);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function asBoolean(value: unknown, fallback = false): boolean {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    const lowered = value.trim().toLowerCase();
    if (['true', 'yes', '1', 'y'].includes(lowered)) return true;
    if (['false', 'no', '0', 'n', ''].includes(lowered)) return false;
  }
  return fallback;
}

function asIsoDate(value: unknown, fallback: string): string {
  const text = asString(value);
  if (text.length > 0) {
    const date = new Date(text);
    if (!Number.isNaN(date.getTime())) {
      return date.toISOString();
    }
  }
  return fallback;
}

function asOptionalIsoDate(value: unknown): string | undefined {
  const text = asString(value);
  if (text.length === 0) return undefined;
  const date = new Date(text);
  return Number.isNaN(date.getTime()) ? undefined : date.toISOString();
}

export function sanitizeLeg(raw: unknown): Leg | null {
  if (!isRecord(raw)) return null;
  const odds = asNumber(raw.odds, NaN);
  if (!isValidDecimalOdds(odds)) return null;

  const status = asString(raw.status, 'pending') as LegStatus;
  const closingOdds = asOptionalNumber(raw.closingOdds);

  return {
    id: asString(raw.id) || createId('leg'),
    sport: asString(raw.sport, 'Other'),
    league: asString(raw.league),
    event: asString(raw.event),
    market: asString(raw.market),
    selection: asString(raw.selection),
    odds,
    closingOdds: closingOdds !== undefined && isValidDecimalOdds(closingOdds) ? closingOdds : undefined,
    status: LEG_STATUS_VALUES.includes(status) ? status : 'pending',
    startsAt: asOptionalIsoDate(raw.startsAt),
  };
}

export function sanitizeBet(raw: unknown): Bet | null {
  if (!isRecord(raw)) return null;

  const legsRaw = Array.isArray(raw.legs) ? raw.legs : [];
  const legs = legsRaw.map(sanitizeLeg).filter((leg): leg is Leg => leg !== null);
  if (legs.length === 0) return null;

  const stake = asNumber(raw.stake, NaN);
  if (!Number.isFinite(stake) || stake <= 0) return null;

  const now = new Date().toISOString();
  const placedAt = asIsoDate(raw.placedAt, now);
  const confidence = asOptionalNumber(raw.confidence);
  const cashOutReturn = asOptionalNumber(raw.cashOutReturn);

  return {
    id: asString(raw.id) || createId('bet'),
    placedAt,
    settledAt: asOptionalIsoDate(raw.settledAt),
    createdAt: asIsoDate(raw.createdAt, placedAt),
    updatedAt: asIsoDate(raw.updatedAt, placedAt),
    stake,
    legs,
    bookmaker: asString(raw.bookmaker),
    tags: Array.isArray(raw.tags)
      ? raw.tags.filter((tag): tag is string => typeof tag === 'string' && tag.length > 0)
      : [],
    notes: asString(raw.notes) || undefined,
    isFreeBet: asBoolean(raw.isFreeBet),
    cashOutReturn: cashOutReturn !== undefined && cashOutReturn >= 0 ? cashOutReturn : undefined,
    confidence:
      confidence !== undefined ? Math.min(5, Math.max(1, Math.round(confidence))) : undefined,
  };
}

export function sanitizeTransaction(raw: unknown): Transaction | null {
  if (!isRecord(raw)) return null;
  const amount = asNumber(raw.amount, NaN);
  if (!Number.isFinite(amount)) return null;

  const type = asString(raw.type, 'deposit') as TransactionType;
  const now = new Date().toISOString();
  const date = asIsoDate(raw.date, now);

  return {
    id: asString(raw.id) || createId('txn'),
    type: TRANSACTION_TYPES.includes(type) ? type : 'deposit',
    amount: type === 'adjustment' ? amount : Math.abs(amount),
    date,
    bookmaker: asString(raw.bookmaker) || undefined,
    note: asString(raw.note) || undefined,
    createdAt: asIsoDate(raw.createdAt, date),
  };
}

function sanitizeLimits(raw: unknown): ResponsibleGamblingLimits {
  if (!isRecord(raw)) return {};
  const positive = (value: unknown): number | undefined => {
    const parsed = asOptionalNumber(value);
    return parsed !== undefined && parsed > 0 ? parsed : undefined;
  };
  return {
    dailyStakeLimit: positive(raw.dailyStakeLimit),
    weeklyLossLimit: positive(raw.weeklyLossLimit),
    monthlyLossLimit: positive(raw.monthlyLossLimit),
    maxStakePercent: positive(raw.maxStakePercent),
    dailyBetCountLimit: positive(raw.dailyBetCountLimit),
  };
}

export function sanitizeSettings(raw: unknown): Settings {
  if (!isRecord(raw)) {
    return { ...DEFAULT_SETTINGS, limits: {} };
  }
  const oddsFormat = asString(raw.oddsFormat, DEFAULT_SETTINGS.oddsFormat) as OddsFormat;
  const themeMode = asString(raw.themeMode, DEFAULT_SETTINGS.themeMode) as ThemeMode;
  const kellyFraction = asNumber(raw.kellyFraction, DEFAULT_SETTINGS.kellyFraction);

  return {
    currency: asString(raw.currency, DEFAULT_SETTINGS.currency) || DEFAULT_SETTINGS.currency,
    oddsFormat: ODDS_FORMATS.includes(oddsFormat) ? oddsFormat : DEFAULT_SETTINGS.oddsFormat,
    themeMode: THEME_MODES.includes(themeMode) ? themeMode : DEFAULT_SETTINGS.themeMode,
    defaultStake: Math.max(0, asNumber(raw.defaultStake, DEFAULT_SETTINGS.defaultStake)),
    defaultBookmaker: asString(raw.defaultBookmaker),
    startingBankroll: asNumber(raw.startingBankroll, DEFAULT_SETTINGS.startingBankroll),
    limits: sanitizeLimits(raw.limits),
    kellyFraction: Math.min(Math.max(kellyFraction, 0), 1),
    hapticsEnabled: asBoolean(raw.hapticsEnabled, true),
    disclaimerAcceptedAt: asOptionalIsoDate(raw.disclaimerAcceptedAt),
  };
}

export function sanitizeAppData(raw: unknown): AppData {
  if (!isRecord(raw)) {
    return { version: SCHEMA_VERSION, bets: [], transactions: [], settings: sanitizeSettings(null) };
  }

  const bets = Array.isArray(raw.bets)
    ? raw.bets.map(sanitizeBet).filter((bet): bet is Bet => bet !== null)
    : [];
  const transactions = Array.isArray(raw.transactions)
    ? raw.transactions
        .map(sanitizeTransaction)
        .filter((transaction): transaction is Transaction => transaction !== null)
    : [];

  return {
    version: asNumber(raw.version, SCHEMA_VERSION),
    bets,
    transactions,
    settings: sanitizeSettings(raw.settings),
  };
}

export interface BetDraftErrors {
  stake?: string;
  legs?: string;
  bookmaker?: string;
  general?: string;
  /** Per-leg messages about the event/selection, keyed by leg id. */
  legErrors: Record<string, string>;
  /** Per-leg messages about the price, keyed by leg id. */
  oddsErrors: Record<string, string>;
}

/** Validate a bet before it is saved. Returns an empty object when the bet is valid. */
export function validateBet(bet: Bet): BetDraftErrors {
  const errors: BetDraftErrors = { legErrors: {}, oddsErrors: {} };

  if (!Number.isFinite(bet.stake) || bet.stake <= 0) {
    errors.stake = 'Enter a stake greater than zero.';
  }
  if (bet.legs.length === 0) {
    errors.legs = 'Add at least one selection.';
  }
  if (bet.cashOutReturn !== undefined && bet.cashOutReturn < 0) {
    errors.general = 'Cash-out amount cannot be negative.';
  }

  for (const leg of bet.legs) {
    if (!isValidDecimalOdds(leg.odds)) {
      errors.oddsErrors[leg.id] = 'Enter valid odds.';
    }
    if (leg.selection.trim().length === 0 && leg.event.trim().length === 0) {
      errors.legErrors[leg.id] = 'Name the event or the selection.';
    }
  }

  return errors;
}

export function hasErrors(errors: BetDraftErrors): boolean {
  return (
    errors.stake !== undefined ||
    errors.legs !== undefined ||
    errors.bookmaker !== undefined ||
    errors.general !== undefined ||
    Object.keys(errors.legErrors).length > 0 ||
    Object.keys(errors.oddsErrors).length > 0
  );
}
