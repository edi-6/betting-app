import { SCHEMA_VERSION } from './defaults';
import { createId } from './ids';
import { combinedOdds, settleBet } from './settlement';
import type { AppData, Bet, Leg, Transaction } from './types';
import { sanitizeAppData, sanitizeBet, sanitizeTransaction } from './validation';

/* -------------------------------------------------------------------------- */
/* Generic CSV encoding (RFC 4180)                                             */
/* -------------------------------------------------------------------------- */

export type CsvValue = string | number | boolean | null | undefined;

export function escapeCsvValue(value: CsvValue): string {
  if (value === null || value === undefined) {
    return '';
  }
  const text = String(value);
  if (/[",\r\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

export function toCsv(rows: CsvValue[][]): string {
  return rows.map((row) => row.map(escapeCsvValue).join(',')).join('\r\n');
}

/** RFC 4180 parser: handles quoted fields, escaped quotes and CRLF or LF line endings. */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let inQuotes = false;
  let index = 0;

  // Strip a UTF-8 byte order mark if a spreadsheet added one.
  const input = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;

  while (index < input.length) {
    const char = input[index];

    if (inQuotes) {
      if (char === '"') {
        if (input[index + 1] === '"') {
          field += '"';
          index += 2;
          continue;
        }
        inQuotes = false;
        index += 1;
        continue;
      }
      field += char;
      index += 1;
      continue;
    }

    if (char === '"') {
      inQuotes = true;
      index += 1;
      continue;
    }
    if (char === ',') {
      row.push(field);
      field = '';
      index += 1;
      continue;
    }
    if (char === '\r') {
      index += 1;
      continue;
    }
    if (char === '\n') {
      row.push(field);
      rows.push(row);
      row = [];
      field = '';
      index += 1;
      continue;
    }
    field += char;
    index += 1;
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  return rows.filter((r) => r.some((cell) => cell.trim().length > 0));
}

/* -------------------------------------------------------------------------- */
/* Bets                                                                        */
/* -------------------------------------------------------------------------- */

export const BET_CSV_HEADERS = [
  'id',
  'placed_at',
  'settled_at',
  'bet_type',
  'sport',
  'league',
  'event',
  'market',
  'selection',
  'odds',
  'closing_odds',
  'stake',
  'bookmaker',
  'status',
  'returns',
  'profit',
  'free_bet',
  'cash_out_return',
  'confidence',
  'tags',
  'notes',
  'legs_json',
] as const;

function summarizeLegs(legs: Leg[], field: keyof Leg): string {
  const values = Array.from(new Set(legs.map((leg) => String(leg[field] ?? '')).filter(Boolean)));
  return values.join(' / ');
}

/**
 * One row per bet. Single bets are fully readable in a spreadsheet; parlays keep
 * their per-leg detail in `legs_json` so an export/import round-trip is lossless.
 */
export function betsToCsv(bets: Bet[]): string {
  const rows: CsvValue[][] = [[...BET_CSV_HEADERS]];

  for (const bet of bets) {
    const settlement = settleBet(bet);
    const isParlay = bet.legs.length > 1;
    const first = bet.legs[0];
    const closing = bet.legs.every((leg) => leg.closingOdds !== undefined)
      ? bet.legs.reduce((product, leg) => product * (leg.closingOdds as number), 1)
      : '';

    rows.push([
      bet.id,
      bet.placedAt,
      bet.settledAt ?? '',
      isParlay ? `parlay_${bet.legs.length}` : 'single',
      summarizeLegs(bet.legs, 'sport'),
      summarizeLegs(bet.legs, 'league'),
      isParlay ? `${bet.legs.length}-leg parlay` : (first?.event ?? ''),
      isParlay ? summarizeLegs(bet.legs, 'market') : (first?.market ?? ''),
      isParlay
        ? bet.legs.map((leg) => leg.selection).join(' + ')
        : (first?.selection ?? ''),
      round(combinedOdds(bet), 4),
      closing === '' ? '' : round(closing, 4),
      round(bet.stake, 2),
      bet.bookmaker,
      settlement.status,
      round(settlement.returns, 2),
      round(settlement.profit, 2),
      bet.isFreeBet ? 'true' : 'false',
      bet.cashOutReturn === undefined ? '' : round(bet.cashOutReturn, 2),
      bet.confidence ?? '',
      bet.tags.join('|'),
      bet.notes ?? '',
      JSON.stringify(bet.legs),
    ]);
  }

  return toCsv(rows);
}

export interface CsvImportResult<T> {
  items: T[];
  /** Human-readable problems, one per rejected row. */
  errors: string[];
  skipped: number;
}

export function csvToBets(text: string): CsvImportResult<Bet> {
  const rows = parseCsv(text);
  const items: Bet[] = [];
  const errors: string[] = [];

  if (rows.length === 0) {
    return { items, errors: ['The file is empty.'], skipped: 0 };
  }

  const header = (rows[0] ?? []).map((cell) => cell.trim().toLowerCase());
  const column = (name: string): number => header.indexOf(name);
  const at = (row: string[], name: string): string => {
    const index = column(name);
    return index >= 0 ? (row[index] ?? '').trim() : '';
  };

  if (column('stake') === -1 || (column('odds') === -1 && column('legs_json') === -1)) {
    return {
      items,
      errors: ['Missing required columns. Expected at least "stake" and "odds".'],
      skipped: rows.length - 1,
    };
  }

  for (let i = 1; i < rows.length; i += 1) {
    const row = rows[i];
    if (!row) continue;

    let legs: unknown[] = [];
    const legsJson = at(row, 'legs_json');
    if (legsJson.length > 0) {
      try {
        const parsed: unknown = JSON.parse(legsJson);
        if (Array.isArray(parsed)) {
          legs = parsed;
        }
      } catch {
        errors.push(`Row ${i + 1}: could not read legs_json, falling back to the flat columns.`);
      }
    }

    if (legs.length === 0) {
      legs = [
        {
          id: createId('leg'),
          sport: at(row, 'sport') || 'Other',
          league: at(row, 'league'),
          event: at(row, 'event'),
          market: at(row, 'market'),
          selection: at(row, 'selection'),
          odds: Number(at(row, 'odds')),
          closingOdds: at(row, 'closing_odds') ? Number(at(row, 'closing_odds')) : undefined,
          status: legStatusFromBetStatus(at(row, 'status')),
        },
      ];
    }

    const tags = at(row, 'tags');
    const bet = sanitizeBet({
      id: at(row, 'id') || createId('bet'),
      placedAt: at(row, 'placed_at'),
      settledAt: at(row, 'settled_at') || undefined,
      stake: Number(at(row, 'stake')),
      legs,
      bookmaker: at(row, 'bookmaker'),
      tags: tags.length > 0 ? tags.split(/[|;]/).map((tag) => tag.trim()).filter(Boolean) : [],
      notes: at(row, 'notes') || undefined,
      isFreeBet: at(row, 'free_bet'),
      cashOutReturn: at(row, 'cash_out_return') || undefined,
      confidence: at(row, 'confidence') || undefined,
    });

    if (bet) {
      items.push(bet);
    } else {
      errors.push(`Row ${i + 1}: skipped — a valid stake and odds are required.`);
    }
  }

  return { items, errors, skipped: rows.length - 1 - items.length };
}

function legStatusFromBetStatus(status: string): string {
  const normalized = status.trim().toLowerCase().replace(/[\s-]+/g, '_');
  switch (normalized) {
    case 'won':
    case 'win':
      return 'won';
    case 'lost':
    case 'lose':
    case 'loss':
      return 'lost';
    case 'push':
    case 'tie':
      return 'push';
    case 'void':
    case 'cancelled':
    case 'canceled':
      return 'void';
    case 'half_won':
      return 'half_won';
    case 'half_lost':
      return 'half_lost';
    case 'cashed_out':
    case 'pending':
    default:
      return normalized === 'cashed_out' ? 'won' : 'pending';
  }
}

/* -------------------------------------------------------------------------- */
/* Transactions                                                                */
/* -------------------------------------------------------------------------- */

export const TRANSACTION_CSV_HEADERS = [
  'id',
  'date',
  'type',
  'amount',
  'bookmaker',
  'note',
] as const;

export function transactionsToCsv(transactions: Transaction[]): string {
  const rows: CsvValue[][] = [[...TRANSACTION_CSV_HEADERS]];
  for (const transaction of transactions) {
    rows.push([
      transaction.id,
      transaction.date,
      transaction.type,
      round(transaction.amount, 2),
      transaction.bookmaker ?? '',
      transaction.note ?? '',
    ]);
  }
  return toCsv(rows);
}

export function csvToTransactions(text: string): CsvImportResult<Transaction> {
  const rows = parseCsv(text);
  const items: Transaction[] = [];
  const errors: string[] = [];

  if (rows.length === 0) {
    return { items, errors: ['The file is empty.'], skipped: 0 };
  }

  const header = (rows[0] ?? []).map((cell) => cell.trim().toLowerCase());
  const at = (row: string[], name: string): string => {
    const index = header.indexOf(name);
    return index >= 0 ? (row[index] ?? '').trim() : '';
  };

  for (let i = 1; i < rows.length; i += 1) {
    const row = rows[i];
    if (!row) continue;
    const transaction = sanitizeTransaction({
      id: at(row, 'id') || createId('txn'),
      date: at(row, 'date'),
      type: at(row, 'type'),
      amount: Number(at(row, 'amount')),
      bookmaker: at(row, 'bookmaker') || undefined,
      note: at(row, 'note') || undefined,
    });
    if (transaction) {
      items.push(transaction);
    } else {
      errors.push(`Row ${i + 1}: skipped — a valid amount is required.`);
    }
  }

  return { items, errors, skipped: rows.length - 1 - items.length };
}

/* -------------------------------------------------------------------------- */
/* Full JSON backup                                                            */
/* -------------------------------------------------------------------------- */

export interface Backup {
  app: 'betledger';
  version: number;
  exportedAt: string;
  data: AppData;
}

export function createBackup(data: AppData): string {
  const backup: Backup = {
    app: 'betledger',
    version: SCHEMA_VERSION,
    exportedAt: new Date().toISOString(),
    data,
  };
  return JSON.stringify(backup, null, 2);
}

export function readBackup(text: string): AppData | null {
  try {
    const parsed: unknown = JSON.parse(text);
    if (typeof parsed !== 'object' || parsed === null) {
      return null;
    }
    const candidate = 'data' in parsed ? (parsed as { data: unknown }).data : parsed;
    const data = sanitizeAppData(candidate);
    if (data.bets.length === 0 && data.transactions.length === 0 && !('settings' in (candidate as object))) {
      return null;
    }
    return data;
  } catch {
    return null;
  }
}

function round(value: number, decimals: number): number {
  if (!Number.isFinite(value)) return 0;
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}
