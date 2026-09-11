import {
  betsToCsv,
  createBackup,
  csvToBets,
  csvToTransactions,
  escapeCsvValue,
  parseCsv,
  readBackup,
  toCsv,
  transactionsToCsv,
} from '../csv';
import { createEmptyData } from '../defaults';
import { settleBet } from '../settlement';
import { makeBet, makeLeg, makeTransaction, settledSingle } from './factories';

describe('CSV primitives', () => {
  it('quotes fields that need it', () => {
    expect(escapeCsvValue('plain')).toBe('plain');
    expect(escapeCsvValue('with,comma')).toBe('"with,comma"');
    expect(escapeCsvValue('say "hi"')).toBe('"say ""hi"""');
    expect(escapeCsvValue('line\nbreak')).toBe('"line\nbreak"');
    expect(escapeCsvValue(undefined)).toBe('');
    expect(escapeCsvValue(null)).toBe('');
    expect(escapeCsvValue(12.5)).toBe('12.5');
  });

  it('round-trips through the parser', () => {
    const rows = [
      ['a', 'b', 'c'],
      ['plain', 'with,comma', 'say "hi"'],
      ['multi\nline', '', '42'],
    ];
    expect(parseCsv(toCsv(rows))).toEqual(rows);
  });

  it('accepts LF, CRLF and a BOM', () => {
    expect(parseCsv('a,b\n1,2')).toEqual([
      ['a', 'b'],
      ['1', '2'],
    ]);
    expect(parseCsv('a,b\r\n1,2\r\n')).toEqual([
      ['a', 'b'],
      ['1', '2'],
    ]);
    expect(parseCsv('﻿a,b\n1,2')[0]).toEqual(['a', 'b']);
  });

  it('drops blank lines', () => {
    expect(parseCsv('a,b\n\n1,2\n')).toHaveLength(2);
  });
});

describe('bets CSV', () => {
  it('writes a header and one row per bet', () => {
    const csv = betsToCsv([settledSingle(2.5, 'won')]);
    const rows = parseCsv(csv);
    expect(rows).toHaveLength(2);
    expect(rows[0]?.[0]).toBe('id');
    expect(rows[1]?.[rows[0]?.indexOf('profit') ?? 0]).toBe('150');
    expect(rows[1]?.[rows[0]?.indexOf('status') ?? 0]).toBe('won');
  });

  it('round-trips a single', () => {
    const original = settledSingle(2.5, 'won', { tags: ['model', 'value'], notes: 'Note, with comma' });
    const { items, errors } = csvToBets(betsToCsv([original]));

    expect(errors).toEqual([]);
    expect(items).toHaveLength(1);
    const imported = items[0];
    expect(imported?.id).toBe(original.id);
    expect(imported?.stake).toBe(original.stake);
    expect(imported?.tags).toEqual(['model', 'value']);
    expect(imported?.notes).toBe('Note, with comma');
    expect(imported?.legs[0]?.odds).toBeCloseTo(2.5, 10);
    expect(settleBet(imported!).profit).toBeCloseTo(150, 8);
  });

  it('round-trips a parlay without losing legs', () => {
    const original = makeBet({
      legs: [
        makeLeg({ id: 'a', odds: 2, status: 'won', selection: 'Arsenal' }),
        makeLeg({ id: 'b', odds: 3, status: 'won', selection: 'Real Madrid' }),
      ],
      settledAt: '2026-03-02T12:00:00.000Z',
    });
    const { items } = csvToBets(betsToCsv([original]));
    expect(items[0]?.legs).toHaveLength(2);
    expect(items[0]?.legs.map((leg) => leg.selection)).toEqual(['Arsenal', 'Real Madrid']);
    expect(settleBet(items[0]!).profit).toBeCloseTo(500, 8);
  });

  it('imports a hand-written spreadsheet with only the flat columns', () => {
    const csv = [
      'placed_at,sport,event,selection,odds,stake,bookmaker,status',
      '2026-02-01,Football,Arsenal vs Spurs,Arsenal,2.10,50,Bet365,won',
      '2026-02-02,Tennis,Alcaraz vs Sinner,Sinner,1.80,50,Bet365,lost',
    ].join('\n');

    const { items, errors } = csvToBets(csv);
    expect(errors).toEqual([]);
    expect(items).toHaveLength(2);
    expect(items[0]?.legs[0]?.status).toBe('won');
    expect(items[1]?.legs[0]?.status).toBe('lost');
    expect(settleBet(items[0]!).profit).toBeCloseTo(55, 8);
  });

  it('reports rows it cannot read instead of throwing', () => {
    const csv = ['placed_at,odds,stake', '2026-02-01,notanumber,50', '2026-02-02,2.0,10'].join(
      '\n',
    );
    const { items, errors, skipped } = csvToBets(csv);
    expect(items).toHaveLength(1);
    expect(errors).toHaveLength(1);
    expect(skipped).toBe(1);
  });

  it('rejects a file without the required columns', () => {
    const { items, errors } = csvToBets('foo,bar\n1,2');
    expect(items).toEqual([]);
    expect(errors[0]).toContain('Missing required columns');
  });

  it('handles an empty file', () => {
    expect(csvToBets('').errors[0]).toBe('The file is empty.');
  });
});

describe('transactions CSV', () => {
  it('round-trips', () => {
    const original = [
      makeTransaction({ type: 'deposit', amount: 500, note: 'Top-up' }),
      makeTransaction({ type: 'withdrawal', amount: 200 }),
    ];
    const { items, errors } = csvToTransactions(transactionsToCsv(original));
    expect(errors).toEqual([]);
    expect(items).toHaveLength(2);
    expect(items[0]?.type).toBe('deposit');
    expect(items[0]?.amount).toBe(500);
    expect(items[0]?.note).toBe('Top-up');
    expect(items[1]?.type).toBe('withdrawal');
  });
});

describe('JSON backup', () => {
  it('round-trips the whole dataset', () => {
    const data = createEmptyData();
    data.bets = [settledSingle(2, 'won')];
    data.transactions = [makeTransaction()];
    data.settings.currency = 'EUR';

    const restored = readBackup(createBackup(data));
    expect(restored?.bets).toHaveLength(1);
    expect(restored?.transactions).toHaveLength(1);
    expect(restored?.settings.currency).toBe('EUR');
  });

  it('rejects junk', () => {
    expect(readBackup('not json')).toBeNull();
    expect(readBackup('"a string"')).toBeNull();
  });

  it('accepts a bare AppData payload', () => {
    const data = createEmptyData();
    data.bets = [settledSingle(2, 'won')];
    expect(readBackup(JSON.stringify(data))?.bets).toHaveLength(1);
  });
});
