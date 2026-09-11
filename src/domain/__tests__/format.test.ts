import {
  currencyFor,
  formatCompactCurrency,
  formatCurrency,
  formatNumber,
  formatPercent,
  formatSignedCurrency,
  formatSignedPercent,
  parseAmount,
  pluralize,
  roundTo,
} from '../format';

describe('formatNumber', () => {
  it('groups thousands', () => {
    expect(formatNumber(1234.5)).toBe('1,234.50');
    expect(formatNumber(1234567.891, 2)).toBe('1,234,567.89');
    expect(formatNumber(999)).toBe('999.00');
    expect(formatNumber(0, 0)).toBe('0');
  });

  it('keeps the sign', () => {
    expect(formatNumber(-1234.5)).toBe('-1,234.50');
  });

  it('handles non-numbers', () => {
    expect(formatNumber(NaN)).toBe('—');
    expect(formatNumber(Infinity)).toBe('—');
  });
});

describe('formatCurrency', () => {
  it('prefixes the symbol', () => {
    expect(formatCurrency(1234.5, 'USD')).toBe('$1,234.50');
    expect(formatCurrency(1234.5, 'EUR')).toBe('€1,234.50');
    expect(formatCurrency(-20, 'GBP')).toBe('-£20.00');
  });

  it('suffixes where that is the convention', () => {
    expect(formatCurrency(150, 'SEK')).toBe('150.00 kr');
  });

  it('falls back for an unknown code', () => {
    expect(formatCurrency(10, 'XYZ')).toBe('XYZ 10.00');
    expect(currencyFor('XYZ').symbol).toBe('XYZ ');
  });
});

describe('formatSignedCurrency', () => {
  it('always shows the direction', () => {
    expect(formatSignedCurrency(120, 'USD')).toBe('+$120.00');
    expect(formatSignedCurrency(-40, 'USD')).toBe('-$40.00');
    expect(formatSignedCurrency(0, 'USD')).toBe('$0.00');
  });

  it('does not show a sign for a value that rounds to zero', () => {
    expect(formatSignedCurrency(0.001, 'USD')).toBe('$0.00');
  });
});

describe('formatCompactCurrency', () => {
  it('abbreviates large amounts', () => {
    expect(formatCompactCurrency(12500, 'USD')).toBe('$12.5k');
    expect(formatCompactCurrency(2_400_000, 'USD')).toBe('$2.4m');
    expect(formatCompactCurrency(-15000, 'USD')).toBe('-$15k');
  });

  it('leaves small amounts alone', () => {
    expect(formatCompactCurrency(85.5, 'USD')).toBe('$85.50');
    expect(formatCompactCurrency(1250, 'USD')).toBe('$1,250');
  });
});

describe('percentages', () => {
  it('formats ratios', () => {
    expect(formatPercent(0.0734, 2)).toBe('7.34%');
    expect(formatPercent(0.5, 1)).toBe('50.0%');
    expect(formatSignedPercent(0.0734, 2)).toBe('+7.34%');
    expect(formatSignedPercent(-0.12, 1)).toBe('-12.0%');
    expect(formatSignedPercent(0, 1)).toBe('0.0%');
  });
});

describe('roundTo', () => {
  it('rounds half away from zero', () => {
    expect(roundTo(1.005, 2)).toBe(1.01);
    expect(roundTo(2.675, 2)).toBe(2.68);
    expect(roundTo(-1.005, 2)).toBe(-1.01);
    expect(roundTo(1.4, 0)).toBe(1);
  });
});

describe('parseAmount', () => {
  it('strips symbols and separators', () => {
    expect(parseAmount('$1,234.50')).toBeCloseTo(1234.5, 10);
    expect(parseAmount('  42 ')).toBe(42);
    expect(parseAmount('-15')).toBe(-15);
    expect(parseAmount('')).toBeNull();
    expect(parseAmount('abc')).toBeNull();
  });
});

describe('pluralize', () => {
  it('picks the right form', () => {
    expect(pluralize(1, 'bet')).toBe('bet');
    expect(pluralize(2, 'bet')).toBe('bets');
    expect(pluralize(0, 'match', 'matches')).toBe('matches');
  });
});
