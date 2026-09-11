/**
 * Locale-independent formatting helpers.
 *
 * Deliberately hand-rolled rather than `Intl`-based: output must be identical on
 * iOS, Android and in Jest regardless of the device locale or Hermes ICU build.
 */

export interface Currency {
  code: string;
  symbol: string;
  /** Symbol goes after the amount (e.g. "12,50 kr"). */
  suffix?: boolean;
}

export const CURRENCIES: Currency[] = [
  { code: 'USD', symbol: '$' },
  { code: 'EUR', symbol: '€' },
  { code: 'GBP', symbol: '£' },
  { code: 'CAD', symbol: 'C$' },
  { code: 'AUD', symbol: 'A$' },
  { code: 'CHF', symbol: 'CHF ' },
  { code: 'SEK', symbol: ' kr', suffix: true },
  { code: 'NOK', symbol: ' kr', suffix: true },
  { code: 'DKK', symbol: ' kr', suffix: true },
  { code: 'PLN', symbol: ' zł', suffix: true },
  { code: 'CZK', symbol: ' Kč', suffix: true },
  { code: 'BRL', symbol: 'R$' },
  { code: 'MXN', symbol: 'MX$' },
  { code: 'JPY', symbol: '¥' },
  { code: 'INR', symbol: '₹' },
  { code: 'NGN', symbol: '₦' },
  { code: 'ZAR', symbol: 'R' },
  { code: 'TRY', symbol: '₺' },
  { code: 'KES', symbol: 'KSh ' },
];

export function currencyFor(code: string): Currency {
  return CURRENCIES.find((c) => c.code === code) ?? { code, symbol: `${code} ` };
}

/** Thousand-separated fixed-decimal number, e.g. 1234.5 → "1,234.50". */
export function formatNumber(value: number, decimals = 2): string {
  if (!Number.isFinite(value)) {
    return '—';
  }
  const negative = value < 0;
  const fixed = Math.abs(value).toFixed(decimals);
  const [wholePart = '0', fractionPart] = fixed.split('.');
  const grouped = wholePart.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  const body = fractionPart ? `${grouped}.${fractionPart}` : grouped;
  return negative ? `-${body}` : body;
}

/** Money in the account currency, e.g. "$1,234.50". */
export function formatCurrency(value: number, currencyCode: string, decimals = 2): string {
  if (!Number.isFinite(value)) {
    return '—';
  }
  const currency = currencyFor(currencyCode);
  const negative = value < 0;
  const body = formatNumber(Math.abs(value), decimals);
  const withSymbol = currency.suffix
    ? `${body}${currency.symbol}`
    : `${currency.symbol}${body}`;
  return negative ? `-${withSymbol}` : withSymbol;
}

/** Money with an explicit sign, used for profit figures: "+$120.00" / "-$40.00". */
export function formatSignedCurrency(value: number, currencyCode: string, decimals = 2): string {
  if (!Number.isFinite(value)) {
    return '—';
  }
  const rounded = roundTo(value, decimals);
  const formatted = formatCurrency(Math.abs(rounded), currencyCode, decimals);
  if (rounded > 0) {
    return `+${formatted}`;
  }
  if (rounded < 0) {
    return `-${formatted}`;
  }
  return formatted;
}

/** Compact money for tight spaces: 12500 → "$12.5k". */
export function formatCompactCurrency(value: number, currencyCode: string): string {
  const abs = Math.abs(value);
  if (abs < 10000) {
    return formatCurrency(value, currencyCode, abs < 100 ? 2 : 0);
  }
  const units: [number, string][] = [
    [1e9, 'b'],
    [1e6, 'm'],
    [1e3, 'k'],
  ];
  for (const [size, suffix] of units) {
    if (abs >= size) {
      const scaled = Math.abs(value) / size;
      // One decimal, but "15.0k" reads worse than "15k".
      const digits = formatNumber(scaled, 1).replace(/\.0$/, '');
      const text = `${digits}${suffix}`;
      const currency = currencyFor(currencyCode);
      const withSymbol = currency.suffix ? `${text}${currency.symbol}` : `${currency.symbol}${text}`;
      return value < 0 ? `-${withSymbol}` : withSymbol;
    }
  }
  return formatCurrency(value, currencyCode, 0);
}

/** Ratio → percentage string. 0.0734 → "7.34%". */
export function formatPercent(ratio: number, decimals = 1): string {
  if (!Number.isFinite(ratio)) {
    return '—';
  }
  return `${formatNumber(ratio * 100, decimals)}%`;
}

/** Ratio → signed percentage string. 0.0734 → "+7.34%". */
export function formatSignedPercent(ratio: number, decimals = 1): string {
  if (!Number.isFinite(ratio)) {
    return '—';
  }
  const percent = roundTo(ratio * 100, decimals);
  const body = `${formatNumber(Math.abs(percent), decimals)}%`;
  if (percent > 0) return `+${body}`;
  if (percent < 0) return `-${body}`;
  return body;
}

export function roundTo(value: number, decimals = 2): number {
  if (!Number.isFinite(value)) {
    return value;
  }
  const factor = 10 ** decimals;
  return Math.round((value + Number.EPSILON * Math.sign(value)) * factor) / factor;
}

/** Strip a currency symbol and separators from user input. */
export function parseAmount(input: string): number | null {
  const cleaned = input.replace(/[^\d.,-]/g, '').replace(/,/g, '');
  if (cleaned.length === 0) {
    return null;
  }
  const value = Number(cleaned);
  return Number.isFinite(value) ? value : null;
}

export function pluralize(count: number, singular: string, plural?: string): string {
  return count === 1 ? singular : (plural ?? `${singular}s`);
}
