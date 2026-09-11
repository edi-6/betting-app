import type { OddsFormat } from './types';

/** Smallest decimal odds we accept. Anything at or below 1.0 pays nothing. */
export const MIN_DECIMAL_ODDS = 1.0001;
export const MAX_DECIMAL_ODDS = 100000;

export function isValidDecimalOdds(value: number): boolean {
  return Number.isFinite(value) && value >= MIN_DECIMAL_ODDS && value <= MAX_DECIMAL_ODDS;
}

/* -------------------------------------------------------------------------- */
/* Conversions                                                                 */
/* -------------------------------------------------------------------------- */

/** Decimal → American (moneyline). 2.50 → +150, 1.50 → −200. */
export function decimalToAmerican(decimal: number): number {
  if (!isValidDecimalOdds(decimal)) {
    return 0;
  }
  if (decimal >= 2) {
    return Math.round((decimal - 1) * 100);
  }
  return -Math.round(100 / (decimal - 1));
}

/** American (moneyline) → decimal. +150 → 2.50, −200 → 1.50. */
export function americanToDecimal(american: number): number {
  if (!Number.isFinite(american) || american === 0) {
    return NaN;
  }
  if (american > 0) {
    return american / 100 + 1;
  }
  return 100 / Math.abs(american) + 1;
}

export interface Fraction {
  numerator: number;
  denominator: number;
}

/**
 * Best rational approximation of `value` with a denominator no larger than
 * `maxDenominator`, using the continued-fraction convergents of the value.
 */
export function approximateFraction(value: number, maxDenominator = 1000): Fraction {
  if (!Number.isFinite(value) || value <= 0) {
    return { numerator: 0, denominator: 1 };
  }

  let lowerN = 0;
  let lowerD = 1;
  let upperN = 1;
  let upperD = 0;
  let bestN = Math.round(value);
  let bestD = 1;
  let bestError = Math.abs(value - bestN);

  // Stern–Brocot search: always converges within maxDenominator steps.
  for (let i = 0; i < 10000; i += 1) {
    const mediantN = lowerN + upperN;
    const mediantD = lowerD + upperD;
    if (mediantD > maxDenominator) {
      break;
    }
    const mediant = mediantN / mediantD;
    const error = Math.abs(value - mediant);
    if (error < bestError) {
      bestError = error;
      bestN = mediantN;
      bestD = mediantD;
    }
    if (error <= Number.EPSILON * value) {
      break;
    }
    if (mediant < value) {
      lowerN = mediantN;
      lowerD = mediantD;
    } else {
      upperN = mediantN;
      upperD = mediantD;
    }
  }

  const divisor = greatestCommonDivisor(bestN, bestD);
  return { numerator: bestN / divisor, denominator: bestD / divisor };
}

function greatestCommonDivisor(a: number, b: number): number {
  let x = Math.abs(a);
  let y = Math.abs(b);
  while (y > 0) {
    const t = y;
    y = x % y;
    x = t;
  }
  return x === 0 ? 1 : x;
}

/** Decimal → fractional. 3.5 → 5/2, 1.5 → 1/2. */
export function decimalToFraction(decimal: number): Fraction {
  if (!isValidDecimalOdds(decimal)) {
    return { numerator: 0, denominator: 1 };
  }
  return approximateFraction(decimal - 1);
}

/** Fractional → decimal. 5/2 → 3.5. */
export function fractionToDecimal(numerator: number, denominator: number): number {
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator === 0) {
    return NaN;
  }
  return numerator / denominator + 1;
}

/* -------------------------------------------------------------------------- */
/* Probability                                                                 */
/* -------------------------------------------------------------------------- */

/** Bookmaker-implied probability (includes the vig). */
export function impliedProbability(decimal: number): number {
  if (!isValidDecimalOdds(decimal)) {
    return NaN;
  }
  return 1 / decimal;
}

/** Probability (0–1) → fair decimal odds. */
export function probabilityToDecimal(probability: number): number {
  if (!Number.isFinite(probability) || probability <= 0 || probability >= 1) {
    return NaN;
  }
  return 1 / probability;
}

/**
 * Total book percentage. 1.0 is a perfectly fair book; 1.05 means a 5% overround.
 */
export function bookPercentage(decimalOdds: number[]): number {
  return decimalOdds.reduce((sum, odds) => sum + impliedProbability(odds), 0);
}

/** Overround expressed as a percentage, e.g. 4.76 for a typical two-way −110/−110 market. */
export function overround(decimalOdds: number[]): number {
  return (bookPercentage(decimalOdds) - 1) * 100;
}

export type DevigMethod = 'multiplicative' | 'power' | 'shin' | 'additive';

/**
 * Strip the bookmaker margin from a market and return fair probabilities summing to 1.
 *
 * - `multiplicative` — proportional normalisation. Fast, but overstates longshots.
 * - `additive`       — subtracts the margin equally across outcomes.
 * - `power`          — solves Σ pᵢ^k = 1. Favoured for its favourite/longshot handling.
 * - `shin`           — Shin (1992) insider-trading model. Best for two-way markets.
 */
export function removeVig(
  decimalOdds: number[],
  method: DevigMethod = 'multiplicative',
): number[] {
  const raw = decimalOdds.map(impliedProbability);
  if (raw.length === 0 || raw.some((p) => !Number.isFinite(p) || p <= 0)) {
    return [];
  }
  const total = raw.reduce((a, b) => a + b, 0);
  if (total <= 0) {
    return [];
  }

  switch (method) {
    case 'multiplicative':
      return raw.map((p) => p / total);

    case 'additive': {
      const excess = (total - 1) / raw.length;
      const adjusted = raw.map((p) => Math.max(p - excess, 1e-9));
      const sum = adjusted.reduce((a, b) => a + b, 0);
      return adjusted.map((p) => p / sum);
    }

    case 'power': {
      // Σ pᵢ^k = 1, with k ∈ (0, 1] for an overround book.
      const f = (k: number) => raw.reduce((sum, p) => sum + Math.pow(p, k), 0) - 1;
      const k = bisect(f, 0.2, 3, 1e-12, 200);
      const adjusted = raw.map((p) => Math.pow(p, k));
      const sum = adjusted.reduce((a, b) => a + b, 0);
      return adjusted.map((p) => p / sum);
    }

    case 'shin': {
      const shinProbabilities = (z: number) =>
        raw.map((p) => {
          const value =
            (Math.sqrt(z * z + 4 * (1 - z) * ((p * p) / total)) - z) / (2 * (1 - z));
          return value;
        });
      const f = (z: number) => shinProbabilities(z).reduce((a, b) => a + b, 0) - 1;
      const z = bisect(f, 0, 0.9999, 1e-12, 200);
      const adjusted = shinProbabilities(z);
      const sum = adjusted.reduce((a, b) => a + b, 0);
      return adjusted.map((p) => p / sum);
    }

    default:
      return raw.map((p) => p / total);
  }
}

/** Fair (no-vig) decimal odds for each outcome of a market. */
export function fairOdds(decimalOdds: number[], method: DevigMethod = 'multiplicative'): number[] {
  return removeVig(decimalOdds, method).map((p) => (p > 0 ? 1 / p : NaN));
}

function bisect(
  f: (x: number) => number,
  lower: number,
  upper: number,
  tolerance: number,
  maxIterations: number,
): number {
  let lo = lower;
  let hi = upper;
  let fLo = f(lo);
  const fHi = f(hi);
  if (!Number.isFinite(fLo) || !Number.isFinite(fHi) || fLo * fHi > 0) {
    // No sign change in the bracket — fall back to the midpoint.
    return (lo + hi) / 2;
  }
  let mid = (lo + hi) / 2;
  for (let i = 0; i < maxIterations; i += 1) {
    mid = (lo + hi) / 2;
    const fMid = f(mid);
    if (!Number.isFinite(fMid) || Math.abs(fMid) < tolerance || hi - lo < tolerance) {
      return mid;
    }
    if (fLo * fMid <= 0) {
      hi = mid;
    } else {
      lo = mid;
      fLo = fMid;
    }
  }
  return mid;
}

/* -------------------------------------------------------------------------- */
/* Multi-leg maths                                                             */
/* -------------------------------------------------------------------------- */

/** Combined decimal odds for a parlay/accumulator. */
export function combineOdds(decimalOdds: number[]): number {
  if (decimalOdds.length === 0) {
    return 1;
  }
  return decimalOdds.reduce((product, odds) => product * odds, 1);
}

/* -------------------------------------------------------------------------- */
/* Parsing & formatting                                                        */
/* -------------------------------------------------------------------------- */

/**
 * Parse user input in the given format into decimal odds.
 * Returns `null` when the input cannot be interpreted as valid odds.
 */
export function parseOdds(input: string, format: OddsFormat): number | null {
  const text = input.trim();
  if (text.length === 0) {
    return null;
  }

  if (format === 'fractional' || text.includes('/')) {
    const match = /^(-?\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)$/.exec(text);
    if (!match) {
      return null;
    }
    const numerator = Number(match[1]);
    const denominator = Number(match[2]);
    const decimal = fractionToDecimal(numerator, denominator);
    return isValidDecimalOdds(decimal) ? decimal : null;
  }

  const numeric = Number(text.replace(/^\+/, ''));
  if (!Number.isFinite(numeric)) {
    return null;
  }

  if (format === 'american') {
    if (Math.abs(numeric) < 100) {
      return null;
    }
    const decimal = americanToDecimal(numeric);
    return isValidDecimalOdds(decimal) ? decimal : null;
  }

  return isValidDecimalOdds(numeric) ? numeric : null;
}

/** Render decimal odds in the requested display format. */
export function formatOdds(decimal: number, format: OddsFormat): string {
  if (!isValidDecimalOdds(decimal)) {
    return '—';
  }
  switch (format) {
    case 'american': {
      const american = decimalToAmerican(decimal);
      return american > 0 ? `+${american}` : `${american}`;
    }
    case 'fractional': {
      const { numerator, denominator } = decimalToFraction(decimal);
      return `${numerator}/${denominator}`;
    }
    case 'decimal':
    default:
      return decimal.toFixed(2);
  }
}

/** A short hint shown next to odds inputs. */
export function oddsPlaceholder(format: OddsFormat): string {
  switch (format) {
    case 'american':
      return '-110';
    case 'fractional':
      return '10/11';
    case 'decimal':
    default:
      return '1.91';
  }
}
