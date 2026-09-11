import {
  americanToDecimal,
  approximateFraction,
  bookPercentage,
  combineOdds,
  decimalToAmerican,
  decimalToFraction,
  fairOdds,
  formatOdds,
  fractionToDecimal,
  impliedProbability,
  isValidDecimalOdds,
  overround,
  parseOdds,
  probabilityToDecimal,
  removeVig,
} from '../odds';

describe('decimal ↔ American', () => {
  it('converts favourites and underdogs', () => {
    expect(decimalToAmerican(2.5)).toBe(150);
    expect(decimalToAmerican(3)).toBe(200);
    expect(decimalToAmerican(2)).toBe(100);
    expect(decimalToAmerican(1.5)).toBe(-200);
    expect(decimalToAmerican(1.91)).toBe(-110);
  });

  it('round-trips', () => {
    for (const american of [-500, -250, -110, -101, 100, 135, 250, 900]) {
      expect(decimalToAmerican(americanToDecimal(american))).toBe(american);
    }
  });

  it('rejects nonsense', () => {
    expect(decimalToAmerican(1)).toBe(0);
    expect(decimalToAmerican(NaN)).toBe(0);
    expect(Number.isNaN(americanToDecimal(0))).toBe(true);
  });
});

describe('decimal ↔ fractional', () => {
  it('produces the canonical betting fractions', () => {
    expect(decimalToFraction(3.5)).toEqual({ numerator: 5, denominator: 2 });
    expect(decimalToFraction(2)).toEqual({ numerator: 1, denominator: 1 });
    expect(decimalToFraction(1.5)).toEqual({ numerator: 1, denominator: 2 });
    expect(decimalToFraction(6)).toEqual({ numerator: 5, denominator: 1 });
  });

  it('reduces fractions', () => {
    expect(approximateFraction(0.5)).toEqual({ numerator: 1, denominator: 2 });
    expect(approximateFraction(2.25)).toEqual({ numerator: 9, denominator: 4 });
    expect(approximateFraction(0)).toEqual({ numerator: 0, denominator: 1 });
  });

  it('round-trips', () => {
    expect(fractionToDecimal(5, 2)).toBeCloseTo(3.5, 10);
    expect(fractionToDecimal(10, 11)).toBeCloseTo(1.909090909, 6);
  });
});

describe('probability', () => {
  it('maps odds to implied probability and back', () => {
    expect(impliedProbability(2)).toBeCloseTo(0.5, 12);
    expect(impliedProbability(4)).toBeCloseTo(0.25, 12);
    expect(probabilityToDecimal(0.25)).toBeCloseTo(4, 12);
    expect(Number.isNaN(probabilityToDecimal(0))).toBe(true);
    expect(Number.isNaN(probabilityToDecimal(1))).toBe(true);
  });

  it('measures the overround of a standard −110 / −110 market', () => {
    const market = [americanToDecimal(-110), americanToDecimal(-110)];
    expect(bookPercentage(market)).toBeCloseTo(1.047619, 5);
    expect(overround(market)).toBeCloseTo(4.7619, 3);
  });
});

describe('removeVig', () => {
  const market = [americanToDecimal(-110), americanToDecimal(-110)];

  it.each(['multiplicative', 'power', 'shin', 'additive'] as const)(
    'returns probabilities summing to one (%s)',
    (method) => {
      const probabilities = removeVig(market, method);
      expect(probabilities).toHaveLength(2);
      expect(probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
      probabilities.forEach((p) => expect(p).toBeCloseTo(0.5, 6));
    },
  );

  it('prices a lopsided three-way market without vig', () => {
    const threeWay = [1.53, 4.4, 6.5];
    for (const method of ['multiplicative', 'power', 'shin', 'additive'] as const) {
      const probabilities = removeVig(threeWay, method);
      expect(probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 8);
      expect(probabilities[0]).toBeGreaterThan(probabilities[1] as number);
      expect(probabilities[1]).toBeGreaterThan(probabilities[2] as number);
    }
  });

  it('lengthens every price when the margin is removed', () => {
    const fair = fairOdds(market, 'multiplicative');
    expect(fair[0]).toBeGreaterThan(market[0] as number);
    expect(fair[0]).toBeCloseTo(2, 6);
  });

  it('returns an empty array for invalid input', () => {
    expect(removeVig([])).toEqual([]);
    expect(removeVig([1, 2])).toEqual([]);
  });
});

describe('combineOdds', () => {
  it('multiplies parlay legs', () => {
    expect(combineOdds([2, 3])).toBeCloseTo(6, 12);
    expect(combineOdds([1.5, 1.5, 2])).toBeCloseTo(4.5, 12);
    expect(combineOdds([])).toBe(1);
  });
});

describe('parseOdds', () => {
  it('reads decimal input', () => {
    expect(parseOdds('2.50', 'decimal')).toBeCloseTo(2.5, 12);
    expect(parseOdds(' 1.91 ', 'decimal')).toBeCloseTo(1.91, 12);
    expect(parseOdds('0.9', 'decimal')).toBeNull();
    expect(parseOdds('', 'decimal')).toBeNull();
    expect(parseOdds('abc', 'decimal')).toBeNull();
  });

  it('reads American input with or without a plus sign', () => {
    expect(parseOdds('+150', 'american')).toBeCloseTo(2.5, 12);
    expect(parseOdds('150', 'american')).toBeCloseTo(2.5, 12);
    expect(parseOdds('-200', 'american')).toBeCloseTo(1.5, 12);
    expect(parseOdds('50', 'american')).toBeNull();
  });

  it('reads fractional input in any format', () => {
    expect(parseOdds('5/2', 'fractional')).toBeCloseTo(3.5, 12);
    expect(parseOdds('5 / 2', 'fractional')).toBeCloseTo(3.5, 12);
    expect(parseOdds('5/2', 'decimal')).toBeCloseTo(3.5, 12);
    expect(parseOdds('5', 'fractional')).toBeNull();
  });
});

describe('formatOdds', () => {
  it('renders each display format', () => {
    expect(formatOdds(2.5, 'decimal')).toBe('2.50');
    expect(formatOdds(2.5, 'american')).toBe('+150');
    expect(formatOdds(1.5, 'american')).toBe('-200');
    expect(formatOdds(3.5, 'fractional')).toBe('5/2');
    expect(formatOdds(1, 'decimal')).toBe('—');
  });
});

describe('isValidDecimalOdds', () => {
  it('guards the accepted range', () => {
    expect(isValidDecimalOdds(1.01)).toBe(true);
    expect(isValidDecimalOdds(1)).toBe(false);
    expect(isValidDecimalOdds(-2)).toBe(false);
    expect(isValidDecimalOdds(Infinity)).toBe(false);
  });
});
