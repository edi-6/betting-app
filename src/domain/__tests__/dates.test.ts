import {
  addDays,
  addMonths,
  buildDate,
  dayKey,
  daysBetween,
  daysInMonth,
  formatDate,
  formatDateTime,
  formatRelativeDay,
  formatShortDate,
  formatTime,
  isSameDay,
  monthKey,
  monthKeyLabel,
  monthKeysBetween,
  startOfDay,
  startOfMonth,
  startOfWeek,
  startOfYear,
} from '../dates';

describe('boundaries', () => {
  const reference = buildDate(2026, 2, 11, 15, 30); // 11 March 2026, local time

  it('finds the start of the day', () => {
    const start = startOfDay(reference);
    expect(start.getHours()).toBe(0);
    expect(start.getDate()).toBe(11);
  });

  it('finds the Monday of the week', () => {
    const start = startOfWeek(reference);
    expect(start.getDay()).toBe(1);
    expect(daysBetween(start, reference)).toBeLessThan(7);
  });

  it('finds the start of the month and year', () => {
    expect(startOfMonth(reference).getDate()).toBe(1);
    expect(startOfYear(reference).getMonth()).toBe(0);
    expect(startOfYear(reference).getDate()).toBe(1);
  });
});

describe('arithmetic', () => {
  it('adds days across month boundaries', () => {
    const result = addDays(buildDate(2026, 0, 31), 1);
    expect(result.getMonth()).toBe(1);
    expect(result.getDate()).toBe(1);
  });

  it('adds months and clamps to the last valid day', () => {
    const result = addMonths(buildDate(2026, 0, 31), 1);
    expect(result.getMonth()).toBe(1);
    expect(result.getDate()).toBe(28);
  });

  it('counts calendar days between dates', () => {
    expect(daysBetween(buildDate(2026, 2, 1), buildDate(2026, 2, 11))).toBe(10);
    expect(daysBetween(buildDate(2026, 2, 11), buildDate(2026, 2, 1))).toBe(-10);
  });

  it('knows the length of each month', () => {
    expect(daysInMonth(2026, 1)).toBe(28);
    expect(daysInMonth(2028, 1)).toBe(29);
    expect(daysInMonth(2026, 0)).toBe(31);
  });
});

describe('keys', () => {
  it('builds local day and month keys', () => {
    const date = buildDate(2026, 8, 5, 23, 59);
    expect(dayKey(date)).toBe('2026-09-05');
    expect(monthKey(date)).toBe('2026-09');
  });

  it('labels a month key', () => {
    expect(monthKeyLabel('2026-09')).toBe('Sep 2026');
  });

  it('lists the months in a range inclusively', () => {
    const keys = monthKeysBetween(buildDate(2026, 0, 15), buildDate(2026, 3, 2));
    expect(keys).toEqual(['2026-01', '2026-02', '2026-03', '2026-04']);
  });

  it('compares days', () => {
    expect(isSameDay(buildDate(2026, 2, 11, 1), buildDate(2026, 2, 11, 23))).toBe(true);
    expect(isSameDay(buildDate(2026, 2, 11), buildDate(2026, 2, 12))).toBe(false);
  });
});

describe('formatting', () => {
  const date = buildDate(2026, 8, 11, 19, 45);

  it('formats dates and times', () => {
    expect(formatDate(date)).toBe('Sep 11, 2026');
    expect(formatShortDate(date)).toBe('Sep 11');
    expect(formatTime(date)).toBe('19:45');
    expect(formatDateTime(date)).toBe('Sep 11, 2026 · 19:45');
  });

  it('handles an invalid date', () => {
    expect(formatDate('nonsense')).toBe('—');
    expect(formatTime('nonsense')).toBe('—');
  });

  it('describes recent days in words', () => {
    const now = buildDate(2026, 8, 11, 12);
    expect(formatRelativeDay(now, now)).toBe('Today');
    expect(formatRelativeDay(addDays(now, -1), now)).toBe('Yesterday');
    expect(formatRelativeDay(addDays(now, 1), now)).toBe('Tomorrow');
    expect(formatRelativeDay(addDays(now, -3), now)).toBe('3 days ago');
    expect(formatRelativeDay(addDays(now, -30), now)).toBe(formatDate(addDays(now, -30)));
  });
});
