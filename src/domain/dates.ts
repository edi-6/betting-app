/**
 * Date helpers. Everything works in the device's local timezone so that "today"
 * means what the user thinks it means; values are persisted as ISO-8601 strings.
 */

export const MONTH_NAMES = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
] as const;

export const MONTH_ABBREVIATIONS = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec',
] as const;

export const WEEKDAY_ABBREVIATIONS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'] as const;

export function toDate(value: string | number | Date): Date {
  return value instanceof Date ? new Date(value.getTime()) : new Date(value);
}

export function isValidDate(value: Date): boolean {
  return value instanceof Date && !Number.isNaN(value.getTime());
}

export function startOfDay(value: string | number | Date): Date {
  const date = toDate(value);
  date.setHours(0, 0, 0, 0);
  return date;
}

/** ISO-style weeks: Monday is day 1. */
export function startOfWeek(value: string | number | Date, weekStartsOn = 1): Date {
  const date = startOfDay(value);
  const diff = (date.getDay() - weekStartsOn + 7) % 7;
  date.setDate(date.getDate() - diff);
  return date;
}

export function startOfMonth(value: string | number | Date): Date {
  const date = startOfDay(value);
  date.setDate(1);
  return date;
}

export function startOfYear(value: string | number | Date): Date {
  const date = startOfMonth(value);
  date.setMonth(0);
  return date;
}

export function addDays(value: string | number | Date, days: number): Date {
  const date = toDate(value);
  date.setDate(date.getDate() + days);
  return date;
}

export function addMonths(value: string | number | Date, months: number): Date {
  const date = toDate(value);
  const day = date.getDate();
  date.setDate(1);
  date.setMonth(date.getMonth() + months);
  const lastDay = new Date(date.getFullYear(), date.getMonth() + 1, 0).getDate();
  date.setDate(Math.min(day, lastDay));
  return date;
}

/** Local-time "YYYY-MM-DD" bucket key. */
export function dayKey(value: string | number | Date): string {
  const date = toDate(value);
  const month = `${date.getMonth() + 1}`.padStart(2, '0');
  const day = `${date.getDate()}`.padStart(2, '0');
  return `${date.getFullYear()}-${month}-${day}`;
}

/** Local-time "YYYY-MM" bucket key. */
export function monthKey(value: string | number | Date): string {
  const date = toDate(value);
  return `${date.getFullYear()}-${`${date.getMonth() + 1}`.padStart(2, '0')}`;
}

export function monthKeyLabel(key: string): string {
  const [year, month] = key.split('-');
  const index = Number(month) - 1;
  const name = MONTH_ABBREVIATIONS[index] ?? month ?? '';
  return `${name} ${year ?? ''}`.trim();
}

export function daysBetween(a: string | number | Date, b: string | number | Date): number {
  const millis = startOfDay(b).getTime() - startOfDay(a).getTime();
  return Math.round(millis / 86400000);
}

/** "Sep 11, 2026" */
export function formatDate(value: string | number | Date): string {
  const date = toDate(value);
  if (!isValidDate(date)) {
    return '—';
  }
  return `${MONTH_ABBREVIATIONS[date.getMonth()]} ${date.getDate()}, ${date.getFullYear()}`;
}

/** "Sep 11" */
export function formatShortDate(value: string | number | Date): string {
  const date = toDate(value);
  if (!isValidDate(date)) {
    return '—';
  }
  return `${MONTH_ABBREVIATIONS[date.getMonth()]} ${date.getDate()}`;
}

/** "19:45" in 24-hour time. */
export function formatTime(value: string | number | Date): string {
  const date = toDate(value);
  if (!isValidDate(date)) {
    return '—';
  }
  return `${`${date.getHours()}`.padStart(2, '0')}:${`${date.getMinutes()}`.padStart(2, '0')}`;
}

export function formatDateTime(value: string | number | Date): string {
  return `${formatDate(value)} · ${formatTime(value)}`;
}

/** "Today" / "Yesterday" / "Sep 11, 2026" */
export function formatRelativeDay(value: string | number | Date, now: Date = new Date()): string {
  const diff = daysBetween(value, now);
  if (diff === 0) return 'Today';
  if (diff === 1) return 'Yesterday';
  if (diff === -1) return 'Tomorrow';
  if (diff > 1 && diff < 7) return `${diff} days ago`;
  if (diff < -1 && diff > -7) return `In ${Math.abs(diff)} days`;
  return formatDate(value);
}

/** Build a local Date from Y/M/D plus optional H:M, avoiding timezone drift. */
export function buildDate(
  year: number,
  month: number,
  day: number,
  hours = 12,
  minutes = 0,
): Date {
  return new Date(year, month, day, hours, minutes, 0, 0);
}

/** Number of days in the given month (month is 0-indexed). */
export function daysInMonth(year: number, month: number): number {
  return new Date(year, month + 1, 0).getDate();
}
