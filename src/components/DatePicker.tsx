import React, { useMemo, useState } from 'react';
import { Pressable, StyleSheet, View } from 'react-native';

import {
  addMonths,
  buildDate,
  dayKey,
  daysInMonth,
  formatDate,
  formatTime,
  MONTH_NAMES,
  WEEKDAY_ABBREVIATIONS,
} from '../domain/dates';
import { useTheme } from '../theme';
import { Button } from './Button';
import { Icon } from './Icon';
import { Sheet } from './Sheet';
import { Text } from './Text';

export interface DatePickerProps {
  label?: string;
  /** ISO-8601 timestamp. */
  value: string;
  onChange: (iso: string) => void;
  /** Show hour/minute steppers under the calendar. */
  withTime?: boolean;
  maximumDate?: Date;
  minimumDate?: Date;
  testID?: string;
}

/**
 * Self-contained calendar picker.
 *
 * Implemented in plain React Native rather than the native picker so the control looks
 * and behaves identically on iOS and Android and needs no extra native module.
 */
export function DatePicker({
  label,
  value,
  onChange,
  withTime = true,
  maximumDate,
  minimumDate,
  testID,
}: DatePickerProps) {
  const theme = useTheme();
  const [open, setOpen] = useState(false);

  const selected = useMemo(() => {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? new Date() : date;
  }, [value]);

  const [cursor, setCursor] = useState(() => new Date(selected.getFullYear(), selected.getMonth(), 1));

  const weeks = useMemo(() => {
    const year = cursor.getFullYear();
    const month = cursor.getMonth();
    const total = daysInMonth(year, month);
    // Monday-first grid.
    const leading = (new Date(year, month, 1).getDay() + 6) % 7;
    const cells: (Date | null)[] = Array.from({ length: leading }, () => null);
    for (let day = 1; day <= total; day += 1) {
      cells.push(buildDate(year, month, day, selected.getHours(), selected.getMinutes()));
    }
    while (cells.length % 7 !== 0) {
      cells.push(null);
    }
    const rows: (Date | null)[][] = [];
    for (let i = 0; i < cells.length; i += 7) {
      rows.push(cells.slice(i, i + 7));
    }
    return rows;
  }, [cursor, selected]);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        container: { gap: theme.spacing(1.5) },
        trigger: {
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          backgroundColor: theme.colors.surfaceAlt,
          borderRadius: theme.radius.md,
          borderWidth: 1,
          borderColor: theme.colors.border,
          paddingHorizontal: theme.spacing(3.5),
          minHeight: 48,
        },
        monthRow: {
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: theme.spacing(3),
        },
        navButton: {
          width: 36,
          height: 36,
          borderRadius: 18,
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: theme.colors.surfaceAlt,
        },
        weekHeader: { flexDirection: 'row', marginBottom: theme.spacing(1) },
        week: { flexDirection: 'row' },
        cell: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingVertical: 2 },
        day: {
          width: 38,
          height: 38,
          borderRadius: 19,
          alignItems: 'center',
          justifyContent: 'center',
        },
        daySelected: { backgroundColor: theme.colors.primary },
        dayToday: { borderWidth: 1, borderColor: theme.colors.primary },
        timeRow: {
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginTop: theme.spacing(4),
          paddingTop: theme.spacing(4),
          borderTopWidth: StyleSheet.hairlineWidth,
          borderColor: theme.colors.border,
        },
        stepperGroup: { flexDirection: 'row', alignItems: 'center', gap: theme.spacing(2) },
        stepper: {
          width: 36,
          height: 36,
          borderRadius: 18,
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: theme.colors.surfaceAlt,
        },
        quickRow: { flexDirection: 'row', gap: theme.spacing(2), marginTop: theme.spacing(3) },
      }),
    [theme],
  );

  const isOutOfRange = (date: Date) => {
    if (maximumDate && date.getTime() > maximumDate.getTime()) return true;
    if (minimumDate && date.getTime() < minimumDate.getTime()) return true;
    return false;
  };

  const commit = (date: Date) => {
    onChange(date.toISOString());
  };

  const shiftTime = (field: 'hours' | 'minutes', delta: number) => {
    const next = new Date(selected.getTime());
    if (field === 'hours') {
      next.setHours((next.getHours() + delta + 24) % 24);
    } else {
      next.setMinutes((next.getMinutes() + delta + 60) % 60);
    }
    commit(next);
  };

  const todayKey = dayKey(new Date());

  return (
    <View style={styles.container}>
      {label ? (
        <Text variant="label" tone="secondary">
          {label}
        </Text>
      ) : null}
      <Pressable
        testID={testID}
        accessibilityRole="button"
        accessibilityLabel={`${label ?? 'Date'}: ${formatDate(selected)}`}
        onPress={() => {
          setCursor(new Date(selected.getFullYear(), selected.getMonth(), 1));
          setOpen(true);
        }}
        style={styles.trigger}
      >
        <Text>
          {formatDate(selected)}
          {withTime ? ` · ${formatTime(selected)}` : ''}
        </Text>
        <Icon name="calendar-outline" size={18} color={theme.colors.textMuted} />
      </Pressable>

      <Sheet
        visible={open}
        onClose={() => setOpen(false)}
        title="Pick a date"
        footer={<Button label="Done" onPress={() => setOpen(false)} fullWidth />}
      >
        <View style={styles.monthRow}>
          <Pressable
            style={styles.navButton}
            accessibilityRole="button"
            accessibilityLabel="Previous month"
            onPress={() => setCursor(addMonths(cursor, -1))}
          >
            <Icon name="chevron-back" size={18} color={theme.colors.textSecondary} />
          </Pressable>
          <Text variant="subheading">
            {MONTH_NAMES[cursor.getMonth()]} {cursor.getFullYear()}
          </Text>
          <Pressable
            style={styles.navButton}
            accessibilityRole="button"
            accessibilityLabel="Next month"
            onPress={() => setCursor(addMonths(cursor, 1))}
          >
            <Icon name="chevron-forward" size={18} color={theme.colors.textSecondary} />
          </Pressable>
        </View>

        <View style={styles.weekHeader}>
          {[1, 2, 3, 4, 5, 6, 0].map((index) => (
            <View key={index} style={styles.cell}>
              <Text variant="caption" tone="muted">
                {WEEKDAY_ABBREVIATIONS[index]?.slice(0, 2)}
              </Text>
            </View>
          ))}
        </View>

        {weeks.map((week, weekIndex) => (
          <View key={weekIndex} style={styles.week}>
            {week.map((date, dayIndex) => {
              if (!date) {
                return <View key={`empty-${dayIndex}`} style={styles.cell} />;
              }
              const isSelected = dayKey(date) === dayKey(selected);
              const isToday = dayKey(date) === todayKey;
              const disabled = isOutOfRange(date);
              return (
                <View key={dayKey(date)} style={styles.cell}>
                  <Pressable
                    disabled={disabled}
                    accessibilityRole="button"
                    accessibilityLabel={formatDate(date)}
                    accessibilityState={{ selected: isSelected, disabled }}
                    onPress={() => commit(date)}
                    style={[
                      styles.day,
                      isToday && !isSelected ? styles.dayToday : null,
                      isSelected ? styles.daySelected : null,
                    ]}
                  >
                    <Text
                      variant="label"
                      color={
                        disabled
                          ? theme.colors.textMuted
                          : isSelected
                            ? theme.colors.primaryText
                            : theme.colors.text
                      }
                    >
                      {date.getDate()}
                    </Text>
                  </Pressable>
                </View>
              );
            })}
          </View>
        ))}

        <View style={styles.quickRow}>
          <Button label="Today" size="sm" variant="secondary" onPress={() => commit(new Date())} />
          <Button
            label="Yesterday"
            size="sm"
            variant="secondary"
            onPress={() => {
              const date = new Date();
              date.setDate(date.getDate() - 1);
              commit(date);
            }}
          />
        </View>

        {withTime ? (
          <View style={styles.timeRow}>
            <Text variant="label" tone="secondary">
              Time
            </Text>
            <View style={styles.stepperGroup}>
              <Pressable
                style={styles.stepper}
                accessibilityRole="button"
                accessibilityLabel="Hour down"
                onPress={() => shiftTime('hours', -1)}
              >
                <Icon name="remove" size={18} color={theme.colors.textSecondary} />
              </Pressable>
              <Text variant="mono" style={{ minWidth: 56, textAlign: 'center' }}>
                {formatTime(selected)}
              </Text>
              <Pressable
                style={styles.stepper}
                accessibilityRole="button"
                accessibilityLabel="Hour up"
                onPress={() => shiftTime('hours', 1)}
              >
                <Icon name="add" size={18} color={theme.colors.textSecondary} />
              </Pressable>
              <Pressable
                style={styles.stepper}
                accessibilityRole="button"
                accessibilityLabel="Minutes plus five"
                onPress={() => shiftTime('minutes', 5)}
              >
                <Text variant="caption" tone="secondary">
                  +5m
                </Text>
              </Pressable>
            </View>
          </View>
        ) : null}
      </Sheet>
    </View>
  );
}
