import React, { useMemo } from 'react';
import { Pressable, StyleSheet, View } from 'react-native';

import { useTheme } from '../../theme';
import { Text } from '../Text';

export interface BarDatum {
  key: string;
  label: string;
  value: number;
  /** Shown on the right of the row, e.g. "12 bets · ROI 4.2%". */
  caption?: string;
}

export interface BarChartProps {
  data: BarDatum[];
  formatValue: (value: number) => string;
  /** Colour positive bars green and negative bars red. */
  diverging?: boolean;
  color?: string;
  maxRows?: number;
  onPressRow?: (datum: BarDatum) => void;
  emptyMessage?: string;
  testID?: string;
}

/**
 * Horizontal bar chart built from plain views — no SVG needed, and it wraps gracefully
 * on narrow screens. Diverging mode centres the axis so losses grow to the left.
 */
export function BarChart({
  data,
  formatValue,
  diverging = true,
  color,
  maxRows,
  onPressRow,
  emptyMessage = 'Not enough data yet.',
  testID,
}: BarChartProps) {
  const theme = useTheme();
  const rows = maxRows ? data.slice(0, maxRows) : data;

  const scale = useMemo(() => {
    const magnitude = Math.max(...rows.map((row) => Math.abs(row.value)), 1);
    return (value: number) => Math.min(Math.abs(value) / magnitude, 1);
  }, [rows]);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        row: { gap: theme.spacing(1.5), paddingVertical: theme.spacing(2) },
        header: {
          flexDirection: 'row',
          alignItems: 'baseline',
          justifyContent: 'space-between',
          gap: theme.spacing(2),
        },
        labelGroup: { flex: 1, flexDirection: 'row', alignItems: 'baseline', gap: theme.spacing(2) },
        track: {
          height: 8,
          borderRadius: 4,
          backgroundColor: theme.colors.surfaceSunken,
          flexDirection: 'row',
          overflow: 'hidden',
        },
        half: { flex: 1, justifyContent: 'center' },
        barRight: { height: 8, borderRadius: 4, alignSelf: 'flex-start' },
        barLeft: { height: 8, borderRadius: 4, alignSelf: 'flex-end' },
        empty: { paddingVertical: theme.spacing(6), alignItems: 'center' },
      }),
    [theme],
  );

  if (rows.length === 0) {
    return (
      <View style={styles.empty} testID={testID}>
        <Text tone="muted" variant="caption">
          {emptyMessage}
        </Text>
      </View>
    );
  }

  return (
    <View testID={testID}>
      {rows.map((row) => {
        const positive = row.value >= 0;
        const barColor =
          color ?? (diverging ? (positive ? theme.colors.positive : theme.colors.negative) : theme.colors.primary);
        const width = `${scale(row.value) * 100}%` as const;

        const body = (
          <View style={styles.row} key={row.key}>
            <View style={styles.header}>
              <View style={styles.labelGroup}>
                <Text variant="label" numberOfLines={1} style={{ flexShrink: 1 }}>
                  {row.label}
                </Text>
                {row.caption ? (
                  <Text variant="caption" tone="muted" numberOfLines={1}>
                    {row.caption}
                  </Text>
                ) : null}
              </View>
              <Text
                variant="label"
                color={diverging ? barColor : theme.colors.textSecondary}
                numberOfLines={1}
              >
                {formatValue(row.value)}
              </Text>
            </View>

            {diverging ? (
              <View style={styles.track}>
                <View style={styles.half}>
                  {!positive ? (
                    <View style={[styles.barLeft, { width, backgroundColor: barColor }]} />
                  ) : null}
                </View>
                <View style={styles.half}>
                  {positive ? (
                    <View style={[styles.barRight, { width, backgroundColor: barColor }]} />
                  ) : null}
                </View>
              </View>
            ) : (
              <View style={styles.track}>
                <View style={[styles.barRight, { width, backgroundColor: barColor }]} />
              </View>
            )}
          </View>
        );

        if (!onPressRow) {
          return body;
        }
        return (
          <Pressable
            key={row.key}
            accessibilityRole="button"
            onPress={() => onPressRow(row)}
            style={({ pressed }) => (pressed ? { opacity: 0.6 } : null)}
          >
            {body}
          </Pressable>
        );
      })}
    </View>
  );
}
