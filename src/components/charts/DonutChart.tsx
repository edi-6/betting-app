import React, { useMemo } from 'react';
import { StyleSheet, View } from 'react-native';
import Svg, { Circle, G } from 'react-native-svg';

import { useTheme } from '../../theme';
import { Text } from '../Text';

export interface DonutSlice {
  key: string;
  label: string;
  value: number;
  color: string;
}

export interface DonutChartProps {
  slices: DonutSlice[];
  size?: number;
  thickness?: number;
  centerValue?: string;
  centerLabel?: string;
  testID?: string;
}

/** Ring chart drawn with stroke dash offsets — one `Circle` per slice. */
export function DonutChart({
  slices,
  size = 150,
  thickness = 16,
  centerValue,
  centerLabel,
  testID,
}: DonutChartProps) {
  const theme = useTheme();
  const radius = (size - thickness) / 2;
  const circumference = 2 * Math.PI * radius;
  const total = slices.reduce((sum, slice) => sum + Math.max(slice.value, 0), 0);

  const segments = useMemo(() => {
    const visible = slices.filter((slice) => slice.value > 0);
    const result: (DonutSlice & { dash: number; gap: number; offset: number })[] = [];
    let consumed = 0;
    for (const slice of visible) {
      const fraction = total > 0 ? slice.value / total : 0;
      const dash = fraction * circumference;
      result.push({ ...slice, dash, gap: circumference - dash, offset: -consumed });
      consumed += dash;
    }
    return result;
  }, [slices, total, circumference]);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        container: { alignItems: 'center', justifyContent: 'center' },
        center: { position: 'absolute', alignItems: 'center' },
      }),
    [],
  );

  return (
    <View style={[styles.container, { width: size, height: size }]} testID={testID}>
      <Svg width={size} height={size}>
        <G rotation={-90} origin={`${size / 2}, ${size / 2}`}>
          <Circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={theme.colors.surfaceSunken}
            strokeWidth={thickness}
            fill="none"
          />
          {segments.map((segment) => (
            <Circle
              key={segment.key}
              cx={size / 2}
              cy={size / 2}
              r={radius}
              stroke={segment.color}
              strokeWidth={thickness}
              strokeDasharray={`${segment.dash} ${segment.gap}`}
              strokeDashoffset={segment.offset}
              strokeLinecap="butt"
              fill="none"
            />
          ))}
        </G>
      </Svg>
      {centerValue || centerLabel ? (
        <View style={styles.center}>
          {centerValue ? <Text variant="heading">{centerValue}</Text> : null}
          {centerLabel ? (
            <Text variant="caption" tone="muted">
              {centerLabel}
            </Text>
          ) : null}
        </View>
      ) : null}
    </View>
  );
}
