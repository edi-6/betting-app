import React, { useMemo } from 'react';
import { StyleSheet, View, type StyleProp, type ViewStyle } from 'react-native';

import { useTheme } from '../theme';

export interface ProgressBarProps {
  /** 0–1. Values above 1 fill the track and switch to the overflow colour. */
  progress: number;
  color?: string;
  trackColor?: string;
  height?: number;
  style?: StyleProp<ViewStyle>;
  testID?: string;
}

export function ProgressBar({
  progress,
  color,
  trackColor,
  height = 8,
  style,
  testID,
}: ProgressBarProps) {
  const theme = useTheme();
  const clamped = Number.isFinite(progress) ? Math.min(Math.max(progress, 0), 1) : 0;

  const styles = useMemo(
    () =>
      StyleSheet.create({
        track: {
          height,
          borderRadius: height / 2,
          backgroundColor: trackColor ?? theme.colors.surfaceSunken,
          overflow: 'hidden',
        },
        fill: {
          height: '100%',
          borderRadius: height / 2,
          backgroundColor: color ?? theme.colors.primary,
        },
      }),
    [theme, height, color, trackColor],
  );

  return (
    <View
      testID={testID}
      style={[styles.track, style]}
      accessibilityRole="progressbar"
      accessibilityValue={{ now: Math.round(clamped * 100), min: 0, max: 100 }}
    >
      <View style={[styles.fill, { width: `${clamped * 100}%` }]} />
    </View>
  );
}
