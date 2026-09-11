import React, { useMemo } from 'react';
import { StyleSheet, View, type StyleProp, type ViewStyle } from 'react-native';

import { useTheme } from '../theme';
import { Icon, type IconName } from './Icon';
import { Text } from './Text';

export interface StatTileProps {
  label: string;
  value: string;
  /** Small line under the value, e.g. "12 bets". */
  caption?: string;
  tone?: 'default' | 'positive' | 'negative' | 'warning' | 'info';
  icon?: IconName | string;
  style?: StyleProp<ViewStyle>;
  /** Renders a compact variant for dense grids. */
  compact?: boolean;
  testID?: string;
}

export function StatTile({
  label,
  value,
  caption,
  tone = 'default',
  icon,
  style,
  compact = false,
  testID,
}: StatTileProps) {
  const theme = useTheme();

  const accent = useMemo(() => {
    switch (tone) {
      case 'positive':
        return theme.colors.positive;
      case 'negative':
        return theme.colors.negative;
      case 'warning':
        return theme.colors.warning;
      case 'info':
        return theme.colors.info;
      default:
        return theme.colors.text;
    }
  }, [theme, tone]);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        tile: {
          flex: 1,
          minWidth: compact ? 96 : 130,
          gap: compact ? 2 : theme.spacing(1),
          backgroundColor: theme.colors.surfaceAlt,
          borderRadius: theme.radius.md,
          borderWidth: StyleSheet.hairlineWidth,
          borderColor: theme.colors.border,
          padding: compact ? theme.spacing(3) : theme.spacing(3.5),
        },
        labelRow: { flexDirection: 'row', alignItems: 'center', gap: theme.spacing(1.5) },
      }),
    [theme, compact],
  );

  return (
    <View style={[styles.tile, style]} testID={testID} accessibilityLabel={`${label}: ${value}`}>
      <View style={styles.labelRow}>
        {icon ? <Icon name={icon} size={13} color={theme.colors.textMuted} /> : null}
        <Text variant="caption" tone="muted" numberOfLines={1}>
          {label}
        </Text>
      </View>
      <Text
        variant={compact ? 'subheading' : 'heading'}
        color={accent}
        numberOfLines={1}
        adjustsFontSizeToFit
      >
        {value}
      </Text>
      {caption ? (
        <Text variant="caption" tone="muted" numberOfLines={1}>
          {caption}
        </Text>
      ) : null}
    </View>
  );
}
