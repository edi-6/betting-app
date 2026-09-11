import React, { useMemo } from 'react';
import { StyleSheet, View, type StyleProp, type ViewStyle } from 'react-native';

import { BET_STATUS_LABELS } from '../domain/settlement';
import type { BetStatus } from '../domain/types';
import { useTheme } from '../theme';
import type { ThemeColors } from '../theme/tokens';
import { Text } from './Text';

export interface StatusPillProps {
  status: BetStatus;
  size?: 'sm' | 'md';
  style?: StyleProp<ViewStyle>;
}

export function statusPalette(
  status: BetStatus,
  colors: ThemeColors,
): { solid: string; soft: string } {
  switch (status) {
    case 'won':
      return { solid: colors.positive, soft: colors.positiveSoft };
    case 'half_won':
      return { solid: colors.positive, soft: colors.positiveSoft };
    case 'lost':
      return { solid: colors.negative, soft: colors.negativeSoft };
    case 'half_lost':
      return { solid: colors.negative, soft: colors.negativeSoft };
    case 'cashed_out':
      return { solid: colors.info, soft: colors.infoSoft };
    case 'pending':
      return { solid: colors.warning, soft: colors.warningSoft };
    case 'push':
    case 'void':
    default:
      return { solid: colors.neutral, soft: colors.neutralSoft };
  }
}

export function StatusPill({ status, size = 'sm', style }: StatusPillProps) {
  const theme = useTheme();
  const palette = statusPalette(status, theme.colors);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        pill: {
          alignSelf: 'flex-start',
          backgroundColor: palette.soft,
          borderRadius: theme.radius.pill,
          paddingVertical: size === 'sm' ? 3 : theme.spacing(1.5),
          paddingHorizontal: size === 'sm' ? theme.spacing(2) : theme.spacing(3),
        },
      }),
    [theme, palette, size],
  );

  return (
    <View style={[styles.pill, style]}>
      <Text variant="caption" color={palette.solid}>
        {BET_STATUS_LABELS[status]}
      </Text>
    </View>
  );
}
