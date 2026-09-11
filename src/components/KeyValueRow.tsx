import React, { useMemo } from 'react';
import { StyleSheet, View, type StyleProp, type ViewStyle } from 'react-native';

import { useTheme } from '../theme';
import { Text } from './Text';

export interface KeyValueRowProps {
  label: string;
  value: string;
  valueColor?: string;
  /** Extra context rendered under the label. */
  hint?: string;
  divider?: boolean;
  style?: StyleProp<ViewStyle>;
}

export function KeyValueRow({
  label,
  value,
  valueColor,
  hint,
  divider = true,
  style,
}: KeyValueRowProps) {
  const theme = useTheme();

  const styles = useMemo(
    () =>
      StyleSheet.create({
        row: {
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing(4),
          paddingVertical: theme.spacing(2.5),
          borderBottomWidth: divider ? StyleSheet.hairlineWidth : 0,
          borderColor: theme.colors.border,
        },
        label: { flex: 1, gap: 1 },
      }),
    [theme, divider],
  );

  return (
    <View style={[styles.row, style]}>
      <View style={styles.label}>
        <Text variant="body" tone="secondary">
          {label}
        </Text>
        {hint ? (
          <Text variant="caption" tone="muted">
            {hint}
          </Text>
        ) : null}
      </View>
      <Text variant="mono" color={valueColor} numberOfLines={1}>
        {value}
      </Text>
    </View>
  );
}
