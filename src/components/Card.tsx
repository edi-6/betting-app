import React, { useMemo } from 'react';
import { StyleSheet, View, type StyleProp, type ViewStyle } from 'react-native';

import { useTheme } from '../theme';
import { Text } from './Text';

export interface CardProps {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
  /** Rendered on the right of the header row. */
  action?: React.ReactNode;
  padded?: boolean;
  style?: StyleProp<ViewStyle>;
  tone?: 'default' | 'sunken';
}

export function Card({
  children,
  title,
  subtitle,
  action,
  padded = true,
  style,
  tone = 'default',
}: CardProps) {
  const theme = useTheme();

  const styles = useMemo(
    () =>
      StyleSheet.create({
        card: {
          backgroundColor:
            tone === 'sunken' ? theme.colors.surfaceSunken : theme.colors.surface,
          borderRadius: theme.radius.lg,
          borderWidth: StyleSheet.hairlineWidth,
          borderColor: theme.colors.border,
          overflow: 'hidden',
          ...theme.shadow.card,
        },
        body: {
          padding: padded ? theme.spacing(4) : 0,
        },
        header: {
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing(3),
          paddingHorizontal: theme.spacing(4),
          paddingTop: theme.spacing(4),
          paddingBottom: theme.spacing(1),
        },
        headerText: { flex: 1 },
      }),
    [theme, padded, tone],
  );

  return (
    <View style={[styles.card, style]}>
      {(title || action) && (
        <View style={styles.header}>
          <View style={styles.headerText}>
            {title ? <Text variant="subheading">{title}</Text> : null}
            {subtitle ? (
              <Text variant="caption" tone="muted">
                {subtitle}
              </Text>
            ) : null}
          </View>
          {action}
        </View>
      )}
      <View style={styles.body}>{children}</View>
    </View>
  );
}
