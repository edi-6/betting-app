import React, { useMemo } from 'react';
import { StyleSheet, View } from 'react-native';

import { useTheme } from '../theme';
import { Button } from './Button';
import { Icon, type IconName } from './Icon';
import { Text } from './Text';

export interface EmptyStateProps {
  icon?: IconName | string;
  title: string;
  message?: string;
  actionLabel?: string;
  onAction?: () => void;
  secondaryActionLabel?: string;
  onSecondaryAction?: () => void;
}

export function EmptyState({
  icon = 'documents-outline',
  title,
  message,
  actionLabel,
  onAction,
  secondaryActionLabel,
  onSecondaryAction,
}: EmptyStateProps) {
  const theme = useTheme();

  const styles = useMemo(
    () =>
      StyleSheet.create({
        container: {
          alignItems: 'center',
          justifyContent: 'center',
          gap: theme.spacing(2),
          paddingVertical: theme.spacing(10),
          paddingHorizontal: theme.spacing(6),
        },
        badge: {
          width: 64,
          height: 64,
          borderRadius: 32,
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: theme.colors.surfaceAlt,
          marginBottom: theme.spacing(1),
        },
        actions: { flexDirection: 'row', gap: theme.spacing(2), marginTop: theme.spacing(3) },
      }),
    [theme],
  );

  return (
    <View style={styles.container}>
      <View style={styles.badge}>
        <Icon name={icon} size={28} color={theme.colors.textMuted} />
      </View>
      <Text variant="heading" align="center">
        {title}
      </Text>
      {message ? (
        <Text tone="muted" align="center">
          {message}
        </Text>
      ) : null}
      {(actionLabel && onAction) || (secondaryActionLabel && onSecondaryAction) ? (
        <View style={styles.actions}>
          {actionLabel && onAction ? <Button label={actionLabel} onPress={onAction} /> : null}
          {secondaryActionLabel && onSecondaryAction ? (
            <Button label={secondaryActionLabel} variant="secondary" onPress={onSecondaryAction} />
          ) : null}
        </View>
      ) : null}
    </View>
  );
}
