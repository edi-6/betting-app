import React, { useMemo } from 'react';
import { Pressable, StyleSheet, View } from 'react-native';

import { useTheme } from '../theme';
import { Icon, type IconName } from './Icon';
import { Text } from './Text';

export type BannerTone = 'info' | 'warning' | 'negative' | 'positive';

export interface BannerProps {
  tone?: BannerTone;
  title?: string;
  message: string;
  icon?: IconName | string;
  actionLabel?: string;
  onAction?: () => void;
  onDismiss?: () => void;
}

export function Banner({
  tone = 'info',
  title,
  message,
  icon,
  actionLabel,
  onAction,
  onDismiss,
}: BannerProps) {
  const theme = useTheme();

  const palette = useMemo(() => {
    switch (tone) {
      case 'warning':
        return { solid: theme.colors.warning, soft: theme.colors.warningSoft, icon: 'warning-outline' };
      case 'negative':
        return { solid: theme.colors.negative, soft: theme.colors.negativeSoft, icon: 'alert-circle-outline' };
      case 'positive':
        return { solid: theme.colors.positive, soft: theme.colors.positiveSoft, icon: 'checkmark-circle-outline' };
      case 'info':
      default:
        return { solid: theme.colors.info, soft: theme.colors.infoSoft, icon: 'information-circle-outline' };
    }
  }, [theme, tone]);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        banner: {
          flexDirection: 'row',
          gap: theme.spacing(3),
          alignItems: 'flex-start',
          backgroundColor: palette.soft,
          borderRadius: theme.radius.md,
          borderWidth: StyleSheet.hairlineWidth,
          borderColor: palette.solid,
          padding: theme.spacing(3.5),
        },
        body: { flex: 1, gap: 2 },
        actions: { flexDirection: 'row', gap: theme.spacing(4), marginTop: theme.spacing(2) },
      }),
    [theme, palette],
  );

  return (
    <View style={styles.banner} accessibilityRole="alert">
      <Icon name={icon ?? palette.icon} size={20} color={palette.solid} />
      <View style={styles.body}>
        {title ? (
          <Text variant="label" color={palette.solid}>
            {title}
          </Text>
        ) : null}
        <Text variant="caption" tone="secondary">
          {message}
        </Text>
        {actionLabel || onDismiss ? (
          <View style={styles.actions}>
            {actionLabel && onAction ? (
              <Pressable onPress={onAction} accessibilityRole="button">
                <Text variant="label" color={palette.solid}>
                  {actionLabel}
                </Text>
              </Pressable>
            ) : null}
            {onDismiss ? (
              <Pressable onPress={onDismiss} accessibilityRole="button">
                <Text variant="label" tone="muted">
                  Dismiss
                </Text>
              </Pressable>
            ) : null}
          </View>
        ) : null}
      </View>
    </View>
  );
}
