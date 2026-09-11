import React, { useMemo } from 'react';
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  View,
  type StyleProp,
  type ViewStyle,
} from 'react-native';

import { useTheme } from '../theme';
import { Icon, type IconName } from './Icon';
import { Text } from './Text';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'success';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps {
  label: string;
  onPress: () => void;
  variant?: ButtonVariant;
  size?: ButtonSize;
  icon?: IconName | string;
  iconPosition?: 'left' | 'right';
  disabled?: boolean;
  loading?: boolean;
  fullWidth?: boolean;
  style?: StyleProp<ViewStyle>;
  testID?: string;
}

export function Button({
  label,
  onPress,
  variant = 'primary',
  size = 'md',
  icon,
  iconPosition = 'left',
  disabled = false,
  loading = false,
  fullWidth = false,
  style,
  testID,
}: ButtonProps) {
  const theme = useTheme();

  const palette = useMemo(() => {
    switch (variant) {
      case 'secondary':
        return {
          background: theme.colors.surfaceAlt,
          border: theme.colors.border,
          text: theme.colors.text,
        };
      case 'ghost':
        return { background: 'transparent', border: 'transparent', text: theme.colors.primary };
      case 'danger':
        return {
          background: theme.colors.negativeSoft,
          border: theme.colors.negative,
          text: theme.colors.negative,
        };
      case 'success':
        return {
          background: theme.colors.positiveSoft,
          border: theme.colors.positive,
          text: theme.colors.positive,
        };
      case 'primary':
      default:
        return {
          background: theme.colors.primary,
          border: theme.colors.primary,
          text: theme.colors.primaryText,
        };
    }
  }, [theme, variant]);

  const metrics = useMemo(() => {
    switch (size) {
      case 'sm':
        return { paddingVertical: theme.spacing(2), paddingHorizontal: theme.spacing(3), font: 13 };
      case 'lg':
        return { paddingVertical: theme.spacing(4), paddingHorizontal: theme.spacing(6), font: 17 };
      case 'md':
      default:
        return { paddingVertical: theme.spacing(3), paddingHorizontal: theme.spacing(5), font: 15 };
    }
  }, [theme, size]);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        base: {
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'center',
          gap: theme.spacing(2),
          borderRadius: theme.radius.md,
          borderWidth: variant === 'ghost' ? 0 : StyleSheet.hairlineWidth,
          borderColor: palette.border,
          backgroundColor: palette.background,
          paddingVertical: metrics.paddingVertical,
          paddingHorizontal: metrics.paddingHorizontal,
          alignSelf: fullWidth ? 'stretch' : 'flex-start',
          minHeight: 44,
        },
        pressed: { opacity: 0.72 },
        disabled: { opacity: 0.45 },
        content: { flexDirection: 'row', alignItems: 'center', gap: theme.spacing(2) },
      }),
    [theme, palette, metrics, fullWidth, variant],
  );

  const isDisabled = disabled || loading;

  return (
    <Pressable
      testID={testID}
      accessibilityRole="button"
      accessibilityState={{ disabled: isDisabled, busy: loading }}
      accessibilityLabel={label}
      disabled={isDisabled}
      onPress={onPress}
      style={({ pressed }) => [
        styles.base,
        pressed && !isDisabled ? styles.pressed : null,
        isDisabled ? styles.disabled : null,
        style,
      ]}
    >
      {loading ? (
        <ActivityIndicator size="small" color={palette.text} />
      ) : (
        <View style={styles.content}>
          {icon && iconPosition === 'left' ? (
            <Icon name={icon} size={metrics.font + 2} color={palette.text} />
          ) : null}
          <Text color={palette.text} weight="600" style={{ fontSize: metrics.font }}>
            {label}
          </Text>
          {icon && iconPosition === 'right' ? (
            <Icon name={icon} size={metrics.font + 2} color={palette.text} />
          ) : null}
        </View>
      )}
    </Pressable>
  );
}
