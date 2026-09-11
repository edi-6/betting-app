import React, { useMemo } from 'react';
import { Pressable, StyleSheet, type StyleProp, type ViewStyle } from 'react-native';

import { useTheme } from '../theme';
import { Icon, type IconName } from './Icon';
import { Text } from './Text';

export interface ChipProps {
  label: string;
  selected?: boolean;
  onPress?: () => void;
  icon?: IconName | string;
  tone?: 'default' | 'positive' | 'negative' | 'warning' | 'info' | 'accent';
  size?: 'sm' | 'md';
  style?: StyleProp<ViewStyle>;
  testID?: string;
}

export function Chip({
  label,
  selected = false,
  onPress,
  icon,
  tone = 'default',
  size = 'md',
  style,
  testID,
}: ChipProps) {
  const theme = useTheme();

  const palette = useMemo(() => {
    const map = {
      default: { solid: theme.colors.primary, soft: theme.colors.primarySoft },
      positive: { solid: theme.colors.positive, soft: theme.colors.positiveSoft },
      negative: { solid: theme.colors.negative, soft: theme.colors.negativeSoft },
      warning: { solid: theme.colors.warning, soft: theme.colors.warningSoft },
      info: { solid: theme.colors.info, soft: theme.colors.infoSoft },
      accent: { solid: theme.colors.accent, soft: theme.colors.accentSoft },
    } as const;
    return map[tone];
  }, [theme, tone]);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        chip: {
          flexDirection: 'row',
          alignItems: 'center',
          gap: theme.spacing(1.5),
          borderRadius: theme.radius.pill,
          borderWidth: StyleSheet.hairlineWidth,
          borderColor: selected ? palette.solid : theme.colors.border,
          backgroundColor: selected ? palette.soft : theme.colors.surfaceAlt,
          paddingVertical: size === 'sm' ? theme.spacing(1.25) : theme.spacing(2),
          paddingHorizontal: size === 'sm' ? theme.spacing(2.5) : theme.spacing(3.5),
        },
        pressed: { opacity: 0.7 },
      }),
    [theme, selected, palette, size],
  );

  const color = selected ? palette.solid : theme.colors.textSecondary;

  const content = (
    <>
      {icon ? <Icon name={icon} size={size === 'sm' ? 13 : 15} color={color} /> : null}
      <Text variant={size === 'sm' ? 'caption' : 'label'} color={color}>
        {label}
      </Text>
    </>
  );

  if (!onPress) {
    return (
      <Pressable testID={testID} disabled style={[styles.chip, style]}>
        {content}
      </Pressable>
    );
  }

  return (
    <Pressable
      testID={testID}
      accessibilityRole="button"
      accessibilityState={{ selected }}
      onPress={onPress}
      style={({ pressed }) => [styles.chip, pressed ? styles.pressed : null, style]}
    >
      {content}
    </Pressable>
  );
}
