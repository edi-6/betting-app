import React, { useMemo } from 'react';
import { Pressable, StyleSheet } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useTheme } from '../theme';
import { Icon, type IconName } from './Icon';
import { Text } from './Text';

export interface FabProps {
  label: string;
  onPress: () => void;
  icon?: IconName | string;
  /** Distance above the safe area, e.g. to clear a tab bar. */
  offset?: number;
  testID?: string;
}

export function Fab({ label, onPress, icon = 'add', offset = 0, testID }: FabProps) {
  const theme = useTheme();
  const insets = useSafeAreaInsets();

  const styles = useMemo(
    () =>
      StyleSheet.create({
        fab: {
          position: 'absolute',
          right: theme.spacing(5),
          bottom: insets.bottom + theme.spacing(4) + offset,
          flexDirection: 'row',
          alignItems: 'center',
          gap: theme.spacing(2),
          backgroundColor: theme.colors.primary,
          borderRadius: theme.radius.pill,
          paddingVertical: theme.spacing(3.5),
          paddingHorizontal: theme.spacing(5),
          ...theme.shadow.floating,
        },
        pressed: { opacity: 0.85, transform: [{ scale: 0.98 }] },
      }),
    [theme, insets.bottom, offset],
  );

  return (
    <Pressable
      testID={testID}
      accessibilityRole="button"
      accessibilityLabel={label}
      onPress={onPress}
      style={({ pressed }) => [styles.fab, pressed ? styles.pressed : null]}
    >
      <Icon name={icon} size={20} color={theme.colors.primaryText} />
      <Text weight="700" color={theme.colors.primaryText}>
        {label}
      </Text>
    </Pressable>
  );
}
