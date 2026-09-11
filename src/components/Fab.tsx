import { BottomTabBarHeightContext } from '@react-navigation/bottom-tabs';
import React, { useContext, useMemo } from 'react';
import { Pressable, StyleSheet, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useGlass, useTheme } from '../theme';
import { GlassSurface, withAlpha } from './GlassSurface';
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
  const glass = useGlass();
  const insets = useSafeAreaInsets();
  const tabBarHeight = useContext(BottomTabBarHeightContext) ?? 0;
  const liquid = glass === 'liquid';

  const styles = useMemo(
    () =>
      StyleSheet.create({
        fab: {
          position: 'absolute',
          right: theme.spacing(5),
          bottom:
            (tabBarHeight > 0 ? tabBarHeight : insets.bottom) + theme.spacing(4) + offset,
          borderRadius: theme.radius.pill,
          ...theme.shadow.floating,
        },
        inner: {
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'center',
          gap: theme.spacing(2),
          paddingVertical: theme.spacing(3.5),
          paddingHorizontal: theme.spacing(5),
          borderRadius: theme.radius.pill,
          // Liquid Glass supplies its own material, so only the solid path is filled.
          backgroundColor: liquid ? 'transparent' : theme.colors.primary,
        },
        pressed: { opacity: 0.85, transform: [{ scale: 0.98 }] },
      }),
    [theme, insets.bottom, tabBarHeight, offset, liquid],
  );

  return (
    <Pressable
      testID={testID}
      accessibilityRole="button"
      accessibilityLabel={label}
      onPress={onPress}
      style={({ pressed }) => [styles.fab, pressed ? styles.pressed : null]}
    >
      {liquid ? (
        <GlassSurface
          radius={theme.radius.pill}
          glassStyle="regular"
          interactive
          tintColor={withAlpha(theme.colors.primary, 0.55)}
          bordered={false}
        >
          <View style={styles.inner}>
            <Icon name={icon} size={20} color={theme.colors.text} />
            <Text weight="700">{label}</Text>
          </View>
        </GlassSurface>
      ) : (
        <View style={styles.inner}>
          <Icon name={icon} size={20} color={theme.colors.primaryText} />
          <Text weight="700" color={theme.colors.primaryText}>
            {label}
          </Text>
        </View>
      )}
    </Pressable>
  );
}
