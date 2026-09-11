import React, { useMemo } from 'react';
import {
  KeyboardAvoidingView,
  Platform,
  RefreshControl,
  ScrollView,
  StyleSheet,
  View,
  type StyleProp,
  type ViewStyle,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useTheme } from '../theme';
import { Text } from './Text';

export interface ScreenProps {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
  /** Rendered to the right of the title. */
  headerAction?: React.ReactNode;
  scroll?: boolean;
  /** Extra bottom padding so content clears a floating action button. */
  bottomInset?: number;
  contentStyle?: StyleProp<ViewStyle>;
  onRefresh?: () => void;
  refreshing?: boolean;
  testID?: string;
}

/** Standard page frame: safe-area aware, optional large title, optional scrolling. */
export function Screen({
  children,
  title,
  subtitle,
  headerAction,
  scroll = true,
  bottomInset = 0,
  contentStyle,
  onRefresh,
  refreshing = false,
  testID,
}: ScreenProps) {
  const theme = useTheme();
  const insets = useSafeAreaInsets();

  const styles = useMemo(
    () =>
      StyleSheet.create({
        root: { flex: 1, backgroundColor: theme.colors.background },
        header: {
          flexDirection: 'row',
          alignItems: 'flex-end',
          justifyContent: 'space-between',
          gap: theme.spacing(3),
          paddingHorizontal: theme.spacing(5),
          paddingTop: theme.spacing(2),
          paddingBottom: theme.spacing(3),
        },
        headerText: { flex: 1, gap: 2 },
        content: {
          paddingHorizontal: theme.spacing(5),
          paddingBottom: insets.bottom + theme.spacing(6) + bottomInset,
          gap: theme.spacing(4),
        },
      }),
    [theme, insets.bottom, bottomInset],
  );

  const header = title ? (
    <View style={styles.header}>
      <View style={styles.headerText}>
        <Text variant="title">{title}</Text>
        {subtitle ? (
          <Text variant="caption" tone="muted">
            {subtitle}
          </Text>
        ) : null}
      </View>
      {headerAction}
    </View>
  ) : null;

  const body = scroll ? (
    <ScrollView
      contentContainerStyle={[styles.content, contentStyle]}
      showsVerticalScrollIndicator={false}
      keyboardShouldPersistTaps="handled"
      keyboardDismissMode="on-drag"
      refreshControl={
        onRefresh ? (
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={theme.colors.textMuted}
          />
        ) : undefined
      }
    >
      {children}
    </ScrollView>
  ) : (
    <View style={[{ flex: 1 }, contentStyle]}>{children}</View>
  );

  return (
    <View style={styles.root} testID={testID}>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 0}
      >
        {header}
        {body}
      </KeyboardAvoidingView>
    </View>
  );
}
