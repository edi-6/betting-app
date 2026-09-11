import React, { useMemo } from 'react';
import { Modal, Pressable, ScrollView, StyleSheet, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useTheme } from '../theme';
import { Icon } from './Icon';
import { Text } from './Text';

export interface SheetProps {
  visible: boolean;
  onClose: () => void;
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  /** Pinned to the bottom, outside the scroll area. */
  footer?: React.ReactNode;
  /** Fraction of the screen the sheet may occupy. */
  maxHeightRatio?: number;
}

/** A bottom sheet built on the platform modal — no gesture dependencies needed. */
export function Sheet({
  visible,
  onClose,
  title,
  subtitle,
  children,
  footer,
  maxHeightRatio = 0.86,
}: SheetProps) {
  const theme = useTheme();
  const insets = useSafeAreaInsets();

  const styles = useMemo(
    () =>
      StyleSheet.create({
        backdrop: { flex: 1, backgroundColor: theme.colors.overlay, justifyContent: 'flex-end' },
        sheet: {
          backgroundColor: theme.colors.backgroundElevated,
          borderTopLeftRadius: theme.radius.xl,
          borderTopRightRadius: theme.radius.xl,
          maxHeight: `${Math.round(maxHeightRatio * 100)}%`,
          paddingBottom: insets.bottom + theme.spacing(2),
          borderTopWidth: StyleSheet.hairlineWidth,
          borderColor: theme.colors.border,
        },
        grabber: {
          alignSelf: 'center',
          width: 40,
          height: 4,
          borderRadius: 2,
          backgroundColor: theme.colors.borderStrong,
          marginTop: theme.spacing(2.5),
          marginBottom: theme.spacing(1),
        },
        header: {
          flexDirection: 'row',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          gap: theme.spacing(3),
          paddingHorizontal: theme.spacing(5),
          paddingTop: theme.spacing(2),
          paddingBottom: theme.spacing(3),
        },
        headerText: { flex: 1, gap: 2 },
        close: {
          width: 32,
          height: 32,
          borderRadius: 16,
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: theme.colors.surfaceAlt,
        },
        content: { paddingHorizontal: theme.spacing(5), paddingBottom: theme.spacing(4) },
        footer: {
          paddingHorizontal: theme.spacing(5),
          paddingTop: theme.spacing(3),
          borderTopWidth: StyleSheet.hairlineWidth,
          borderColor: theme.colors.border,
          gap: theme.spacing(2),
        },
      }),
    [theme, insets.bottom, maxHeightRatio],
  );

  return (
    <Modal
      visible={visible}
      animationType="slide"
      transparent
      onRequestClose={onClose}
      statusBarTranslucent
    >
      <Pressable style={styles.backdrop} onPress={onClose} accessibilityLabel="Close">
        <Pressable style={styles.sheet} onPress={(event) => event.stopPropagation()}>
          <View style={styles.grabber} />
          {title ? (
            <View style={styles.header}>
              <View style={styles.headerText}>
                <Text variant="heading">{title}</Text>
                {subtitle ? (
                  <Text variant="caption" tone="muted">
                    {subtitle}
                  </Text>
                ) : null}
              </View>
              <Pressable
                onPress={onClose}
                style={styles.close}
                accessibilityRole="button"
                accessibilityLabel="Close"
              >
                <Icon name="close" size={18} color={theme.colors.textSecondary} />
              </Pressable>
            </View>
          ) : null}
          <ScrollView
            contentContainerStyle={styles.content}
            keyboardShouldPersistTaps="handled"
            showsVerticalScrollIndicator={false}
          >
            {children}
          </ScrollView>
          {footer ? <View style={styles.footer}>{footer}</View> : null}
        </Pressable>
      </Pressable>
    </Modal>
  );
}
