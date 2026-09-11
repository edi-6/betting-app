import React, { useMemo } from 'react';
import { Alert, Linking, Modal, ScrollView, StyleSheet, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useApp } from '../store/AppStore';
import { useTheme } from '../theme';
import { Button } from './Button';
import { Icon } from './Icon';
import { Text } from './Text';

const POINTS: { icon: string; title: string; body: string }[] = [
  {
    icon: 'create-outline',
    title: 'A record, not a bookmaker',
    body: 'BetLedger never places a bet, holds money, or connects to a sportsbook. You type in what you already staked elsewhere.',
  },
  {
    icon: 'eye-off-outline',
    title: 'No tips, no predictions',
    body: 'Nothing here tells you what to back. The calculators are arithmetic on numbers you supply — they cannot see the future.',
  },
  {
    icon: 'phone-portrait-outline',
    title: 'Your data stays here',
    body: 'Everything is stored on this device. No account, no servers, no analytics, no ads.',
  },
  {
    icon: 'shield-checkmark-outline',
    title: 'It will show you the truth',
    body: 'Including when the truth is uncomfortable. Most bettors lose over time. Set limits in Settings, and take a break if this stops being fun.',
  },
];

/**
 * Shown once, before the ledger is used.
 *
 * A tracker for gamblers should be straight about what it is and is not before the
 * first bet goes in, so the acknowledgement is a gate rather than a footnote. It waits
 * for hydration so returning users never see it flash.
 */
export function DisclaimerGate() {
  const theme = useTheme();
  const insets = useSafeAreaInsets();
  const { settings, hydrated, updateSettings } = useApp();

  const styles = useMemo(
    () =>
      StyleSheet.create({
        backdrop: { flex: 1, backgroundColor: theme.colors.background },
        content: {
          paddingHorizontal: theme.spacing(6),
          paddingTop: insets.top + theme.spacing(8),
          paddingBottom: theme.spacing(6),
          gap: theme.spacing(5),
        },
        header: { gap: theme.spacing(2) },
        mark: {
          width: 56,
          height: 56,
          borderRadius: 18,
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: theme.colors.primarySoft,
          marginBottom: theme.spacing(2),
        },
        point: { flexDirection: 'row', gap: theme.spacing(3.5), alignItems: 'flex-start' },
        pointIcon: {
          width: 34,
          height: 34,
          borderRadius: 17,
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: theme.colors.surfaceAlt,
        },
        pointBody: { flex: 1, gap: 2 },
        footer: {
          paddingHorizontal: theme.spacing(6),
          paddingBottom: insets.bottom + theme.spacing(4),
          paddingTop: theme.spacing(3),
          gap: theme.spacing(2),
          borderTopWidth: StyleSheet.hairlineWidth,
          borderColor: theme.colors.border,
        },
      }),
    [theme, insets.top, insets.bottom],
  );

  // Nothing to gate until we know what was stored.
  if (!hydrated || settings.disclaimerAcceptedAt) {
    return null;
  }

  const openSupport = () => {
    void Linking.openURL('https://www.begambleaware.org/').catch(() => {
      Alert.alert(
        'Support',
        'BeGambleAware (UK): 0808 8020 133\nNational Problem Gambling Helpline (US): 1-800-522-4700\nGambling Help Online (AU): 1800 858 858',
      );
    });
  };

  return (
    <Modal visible animationType="fade" statusBarTranslucent testID="disclaimer-gate">
      <View style={styles.backdrop}>
        <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
          <View style={styles.header}>
            <View style={styles.mark}>
              <Icon name="trending-up" size={28} color={theme.colors.primary} />
            </View>
            <Text variant="display">Before you start</Text>
            <Text tone="muted">
              BetLedger is a betting journal. Thirty seconds on what that means.
            </Text>
          </View>

          {POINTS.map((point) => (
            <View key={point.title} style={styles.point}>
              <View style={styles.pointIcon}>
                <Icon name={point.icon} size={17} color={theme.colors.textSecondary} />
              </View>
              <View style={styles.pointBody}>
                <Text variant="subheading">{point.title}</Text>
                <Text variant="caption" tone="muted">
                  {point.body}
                </Text>
              </View>
            </View>
          ))}
        </ScrollView>

        <View style={styles.footer}>
          <Button
            label="I understand"
            size="lg"
            fullWidth
            testID="accept-disclaimer"
            onPress={() => updateSettings({ disclaimerAcceptedAt: new Date().toISOString() })}
          />
          <Button label="Find support" variant="ghost" fullWidth onPress={openSupport} />
        </View>
      </View>
    </Modal>
  );
}
