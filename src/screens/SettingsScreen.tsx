import React, { useCallback, useMemo, useState } from 'react';
import { Alert, Linking, StyleSheet, Switch, View } from 'react-native';

import {
  Banner,
  Button,
  Card,
  Field,
  KeyValueRow,
  Screen,
  SegmentedControl,
  Select,
  Text,
} from '../components';
import { pickTextFile, shareTextFile, timestampedFilename } from '../data/files';
import { summarize } from '../domain/analytics';
import {
  betsToCsv,
  createBackup,
  csvToBets,
  csvToTransactions,
  readBackup,
  transactionsToCsv,
} from '../domain/csv';
import { BOOKMAKERS } from '../domain/catalog';
import { CURRENCIES, parseAmount } from '../domain/format';
import type { OddsFormat, ResponsibleGamblingLimits, ThemeMode } from '../domain/types';
import { useFormatters } from '../hooks/useFormatters';
import { useHaptics } from '../hooks/useHaptics';
import { useApp } from '../store/AppStore';
import { useGlass, useTheme } from '../theme';
import type { RootStackScreenProps } from '../navigation/types';

const KELLY_OPTIONS = [
  { value: '0.1', label: '10%' },
  { value: '0.25', label: '25%' },
  { value: '0.5', label: '50%' },
  { value: '1', label: 'Full' },
];

export function SettingsScreen({ navigation }: RootStackScreenProps<'Settings'>) {
  const theme = useTheme();
  const haptics = useHaptics();
  const {
    data,
    bets,
    transactions,
    settings,
    updateSettings,
    importBets,
    importTransactions,
    replaceData,
    loadDemoData,
    resetAll,
  } = useApp();
  const { currency } = useFormatters();
  const glass = useGlass();

  const glassDescription = useMemo(() => {
    if (!settings.glassEnabled) {
      return 'Translucent tab bar, sheets and action button';
    }
    switch (glass) {
      case 'liquid':
        return 'Using iOS 26 Liquid Glass';
      case 'blur':
        return 'Using a translucent blur — Liquid Glass needs iOS 26';
      case 'solid':
      default:
        return 'Turned off by the system “reduce transparency” setting';
    }
  }, [glass, settings.glassEnabled]);

  const [busy, setBusy] = useState<string | null>(null);
  const summary = useMemo(() => summarize(bets), [bets]);

  const setLimit = useCallback(
    (key: keyof ResponsibleGamblingLimits, text: string) => {
      const value = parseAmount(text);
      updateSettings({
        limits: { [key]: value !== null && value > 0 ? value : undefined },
      });
    },
    [updateSettings],
  );

  const exportFile = useCallback(
    async (kind: 'bets' | 'transactions' | 'backup') => {
      try {
        setBusy(kind);
        const [filename, content, mime] =
          kind === 'bets'
            ? [timestampedFilename('betledger-bets', 'csv'), betsToCsv(bets), 'text/csv']
            : kind === 'transactions'
              ? [
                  timestampedFilename('betledger-transactions', 'csv'),
                  transactionsToCsv(transactions),
                  'text/csv',
                ]
              : [
                  timestampedFilename('betledger-backup', 'json'),
                  createBackup(data),
                  'application/json',
                ];

        const result = await shareTextFile(filename, content, mime);
        haptics('success');
        if (!result.shared) {
          Alert.alert('Saved', `Sharing is unavailable here. The file is at:\n${result.uri}`);
        }
      } catch (error) {
        haptics('error');
        Alert.alert('Export failed', error instanceof Error ? error.message : 'Please try again.');
      } finally {
        setBusy(null);
      }
    },
    [bets, transactions, data, haptics],
  );

  const importFile = useCallback(
    async (kind: 'bets' | 'transactions' | 'backup') => {
      try {
        setBusy(kind);
        const picked = await pickTextFile(
          kind === 'backup' ? ['application/json'] : ['text/csv', 'text/comma-separated-values', 'text/plain'],
        );
        if (!picked) {
          return;
        }

        if (kind === 'backup') {
          const data = readBackup(picked.content);
          if (!data) {
            Alert.alert('Could not read that file', 'It does not look like a BetLedger backup.');
            return;
          }
          Alert.alert(
            'Replace everything?',
            `This backup holds ${data.bets.length} bets and ${data.transactions.length} transactions. Your current data will be replaced.`,
            [
              { text: 'Cancel', style: 'cancel' },
              {
                text: 'Replace',
                style: 'destructive',
                onPress: () => {
                  replaceData(data);
                  haptics('success');
                },
              },
            ],
          );
          return;
        }

        if (kind === 'bets') {
          const result = csvToBets(picked.content);
          if (result.items.length === 0) {
            Alert.alert('Nothing imported', result.errors[0] ?? 'No usable rows were found.');
            return;
          }
          importBets(result.items);
          haptics('success');
          Alert.alert(
            'Imported',
            `${result.items.length} bets added or updated.${
              result.errors.length > 0 ? `\n\n${result.errors.length} rows were skipped.` : ''
            }`,
          );
          return;
        }

        const result = csvToTransactions(picked.content);
        if (result.items.length === 0) {
          Alert.alert('Nothing imported', result.errors[0] ?? 'No usable rows were found.');
          return;
        }
        importTransactions(result.items);
        haptics('success');
        Alert.alert('Imported', `${result.items.length} transactions added or updated.`);
      } catch (error) {
        haptics('error');
        Alert.alert('Import failed', error instanceof Error ? error.message : 'Please try again.');
      } finally {
        setBusy(null);
      }
    },
    [importBets, importTransactions, replaceData, haptics],
  );

  const confirmReset = useCallback(() => {
    Alert.alert(
      'Erase all data?',
      'Every bet, transaction and setting will be deleted from this device. Export a backup first if you want to keep it.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Erase everything',
          style: 'destructive',
          onPress: () => {
            haptics('warning');
            void resetAll();
            navigation.popTo('Tabs');
          },
        },
      ],
    );
  }, [resetAll, navigation, haptics]);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        group: { gap: theme.spacing(3.5) },
        row: { flexDirection: 'row', gap: theme.spacing(3) },
        toggleRow: {
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing(3),
          paddingVertical: theme.spacing(2),
        },
        buttonRow: { flexDirection: 'row', flexWrap: 'wrap', gap: theme.spacing(2) },
      }),
    [theme],
  );

  return (
    <Screen
      title="Settings"
      subtitle="Preferences, limits and your data"
      headerAction={<Button label="Done" variant="ghost" size="sm" onPress={() => navigation.goBack()} />}
    >
      <Card title="Display">
        <View style={styles.group}>
          <Select
            label="Currency"
            value={settings.currency}
            options={CURRENCIES.map((item) => ({
              value: item.code,
              label: `${item.code} · ${item.symbol.trim()}`,
            }))}
            onChange={(value) => updateSettings({ currency: value })}
            searchable
            title="Account currency"
          />

          <View style={{ gap: theme.spacing(2) }}>
            <Text variant="label" tone="secondary">
              Odds format
            </Text>
            <SegmentedControl
              segments={[
                { value: 'decimal', label: 'Decimal' },
                { value: 'american', label: 'American' },
                { value: 'fractional', label: 'Fractional' },
              ]}
              value={settings.oddsFormat}
              onChange={(value) => updateSettings({ oddsFormat: value as OddsFormat })}
            />
          </View>

          <View style={{ gap: theme.spacing(2) }}>
            <Text variant="label" tone="secondary">
              Appearance
            </Text>
            <SegmentedControl
              segments={[
                { value: 'system', label: 'System' },
                { value: 'light', label: 'Light' },
                { value: 'dark', label: 'Dark' },
              ]}
              value={settings.themeMode}
              onChange={(value) => updateSettings({ themeMode: value as ThemeMode })}
            />
          </View>

          <View style={styles.toggleRow}>
            <View style={{ flex: 1 }}>
              <Text variant="label" tone="secondary">
                Glass effects
              </Text>
              <Text variant="caption" tone="muted">
                {glassDescription}
              </Text>
            </View>
            <Switch
              value={settings.glassEnabled}
              onValueChange={(value) => updateSettings({ glassEnabled: value })}
              trackColor={{ false: theme.colors.surfaceSunken, true: theme.colors.primarySoft }}
              thumbColor={settings.glassEnabled ? theme.colors.primary : theme.colors.neutral}
            />
          </View>

          <View style={styles.toggleRow}>
            <View style={{ flex: 1 }}>
              <Text variant="label" tone="secondary">
                Haptic feedback
              </Text>
              <Text variant="caption" tone="muted">
                A small buzz when you settle or save a bet
              </Text>
            </View>
            <Switch
              value={settings.hapticsEnabled}
              onValueChange={(value) => updateSettings({ hapticsEnabled: value })}
              trackColor={{ false: theme.colors.surfaceSunken, true: theme.colors.primarySoft }}
              thumbColor={settings.hapticsEnabled ? theme.colors.primary : theme.colors.neutral}
            />
          </View>
        </View>
      </Card>

      <Card title="Betting defaults" subtitle="Pre-filled when you add a bet">
        <View style={styles.group}>
          <View style={styles.row}>
            <Field
              containerStyle={{ flex: 1 }}
              label="Default stake"
              defaultValue={String(settings.defaultStake)}
              onEndEditing={(event) => {
                const value = parseAmount(event.nativeEvent.text);
                updateSettings({ defaultStake: value !== null && value >= 0 ? value : 0 });
              }}
              keyboardType="decimal-pad"
              prefix={currency}
            />
            <View style={{ flex: 1 }}>
              <Select
                label="Default bookmaker"
                value={settings.defaultBookmaker}
                options={BOOKMAKERS}
                onChange={(value) => updateSettings({ defaultBookmaker: value })}
                placeholder="None"
                searchable
                allowCustom
              />
            </View>
          </View>

          <View style={{ gap: theme.spacing(2) }}>
            <Text variant="label" tone="secondary">
              Kelly fraction
            </Text>
            <SegmentedControl
              segments={KELLY_OPTIONS}
              value={String(settings.kellyFraction)}
              onChange={(value) => updateSettings({ kellyFraction: Number(value) })}
            />
            <Text variant="caption" tone="muted">
              Quarter-Kelly keeps most of the growth with a fraction of the swings.
            </Text>
          </View>
        </View>
      </Card>

      <Card
        title="Limits"
        subtitle="Leave blank to switch a limit off. Nothing is blocked — you just get a warning."
      >
        <View style={styles.group}>
          <View style={styles.row}>
            <Field
              containerStyle={{ flex: 1 }}
              label="Daily stake cap"
              defaultValue={settings.limits.dailyStakeLimit?.toString() ?? ''}
              onEndEditing={(event) => setLimit('dailyStakeLimit', event.nativeEvent.text)}
              keyboardType="decimal-pad"
              prefix={currency}
              placeholder="Off"
            />
            <Field
              containerStyle={{ flex: 1 }}
              label="Bets per day"
              defaultValue={settings.limits.dailyBetCountLimit?.toString() ?? ''}
              onEndEditing={(event) => setLimit('dailyBetCountLimit', event.nativeEvent.text)}
              keyboardType="number-pad"
              placeholder="Off"
            />
          </View>
          <View style={styles.row}>
            <Field
              containerStyle={{ flex: 1 }}
              label="Weekly loss limit"
              defaultValue={settings.limits.weeklyLossLimit?.toString() ?? ''}
              onEndEditing={(event) => setLimit('weeklyLossLimit', event.nativeEvent.text)}
              keyboardType="decimal-pad"
              prefix={currency}
              placeholder="Off"
            />
            <Field
              containerStyle={{ flex: 1 }}
              label="Monthly loss limit"
              defaultValue={settings.limits.monthlyLossLimit?.toString() ?? ''}
              onEndEditing={(event) => setLimit('monthlyLossLimit', event.nativeEvent.text)}
              keyboardType="decimal-pad"
              prefix={currency}
              placeholder="Off"
            />
          </View>
          <Field
            label="Max stake as % of bankroll"
            defaultValue={settings.limits.maxStakePercent?.toString() ?? ''}
            onEndEditing={(event) => setLimit('maxStakePercent', event.nativeEvent.text)}
            keyboardType="decimal-pad"
            accessory={<Text tone="muted">%</Text>}
            placeholder="Off"
            hint="Most staking plans keep a single bet under 2–5% of the bankroll."
          />
        </View>
      </Card>

      <Card title="Your data" subtitle={`${bets.length} bets · ${transactions.length} transactions`}>
        <View style={styles.group}>
          <View>
            <Text variant="label" tone="secondary">
              Export
            </Text>
            <View style={[styles.buttonRow, { marginTop: theme.spacing(2) }]}>
              <Button
                label="Bets (CSV)"
                variant="secondary"
                size="sm"
                icon="download-outline"
                loading={busy === 'bets'}
                onPress={() => void exportFile('bets')}
              />
              <Button
                label="Transactions (CSV)"
                variant="secondary"
                size="sm"
                icon="download-outline"
                loading={busy === 'transactions'}
                onPress={() => void exportFile('transactions')}
              />
              <Button
                label="Full backup (JSON)"
                variant="secondary"
                size="sm"
                icon="archive-outline"
                loading={busy === 'backup'}
                onPress={() => void exportFile('backup')}
              />
            </View>
          </View>

          <View>
            <Text variant="label" tone="secondary">
              Import
            </Text>
            <View style={[styles.buttonRow, { marginTop: theme.spacing(2) }]}>
              <Button
                label="Bets (CSV)"
                variant="secondary"
                size="sm"
                icon="cloud-upload-outline"
                onPress={() => void importFile('bets')}
              />
              <Button
                label="Transactions (CSV)"
                variant="secondary"
                size="sm"
                icon="cloud-upload-outline"
                onPress={() => void importFile('transactions')}
              />
              <Button
                label="Restore backup"
                variant="secondary"
                size="sm"
                icon="refresh-outline"
                onPress={() => void importFile('backup')}
              />
            </View>
            <Text variant="caption" tone="muted" style={{ marginTop: theme.spacing(2) }}>
              Imported rows replace existing bets with the same id, so re-importing the same file is
              safe.
            </Text>
          </View>

          <View style={styles.buttonRow}>
            <Button
              label="Load demo data"
              variant="ghost"
              size="sm"
              onPress={() =>
                Alert.alert(
                  'Load demo data?',
                  'This replaces everything currently in the app with a sample six-month history.',
                  [
                    { text: 'Cancel', style: 'cancel' },
                    { text: 'Load', onPress: loadDemoData },
                  ],
                )
              }
            />
            <Button label="Erase all data" variant="danger" size="sm" onPress={confirmReset} />
          </View>
        </View>
      </Card>

      <Card title="Summary">
        <KeyValueRow label="Bets tracked" value={String(summary.totalBets)} />
        <KeyValueRow label="Settled" value={String(summary.settledBets)} />
        <KeyValueRow label="Open" value={String(summary.pendingBets)} />
        <KeyValueRow label="Storage" value="On this device only" divider={false} />
      </Card>

      <Banner
        tone="warning"
        title="Gamble responsibly"
        message="BetLedger is a record-keeping and analysis tool. It does not place bets, offer tips, or predict results. If betting stops being fun, take a break — help is available and free."
        actionLabel="Find support"
        onAction={() => {
          void Linking.openURL('https://www.begambleaware.org/').catch(() => {
            Alert.alert(
              'Support',
              'BeGambleAware (UK): 0808 8020 133\nNational Problem Gambling Helpline (US): 1-800-522-4700\nGambling Help Online (AU): 1800 858 858',
            );
          });
        }}
      />

      <Text variant="caption" tone="muted" align="center">
        BetLedger 1.0.0 · All data stays on your device
      </Text>
    </Screen>
  );
}
