import React, { useMemo, useState } from 'react';
import { Pressable, StyleSheet, View } from 'react-native';

import {
  BarChart,
  Banner,
  BetCard,
  Button,
  Card,
  EmptyState,
  Fab,
  Icon,
  LineChart,
  Screen,
  SegmentedControl,
  StatTile,
  Text,
  type LinePoint,
} from '../components';
import {
  breakdown,
  bySport,
  clvStats,
  profitSeries,
  summarize,
} from '../domain/analytics';
import { computeBankroll, evaluateLimits } from '../domain/bankroll';
import { formatShortDate } from '../domain/dates';
import { formatPercent, formatSignedPercent } from '../domain/format';
import {
  applyFilter,
  DATE_RANGE_PRESETS,
  EMPTY_FILTER,
  type DateRangePreset,
} from '../domain/filters';
import { settleBet } from '../domain/settlement';
import { useFormatters } from '../hooks/useFormatters';
import { useApp } from '../store/AppStore';
import { useTheme } from '../theme';
import { profitColor } from '../theme/tokens';
import type { TabScreenProps } from '../navigation/types';

export function DashboardScreen({ navigation }: TabScreenProps<'Dashboard'>) {
  const theme = useTheme();
  const { bets, transactions, settings, loadDemoData } = useApp();
  const { money, signedMoney, compactMoney } = useFormatters();
  const [range, setRange] = useState<DateRangePreset>('30d');

  const scoped = useMemo(
    () => applyFilter(bets, { ...EMPTY_FILTER, range }),
    [bets, range],
  );

  const summary = useMemo(() => summarize(scoped), [scoped]);
  const allTime = useMemo(() => summarize(bets), [bets]);
  const bankroll = useMemo(
    () => computeBankroll(bets, transactions, settings.startingBankroll),
    [bets, transactions, settings.startingBankroll],
  );
  const series = useMemo(() => profitSeries(scoped), [scoped]);
  const clv = useMemo(() => clvStats(scoped), [scoped]);
  const limits = useMemo(
    () => evaluateLimits(bets, settings.limits, bankroll.balance),
    [bets, settings.limits, bankroll.balance],
  );

  const openBets = useMemo(
    () =>
      bets
        .filter((bet) => !settleBet(bet).isSettled)
        .sort((a, b) => new Date(b.placedAt).getTime() - new Date(a.placedAt).getTime()),
    [bets],
  );

  const recent = useMemo(
    () =>
      bets
        .slice()
        .sort((a, b) => new Date(b.placedAt).getTime() - new Date(a.placedAt).getTime())
        .slice(0, 4),
    [bets],
  );

  const chartData = useMemo<LinePoint[]>(
    () =>
      series.map((point) => ({
        x: point.timestamp,
        y: point.cumulative,
        label: formatShortDate(point.timestamp),
      })),
    [series],
  );

  const sportRows = useMemo(
    () =>
      breakdown(scoped, bySport)
        .slice(0, 5)
        .map((row) => ({
          key: row.key,
          label: row.label,
          value: row.profit,
          caption: `${row.bets} · ${formatSignedPercent(row.roi)}`,
        })),
    [scoped],
  );

  const breachedLimit = limits.find((limit) => limit.severity === 'exceeded');

  const styles = useMemo(
    () =>
      StyleSheet.create({
        hero: {
          backgroundColor: theme.colors.surface,
          borderRadius: theme.radius.xl,
          borderWidth: StyleSheet.hairlineWidth,
          borderColor: theme.colors.border,
          padding: theme.spacing(5),
          gap: theme.spacing(4),
          ...theme.shadow.card,
        },
        heroTop: {
          flexDirection: 'row',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
        },
        heroStats: { flexDirection: 'row', gap: theme.spacing(3) },
        roiBadge: {
          flexDirection: 'row',
          alignItems: 'center',
          gap: theme.spacing(1),
          backgroundColor:
            allTime.profit >= 0 ? theme.colors.positiveSoft : theme.colors.negativeSoft,
          borderRadius: theme.radius.pill,
          paddingVertical: theme.spacing(1.5),
          paddingHorizontal: theme.spacing(3),
        },
        grid: { flexDirection: 'row', flexWrap: 'wrap', gap: theme.spacing(3) },
        sectionHeader: {
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing(3),
        },
        list: { gap: theme.spacing(3) },
        gearButton: {
          width: 40,
          height: 40,
          borderRadius: 20,
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: theme.colors.surfaceAlt,
        },
      }),
    [theme, allTime.profit],
  );

  if (bets.length === 0 && transactions.length === 0) {
    return (
      <Screen title="BetLedger" subtitle="Track every bet, learn what actually works">
        <EmptyState
          icon="rocket-outline"
          title="Let's get your ledger started"
          message="Log your first bet to start building a picture of your edge — profit, ROI, closing line value and more."
          actionLabel="Add your first bet"
          onAction={() => navigation.navigate('BetForm')}
          secondaryActionLabel="Load demo data"
          onSecondaryAction={loadDemoData}
        />
      </Screen>
    );
  }

  return (
    <>
      <Screen
        title="Dashboard"
        subtitle={`${allTime.totalBets} bets tracked · ${allTime.pendingBets} open`}
        bottomInset={72}
        headerAction={
          <Pressable
            style={styles.gearButton}
            accessibilityRole="button"
            accessibilityLabel="Settings"
            onPress={() => navigation.navigate('Settings')}
          >
            <Icon name="settings-outline" size={20} color={theme.colors.textSecondary} />
          </Pressable>
        }
      >
        {breachedLimit ? (
          <Banner
            tone="warning"
            title={`${breachedLimit.label} limit reached`}
            message={`${breachedLimit.description}: ${
              breachedLimit.isCurrency
                ? money(breachedLimit.used)
                : breachedLimit.used.toFixed(breachedLimit.key === 'maxStakePercent' ? 1 : 0)
            } of ${
              breachedLimit.isCurrency
                ? money(breachedLimit.limit)
                : String(breachedLimit.limit)
            }. Consider taking a break.`}
            actionLabel="Review limits"
            onAction={() => navigation.navigate('Settings')}
          />
        ) : null}

        <View style={styles.hero}>
          <View style={styles.heroTop}>
            <View style={{ gap: 2 }}>
              <Text variant="caption" tone="muted">
                Bankroll
              </Text>
              <Text variant="display">{money(bankroll.balance)}</Text>
            </View>
            <View style={styles.roiBadge}>
              <Icon
                name={allTime.profit >= 0 ? 'trending-up' : 'trending-down'}
                size={14}
                color={profitColor(allTime.profit, theme.colors)}
              />
              <Text variant="label" color={profitColor(allTime.profit, theme.colors)}>
                {formatSignedPercent(allTime.roi)}
              </Text>
            </View>
          </View>

          <View style={styles.heroStats}>
            <StatTile
              label="Available"
              value={money(bankroll.available)}
              icon="wallet-outline"
              compact
            />
            <StatTile
              label="At risk"
              value={money(bankroll.atRisk)}
              caption={`${allTime.pendingBets} open`}
              icon="hourglass-outline"
              compact
              tone={bankroll.atRisk > 0 ? 'warning' : 'default'}
            />
            <StatTile
              label="All-time P/L"
              value={compactMoney(allTime.profit)}
              icon="stats-chart-outline"
              compact
              tone={allTime.profit > 0 ? 'positive' : allTime.profit < 0 ? 'negative' : 'default'}
            />
          </View>
        </View>

        <SegmentedControl
          segments={DATE_RANGE_PRESETS.map((preset) => ({
            value: preset.key,
            label: preset.label,
          }))}
          value={range}
          onChange={setRange}
        />

        <Card
          title="Profit over time"
          subtitle={`${summary.settledBets} settled bets in this window`}
        >
          <LineChart
            data={chartData}
            color={profitColor(summary.profit, theme.colors)}
            formatValue={(value) => signedMoney(value)}
            formatLabel={(point) => point.label ?? ''}
            emptyMessage="No settled bets in this period."
          />
        </Card>

        <View style={styles.grid}>
          <StatTile
            label="Profit"
            value={signedMoney(summary.profit)}
            caption={`from ${money(summary.turnover)} staked`}
            tone={summary.profit > 0 ? 'positive' : summary.profit < 0 ? 'negative' : 'default'}
            icon="cash-outline"
          />
          <StatTile
            label="ROI"
            value={formatSignedPercent(summary.roi)}
            caption="profit ÷ turnover"
            tone={summary.roi > 0 ? 'positive' : summary.roi < 0 ? 'negative' : 'default'}
            icon="analytics-outline"
          />
          <StatTile
            label="Win rate"
            value={formatPercent(summary.winRate)}
            caption={`${summary.wins}W · ${summary.losses}L · ${summary.neutrals}V`}
            icon="trophy-outline"
          />
          <StatTile
            label="Avg odds"
            value={summary.averageOdds > 0 ? summary.averageOdds.toFixed(2) : '—'}
            caption={`avg stake ${money(summary.averageStake)}`}
            icon="pricetag-outline"
          />
        </View>

        <Card
          title="Form"
          subtitle="Current run and closing line value"
          action={
            <Button
              label="Analytics"
              size="sm"
              variant="ghost"
              icon="arrow-forward"
              iconPosition="right"
              onPress={() => navigation.navigate('Analytics')}
            />
          }
        >
          <View style={styles.grid}>
            <StatTile
              label="Current streak"
              value={
                summary.currentStreak.count === 0
                  ? '—'
                  : `${summary.currentStreak.count} ${summary.currentStreak.type === 'win' ? 'W' : 'L'}`
              }
              caption={`best ${summary.longestWinStreak}W · worst ${summary.longestLossStreak}L`}
              tone={summary.currentStreak.type === 'win' ? 'positive' : summary.currentStreak.type === 'loss' ? 'negative' : 'default'}
              icon="flame-outline"
              compact
            />
            <StatTile
              label="Beat the close"
              value={clv.tracked > 0 ? formatPercent(clv.beatRate, 0) : '—'}
              caption={clv.tracked > 0 ? `${clv.tracked} bets priced` : 'add closing odds'}
              tone={clv.beatRate > 0.5 ? 'positive' : 'default'}
              icon="speedometer-outline"
              compact
            />
            <StatTile
              label="Avg CLV"
              value={clv.tracked > 0 ? formatSignedPercent(clv.averageClv, 2) : '—'}
              caption="price vs closing"
              tone={clv.averageClv > 0 ? 'positive' : clv.averageClv < 0 ? 'negative' : 'default'}
              icon="pulse-outline"
              compact
            />
          </View>
        </Card>

        {openBets.length > 0 ? (
          <Card
            title={`Open bets (${openBets.length})`}
            subtitle={`${money(bankroll.atRisk)} riding`}
            padded={false}
            action={
              <Button
                label="View all"
                size="sm"
                variant="ghost"
                onPress={() =>
                  navigation.navigate('Bets', {
                    presetStatus: 'pending',
                    requestedAt: Date.now(),
                  })
                }
              />
            }
          >
            <View style={{ padding: theme.spacing(4), gap: theme.spacing(3) }}>
              {openBets.slice(0, 3).map((bet) => (
                <BetCard
                  key={bet.id}
                  bet={bet}
                  onPress={() => navigation.navigate('BetDetail', { betId: bet.id })}
                />
              ))}
            </View>
          </Card>
        ) : null}

        {sportRows.length > 0 ? (
          <Card title="Profit by sport" subtitle="In the selected window">
            <BarChart
              data={sportRows}
              formatValue={(value) => signedMoney(value, 0)}
              emptyMessage="No settled bets in this period."
            />
          </Card>
        ) : null}

        <View style={styles.sectionHeader}>
          <Text variant="heading">Recent activity</Text>
          <Button
            label="See all"
            size="sm"
            variant="ghost"
            icon="arrow-forward"
            iconPosition="right"
            onPress={() => navigation.navigate('Bets')}
          />
        </View>

        <View style={styles.list}>
          {recent.map((bet) => (
            <BetCard
              key={bet.id}
              bet={bet}
              onPress={() => navigation.navigate('BetDetail', { betId: bet.id })}
            />
          ))}
        </View>
      </Screen>

      <Fab label="Add bet" onPress={() => navigation.navigate('BetForm')} testID="dashboard-fab" />
    </>
  );
}
