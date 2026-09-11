import React, { useMemo, useState } from 'react';
import { StyleSheet, View } from 'react-native';

import {
  BarChart,
  Banner,
  Card,
  DonutChart,
  EmptyState,
  KeyValueRow,
  LineChart,
  Screen,
  SegmentedControl,
  Select,
  StatTile,
  Text,
  type BarDatum,
  type LinePoint,
} from '../components';
import {
  breakdown,
  byBetType,
  byBookmaker,
  byConfidence,
  byLeague,
  byLegCount,
  byMarket,
  byMonth,
  byOddsRange,
  bySport,
  byTag,
  byWeekday,
  calibration,
  clvStats,
  profitSeries,
  riskStats,
  summarize,
  type BucketSelector,
} from '../domain/analytics';
import { formatShortDate, monthKeyLabel } from '../domain/dates';
import { formatNumber, formatPercent, formatSignedPercent } from '../domain/format';
import {
  applyFilter,
  DATE_RANGE_PRESETS,
  EMPTY_FILTER,
  type DateRangePreset,
} from '../domain/filters';
import { combinedClosingOdds, combinedOdds } from '../domain/settlement';
import { useFormatters } from '../hooks/useFormatters';
import { useApp } from '../store/AppStore';
import { useTheme } from '../theme';
import { profitColor } from '../theme/tokens';
import type { TabScreenProps } from '../navigation/types';

type AnalyticsTab = 'overview' | 'breakdown' | 'clv' | 'risk';

const DIMENSIONS: { value: string; label: string; selector: BucketSelector }[] = [
  { value: 'sport', label: 'Sport', selector: bySport },
  { value: 'league', label: 'League', selector: byLeague },
  { value: 'market', label: 'Market', selector: byMarket },
  { value: 'bookmaker', label: 'Bookmaker', selector: byBookmaker },
  { value: 'tag', label: 'Tag', selector: byTag },
  { value: 'oddsRange', label: 'Odds range', selector: byOddsRange },
  { value: 'betType', label: 'Bet type', selector: byBetType },
  { value: 'legCount', label: 'Legs', selector: byLegCount },
  { value: 'weekday', label: 'Day of week', selector: byWeekday },
  { value: 'confidence', label: 'Confidence', selector: byConfidence },
];

export function AnalyticsScreen({ navigation }: TabScreenProps<'Analytics'>) {
  const theme = useTheme();
  const { bets } = useApp();
  const { money, signedMoney } = useFormatters();

  const [tab, setTab] = useState<AnalyticsTab>('overview');
  const [range, setRange] = useState<DateRangePreset>('90d');
  const [dimension, setDimension] = useState('sport');
  const [minSample, setMinSample] = useState('1');

  const scoped = useMemo(() => applyFilter(bets, { ...EMPTY_FILTER, range }), [bets, range]);
  const summary = useMemo(() => summarize(scoped), [scoped]);
  const series = useMemo(() => profitSeries(scoped), [scoped]);
  const risk = useMemo(() => riskStats(scoped), [scoped]);
  const clv = useMemo(() => clvStats(scoped), [scoped]);

  const chartData = useMemo<LinePoint[]>(
    () =>
      series.map((point) => ({
        x: point.timestamp,
        y: point.cumulative,
        label: formatShortDate(point.timestamp),
      })),
    [series],
  );

  const monthlyRows = useMemo<BarDatum[]>(
    () =>
      breakdown(scoped, byMonth)
        .slice()
        .sort((a, b) => a.key.localeCompare(b.key))
        .map((row) => ({
          key: row.key,
          label: monthKeyLabel(row.key),
          value: row.profit,
          caption: `${row.bets} bets · ${formatSignedPercent(row.roi)}`,
        })),
    [scoped],
  );

  const selectedDimension = DIMENSIONS.find((item) => item.value === dimension) ?? DIMENSIONS[0]!;
  const breakdownRows = useMemo(
    () =>
      breakdown(scoped, selectedDimension.selector, {
        minBets: Math.max(1, Number(minSample) || 1),
      }),
    [scoped, selectedDimension, minSample],
  );

  const calibrationRows = useMemo(() => calibration(scoped), [scoped]);

  const clvLeaders = useMemo(() => {
    const rated = scoped
      .map((bet) => {
        const closing = combinedClosingOdds(bet);
        const price = combinedOdds(bet);
        if (closing === undefined || !Number.isFinite(price)) return null;
        return { bet, clv: price / closing - 1 };
      })
      .filter((entry): entry is { bet: (typeof scoped)[number]; clv: number } => entry !== null)
      .sort((a, b) => b.clv - a.clv);
    return { best: rated.slice(0, 3), worst: rated.slice(-3).reverse() };
  }, [scoped]);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        grid: { flexDirection: 'row', flexWrap: 'wrap', gap: theme.spacing(3) },
        donutRow: {
          flexDirection: 'row',
          alignItems: 'center',
          gap: theme.spacing(5),
        },
        legend: { flex: 1, gap: theme.spacing(2) },
        legendRow: { flexDirection: 'row', alignItems: 'center', gap: theme.spacing(2) },
        swatch: { width: 10, height: 10, borderRadius: 5 },
        calibrationRow: {
          flexDirection: 'row',
          alignItems: 'center',
          gap: theme.spacing(3),
          paddingVertical: theme.spacing(2),
        },
        calibrationBar: { flex: 1, gap: 4 },
        bar: { height: 6, borderRadius: 3 },
      }),
    [theme],
  );

  if (bets.length === 0) {
    return (
      <Screen title="Analytics">
        <EmptyState
          icon="bar-chart-outline"
          title="Nothing to analyse yet"
          message="Once you've logged a few bets, this is where you'll find your real edge — by sport, market, price and closing line value."
          actionLabel="Add a bet"
          onAction={() => navigation.navigate('BetForm')}
        />
      </Screen>
    );
  }

  return (
    <Screen title="Analytics" subtitle={`${summary.settledBets} settled bets in view`}>
      <SegmentedControl
        segments={[
          { value: 'overview', label: 'Overview' },
          { value: 'breakdown', label: 'Breakdown' },
          { value: 'clv', label: 'CLV' },
          { value: 'risk', label: 'Risk' },
        ]}
        value={tab}
        onChange={setTab}
      />

      <SegmentedControl
        segments={DATE_RANGE_PRESETS.map((preset) => ({ value: preset.key, label: preset.label }))}
        value={range}
        onChange={setRange}
      />

      {tab === 'overview' ? (
        <>
          <Card title="Cumulative profit" subtitle="Every settled bet, day by day">
            <LineChart
              data={chartData}
              color={profitColor(summary.profit, theme.colors)}
              formatValue={(value) => signedMoney(value)}
              formatLabel={(point) => point.label ?? ''}
            />
          </Card>

          <View style={styles.grid}>
            <StatTile
              label="Profit"
              value={signedMoney(summary.profit)}
              tone={summary.profit > 0 ? 'positive' : summary.profit < 0 ? 'negative' : 'default'}
              icon="cash-outline"
            />
            <StatTile
              label="ROI"
              value={formatSignedPercent(summary.roi)}
              tone={summary.roi > 0 ? 'positive' : summary.roi < 0 ? 'negative' : 'default'}
              icon="analytics-outline"
            />
            <StatTile
              label="Turnover"
              value={money(summary.turnover, 0)}
              caption={`${summary.settledBets} settled`}
              icon="repeat-outline"
            />
            <StatTile
              label="Win rate"
              value={formatPercent(summary.winRate)}
              caption={`${summary.wins}W · ${summary.losses}L`}
              icon="trophy-outline"
            />
            <StatTile
              label="Avg odds"
              value={summary.averageOdds > 0 ? summary.averageOdds.toFixed(2) : '—'}
              caption="stake-weighted"
              icon="pricetag-outline"
            />
            <StatTile
              label="Avg stake"
              value={money(summary.averageStake)}
              caption={`biggest win ${money(summary.biggestWin, 0)}`}
              icon="wallet-outline"
            />
          </View>

          <Card title="Results split" subtitle="Wins, losses and voids">
            <View style={styles.donutRow}>
              <DonutChart
                size={132}
                thickness={15}
                centerValue={formatPercent(summary.winRate, 0)}
                centerLabel="win rate"
                slices={[
                  { key: 'won', label: 'Won', value: summary.wins, color: theme.colors.positive },
                  { key: 'lost', label: 'Lost', value: summary.losses, color: theme.colors.negative },
                  {
                    key: 'void',
                    label: 'Void / push',
                    value: summary.neutrals,
                    color: theme.colors.neutral,
                  },
                ]}
              />
              <View style={styles.legend}>
                {[
                  { label: 'Won', value: summary.wins, color: theme.colors.positive },
                  { label: 'Lost', value: summary.losses, color: theme.colors.negative },
                  { label: 'Void / push', value: summary.neutrals, color: theme.colors.neutral },
                  { label: 'Pending', value: summary.pendingBets, color: theme.colors.warning },
                ].map((row) => (
                  <View key={row.label} style={styles.legendRow}>
                    <View style={[styles.swatch, { backgroundColor: row.color }]} />
                    <Text variant="caption" tone="secondary" style={{ flex: 1 }}>
                      {row.label}
                    </Text>
                    <Text variant="label">{row.value}</Text>
                  </View>
                ))}
              </View>
            </View>
          </Card>

          <Card title="Monthly profit" subtitle="Are you trending the right way?">
            <BarChart data={monthlyRows} formatValue={(value) => signedMoney(value, 0)} />
          </Card>

          <Card
            title="Calibration"
            subtitle="Do the prices you take actually win as often as they imply?"
          >
            {calibrationRows.map((row) => (
              <View key={row.label} style={styles.calibrationRow}>
                <Text variant="caption" tone="muted" style={{ width: 62 }}>
                  {row.label}
                </Text>
                <View style={styles.calibrationBar}>
                  <View
                    style={[
                      styles.bar,
                      {
                        width: `${Math.min(row.expected, 1) * 100}%`,
                        backgroundColor: theme.colors.neutral,
                      },
                    ]}
                  />
                  <View
                    style={[
                      styles.bar,
                      {
                        width: `${Math.min(row.actual, 1) * 100}%`,
                        backgroundColor:
                          row.bets === 0
                            ? theme.colors.neutralSoft
                            : row.actual >= row.expected
                              ? theme.colors.positive
                              : theme.colors.negative,
                      },
                    ]}
                  />
                </View>
                <Text variant="caption" tone="muted" style={{ width: 58, textAlign: 'right' }}>
                  {row.bets > 0 ? `${formatPercent(row.actual, 0)}` : '—'}
                </Text>
              </View>
            ))}
            <Text variant="caption" tone="muted" style={{ marginTop: theme.spacing(2) }}>
              Grey is the implied (expected) win rate; the coloured bar is what actually happened.
            </Text>
          </Card>
        </>
      ) : null}

      {tab === 'breakdown' ? (
        <>
          <Select
            label="Group by"
            value={dimension}
            options={DIMENSIONS.map((item) => ({ value: item.value, label: item.label }))}
            onChange={setDimension}
            title="Group bets by"
          />

          <SegmentedControl
            segments={[
              { value: '1', label: 'All' },
              { value: '5', label: '5+ bets' },
              { value: '10', label: '10+ bets' },
              { value: '25', label: '25+ bets' },
            ]}
            value={minSample}
            onChange={setMinSample}
          />

          <Card
            title={`Profit by ${selectedDimension.label.toLowerCase()}`}
            subtitle={`${breakdownRows.length} groups`}
          >
            <BarChart
              data={breakdownRows.map((row) => ({
                key: row.key,
                label: row.label,
                value: row.profit,
                caption: `${row.bets} · ${formatSignedPercent(row.roi)}`,
              }))}
              formatValue={(value) => signedMoney(value, 0)}
              emptyMessage="No group meets that minimum sample size."
            />
          </Card>

          {breakdownRows.length > 0 ? (
            <Card title="Detail" subtitle="Sorted by profit">
              {breakdownRows.map((row) => (
                <KeyValueRow
                  key={row.key}
                  label={row.label}
                  hint={`${row.bets} bets · ${money(row.staked, 0)} staked · ${formatPercent(row.winRate, 0)} win rate · avg ${row.averageOdds.toFixed(2)}`}
                  value={signedMoney(row.profit, 0)}
                  valueColor={profitColor(row.profit, theme.colors)}
                />
              ))}
            </Card>
          ) : null}
        </>
      ) : null}

      {tab === 'clv' ? (
        <>
          <Banner
            tone="info"
            title="Why closing line value matters"
            message="Beating the closing price is the single best predictor of long-run profit. Record the closing odds on your bets and this page tells you whether your edge is real or just variance."
          />

          {clv.tracked === 0 ? (
            <EmptyState
              icon="speedometer-outline"
              title="No closing odds recorded"
              message="Add a closing price when you settle a bet and your CLV will appear here."
            />
          ) : (
            <>
              <View style={styles.grid}>
                <StatTile
                  label="Bets priced"
                  value={String(clv.tracked)}
                  caption="with closing odds"
                  icon="pricetags-outline"
                />
                <StatTile
                  label="Beat the close"
                  value={formatPercent(clv.beatRate, 0)}
                  caption={`${clv.beat} of ${clv.tracked}`}
                  tone={clv.beatRate > 0.5 ? 'positive' : 'negative'}
                  icon="trending-up-outline"
                />
                <StatTile
                  label="Average CLV"
                  value={formatSignedPercent(clv.averageClv, 2)}
                  caption="stake-weighted"
                  tone={clv.averageClv > 0 ? 'positive' : 'negative'}
                  icon="pulse-outline"
                />
                <StatTile
                  label="Median CLV"
                  value={formatSignedPercent(clv.medianClv, 2)}
                  caption="typical bet"
                  tone={clv.medianClv > 0 ? 'positive' : 'negative'}
                  icon="git-commit-outline"
                />
              </View>

              <Card title="Best prices you took" subtitle="Biggest gap to the close">
                {clvLeaders.best.map(({ bet, clv: value }) => (
                  <KeyValueRow
                    key={bet.id}
                    label={bet.legs[0]?.selection || bet.legs[0]?.event || 'Bet'}
                    hint={`${bet.bookmaker} · ${formatShortDate(bet.placedAt)}`}
                    value={formatSignedPercent(value, 2)}
                    valueColor={theme.colors.positive}
                  />
                ))}
              </Card>

              <Card title="Worst prices you took" subtitle="Where you were behind the market">
                {clvLeaders.worst.map(({ bet, clv: value }) => (
                  <KeyValueRow
                    key={bet.id}
                    label={bet.legs[0]?.selection || bet.legs[0]?.event || 'Bet'}
                    hint={`${bet.bookmaker} · ${formatShortDate(bet.placedAt)}`}
                    value={formatSignedPercent(value, 2)}
                    valueColor={profitColor(value, theme.colors)}
                  />
                ))}
              </Card>

              <Card title="Books where you get the best price">
                <BarChart
                  data={breakdown(scoped, byBookmaker).map((row) => ({
                    key: row.key,
                    label: row.label,
                    value: row.profit,
                    caption: `${row.bets} bets`,
                  }))}
                  formatValue={(value) => signedMoney(value, 0)}
                />
              </Card>
            </>
          )}
        </>
      ) : null}

      {tab === 'risk' ? (
        <>
          <View style={styles.grid}>
            <StatTile
              label="Max drawdown"
              value={money(risk.maxDrawdown, 0)}
              caption={
                risk.maxDrawdownPercent > 0
                  ? `${formatPercent(risk.maxDrawdownPercent, 0)} off peak`
                  : 'never below the start'
              }
              tone={risk.maxDrawdown > 0 ? 'negative' : 'default'}
              icon="trending-down-outline"
            />
            <StatTile
              label="Longest slump"
              value={`${risk.longestDrawdownDays}d`}
              caption="below the previous peak"
              icon="time-outline"
            />
            <StatTile
              label="Volatility"
              value={formatNumber(risk.volatility, 2)}
              caption="std dev per unit staked"
              icon="swap-vertical-outline"
            />
            <StatTile
              label="Daily swing"
              value={money(risk.averageDailySwing, 0)}
              caption="typical day up or down"
              icon="calendar-outline"
            />
          </View>

          <Card
            title="Is your edge real?"
            subtitle="A statistical read on whether these results could be luck"
          >
            <KeyValueRow
              label="ROI"
              value={formatSignedPercent(summary.roi)}
              valueColor={profitColor(summary.roi, theme.colors)}
            />
            <KeyValueRow
              label="95% confidence interval"
              value={`${formatSignedPercent(risk.roiConfidenceInterval[0])} … ${formatSignedPercent(risk.roiConfidenceInterval[1])}`}
              hint="Where your true ROI most likely sits"
            />
            <KeyValueRow
              label="t-statistic"
              value={formatNumber(risk.tStatistic, 2)}
              hint="Above 2 is the usual bar for significance"
            />
            <KeyValueRow
              label="p-value"
              value={risk.pValue < 0.001 ? '< 0.001' : formatNumber(risk.pValue, 3)}
              valueColor={risk.pValue < 0.05 ? theme.colors.positive : theme.colors.textMuted}
              divider={false}
            />
            <Text variant="caption" tone="muted" style={{ marginTop: theme.spacing(3) }}>
              {summary.settledBets < 100
                ? `With ${summary.settledBets} settled bets this is still a small sample — treat any ROI as provisional until you pass a few hundred.`
                : risk.pValue < 0.05
                  ? 'Your results are unlikely to be chance alone at the 5% level. Keep tracking CLV to confirm.'
                  : 'These results are still consistent with random variation. Closing line value is a faster signal than profit.'}
            </Text>
          </Card>

          <Card title="Streaks" subtitle="Runs of wins and losses">
            <KeyValueRow
              label="Current streak"
              value={
                summary.currentStreak.count === 0
                  ? '—'
                  : `${summary.currentStreak.count} ${summary.currentStreak.type === 'win' ? 'wins' : 'losses'}`
              }
              valueColor={
                summary.currentStreak.type === 'win'
                  ? theme.colors.positive
                  : summary.currentStreak.type === 'loss'
                    ? theme.colors.negative
                    : undefined
              }
            />
            <KeyValueRow label="Longest winning run" value={`${summary.longestWinStreak}`} />
            <KeyValueRow
              label="Longest losing run"
              value={`${summary.longestLossStreak}`}
              divider={false}
            />
          </Card>

          <Card title="Stake discipline" subtitle="Profit by stake size relative to your average">
            <BarChart
              data={breakdown(scoped, byOddsRange).map((row) => ({
                key: row.key,
                label: row.label,
                value: row.profit,
                caption: `${row.bets} bets · ${formatSignedPercent(row.roi)}`,
              }))}
              formatValue={(value) => signedMoney(value, 0)}
            />
          </Card>
        </>
      ) : null}
    </Screen>
  );
}
