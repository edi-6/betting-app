import React, { useCallback, useMemo, useState } from 'react';
import { Alert, Pressable, StyleSheet, View } from 'react-native';

import {
  Banner,
  Button,
  Card,
  Chip,
  EmptyState,
  Icon,
  KeyValueRow,
  Screen,
  Sheet,
  StatusPill,
  Field,
  Text,
} from '../components';
import { iconForSport } from '../domain/catalog';
import { formatDateTime } from '../domain/dates';
import { formatPercent, formatSignedPercent, parseAmount } from '../domain/format';
import { impliedProbability } from '../domain/odds';
import {
  amountAtRisk,
  betType,
  combinedClosingOdds,
  combinedOdds,
  LEG_STATUS_LABELS,
  LEG_STATUSES,
  settleBet,
} from '../domain/settlement';
import type { LegStatus } from '../domain/types';
import { useFormatters } from '../hooks/useFormatters';
import { useHaptics } from '../hooks/useHaptics';
import { useApp } from '../store/AppStore';
import { useTheme } from '../theme';
import { profitColor } from '../theme/tokens';
import type { RootStackScreenProps } from '../navigation/types';

export function BetDetailScreen({ navigation, route }: RootStackScreenProps<'BetDetail'>) {
  const theme = useTheme();
  const haptics = useHaptics();
  const { bets, setLegStatuses, cashOut, clearCashOut, deleteBet } = useApp();
  const { money, signedMoney, odds: renderOdds, currency } = useFormatters();

  const bet = useMemo(() => bets.find((candidate) => candidate.id === route.params.betId), [
    bets,
    route.params.betId,
  ]);

  const [cashOutOpen, setCashOutOpen] = useState(false);
  const [cashOutText, setCashOutText] = useState('');

  const settlement = useMemo(() => (bet ? settleBet(bet) : undefined), [bet]);
  const price = useMemo(() => (bet ? combinedOdds(bet) : NaN), [bet]);
  const closing = useMemo(() => (bet ? combinedClosingOdds(bet) : undefined), [bet]);
  const clv = closing !== undefined && Number.isFinite(price) ? price / closing - 1 : undefined;

  const gradeAll = useCallback(
    (status: LegStatus) => {
      if (!bet) return;
      haptics(status === 'won' ? 'success' : status === 'lost' ? 'warning' : 'selection');
      const statuses: Record<string, LegStatus> = {};
      for (const leg of bet.legs) {
        statuses[leg.id] = status;
      }
      setLegStatuses(bet.id, statuses);
    },
    [bet, setLegStatuses, haptics],
  );

  const confirmDelete = useCallback(() => {
    if (!bet) return;
    Alert.alert('Delete this bet?', 'This cannot be undone.', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          haptics('warning');
          deleteBet(bet.id);
          navigation.goBack();
        },
      },
    ]);
  }, [bet, deleteBet, navigation, haptics]);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        hero: {
          gap: theme.spacing(2),
          backgroundColor: theme.colors.surface,
          borderRadius: theme.radius.xl,
          borderWidth: StyleSheet.hairlineWidth,
          borderColor: theme.colors.border,
          padding: theme.spacing(5),
          ...theme.shadow.card,
        },
        heroRow: {
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing(3),
        },
        actionRow: { flexDirection: 'row', flexWrap: 'wrap', gap: theme.spacing(2) },
        legRow: {
          flexDirection: 'row',
          gap: theme.spacing(3),
          paddingVertical: theme.spacing(3),
          borderBottomWidth: StyleSheet.hairlineWidth,
          borderColor: theme.colors.border,
        },
        legBadge: {
          width: 34,
          height: 34,
          borderRadius: 17,
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: theme.colors.surfaceAlt,
        },
        statusRow: { flexDirection: 'row', flexWrap: 'wrap', gap: theme.spacing(1.5), marginTop: theme.spacing(2) },
        tags: { flexDirection: 'row', flexWrap: 'wrap', gap: theme.spacing(2) },
        stars: { flexDirection: 'row', gap: 2 },
      }),
    [theme],
  );

  if (!bet || !settlement) {
    return (
      <Screen title="Bet">
        <EmptyState
          icon="alert-circle-outline"
          title="This bet is gone"
          message="It may have been deleted."
          actionLabel="Go back"
          onAction={() => navigation.goBack()}
        />
      </Screen>
    );
  }

  const isParlay = betType(bet) === 'parlay';
  const fairCashOut = Number.isFinite(price) ? bet.stake * price : 0;

  return (
    <>
      <Screen
        title={isParlay ? `${bet.legs.length}-leg parlay` : (bet.legs[0]?.selection || 'Bet')}
        subtitle={formatDateTime(bet.placedAt)}
        headerAction={
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Edit bet"
            onPress={() => navigation.navigate('BetForm', { betId: bet.id })}
            hitSlop={8}
          >
            <Icon name="create-outline" size={22} color={theme.colors.textSecondary} />
          </Pressable>
        }
      >
        <View style={styles.hero}>
          <View style={styles.heroRow}>
            <StatusPill status={settlement.status} size="md" />
            <Text variant="caption" tone="muted">
              {bet.bookmaker || 'No bookmaker'}
            </Text>
          </View>
          <View style={styles.heroRow}>
            <View>
              <Text variant="caption" tone="muted">
                {settlement.isSettled ? 'Profit / loss' : 'Potential profit'}
              </Text>
              <Text
                variant="display"
                color={
                  settlement.isSettled
                    ? profitColor(settlement.profit, theme.colors)
                    : theme.colors.text
                }
              >
                {settlement.isSettled
                  ? signedMoney(settlement.profit)
                  : money(bet.stake * (price - 1))}
              </Text>
            </View>
            <View style={{ alignItems: 'flex-end' }}>
              <Text variant="caption" tone="muted">
                Odds
              </Text>
              <Text variant="title">{renderOdds(price)}</Text>
            </View>
          </View>
        </View>

        {!settlement.isSettled ? (
          <Card title="Grade this bet" subtitle="Tap once to settle every leg">
            <View style={styles.actionRow}>
              <Button
                label="Won"
                variant="success"
                icon="checkmark"
                testID="grade-won"
                onPress={() => gradeAll('won')}
              />
              <Button
                label="Lost"
                variant="danger"
                icon="close"
                testID="grade-lost"
                onPress={() => gradeAll('lost')}
              />
              <Button
                label="Void"
                variant="secondary"
                icon="remove-circle-outline"
                testID="grade-void"
                onPress={() => gradeAll('void')}
              />
              <Button
                label="Cash out"
                variant="secondary"
                icon="cash-outline"
                onPress={() => {
                  setCashOutText('');
                  setCashOutOpen(true);
                }}
              />
            </View>
          </Card>
        ) : null}

        {settlement.status === 'cashed_out' ? (
          <Banner
            tone="info"
            title="Settled by cash out"
            message={`You took ${money(bet.cashOutReturn ?? 0)} before the result. Undo this to grade the legs normally.`}
            actionLabel="Undo cash out"
            onAction={() => {
              haptics('light');
              clearCashOut(bet.id);
            }}
          />
        ) : null}

        <Card title={isParlay ? 'Selections' : 'Selection'} padded>
          {bet.legs.map((leg, index) => (
            <View
              key={leg.id}
              style={[styles.legRow, index === bet.legs.length - 1 ? { borderBottomWidth: 0 } : null]}
            >
              <View style={styles.legBadge}>
                <Icon name={iconForSport(leg.sport)} size={17} color={theme.colors.textSecondary} />
              </View>
              <View style={{ flex: 1, gap: 2 }}>
                <View style={styles.heroRow}>
                  <Text variant="subheading" numberOfLines={1} style={{ flex: 1 }}>
                    {leg.selection || leg.event || `Leg ${index + 1}`}
                  </Text>
                  <Text variant="mono" tone="secondary">
                    {renderOdds(leg.odds)}
                  </Text>
                </View>
                <Text variant="caption" tone="muted" numberOfLines={2}>
                  {[leg.event, leg.market, leg.league].filter(Boolean).join(' · ')}
                </Text>
                {leg.closingOdds !== undefined ? (
                  <Text variant="caption" tone="muted">
                    Closed at {renderOdds(leg.closingOdds)} ·{' '}
                    {formatSignedPercent(leg.odds / leg.closingOdds - 1, 2)} CLV
                  </Text>
                ) : null}

                <View style={styles.statusRow}>
                  {LEG_STATUSES.map((status) => (
                    <Chip
                      key={status}
                      label={LEG_STATUS_LABELS[status]}
                      size="sm"
                      selected={leg.status === status}
                      tone={
                        status === 'won' || status === 'half_won'
                          ? 'positive'
                          : status === 'lost' || status === 'half_lost'
                            ? 'negative'
                            : status === 'pending'
                              ? 'warning'
                              : 'default'
                      }
                      onPress={() => {
                        haptics('selection');
                        setLegStatuses(bet.id, { [leg.id]: status });
                      }}
                    />
                  ))}
                </View>
              </View>
            </View>
          ))}
        </Card>

        <Card title="Numbers">
          <KeyValueRow label="Stake" value={money(bet.stake)} hint={bet.isFreeBet ? 'Free bet' : undefined} />
          <KeyValueRow label="At risk" value={money(amountAtRisk(bet))} />
          <KeyValueRow
            label="Implied probability"
            value={Number.isFinite(price) ? formatPercent(impliedProbability(price), 1) : '—'}
          />
          <KeyValueRow
            label={settlement.isSettled ? 'Returns' : 'Potential returns'}
            value={money(
              settlement.isSettled
                ? settlement.returns
                : bet.isFreeBet
                  ? bet.stake * (price - 1)
                  : bet.stake * price,
            )}
          />
          {closing !== undefined ? (
            <KeyValueRow
              label="Closing line value"
              value={formatSignedPercent(clv ?? 0, 2)}
              valueColor={profitColor(clv ?? 0, theme.colors)}
              hint={`Closed at ${renderOdds(closing)}`}
            />
          ) : null}
          <KeyValueRow label="Placed" value={formatDateTime(bet.placedAt)} />
          <KeyValueRow
            label="Settled"
            value={bet.settledAt ? formatDateTime(bet.settledAt) : 'Not yet'}
            divider={bet.confidence !== undefined || bet.tags.length > 0 || Boolean(bet.notes)}
          />
          {bet.confidence !== undefined ? (
            <View style={{ paddingVertical: theme.spacing(2.5), flexDirection: 'row', justifyContent: 'space-between' }}>
              <Text tone="secondary">Confidence</Text>
              <View style={styles.stars}>
                {[1, 2, 3, 4, 5].map((value) => (
                  <Icon
                    key={value}
                    name={value <= (bet.confidence ?? 0) ? 'star' : 'star-outline'}
                    size={15}
                    color={value <= (bet.confidence ?? 0) ? theme.colors.warning : theme.colors.textMuted}
                  />
                ))}
              </View>
            </View>
          ) : null}
        </Card>

        {bet.tags.length > 0 || bet.notes ? (
          <Card title="Notes & tags">
            {bet.tags.length > 0 ? (
              <View style={styles.tags}>
                {bet.tags.map((tag) => (
                  <Chip key={tag} label={tag} size="sm" />
                ))}
              </View>
            ) : null}
            {bet.notes ? (
              <Text tone="secondary" style={{ marginTop: bet.tags.length > 0 ? theme.spacing(3) : 0 }}>
                {bet.notes}
              </Text>
            ) : null}
          </Card>
        ) : null}

        <View style={styles.actionRow}>
          <Button
            label="Duplicate"
            variant="secondary"
            icon="copy-outline"
            onPress={() => navigation.navigate('BetForm', { duplicateOf: bet.id })}
          />
          <Button
            label="Edit"
            variant="secondary"
            icon="create-outline"
            onPress={() => navigation.navigate('BetForm', { betId: bet.id })}
          />
          <Button label="Delete" variant="danger" icon="trash-outline" onPress={confirmDelete} />
        </View>
      </Screen>

      <Sheet
        visible={cashOutOpen}
        onClose={() => setCashOutOpen(false)}
        title="Cash out"
        subtitle="Record what the bookmaker actually paid you"
        footer={
          <Button
            label="Save cash out"
            fullWidth
            onPress={() => {
              const amount = parseAmount(cashOutText);
              if (amount === null || amount < 0) {
                Alert.alert('Enter an amount', 'Type the cash-out amount you received.');
                return;
              }
              haptics('success');
              cashOut(bet.id, amount);
              setCashOutOpen(false);
            }}
          />
        }
      >
        <Field
          label="Amount received"
          value={cashOutText}
          onChangeText={setCashOutText}
          keyboardType="decimal-pad"
          placeholder={fairCashOut.toFixed(2)}
          prefix={currency}
          hint={`Full payout if it wins: ${money(fairCashOut)}. Your stake was ${money(bet.stake)}.`}
        />
      </Sheet>
    </>
  );
}
