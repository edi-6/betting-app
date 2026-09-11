import React, { useCallback, useMemo, useState } from 'react';
import { Alert, Pressable, StyleSheet, Switch, View } from 'react-native';

import {
  Banner,
  Button,
  Card,
  Chip,
  DatePicker,
  Field,
  Icon,
  KeyValueRow,
  Screen,
  SegmentedControl,
  Select,
  TagInput,
  Text,
} from '../components';
import { computeBankroll, warningsForStake } from '../domain/bankroll';
import { BOOKMAKERS, leaguesForSport, marketsForSport, SPORT_NAMES } from '../domain/catalog';
import { expectedValue, kellyStake } from '../domain/calculators';
import { facetsFor } from '../domain/filters';
import { formatPercent, formatSignedPercent, parseAmount } from '../domain/format';
import { createId } from '../domain/ids';
import { impliedProbability, isValidDecimalOdds } from '../domain/odds';
import { LEG_STATUS_LABELS, LEG_STATUSES, nominalOdds } from '../domain/settlement';
import type { Bet, Leg, LegStatus } from '../domain/types';
import { hasErrors, validateBet } from '../domain/validation';
import { useFormatters } from '../hooks/useFormatters';
import { useHaptics } from '../hooks/useHaptics';
import { useApp } from '../store/AppStore';
import { useTheme } from '../theme';
import { profitColor } from '../theme/tokens';
import type { RootStackScreenProps } from '../navigation/types';

interface LegDraft extends Omit<Leg, 'odds' | 'closingOdds'> {
  oddsText: string;
  closingOddsText: string;
}

function toLegDraft(leg: Leg, format: (value: number) => string): LegDraft {
  const { odds, closingOdds, ...rest } = leg;
  return {
    ...rest,
    oddsText: format(odds),
    closingOddsText: closingOdds === undefined ? '' : format(closingOdds),
  };
}

function emptyLegDraft(sport = 'Football'): LegDraft {
  return {
    id: createId('leg'),
    sport,
    league: '',
    event: '',
    market: '',
    selection: '',
    status: 'pending',
    oddsText: '',
    closingOddsText: '',
  };
}

export function BetFormScreen({ navigation, route }: RootStackScreenProps<'BetForm'>) {
  const theme = useTheme();
  const haptics = useHaptics();
  const { bets, transactions, settings, addBet, updateBet, deleteBet } = useApp();
  const { money, signedMoney, odds: renderOdds, readOdds, oddsHint } = useFormatters();

  const betId = route.params?.betId;
  const duplicateOf = route.params?.duplicateOf;
  const source = useMemo(
    () => bets.find((bet) => bet.id === (betId ?? duplicateOf)),
    [bets, betId, duplicateOf],
  );
  const isEditing = betId !== undefined && source !== undefined;

  const oddsAsText = useCallback(
    (value: number) => (isValidDecimalOdds(value) ? renderOdds(value).replace('+', '') : ''),
    [renderOdds],
  );

  const [legs, setLegs] = useState<LegDraft[]>(() =>
    source ? source.legs.map((leg) => toLegDraft(leg, oddsAsText)) : [emptyLegDraft()],
  );
  const [stakeText, setStakeText] = useState(() =>
    source ? String(source.stake) : settings.defaultStake > 0 ? String(settings.defaultStake) : '',
  );
  const [bookmaker, setBookmaker] = useState(
    () => source?.bookmaker ?? settings.defaultBookmaker ?? '',
  );
  const [placedAt, setPlacedAt] = useState(
    () => (duplicateOf ? new Date().toISOString() : (source?.placedAt ?? new Date().toISOString())),
  );
  const [tags, setTags] = useState<string[]>(() => source?.tags ?? []);
  const [notes, setNotes] = useState(() => source?.notes ?? '');
  const [isFreeBet, setIsFreeBet] = useState(() => source?.isFreeBet ?? false);
  const [confidence, setConfidence] = useState<number | undefined>(() =>
    duplicateOf ? undefined : source?.confidence,
  );
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const facets = useMemo(() => facetsFor(bets), [bets]);
  const bankroll = useMemo(
    () => computeBankroll(bets, transactions, settings.startingBankroll),
    [bets, transactions, settings.startingBankroll],
  );

  const stake = useMemo(() => parseAmount(stakeText) ?? 0, [stakeText]);
  const parsedLegs = useMemo(
    () =>
      legs.map((leg) => {
        const { oddsText, closingOddsText, ...rest } = leg;
        return {
          ...rest,
          odds: readOdds(oddsText) ?? NaN,
          closingOdds: readOdds(closingOddsText) ?? undefined,
        } as Leg;
      }),
    [legs, readOdds],
  );

  const draftBet = useMemo<Bet>(() => {
    const nowIso = new Date().toISOString();
    return {
      id: source && isEditing ? source.id : createId('bet'),
      placedAt,
      settledAt: isEditing ? source?.settledAt : undefined,
      createdAt: isEditing ? (source?.createdAt ?? nowIso) : nowIso,
      updatedAt: nowIso,
      stake,
      legs: parsedLegs,
      bookmaker,
      tags,
      notes: notes.trim().length > 0 ? notes.trim() : undefined,
      isFreeBet,
      cashOutReturn: isEditing ? source?.cashOutReturn : undefined,
      confidence,
    };
  }, [
    source,
    isEditing,
    placedAt,
    stake,
    parsedLegs,
    bookmaker,
    tags,
    notes,
    isFreeBet,
    confidence,
  ]);

  const errors = useMemo(() => validateBet(draftBet), [draftBet]);
  const validLegOdds = parsedLegs.filter((leg) => isValidDecimalOdds(leg.odds));
  const combined = validLegOdds.length === parsedLegs.length ? nominalOdds(parsedLegs) : NaN;
  const potentialProfit = Number.isFinite(combined) ? stake * (combined - 1) : NaN;
  const potentialReturn = Number.isFinite(combined)
    ? isFreeBet
      ? stake * (combined - 1)
      : stake * combined
    : NaN;

  const closingCombined = useMemo(() => {
    if (parsedLegs.length === 0 || parsedLegs.some((leg) => leg.closingOdds === undefined)) {
      return undefined;
    }
    return parsedLegs.reduce((product, leg) => product * (leg.closingOdds as number), 1);
  }, [parsedLegs]);

  const closingEdge = useMemo(() => {
    if (closingCombined === undefined || !Number.isFinite(combined)) return undefined;
    return expectedValue(combined, impliedProbability(closingCombined), stake);
  }, [closingCombined, combined, stake]);

  const kelly = useMemo(() => {
    if (closingCombined === undefined || !Number.isFinite(combined)) return undefined;
    return kellyStake(
      bankroll.balance,
      combined,
      impliedProbability(closingCombined),
      settings.kellyFraction,
    );
  }, [closingCombined, combined, bankroll.balance, settings.kellyFraction]);

  const stakeWarnings = useMemo(
    () =>
      warningsForStake(
        stake,
        bets.filter((bet) => bet.id !== betId),
        settings.limits,
        bankroll.available + (isEditing ? (source?.stake ?? 0) : 0),
      ),
    [stake, bets, betId, settings.limits, bankroll.available, isEditing, source?.stake],
  );

  const updateLeg = useCallback((id: string, patch: Partial<LegDraft>) => {
    setLegs((current) => current.map((leg) => (leg.id === id ? { ...leg, ...patch } : leg)));
  }, []);

  const addLeg = useCallback(() => {
    haptics('selection');
    setLegs((current) => [...current, emptyLegDraft(current[current.length - 1]?.sport)]);
  }, [haptics]);

  const removeLeg = useCallback(
    (id: string) => {
      haptics('light');
      setLegs((current) => (current.length <= 1 ? current : current.filter((leg) => leg.id !== id)));
    },
    [haptics],
  );

  const save = useCallback(() => {
    setSubmitted(true);
    if (hasErrors(errors)) {
      haptics('error');
      return;
    }
    if (stakeWarnings.some((warning) => warning.severity === 'block')) {
      haptics('error');
      Alert.alert(
        'Stake too large',
        stakeWarnings.find((warning) => warning.severity === 'block')?.message ??
          'This stake exceeds your available bankroll.',
      );
      return;
    }

    haptics('success');
    if (isEditing) {
      updateBet(draftBet);
    } else {
      addBet(draftBet);
    }
    navigation.goBack();
  }, [errors, stakeWarnings, isEditing, updateBet, addBet, draftBet, navigation, haptics]);

  const confirmDelete = useCallback(() => {
    if (!betId) return;
    Alert.alert('Delete this bet?', 'This cannot be undone.', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          haptics('warning');
          deleteBet(betId);
          navigation.popTo('Tabs');
        },
      },
    ]);
  }, [betId, deleteBet, navigation, haptics]);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        row: { flexDirection: 'row', gap: theme.spacing(3) },
        legHeader: {
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: theme.spacing(3),
        },
        toggleRow: {
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing(3),
          paddingVertical: theme.spacing(1),
        },
        stars: { flexDirection: 'row', gap: theme.spacing(2) },
        statusRow: { flexDirection: 'row', flexWrap: 'wrap', gap: theme.spacing(2) },
        fieldGroup: { gap: theme.spacing(3.5) },
        advancedToggle: {
          flexDirection: 'row',
          alignItems: 'center',
          gap: theme.spacing(2),
          paddingVertical: theme.spacing(2),
        },
        footer: { gap: theme.spacing(3), marginTop: theme.spacing(2) },
      }),
    [theme],
  );

  const showErrors = submitted;

  return (
    <Screen
      title={isEditing ? 'Edit bet' : 'New bet'}
      subtitle={isEditing ? 'Update the details or grade the result' : 'Log a slip in a few taps'}
      headerAction={
        <Button label="Cancel" variant="ghost" size="sm" onPress={() => navigation.goBack()} />
      }
    >
      <SegmentedControl
        segments={[
          { value: 'single', label: 'Single' },
          { value: 'parlay', label: 'Parlay' },
        ]}
        value={legs.length > 1 ? 'parlay' : 'single'}
        onChange={(value) => {
          if (value === 'parlay' && legs.length === 1) {
            addLeg();
          } else if (value === 'single' && legs.length > 1) {
            setLegs((current) => current.slice(0, 1));
          }
        }}
      />

      {legs.map((leg, index) => (
        <Card key={leg.id}>
          {legs.length > 1 ? (
            <View style={styles.legHeader}>
              <Text variant="subheading">Leg {index + 1}</Text>
              <Pressable
                onPress={() => removeLeg(leg.id)}
                accessibilityRole="button"
                accessibilityLabel={`Remove leg ${index + 1}`}
                hitSlop={8}
              >
                <Icon name="trash-outline" size={18} color={theme.colors.negative} />
              </Pressable>
            </View>
          ) : null}

          <View style={styles.fieldGroup}>
            <View style={styles.row}>
              <View style={{ flex: 1 }}>
                <Select
                  label="Sport"
                  value={leg.sport}
                  options={Array.from(new Set([...SPORT_NAMES, ...facets.sports]))}
                  onChange={(sport) => updateLeg(leg.id, { sport, league: '', market: '' })}
                  searchable
                  allowCustom
                />
              </View>
              <View style={{ flex: 1 }}>
                <Select
                  label="League"
                  value={leg.league}
                  options={Array.from(
                    new Set([...leaguesForSport(leg.sport), ...facets.leagues]),
                  )}
                  onChange={(league) => updateLeg(leg.id, { league })}
                  placeholder="Optional"
                  searchable
                  allowCustom
                />
              </View>
            </View>

            <Field
              label="Event"
              value={leg.event}
              onChangeText={(event) => updateLeg(leg.id, { event })}
              placeholder="Arsenal vs Liverpool"
              error={showErrors ? errors.legErrors[leg.id] : undefined}
            />

            <View style={styles.row}>
              <View style={{ flex: 1 }}>
                <Select
                  label="Market"
                  value={leg.market}
                  options={marketsForSport(leg.sport)}
                  onChange={(market) => updateLeg(leg.id, { market })}
                  placeholder="Moneyline"
                  searchable
                  allowCustom
                />
              </View>
              <View style={{ flex: 1 }}>
                <Field
                  label="Selection"
                  value={leg.selection}
                  onChangeText={(selection) => updateLeg(leg.id, { selection })}
                  placeholder="Arsenal"
                />
              </View>
            </View>

            <View style={styles.row}>
              <View style={{ flex: 1 }}>
                <Field
                  label="Odds"
                  value={leg.oddsText}
                  onChangeText={(oddsText) => updateLeg(leg.id, { oddsText })}
                  placeholder={oddsHint}
                  keyboardType="numbers-and-punctuation"
                  hint={
                    isValidDecimalOdds(readOdds(leg.oddsText) ?? NaN)
                      ? `${formatPercent(impliedProbability(readOdds(leg.oddsText) as number), 1)} implied`
                      : undefined
                  }
                  error={showErrors ? errors.oddsErrors[leg.id] : undefined}
                />
              </View>
              <View style={{ flex: 1 }}>
                <Field
                  label="Closing odds"
                  value={leg.closingOddsText}
                  onChangeText={(closingOddsText) => updateLeg(leg.id, { closingOddsText })}
                  placeholder="Optional"
                  keyboardType="numbers-and-punctuation"
                  hint="Powers CLV"
                />
              </View>
            </View>

            <View style={{ gap: theme.spacing(2) }}>
              <Text variant="label" tone="secondary">
                Result
              </Text>
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
                    onPress={() => updateLeg(leg.id, { status: status as LegStatus })}
                  />
                ))}
              </View>
            </View>
          </View>
        </Card>
      ))}

      {legs.length > 1 || legs.length === 1 ? (
        <Button
          label="Add another leg"
          variant="secondary"
          icon="add"
          fullWidth
          onPress={addLeg}
        />
      ) : null}

      <Card title="Stake & book">
        <View style={styles.fieldGroup}>
          <View style={styles.row}>
            <View style={{ flex: 1 }}>
              <Field
                label="Stake"
                value={stakeText}
                onChangeText={setStakeText}
                placeholder="25"
                keyboardType="decimal-pad"
                prefix={settings.currency}
                error={showErrors ? errors.stake : undefined}
              />
            </View>
            <View style={{ flex: 1 }}>
              <Select
                label="Bookmaker"
                value={bookmaker}
                options={Array.from(new Set([...facets.bookmakers, ...BOOKMAKERS]))}
                onChange={setBookmaker}
                placeholder="Where?"
                searchable
                allowCustom
              />
            </View>
          </View>

          <DatePicker label="Placed at" value={placedAt} onChange={setPlacedAt} />

          <View style={styles.toggleRow}>
            <View style={{ flex: 1 }}>
              <Text variant="label" tone="secondary">
                Free bet / bonus
              </Text>
              <Text variant="caption" tone="muted">
                Stake is not returned on a win and is not lost on a loss
              </Text>
            </View>
            <Switch
              value={isFreeBet}
              onValueChange={setIsFreeBet}
              trackColor={{ false: theme.colors.surfaceSunken, true: theme.colors.primarySoft }}
              thumbColor={isFreeBet ? theme.colors.primary : theme.colors.neutral}
            />
          </View>
        </View>
      </Card>

      <Card title="Payout preview">
        <KeyValueRow
          label="Combined odds"
          value={Number.isFinite(combined) ? renderOdds(combined) : '—'}
        />
        <KeyValueRow
          label="Implied probability"
          value={Number.isFinite(combined) ? formatPercent(impliedProbability(combined), 1) : '—'}
        />
        <KeyValueRow
          label={isFreeBet ? 'Returns (winnings only)' : 'Returns'}
          value={Number.isFinite(potentialReturn) ? money(potentialReturn) : '—'}
        />
        <KeyValueRow
          label="Profit if it wins"
          value={Number.isFinite(potentialProfit) ? signedMoney(potentialProfit) : '—'}
          valueColor={theme.colors.positive}
          divider={closingEdge !== undefined}
        />
        {closingEdge !== undefined ? (
          <>
            <KeyValueRow
              label="Edge vs closing line"
              value={formatSignedPercent(closingEdge.edge, 2)}
              valueColor={profitColor(closingEdge.edge, theme.colors)}
              hint="Your price against the market's closing probability"
            />
            <KeyValueRow
              label="Expected value"
              value={signedMoney(closingEdge.expectedValue)}
              valueColor={profitColor(closingEdge.expectedValue, theme.colors)}
              divider={kelly !== undefined}
            />
          </>
        ) : null}
        {kelly && !kelly.noEdge ? (
          <KeyValueRow
            label={`${Math.round(settings.kellyFraction * 100)}% Kelly stake`}
            value={money(kelly.stake)}
            hint="Based on your current bankroll"
            divider={false}
          />
        ) : null}
      </Card>

      {stakeWarnings.map((warning, index) => (
        <Banner
          key={`${warning.severity}-${index}`}
          tone={warning.severity === 'block' ? 'negative' : 'warning'}
          message={warning.message}
        />
      ))}

      <Pressable
        style={styles.advancedToggle}
        accessibilityRole="button"
        onPress={() => setShowAdvanced((current) => !current)}
      >
        <Icon
          name={showAdvanced ? 'chevron-down' : 'chevron-forward'}
          size={16}
          color={theme.colors.textSecondary}
        />
        <Text variant="label" tone="secondary">
          Tags, confidence & notes
        </Text>
      </Pressable>

      {showAdvanced ? (
        <Card>
          <View style={styles.fieldGroup}>
            <TagInput label="Tags" value={tags} onChange={setTags} suggestions={facets.tags} />

            <View style={{ gap: theme.spacing(2) }}>
              <Text variant="label" tone="secondary">
                Confidence
              </Text>
              <View style={styles.stars}>
                {[1, 2, 3, 4, 5].map((value) => (
                  <Pressable
                    key={value}
                    accessibilityRole="button"
                    accessibilityLabel={`${value} star confidence`}
                    onPress={() => setConfidence(confidence === value ? undefined : value)}
                    hitSlop={6}
                  >
                    <Icon
                      name={confidence !== undefined && value <= confidence ? 'star' : 'star-outline'}
                      size={26}
                      color={
                        confidence !== undefined && value <= confidence
                          ? theme.colors.warning
                          : theme.colors.textMuted
                      }
                    />
                  </Pressable>
                ))}
              </View>
            </View>

            <Field
              label="Notes"
              value={notes}
              onChangeText={setNotes}
              placeholder="Why did you like this bet?"
              multiline
              numberOfLines={3}
            />
          </View>
        </Card>
      ) : null}

      <View style={styles.footer}>
        <Button
          label={isEditing ? 'Save changes' : 'Add bet'}
          size="lg"
          fullWidth
          onPress={save}
          testID="save-bet"
        />
        {isEditing ? (
          <Button label="Delete bet" variant="danger" fullWidth onPress={confirmDelete} />
        ) : null}
      </View>
    </Screen>
  );
}
