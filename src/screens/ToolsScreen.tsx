import React, { useMemo, useState } from 'react';
import { StyleSheet, View } from 'react-native';

import {
  Banner,
  Button,
  Card,
  Field,
  Icon,
  KeyValueRow,
  Screen,
  SegmentedControl,
  Select,
  Text,
} from '../components';
import { computeBankroll } from '../domain/bankroll';
import {
  arbitrage,
  assessCashOut,
  breakEvenWinRate,
  expectedValue,
  hedge,
  kellyStake,
  parlay,
} from '../domain/calculators';
import { formatNumber, formatPercent, formatSignedPercent, parseAmount } from '../domain/format';
import {
  bookPercentage,
  decimalToAmerican,
  decimalToFraction,
  fairOdds,
  formatOdds,
  impliedProbability,
  isValidDecimalOdds,
  overround,
  parseOdds,
  removeVig,
  type DevigMethod,
} from '../domain/odds';
import { useFormatters } from '../hooks/useFormatters';
import { useApp } from '../store/AppStore';
import { useTheme } from '../theme';
import { profitColor } from '../theme/tokens';

type ToolKey = 'converter' | 'value' | 'devig' | 'parlay' | 'arb' | 'hedge' | 'cashout';

const TOOLS: { value: ToolKey; label: string }[] = [
  { value: 'converter', label: 'Odds' },
  { value: 'value', label: 'EV & Kelly' },
  { value: 'devig', label: 'No-vig' },
  { value: 'parlay', label: 'Parlay' },
  { value: 'arb', label: 'Arbitrage' },
  { value: 'hedge', label: 'Hedge' },
  { value: 'cashout', label: 'Cash out' },
];

/** Reads odds in the user's preferred format, but always falls back to decimal. */
function useOddsReader() {
  const { oddsFormat } = useFormatters();
  return useMemo(
    () => (text: string) => {
      const parsed = parseOdds(text, oddsFormat);
      return parsed ?? parseOdds(text, 'decimal');
    },
    [oddsFormat],
  );
}

export function ToolsScreen() {
  const theme = useTheme();
  const { bets, transactions, settings } = useApp();
  const { money, signedMoney, odds: renderOdds, oddsHint, currency } = useFormatters();
  const readOdds = useOddsReader();

  const [tool, setTool] = useState<ToolKey>('converter');

  const bankroll = useMemo(
    () => computeBankroll(bets, transactions, settings.startingBankroll),
    [bets, transactions, settings.startingBankroll],
  );

  const styles = useMemo(
    () =>
      StyleSheet.create({
        group: { gap: theme.spacing(3.5) },
        row: { flexDirection: 'row', gap: theme.spacing(3) },
        legend: {
          flexDirection: 'row',
          alignItems: 'center',
          gap: theme.spacing(2),
          marginTop: theme.spacing(2),
        },
      }),
    [theme],
  );

  /* ---------------------------------------------------------------------- */
  /* Odds converter                                                          */
  /* ---------------------------------------------------------------------- */
  const [convertText, setConvertText] = useState('2.00');
  const converted = readOdds(convertText);

  /* ---------------------------------------------------------------------- */
  /* EV & Kelly                                                              */
  /* ---------------------------------------------------------------------- */
  const [evOddsText, setEvOddsText] = useState('2.10');
  const [evProbText, setEvProbText] = useState('52');
  const [evStakeText, setEvStakeText] = useState(String(settings.defaultStake || 25));
  const evOdds = readOdds(evOddsText);
  const evProb = (parseAmount(evProbText) ?? NaN) / 100;
  const evStake = parseAmount(evStakeText) ?? 0;
  const ev = evOdds !== null ? expectedValue(evOdds, evProb, evStake) : null;
  const kelly =
    evOdds !== null ? kellyStake(bankroll.balance, evOdds, evProb, settings.kellyFraction) : null;

  /* ---------------------------------------------------------------------- */
  /* De-vig                                                                  */
  /* ---------------------------------------------------------------------- */
  const [devigA, setDevigA] = useState('1.91');
  const [devigB, setDevigB] = useState('1.91');
  const [devigC, setDevigC] = useState('');
  const [method, setMethod] = useState<DevigMethod>('multiplicative');
  const devigInputs = [devigA, devigB, devigC]
    .map((text) => readOdds(text))
    .filter((value): value is number => value !== null && isValidDecimalOdds(value));
  const fair = devigInputs.length >= 2 ? removeVig(devigInputs, method) : [];
  const fairPrices = devigInputs.length >= 2 ? fairOdds(devigInputs, method) : [];

  /* ---------------------------------------------------------------------- */
  /* Parlay                                                                  */
  /* ---------------------------------------------------------------------- */
  const [parlayLegs, setParlayLegs] = useState<string[]>(['1.80', '2.10', '']);
  const [parlayStakeText, setParlayStakeText] = useState(String(settings.defaultStake || 25));
  const parlayOdds = parlayLegs
    .map((text) => readOdds(text))
    .filter((value): value is number => value !== null && isValidDecimalOdds(value));
  const parlayResult = parlay(parlayOdds, parseAmount(parlayStakeText) ?? 0);

  /* ---------------------------------------------------------------------- */
  /* Arbitrage                                                               */
  /* ---------------------------------------------------------------------- */
  const [arbA, setArbA] = useState('2.10');
  const [arbB, setArbB] = useState('2.05');
  const [arbC, setArbC] = useState('');
  const [arbStakeText, setArbStakeText] = useState('1000');
  const arbOdds = [arbA, arbB, arbC]
    .map((text) => readOdds(text))
    .filter((value): value is number => value !== null && isValidDecimalOdds(value));
  const arbResult = arbitrage(arbOdds, parseAmount(arbStakeText) ?? 0);

  /* ---------------------------------------------------------------------- */
  /* Hedge                                                                   */
  /* ---------------------------------------------------------------------- */
  const [hedgeStakeText, setHedgeStakeText] = useState('100');
  const [hedgeOriginalOdds, setHedgeOriginalOdds] = useState('4.00');
  const [hedgeOpposingOdds, setHedgeOpposingOdds] = useState('1.60');
  const hedgeResult = hedge(
    parseAmount(hedgeStakeText) ?? 0,
    readOdds(hedgeOriginalOdds) ?? NaN,
    readOdds(hedgeOpposingOdds) ?? NaN,
  );

  /* ---------------------------------------------------------------------- */
  /* Cash out                                                                */
  /* ---------------------------------------------------------------------- */
  const [coStakeText, setCoStakeText] = useState('100');
  const [coOriginalOdds, setCoOriginalOdds] = useState('4.00');
  const [coCurrentOdds, setCoCurrentOdds] = useState('2.00');
  const [coOfferText, setCoOfferText] = useState('180');
  const cashOutResult = assessCashOut(
    parseAmount(coStakeText) ?? 0,
    readOdds(coOriginalOdds) ?? NaN,
    readOdds(coCurrentOdds) ?? NaN,
    parseAmount(coOfferText) ?? NaN,
  );

  return (
    <Screen title="Tools" subtitle="Price checks and staking maths">
      <SegmentedControl
        scrollable
        segments={TOOLS}
        value={tool}
        onChange={setTool}
      />

      {tool === 'converter' ? (
        <>
          <Card title="Odds converter" subtitle="Type any format — decimal, American or fractional">
            <View style={styles.group}>
              <Field
                label="Odds"
                value={convertText}
                onChangeText={setConvertText}
                placeholder={oddsHint}
                keyboardType="numbers-and-punctuation"
                autoCapitalize="none"
              />
            </View>
          </Card>

          <Card title="Same price, every format">
            {converted !== null && isValidDecimalOdds(converted) ? (
              <>
                <KeyValueRow label="Decimal" value={converted.toFixed(3)} />
                <KeyValueRow
                  label="American"
                  value={decimalToAmerican(converted) > 0 ? `+${decimalToAmerican(converted)}` : `${decimalToAmerican(converted)}`}
                />
                <KeyValueRow
                  label="Fractional"
                  value={`${decimalToFraction(converted).numerator}/${decimalToFraction(converted).denominator}`}
                />
                <KeyValueRow
                  label="Implied probability"
                  value={formatPercent(impliedProbability(converted), 2)}
                />
                <KeyValueRow
                  label="Break-even win rate"
                  value={formatPercent(breakEvenWinRate(converted), 2)}
                  hint="How often this must land just to break even"
                />
                <KeyValueRow
                  label="Profit on 100"
                  value={money(100 * (converted - 1))}
                  divider={false}
                />
              </>
            ) : (
              <Text tone="muted">Enter odds above to see the conversions.</Text>
            )}
          </Card>
        </>
      ) : null}

      {tool === 'value' ? (
        <>
          <Card title="Expected value & Kelly" subtitle="How much is this bet actually worth?">
            <View style={styles.group}>
              <View style={styles.row}>
                <Field
                  containerStyle={{ flex: 1 }}
                  label="Odds"
                  value={evOddsText}
                  onChangeText={setEvOddsText}
                  placeholder={oddsHint}
                  keyboardType="numbers-and-punctuation"
                />
                <Field
                  containerStyle={{ flex: 1 }}
                  label="Your win %"
                  value={evProbText}
                  onChangeText={setEvProbText}
                  placeholder="52"
                  keyboardType="decimal-pad"
                  accessory={<Text tone="muted">%</Text>}
                />
              </View>
              <Field
                label="Stake"
                value={evStakeText}
                onChangeText={setEvStakeText}
                keyboardType="decimal-pad"
                prefix={currency}
              />
            </View>
          </Card>

          {ev && Number.isFinite(ev.expectedValue) ? (
            <>
              <Card title="Verdict">
                <KeyValueRow
                  label="Expected value"
                  value={signedMoney(ev.expectedValue)}
                  valueColor={profitColor(ev.expectedValue, theme.colors)}
                />
                <KeyValueRow
                  label="Expected ROI"
                  value={formatSignedPercent(ev.expectedRoi, 2)}
                  valueColor={profitColor(ev.expectedRoi, theme.colors)}
                />
                <KeyValueRow
                  label="Break-even probability"
                  value={formatPercent(ev.breakEvenProbability, 2)}
                />
                <KeyValueRow
                  label="Your edge"
                  value={formatSignedPercent(ev.edge, 2)}
                  valueColor={profitColor(ev.edge, theme.colors)}
                />
                <KeyValueRow
                  label="Fair odds at your estimate"
                  value={renderOdds(ev.fairOdds)}
                  divider={false}
                />
              </Card>

              <Card
                title="Suggested stake"
                subtitle={`${Math.round(settings.kellyFraction * 100)}% Kelly on a ${money(bankroll.balance, 0)} bankroll`}
              >
                {kelly && !kelly.noEdge ? (
                  <>
                    <KeyValueRow label="Full Kelly" value={formatPercent(kelly.fullKellyFraction, 2)} />
                    <KeyValueRow
                      label="Your fraction"
                      value={formatPercent(kelly.stakeFraction, 2)}
                    />
                    <KeyValueRow
                      label="Stake"
                      value={money(kelly.stake)}
                      valueColor={theme.colors.primary}
                      divider={false}
                    />
                  </>
                ) : (
                  <Banner
                    tone="warning"
                    message="No edge at this price — Kelly says stake nothing. Either find a better number or pass."
                  />
                )}
              </Card>
            </>
          ) : (
            <Banner tone="info" message="Enter valid odds and a win probability between 0 and 100." />
          )}
        </>
      ) : null}

      {tool === 'devig' ? (
        <>
          <Card
            title="Remove the vig"
            subtitle="Enter every price in a market to see the bookmaker's true probabilities"
          >
            <View style={styles.group}>
              <View style={styles.row}>
                <Field
                  containerStyle={{ flex: 1 }}
                  label="Outcome 1"
                  value={devigA}
                  onChangeText={setDevigA}
                  keyboardType="numbers-and-punctuation"
                  placeholder={oddsHint}
                />
                <Field
                  containerStyle={{ flex: 1 }}
                  label="Outcome 2"
                  value={devigB}
                  onChangeText={setDevigB}
                  keyboardType="numbers-and-punctuation"
                  placeholder={oddsHint}
                />
              </View>
              <Field
                label="Outcome 3 (optional)"
                value={devigC}
                onChangeText={setDevigC}
                keyboardType="numbers-and-punctuation"
                placeholder="For three-way markets"
              />
              <Select
                label="Method"
                value={method}
                onChange={(value) => setMethod(value as DevigMethod)}
                options={[
                  {
                    value: 'multiplicative',
                    label: 'Multiplicative',
                    description: 'Proportional — the standard quick method',
                  },
                  {
                    value: 'power',
                    label: 'Power',
                    description: 'Handles favourite/longshot bias better',
                  },
                  { value: 'shin', label: 'Shin', description: 'Best for two-way markets' },
                  {
                    value: 'additive',
                    label: 'Additive',
                    description: 'Splits the margin evenly across outcomes',
                  },
                ]}
              />
            </View>
          </Card>

          {devigInputs.length >= 2 ? (
            <Card title="Fair prices">
              <KeyValueRow
                label="Book percentage"
                value={formatPercent(bookPercentage(devigInputs), 2)}
              />
              <KeyValueRow
                label="Overround"
                value={`${formatNumber(overround(devigInputs), 2)}%`}
                valueColor={overround(devigInputs) < 0 ? theme.colors.positive : undefined}
                hint={overround(devigInputs) < 0 ? 'Negative — this is an arbitrage' : undefined}
              />
              {fair.map((probability, index) => (
                <KeyValueRow
                  key={index}
                  label={`Outcome ${index + 1}`}
                  hint={`Book price ${renderOdds(devigInputs[index] as number)}`}
                  value={`${formatPercent(probability, 2)} · ${formatOdds(fairPrices[index] ?? NaN, 'decimal')}`}
                  divider={index < fair.length - 1}
                />
              ))}
            </Card>
          ) : (
            <Banner tone="info" message="Enter at least two prices from the same market." />
          )}
        </>
      ) : null}

      {tool === 'parlay' ? (
        <>
          <Card title="Parlay calculator" subtitle="Combine any number of legs">
            <View style={styles.group}>
              {parlayLegs.map((leg, index) => (
                <Field
                  key={index}
                  label={`Leg ${index + 1}`}
                  value={leg}
                  onChangeText={(value) =>
                    setParlayLegs((current) =>
                      current.map((item, itemIndex) => (itemIndex === index ? value : item)),
                    )
                  }
                  keyboardType="numbers-and-punctuation"
                  placeholder={oddsHint}
                />
              ))}
              <View style={styles.row}>
                <Button
                  label="Add leg"
                  variant="secondary"
                  icon="add"
                  onPress={() => setParlayLegs((current) => [...current, ''])}
                />
                {parlayLegs.length > 2 ? (
                  <Button
                    label="Remove"
                    variant="ghost"
                    onPress={() => setParlayLegs((current) => current.slice(0, -1))}
                  />
                ) : null}
              </View>
              <Field
                label="Stake"
                value={parlayStakeText}
                onChangeText={setParlayStakeText}
                keyboardType="decimal-pad"
                prefix={currency}
              />
            </View>
          </Card>

          <Card title="Payout">
            <KeyValueRow label="Legs priced" value={String(parlayOdds.length)} />
            <KeyValueRow
              label="Combined odds"
              value={
                Number.isFinite(parlayResult.combinedOdds)
                  ? renderOdds(parlayResult.combinedOdds)
                  : '—'
              }
            />
            <KeyValueRow
              label="Implied probability"
              value={
                Number.isFinite(parlayResult.impliedProbability)
                  ? formatPercent(parlayResult.impliedProbability, 2)
                  : '—'
              }
            />
            <KeyValueRow
              label="Returns"
              value={Number.isFinite(parlayResult.returns) ? money(parlayResult.returns) : '—'}
            />
            <KeyValueRow
              label="Profit"
              value={Number.isFinite(parlayResult.profit) ? signedMoney(parlayResult.profit) : '—'}
              valueColor={theme.colors.positive}
              divider={false}
            />
          </Card>
        </>
      ) : null}

      {tool === 'arb' ? (
        <>
          <Card title="Arbitrage calculator" subtitle="Split a bankroll across every outcome">
            <View style={styles.group}>
              <View style={styles.row}>
                <Field
                  containerStyle={{ flex: 1 }}
                  label="Book A"
                  value={arbA}
                  onChangeText={setArbA}
                  keyboardType="numbers-and-punctuation"
                  placeholder={oddsHint}
                />
                <Field
                  containerStyle={{ flex: 1 }}
                  label="Book B"
                  value={arbB}
                  onChangeText={setArbB}
                  keyboardType="numbers-and-punctuation"
                  placeholder={oddsHint}
                />
              </View>
              <Field
                label="Book C (optional)"
                value={arbC}
                onChangeText={setArbC}
                keyboardType="numbers-and-punctuation"
                placeholder="Three-way markets"
              />
              <Field
                label="Total outlay"
                value={arbStakeText}
                onChangeText={setArbStakeText}
                keyboardType="decimal-pad"
                prefix={currency}
              />
            </View>
          </Card>

          {arbOdds.length >= 2 && Number.isFinite(arbResult.totalImplied) ? (
            <Card title={arbResult.isArbitrage ? 'Arbitrage found' : 'No arbitrage here'}>
              <View style={styles.legend}>
                <Icon
                  name={arbResult.isArbitrage ? 'checkmark-circle' : 'close-circle'}
                  size={18}
                  color={arbResult.isArbitrage ? theme.colors.positive : theme.colors.negative}
                />
                <Text variant="caption" tone="muted" style={{ flex: 1 }}>
                  {arbResult.isArbitrage
                    ? 'These prices guarantee a profit whichever way the market lands.'
                    : 'The combined book is over 100% — you would lose money covering every outcome.'}
                </Text>
              </View>
              <KeyValueRow
                label="Book percentage"
                value={formatPercent(arbResult.totalImplied, 2)}
              />
              {arbResult.stakes.map((stake, index) => (
                <KeyValueRow
                  key={index}
                  label={`Stake on outcome ${index + 1}`}
                  hint={`at ${renderOdds(arbOdds[index] as number)}`}
                  value={money(stake)}
                />
              ))}
              <KeyValueRow label="Return either way" value={money(arbResult.guaranteedReturn)} />
              <KeyValueRow
                label="Profit"
                value={signedMoney(arbResult.profit)}
                valueColor={profitColor(arbResult.profit, theme.colors)}
              />
              <KeyValueRow
                label="Return on outlay"
                value={formatSignedPercent(arbResult.profitPercent, 2)}
                valueColor={profitColor(arbResult.profitPercent, theme.colors)}
                divider={false}
              />
            </Card>
          ) : (
            <Banner tone="info" message="Enter at least two valid prices." />
          )}
        </>
      ) : null}

      {tool === 'hedge' ? (
        <>
          <Card title="Hedge calculator" subtitle="Lock in the same result whichever way it goes">
            <View style={styles.group}>
              <Field
                label="Original stake"
                value={hedgeStakeText}
                onChangeText={setHedgeStakeText}
                keyboardType="decimal-pad"
                prefix={currency}
              />
              <View style={styles.row}>
                <Field
                  containerStyle={{ flex: 1 }}
                  label="Original odds"
                  value={hedgeOriginalOdds}
                  onChangeText={setHedgeOriginalOdds}
                  keyboardType="numbers-and-punctuation"
                  placeholder={oddsHint}
                />
                <Field
                  containerStyle={{ flex: 1 }}
                  label="Opposing odds"
                  value={hedgeOpposingOdds}
                  onChangeText={setHedgeOpposingOdds}
                  keyboardType="numbers-and-punctuation"
                  placeholder={oddsHint}
                />
              </View>
            </View>
          </Card>

          {Number.isFinite(hedgeResult.hedgeStake) ? (
            <Card title="Lay it off">
              <KeyValueRow label="Hedge stake" value={money(hedgeResult.hedgeStake)} />
              <KeyValueRow
                label="If the original wins"
                value={signedMoney(hedgeResult.profitIfOriginalWins)}
                valueColor={profitColor(hedgeResult.profitIfOriginalWins, theme.colors)}
              />
              <KeyValueRow
                label="If the hedge wins"
                value={signedMoney(hedgeResult.profitIfHedgeWins)}
                valueColor={profitColor(hedgeResult.profitIfHedgeWins, theme.colors)}
              />
              <KeyValueRow
                label="Locked-in result"
                value={signedMoney(hedgeResult.guaranteedProfit)}
                valueColor={profitColor(hedgeResult.guaranteedProfit, theme.colors)}
                divider={false}
              />
            </Card>
          ) : (
            <Banner tone="info" message="Enter a stake and two valid prices." />
          )}
        </>
      ) : null}

      {tool === 'cashout' ? (
        <>
          <Card
            title="Cash-out checker"
            subtitle="Is the bookmaker's offer fair, or are they keeping a slice?"
          >
            <View style={styles.group}>
              <Field
                label="Stake"
                value={coStakeText}
                onChangeText={setCoStakeText}
                keyboardType="decimal-pad"
                prefix={currency}
              />
              <View style={styles.row}>
                <Field
                  containerStyle={{ flex: 1 }}
                  label="Odds you took"
                  value={coOriginalOdds}
                  onChangeText={setCoOriginalOdds}
                  keyboardType="numbers-and-punctuation"
                  placeholder={oddsHint}
                />
                <Field
                  containerStyle={{ flex: 1 }}
                  label="Current odds"
                  value={coCurrentOdds}
                  onChangeText={setCoCurrentOdds}
                  keyboardType="numbers-and-punctuation"
                  placeholder={oddsHint}
                />
              </View>
              <Field
                label="Cash-out offer"
                value={coOfferText}
                onChangeText={setCoOfferText}
                keyboardType="decimal-pad"
                prefix={currency}
              />
            </View>
          </Card>

          {Number.isFinite(cashOutResult.fairValue) ? (
            <Card title={cashOutResult.recommendation === 'take' ? 'Fair offer' : 'They are keeping a slice'}>
              <KeyValueRow label="Fair value" value={money(cashOutResult.fairValue)} />
              <KeyValueRow label="Their offer" value={money(cashOutResult.offer)} />
              <KeyValueRow
                label="Difference"
                value={signedMoney(cashOutResult.difference)}
                valueColor={profitColor(cashOutResult.difference, theme.colors)}
              />
              <KeyValueRow
                label="Bookmaker margin"
                value={formatPercent(cashOutResult.marginPercent, 2)}
                hint="What you give up by cashing out now"
                divider={false}
              />
              <Text variant="caption" tone="muted" style={{ marginTop: theme.spacing(3) }}>
                Cash-out offers are usually priced below fair value. Taking one repeatedly is a slow
                leak — but it can still be right when you need to cut variance.
              </Text>
            </Card>
          ) : (
            <Banner tone="info" message="Fill in the stake, both prices and the offer." />
          )}
        </>
      ) : null}
    </Screen>
  );
}
