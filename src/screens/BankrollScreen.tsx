import React, { useCallback, useMemo, useState } from 'react';
import { Alert, Pressable, StyleSheet, View } from 'react-native';

import {
  Banner,
  Button,
  Card,
  DatePicker,
  EmptyState,
  Fab,
  Field,
  Icon,
  KeyValueRow,
  LineChart,
  ProgressBar,
  Screen,
  SegmentedControl,
  Select,
  Sheet,
  StatTile,
  Text,
  type LinePoint,
} from '../components';
import { bankrollSeries, signedTransactionAmount, summarize } from '../domain/analytics';
import { computeBankroll, evaluateLimits } from '../domain/bankroll';
import { BOOKMAKERS } from '../domain/catalog';
import { formatDate, formatShortDate } from '../domain/dates';
import { formatSignedPercent, parseAmount } from '../domain/format';
import type { Transaction, TransactionType } from '../domain/types';
import { useFormatters } from '../hooks/useFormatters';
import { useHaptics } from '../hooks/useHaptics';
import { useApp } from '../store/AppStore';
import { useTheme } from '../theme';
import { profitColor } from '../theme/tokens';
import type { TabScreenProps } from '../navigation/types';

const TRANSACTION_LABELS: Record<TransactionType, string> = {
  deposit: 'Deposit',
  withdrawal: 'Withdrawal',
  adjustment: 'Adjustment',
};

export function BankrollScreen({ navigation }: TabScreenProps<'Bankroll'>) {
  const theme = useTheme();
  const haptics = useHaptics();
  const {
    bets,
    transactions,
    settings,
    addTransaction,
    updateTransaction,
    deleteTransaction,
    updateSettings,
  } = useApp();
  const { money, signedMoney, currency } = useFormatters();

  const [formOpen, setFormOpen] = useState(false);
  // Set while editing an existing row; null means the form is adding a new one.
  const [editing, setEditing] = useState<Transaction | null>(null);
  const [type, setType] = useState<TransactionType>('deposit');
  const [amountText, setAmountText] = useState('');
  const [bookmaker, setBookmaker] = useState('');
  const [note, setNote] = useState('');
  const [date, setDate] = useState(() => new Date().toISOString());
  const [startingText, setStartingText] = useState(() => String(settings.startingBankroll));

  const bankroll = useMemo(
    () => computeBankroll(bets, transactions, settings.startingBankroll),
    [bets, transactions, settings.startingBankroll],
  );
  const summary = useMemo(() => summarize(bets), [bets]);
  const limits = useMemo(
    () => evaluateLimits(bets, settings.limits, bankroll.balance),
    [bets, settings.limits, bankroll.balance],
  );

  const series = useMemo(
    () => bankrollSeries(bets, transactions, settings.startingBankroll),
    [bets, transactions, settings.startingBankroll],
  );

  const chartData = useMemo<LinePoint[]>(
    () =>
      series.map((point) => ({
        x: point.timestamp,
        y: point.balance,
        label: formatShortDate(point.timestamp),
      })),
    [series],
  );

  const sortedTransactions = useMemo(
    () =>
      transactions
        .slice()
        .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()),
    [transactions],
  );

  const openForm = useCallback((transaction: Transaction | null) => {
    setEditing(transaction);
    setType(transaction?.type ?? 'deposit');
    setAmountText(transaction ? String(Math.abs(transaction.amount)) : '');
    setBookmaker(transaction?.bookmaker ?? '');
    setNote(transaction?.note ?? '');
    setDate(transaction?.date ?? new Date().toISOString());
    setFormOpen(true);
  }, []);

  const submit = useCallback(() => {
    const amount = parseAmount(amountText);
    if (amount === null || amount === 0) {
      Alert.alert('Enter an amount', 'Type how much moved in or out.');
      return;
    }
    const signed = type === 'adjustment' ? amount : Math.abs(amount);
    haptics('success');

    if (editing) {
      updateTransaction({
        ...editing,
        type,
        amount: signed,
        date,
        bookmaker: bookmaker.trim().length > 0 ? bookmaker.trim() : undefined,
        note: note.trim().length > 0 ? note.trim() : undefined,
      });
    } else {
      addTransaction({
        type,
        amount: signed,
        date,
        bookmaker: bookmaker.trim().length > 0 ? bookmaker.trim() : undefined,
        note: note.trim().length > 0 ? note.trim() : undefined,
      });
    }

    setAmountText('');
    setNote('');
    setEditing(null);
    setFormOpen(false);
  }, [amountText, type, date, bookmaker, note, editing, addTransaction, updateTransaction, haptics]);

  const confirmDeleteTransaction = useCallback(
    (transaction: Transaction) => {
      Alert.alert(
        `Delete this ${TRANSACTION_LABELS[transaction.type].toLowerCase()}?`,
        'Your balance will be recalculated.',
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Delete',
            style: 'destructive',
            onPress: () => {
              haptics('warning');
              deleteTransaction(transaction.id);
            },
          },
        ],
      );
    },
    [deleteTransaction, haptics],
  );

  const styles = useMemo(
    () =>
      StyleSheet.create({
        hero: {
          gap: theme.spacing(4),
          backgroundColor: theme.colors.surface,
          borderRadius: theme.radius.xl,
          borderWidth: StyleSheet.hairlineWidth,
          borderColor: theme.colors.border,
          padding: theme.spacing(5),
          ...theme.shadow.card,
        },
        grid: { flexDirection: 'row', flexWrap: 'wrap', gap: theme.spacing(3) },
        transactionRow: {
          flexDirection: 'row',
          alignItems: 'center',
          gap: theme.spacing(3),
          paddingVertical: theme.spacing(3),
          borderBottomWidth: StyleSheet.hairlineWidth,
          borderColor: theme.colors.border,
        },
        badge: {
          width: 34,
          height: 34,
          borderRadius: 17,
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: theme.colors.surfaceAlt,
        },
        limitRow: { gap: theme.spacing(2), paddingVertical: theme.spacing(2.5) },
        limitHeader: {
          flexDirection: 'row',
          alignItems: 'baseline',
          justifyContent: 'space-between',
          gap: theme.spacing(3),
        },
        formSection: { gap: theme.spacing(3.5) },
      }),
    [theme],
  );

  return (
    <>
      <Screen
        title="Bankroll"
        subtitle="Money in, money out, money at risk"
        bottomInset={72}
      >
        <View style={styles.hero}>
          <View>
            <Text variant="caption" tone="muted">
              Current balance
            </Text>
            <Text variant="display">{money(bankroll.balance)}</Text>
            <Text variant="caption" tone="muted">
              {money(bankroll.startingBankroll + bankroll.deposits)} put in ·{' '}
              {formatSignedPercent(bankroll.returnOnCapital)} return on capital
            </Text>
          </View>

          <View style={styles.grid}>
            <StatTile
              label="Available"
              value={money(bankroll.available)}
              icon="wallet-outline"
              compact
            />
            <StatTile
              label="At risk"
              value={money(bankroll.atRisk)}
              caption={`${summary.pendingBets} open`}
              tone={bankroll.atRisk > 0 ? 'warning' : 'default'}
              icon="hourglass-outline"
              compact
            />
            <StatTile
              label="Realised P/L"
              value={signedMoney(bankroll.realizedProfit, 0)}
              tone={
                bankroll.realizedProfit > 0
                  ? 'positive'
                  : bankroll.realizedProfit < 0
                    ? 'negative'
                    : 'default'
              }
              icon="trending-up-outline"
              compact
            />
          </View>
        </View>

        {chartData.length > 1 ? (
          <Card title="Balance over time" subtitle="Deposits, withdrawals and results combined">
            <LineChart
              data={chartData}
              color={theme.colors.accent}
              showZeroLine={false}
              formatValue={(value) => money(value)}
              formatLabel={(point) => point.label ?? ''}
            />
          </Card>
        ) : null}

        <Card title="Where the money went">
          <KeyValueRow label="Starting bankroll" value={money(bankroll.startingBankroll)} />
          <KeyValueRow label="Deposits" value={signedMoney(bankroll.deposits)} valueColor={theme.colors.positive} />
          <KeyValueRow
            label="Withdrawals"
            value={signedMoney(-bankroll.withdrawals)}
            valueColor={bankroll.withdrawals > 0 ? theme.colors.negative : undefined}
          />
          {bankroll.adjustments !== 0 ? (
            <KeyValueRow label="Adjustments" value={signedMoney(bankroll.adjustments)} />
          ) : null}
          <KeyValueRow
            label="Betting profit"
            value={signedMoney(bankroll.realizedProfit)}
            valueColor={profitColor(bankroll.realizedProfit, theme.colors)}
          />
          <KeyValueRow
            label="Balance"
            value={money(bankroll.balance)}
            divider={false}
          />
        </Card>

        <Card
          title="Set your starting bankroll"
          subtitle="The balance you had when you began tracking"
        >
          <View style={{ flexDirection: 'row', gap: theme.spacing(3), alignItems: 'flex-end' }}>
            <Field
              containerStyle={{ flex: 1 }}
              value={startingText}
              onChangeText={setStartingText}
              keyboardType="decimal-pad"
              prefix={currency}
              placeholder="1000"
            />
            <Button
              label="Save"
              testID="save-starting-bankroll"
              onPress={() => {
                const value = parseAmount(startingText);
                if (value === null) {
                  Alert.alert('Enter a number', 'That does not look like an amount.');
                  return;
                }
                haptics('success');
                updateSettings({ startingBankroll: value });
              }}
            />
          </View>
        </Card>

        {limits.length > 0 ? (
          <Card title="Your limits" subtitle="Guard rails you set for yourself">
            {limits.map((limit) => (
              <View key={limit.key} style={styles.limitRow}>
                <View style={styles.limitHeader}>
                  <Text variant="label">{limit.label}</Text>
                  <Text
                    variant="caption"
                    color={
                      limit.severity === 'exceeded'
                        ? theme.colors.negative
                        : limit.severity === 'approaching'
                          ? theme.colors.warning
                          : theme.colors.textMuted
                    }
                  >
                    {limit.isCurrency
                      ? `${money(limit.used, 0)} of ${money(limit.limit, 0)}`
                      : `${limit.used.toFixed(limit.key === 'maxStakePercent' ? 1 : 0)} of ${limit.limit}`}
                  </Text>
                </View>
                <ProgressBar
                  progress={limit.ratio}
                  color={
                    limit.severity === 'exceeded'
                      ? theme.colors.negative
                      : limit.severity === 'approaching'
                        ? theme.colors.warning
                        : theme.colors.primary
                  }
                />
                <Text variant="caption" tone="muted">
                  {limit.description}
                </Text>
              </View>
            ))}
            <Button
              label="Edit limits"
              variant="ghost"
              size="sm"
              onPress={() => navigation.navigate('Settings')}
            />
          </Card>
        ) : (
          <Banner
            tone="info"
            title="Set yourself some guard rails"
            message="Daily stake caps and loss limits keep a bad run from turning into a bad month."
            actionLabel="Set limits"
            onAction={() => navigation.navigate('Settings')}
          />
        )}

        <Card title={`Transactions (${transactions.length})`}>
          {sortedTransactions.length === 0 ? (
            <EmptyState
              icon="swap-horizontal-outline"
              title="No transactions yet"
              message="Record deposits and withdrawals so your balance stays honest."
            />
          ) : (
            sortedTransactions.map((transaction, index) => {
              const signed = signedTransactionAmount(transaction);
              return (
                <Pressable
                  key={transaction.id}
                  onPress={() => openForm(transaction)}
                  onLongPress={() => confirmDeleteTransaction(transaction)}
                  accessibilityRole="button"
                  accessibilityLabel={`Edit ${TRANSACTION_LABELS[transaction.type].toLowerCase()} of ${signedMoney(signed)}`}
                  accessibilityHint="Long press to delete"
                  style={[
                    styles.transactionRow,
                    index === sortedTransactions.length - 1 ? { borderBottomWidth: 0 } : null,
                  ]}
                >
                  <View style={styles.badge}>
                    <Icon
                      name={
                        transaction.type === 'deposit'
                          ? 'arrow-down'
                          : transaction.type === 'withdrawal'
                            ? 'arrow-up'
                            : 'swap-horizontal'
                      }
                      size={16}
                      color={signed >= 0 ? theme.colors.positive : theme.colors.negative}
                    />
                  </View>
                  <View style={{ flex: 1, gap: 1 }}>
                    <Text variant="label">{TRANSACTION_LABELS[transaction.type]}</Text>
                    <Text variant="caption" tone="muted" numberOfLines={1}>
                      {[formatDate(transaction.date), transaction.bookmaker, transaction.note]
                        .filter(Boolean)
                        .join(' · ')}
                    </Text>
                  </View>
                  <Text variant="mono" color={profitColor(signed, theme.colors)}>
                    {signedMoney(signed)}
                  </Text>
                </Pressable>
              );
            })
          )}
        </Card>

        {transactions.length > 0 ? (
          <Text variant="caption" tone="muted" align="center">
            Tap a transaction to edit it, or long-press to delete.
          </Text>
        ) : null}
      </Screen>

      <Fab
        label="Add funds"
        icon="add"
        onPress={() => openForm(null)}
        testID="bankroll-fab"
      />

      <Sheet
        visible={formOpen}
        onClose={() => {
          setEditing(null);
          setFormOpen(false);
        }}
        title={editing ? 'Edit transaction' : 'Record a transaction'}
        subtitle={`Balance right now: ${money(bankroll.balance)}`}
        footer={
          <View style={{ flexDirection: 'row', gap: theme.spacing(3) }}>
            {editing ? (
              <Button
                label="Delete"
                variant="danger"
                fullWidth
                style={{ flex: 1 }}
                onPress={() => {
                  const target = editing;
                  setEditing(null);
                  setFormOpen(false);
                  confirmDeleteTransaction(target);
                }}
              />
            ) : null}
            <Button
              label="Save"
              // `fullWidth` alone only stretches the cross axis inside a row, so the
              // button needs an explicit flex to fill the footer's width.
              fullWidth
              style={{ flex: editing ? 2 : 1 }}
              testID="save-transaction"
              onPress={submit}
            />
          </View>
        }
      >
        <View style={styles.formSection}>
          <SegmentedControl
            segments={[
              { value: 'deposit', label: 'Deposit' },
              { value: 'withdrawal', label: 'Withdrawal' },
              { value: 'adjustment', label: 'Adjustment' },
            ]}
            value={type}
            onChange={setType}
          />
          <Field
            label="Amount"
            value={amountText}
            onChangeText={setAmountText}
            keyboardType={type === 'adjustment' ? 'numbers-and-punctuation' : 'decimal-pad'}
            prefix={currency}
            placeholder="100"
            hint={
              type === 'adjustment'
                ? 'Use a minus sign to reduce the balance, e.g. a bonus that expired.'
                : undefined
            }
          />
          <DatePicker label="Date" value={date} onChange={setDate} withTime={false} />
          <Select
            label="Bookmaker"
            value={bookmaker}
            options={BOOKMAKERS}
            onChange={setBookmaker}
            placeholder="Optional"
            searchable
            allowCustom
          />
          <Field label="Note" value={note} onChangeText={setNote} placeholder="Optional" />
        </View>
      </Sheet>
    </>
  );
}
