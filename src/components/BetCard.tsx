import React, { useMemo } from 'react';
import { Pressable, StyleSheet, View } from 'react-native';

import { iconForSport } from '../domain/catalog';
import { formatRelativeDay, formatTime } from '../domain/dates';
import { combinedOdds, settleBet } from '../domain/settlement';
import type { Bet } from '../domain/types';
import { useFormatters } from '../hooks/useFormatters';
import { useTheme } from '../theme';
import { profitColor } from '../theme/tokens';
import { Icon } from './Icon';
import { StatusPill } from './StatusPill';
import { Text } from './Text';

export interface BetCardProps {
  bet: Bet;
  onPress?: () => void;
  /** Hide the date line when the list is already grouped by day. */
  hideDate?: boolean;
  testID?: string;
}

export function BetCard({ bet, onPress, hideDate = false, testID }: BetCardProps) {
  const theme = useTheme();
  const { money, signedMoney, odds } = useFormatters();

  const settlement = useMemo(() => settleBet(bet), [bet]);
  const price = useMemo(() => combinedOdds(bet), [bet]);
  const isParlay = bet.legs.length > 1;
  const firstLeg = bet.legs[0];

  const styles = useMemo(
    () =>
      StyleSheet.create({
        card: {
          flexDirection: 'row',
          gap: theme.spacing(3),
          backgroundColor: theme.colors.surface,
          borderRadius: theme.radius.lg,
          borderWidth: StyleSheet.hairlineWidth,
          borderColor: theme.colors.border,
          padding: theme.spacing(3.5),
        },
        pressed: { opacity: 0.7 },
        badge: {
          width: 40,
          height: 40,
          borderRadius: 20,
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: theme.colors.surfaceAlt,
        },
        body: { flex: 1, gap: theme.spacing(1) },
        topRow: {
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing(2),
        },
        metaRow: {
          flexDirection: 'row',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: theme.spacing(2),
        },
        dot: { width: 3, height: 3, borderRadius: 1.5, backgroundColor: theme.colors.textMuted },
        right: { alignItems: 'flex-end', gap: theme.spacing(1) },
        tags: { flexDirection: 'row', gap: theme.spacing(1.5), flexWrap: 'wrap' },
        tag: {
          backgroundColor: theme.colors.surfaceAlt,
          borderRadius: theme.radius.pill,
          paddingHorizontal: theme.spacing(2),
          paddingVertical: 2,
        },
      }),
    [theme],
  );

  const title = isParlay
    ? `${bet.legs.length}-leg parlay`
    : (firstLeg?.selection || firstLeg?.event || 'Bet');

  const subtitle = isParlay
    ? bet.legs.map((leg) => leg.selection || leg.event).join(' · ')
    : firstLeg?.event || firstLeg?.market || '';

  const content = (
    <View style={styles.card} testID={testID}>
      <View style={styles.badge}>
        <Icon
          name={isParlay ? 'layers-outline' : iconForSport(firstLeg?.sport ?? '')}
          size={19}
          color={theme.colors.textSecondary}
        />
      </View>

      <View style={styles.body}>
        <View style={styles.topRow}>
          <Text variant="subheading" numberOfLines={1} style={{ flex: 1 }}>
            {title}
          </Text>
          <Text variant="mono" tone="secondary">
            {odds(price)}
          </Text>
        </View>

        {subtitle ? (
          <Text variant="caption" tone="muted" numberOfLines={1}>
            {subtitle}
          </Text>
        ) : null}

        <View style={styles.metaRow}>
          <StatusPill status={settlement.status} />
          <Text variant="caption" tone="muted">
            {money(bet.stake)}
            {bet.isFreeBet ? ' free' : ''}
          </Text>
          {bet.bookmaker ? (
            <>
              <View style={styles.dot} />
              <Text variant="caption" tone="muted" numberOfLines={1}>
                {bet.bookmaker}
              </Text>
            </>
          ) : null}
          {!hideDate ? (
            <>
              <View style={styles.dot} />
              <Text variant="caption" tone="muted">
                {formatRelativeDay(bet.placedAt)} · {formatTime(bet.placedAt)}
              </Text>
            </>
          ) : null}
        </View>

        {bet.tags.length > 0 ? (
          <View style={styles.tags}>
            {bet.tags.map((tag) => (
              <View key={tag} style={styles.tag}>
                <Text variant="caption" tone="muted">
                  {tag}
                </Text>
              </View>
            ))}
          </View>
        ) : null}
      </View>

      <View style={styles.right}>
        {settlement.isSettled ? (
          <Text variant="subheading" color={profitColor(settlement.profit, theme.colors)}>
            {signedMoney(settlement.profit)}
          </Text>
        ) : (
          <>
            <Text variant="caption" tone="muted">
              To win
            </Text>
            <Text variant="subheading" tone="secondary">
              {money(bet.stake * (price - 1))}
            </Text>
          </>
        )}
      </View>
    </View>
  );

  if (!onPress) {
    return content;
  }

  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={`${title}, ${settlement.status}`}
      onPress={onPress}
      style={({ pressed }) => (pressed ? styles.pressed : null)}
    >
      {content}
    </Pressable>
  );
}
