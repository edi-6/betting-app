import { BottomTabBarHeightContext } from '@react-navigation/bottom-tabs';
import React, { useCallback, useContext, useMemo, useState } from 'react';
import { Pressable, SectionList, StyleSheet, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import {
  BetCard,
  Button,
  Chip,
  EmptyState,
  Fab,
  Field,
  Icon,
  Screen,
  SegmentedControl,
  Sheet,
  Text,
} from '../components';
import { summarize } from '../domain/analytics';
import { formatRelativeDay } from '../domain/dates';
import { formatSignedPercent } from '../domain/format';
import {
  applyFilter,
  countActiveFilters,
  DATE_RANGE_PRESETS,
  EMPTY_FILTER,
  facetsFor,
  SORT_OPTIONS,
  sortBets,
  type BetFilter,
  type SortKey,
} from '../domain/filters';
import { BET_STATUS_LABELS } from '../domain/settlement';
import type { Bet, BetStatus } from '../domain/types';
import { useFormatters } from '../hooks/useFormatters';
import { useApp } from '../store/AppStore';
import { useTheme } from '../theme';
import { profitColor } from '../theme/tokens';
import type { TabScreenProps } from '../navigation/types';

const QUICK_STATUSES: BetStatus[] = ['pending', 'won', 'lost', 'void', 'cashed_out'];

export function BetsScreen({ navigation, route }: TabScreenProps<'Bets'>) {
  const theme = useTheme();
  const insets = useSafeAreaInsets();
  const tabBarHeight = useContext(BottomTabBarHeightContext) ?? 0;
  const { bets } = useApp();
  const { money, signedMoney } = useFormatters();

  const [filter, setFilter] = useState<BetFilter>(EMPTY_FILTER);
  const [sort, setSort] = useState<SortKey>('date_desc');
  const [filterOpen, setFilterOpen] = useState(false);

  // Deep link from the dashboard's "open bets" shortcut. Adjusting state during render
  // (rather than in an effect) keeps the list from flashing the unfiltered results first.
  const [appliedPresetAt, setAppliedPresetAt] = useState<number | undefined>(undefined);
  const { presetStatus, requestedAt } = route.params ?? {};
  if (requestedAt !== undefined && requestedAt !== appliedPresetAt) {
    setAppliedPresetAt(requestedAt);
    if (presetStatus) {
      setFilter((current) => ({ ...current, statuses: [presetStatus] }));
    }
  }

  const facets = useMemo(() => facetsFor(bets), [bets]);
  const filtered = useMemo(() => applyFilter(bets, filter), [bets, filter]);
  const sorted = useMemo(() => sortBets(filtered, sort), [filtered, sort]);
  const summary = useMemo(() => summarize(filtered), [filtered]);
  const activeCount = countActiveFilters(filter);

  const sections = useMemo(() => {
    if (sort !== 'date_desc' && sort !== 'date_asc') {
      return [{ title: '', key: 'all', data: sorted }];
    }
    const groups = new Map<string, Bet[]>();
    for (const bet of sorted) {
      const key = formatRelativeDay(bet.placedAt);
      const existing = groups.get(key);
      if (existing) {
        existing.push(bet);
      } else {
        groups.set(key, [bet]);
      }
    }
    return Array.from(groups.entries()).map(([title, data]) => ({ title, key: title, data }));
  }, [sorted, sort]);

  const toggleStatus = useCallback((status: BetStatus) => {
    setFilter((current) => ({
      ...current,
      statuses: current.statuses.includes(status)
        ? current.statuses.filter((value) => value !== status)
        : [...current.statuses, status],
    }));
  }, []);

  const toggleValue = useCallback((key: 'sports' | 'leagues' | 'bookmakers' | 'tags', value: string) => {
    setFilter((current) => {
      const list = current[key];
      return {
        ...current,
        [key]: list.includes(value) ? list.filter((item) => item !== value) : [...list, value],
      };
    });
  }, []);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        controls: {
          paddingHorizontal: theme.spacing(5),
          gap: theme.spacing(3),
          paddingBottom: theme.spacing(3),
        },
        row: { flexDirection: 'row', alignItems: 'center', gap: theme.spacing(2) },
        chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: theme.spacing(2) },
        filterButton: {
          width: 48,
          height: 48,
          borderRadius: theme.radius.md,
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: theme.colors.surfaceAlt,
          borderWidth: 1,
          borderColor: activeCount > 0 ? theme.colors.primary : theme.colors.border,
        },
        badge: {
          position: 'absolute',
          top: 4,
          right: 4,
          minWidth: 16,
          height: 16,
          borderRadius: 8,
          paddingHorizontal: 4,
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: theme.colors.primary,
        },
        summaryStrip: {
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing(3),
          backgroundColor: theme.colors.surfaceAlt,
          borderRadius: theme.radius.md,
          paddingVertical: theme.spacing(2.5),
          paddingHorizontal: theme.spacing(3.5),
        },
        summaryItem: { alignItems: 'center', gap: 1 },
        listContent: {
          paddingHorizontal: theme.spacing(5),
          paddingBottom: (tabBarHeight > 0 ? tabBarHeight : insets.bottom) + theme.spacing(22),
          gap: theme.spacing(3),
        },
        sectionHeader: {
          backgroundColor: theme.colors.background,
          paddingTop: theme.spacing(3),
          paddingBottom: theme.spacing(1),
        },
        sheetSection: { gap: theme.spacing(2), marginBottom: theme.spacing(5) },
      }),
    [theme, insets.bottom, tabBarHeight, activeCount],
  );

  return (
    <>
      <Screen title="Bets" subtitle={`${bets.length} in your ledger`} scroll={false}>
        <View style={styles.controls}>
          <View style={styles.row}>
            <Field
              containerStyle={{ flex: 1 }}
              value={filter.query}
              onChangeText={(query) => setFilter((current) => ({ ...current, query }))}
              placeholder="Search team, market, tag…"
              autoCorrect={false}
              accessory={
                filter.query.length > 0 ? (
                  <Pressable
                    onPress={() => setFilter((current) => ({ ...current, query: '' }))}
                    accessibilityRole="button"
                    accessibilityLabel="Clear search"
                  >
                    <Icon name="close-circle" size={18} color={theme.colors.textMuted} />
                  </Pressable>
                ) : (
                  <Icon name="search" size={18} color={theme.colors.textMuted} />
                )
              }
            />
            <Pressable
              style={styles.filterButton}
              accessibilityRole="button"
              accessibilityLabel={`Filters, ${activeCount} active`}
              onPress={() => setFilterOpen(true)}
            >
              <Icon
                name="options-outline"
                size={20}
                color={activeCount > 0 ? theme.colors.primary : theme.colors.textSecondary}
              />
              {activeCount > 0 ? (
                <View style={styles.badge}>
                  <Text variant="caption" color={theme.colors.primaryText} style={{ fontSize: 10 }}>
                    {activeCount}
                  </Text>
                </View>
              ) : null}
            </Pressable>
          </View>

          <View style={styles.chipRow}>
            {QUICK_STATUSES.map((status) => (
              <Chip
                key={status}
                label={BET_STATUS_LABELS[status]}
                selected={filter.statuses.includes(status)}
                size="sm"
                onPress={() => toggleStatus(status)}
              />
            ))}
          </View>

          {filtered.length > 0 ? (
            <View style={styles.summaryStrip}>
              <View style={styles.summaryItem}>
                <Text variant="caption" tone="muted">
                  Bets
                </Text>
                <Text variant="label">{filtered.length}</Text>
              </View>
              <View style={styles.summaryItem}>
                <Text variant="caption" tone="muted">
                  Staked
                </Text>
                <Text variant="label">{money(summary.turnover, 0)}</Text>
              </View>
              <View style={styles.summaryItem}>
                <Text variant="caption" tone="muted">
                  P/L
                </Text>
                <Text variant="label" color={profitColor(summary.profit, theme.colors)}>
                  {signedMoney(summary.profit, 0)}
                </Text>
              </View>
              <View style={styles.summaryItem}>
                <Text variant="caption" tone="muted">
                  ROI
                </Text>
                <Text variant="label" color={profitColor(summary.roi, theme.colors)}>
                  {formatSignedPercent(summary.roi)}
                </Text>
              </View>
            </View>
          ) : null}
        </View>

        <SectionList
          sections={sections}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
          stickySectionHeadersEnabled={false}
          keyboardShouldPersistTaps="handled"
          renderSectionHeader={({ section }) =>
            section.title ? (
              <View style={styles.sectionHeader}>
                <Text variant="label" tone="muted">
                  {section.title}
                </Text>
              </View>
            ) : null
          }
          renderItem={({ item }) => (
            <BetCard
              bet={item}
              hideDate={sort === 'date_desc' || sort === 'date_asc'}
              onPress={() => navigation.navigate('BetDetail', { betId: item.id })}
            />
          )}
          ListEmptyComponent={
            <EmptyState
              icon="search-outline"
              title={bets.length === 0 ? 'No bets yet' : 'Nothing matches those filters'}
              message={
                bets.length === 0
                  ? 'Add your first bet to start tracking results.'
                  : 'Try clearing a filter or widening the date range.'
              }
              actionLabel={bets.length === 0 ? 'Add a bet' : 'Clear filters'}
              onAction={() =>
                bets.length === 0 ? navigation.navigate('BetForm') : setFilter(EMPTY_FILTER)
              }
            />
          }
        />
      </Screen>

      <Fab label="Add bet" onPress={() => navigation.navigate('BetForm')} testID="bets-fab" />

      <Sheet
        visible={filterOpen}
        onClose={() => setFilterOpen(false)}
        title="Filter & sort"
        subtitle={`${filtered.length} of ${bets.length} bets`}
        footer={
          <View style={{ flexDirection: 'row', gap: theme.spacing(3) }}>
            <Button
              label="Reset"
              variant="secondary"
              onPress={() => setFilter(EMPTY_FILTER)}
              style={{ flex: 1 }}
            />
            <Button label="Show results" onPress={() => setFilterOpen(false)} style={{ flex: 2 }} />
          </View>
        }
      >
        <View style={styles.sheetSection}>
          <Text variant="label" tone="secondary">
            Sort by
          </Text>
          <SegmentedControl
            scrollable
            segments={SORT_OPTIONS.map((option) => ({ value: option.key, label: option.label }))}
            value={sort}
            onChange={setSort}
          />
        </View>

        <View style={styles.sheetSection}>
          <Text variant="label" tone="secondary">
            Date range
          </Text>
          <SegmentedControl
            segments={DATE_RANGE_PRESETS.map((preset) => ({
              value: preset.key,
              label: preset.label,
            }))}
            value={filter.range}
            onChange={(range) => setFilter((current) => ({ ...current, range }))}
          />
        </View>

        <View style={styles.sheetSection}>
          <Text variant="label" tone="secondary">
            Bet type
          </Text>
          <SegmentedControl
            segments={[
              { value: 'all', label: 'All' },
              { value: 'single', label: 'Singles' },
              { value: 'parlay', label: 'Parlays' },
            ]}
            value={filter.betType}
            onChange={(betType) => setFilter((current) => ({ ...current, betType }))}
          />
        </View>

        <View style={styles.sheetSection}>
          <Text variant="label" tone="secondary">
            Status
          </Text>
          <View style={styles.chipRow}>
            {(Object.keys(BET_STATUS_LABELS) as BetStatus[]).map((status) => (
              <Chip
                key={status}
                label={BET_STATUS_LABELS[status]}
                size="sm"
                selected={filter.statuses.includes(status)}
                onPress={() => toggleStatus(status)}
              />
            ))}
          </View>
        </View>

        {facets.sports.length > 0 ? (
          <View style={styles.sheetSection}>
            <Text variant="label" tone="secondary">
              Sport
            </Text>
            <View style={styles.chipRow}>
              {facets.sports.map((sport) => (
                <Chip
                  key={sport}
                  label={sport}
                  size="sm"
                  selected={filter.sports.includes(sport)}
                  onPress={() => toggleValue('sports', sport)}
                />
              ))}
            </View>
          </View>
        ) : null}

        {facets.leagues.length > 0 ? (
          <View style={styles.sheetSection}>
            <Text variant="label" tone="secondary">
              League
            </Text>
            <View style={styles.chipRow}>
              {facets.leagues.map((league) => (
                <Chip
                  key={league}
                  label={league}
                  size="sm"
                  selected={filter.leagues.includes(league)}
                  onPress={() => toggleValue('leagues', league)}
                />
              ))}
            </View>
          </View>
        ) : null}

        {facets.bookmakers.length > 0 ? (
          <View style={styles.sheetSection}>
            <Text variant="label" tone="secondary">
              Bookmaker
            </Text>
            <View style={styles.chipRow}>
              {facets.bookmakers.map((bookmaker) => (
                <Chip
                  key={bookmaker}
                  label={bookmaker}
                  size="sm"
                  selected={filter.bookmakers.includes(bookmaker)}
                  onPress={() => toggleValue('bookmakers', bookmaker)}
                />
              ))}
            </View>
          </View>
        ) : null}

        {facets.tags.length > 0 ? (
          <View style={styles.sheetSection}>
            <Text variant="label" tone="secondary">
              Tags
            </Text>
            <View style={styles.chipRow}>
              {facets.tags.map((tag) => (
                <Chip
                  key={tag}
                  label={tag}
                  size="sm"
                  selected={filter.tags.includes(tag)}
                  onPress={() => toggleValue('tags', tag)}
                />
              ))}
            </View>
          </View>
        ) : null}
      </Sheet>
    </>
  );
}
