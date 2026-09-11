import React, { useMemo, useState } from 'react';
import { Pressable, StyleSheet, TextInput, View } from 'react-native';

import { useTheme } from '../theme';
import { Icon } from './Icon';
import { Sheet } from './Sheet';
import { Text } from './Text';

export interface SelectOption {
  value: string;
  label: string;
  description?: string;
}

export interface SelectProps {
  label?: string;
  value: string;
  options: SelectOption[] | string[];
  onChange: (value: string) => void;
  placeholder?: string;
  /** Let the user type a value that is not in the list. */
  allowCustom?: boolean;
  searchable?: boolean;
  title?: string;
  error?: string;
  testID?: string;
}

function normalize(options: SelectOption[] | string[]): SelectOption[] {
  return options.map((option) =>
    typeof option === 'string' ? { value: option, label: option } : option,
  );
}

/** A tap-to-open picker. Falls back to a free-text field when `allowCustom` is set. */
export function Select({
  label,
  value,
  options,
  onChange,
  placeholder = 'Select…',
  allowCustom = false,
  searchable = false,
  title,
  error,
  testID,
}: SelectProps) {
  const theme = useTheme();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');

  const normalized = useMemo(() => normalize(options), [options]);
  const selected = normalized.find((option) => option.value === value);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (needle.length === 0) return normalized;
    return normalized.filter((option) => option.label.toLowerCase().includes(needle));
  }, [normalized, query]);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        container: { gap: theme.spacing(1.5) },
        trigger: {
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing(2),
          backgroundColor: theme.colors.surfaceAlt,
          borderRadius: theme.radius.md,
          borderWidth: 1,
          borderColor: error ? theme.colors.negative : theme.colors.border,
          paddingHorizontal: theme.spacing(3.5),
          minHeight: 48,
        },
        option: {
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing(3),
          paddingVertical: theme.spacing(3.5),
          borderBottomWidth: StyleSheet.hairlineWidth,
          borderColor: theme.colors.border,
        },
        search: {
          backgroundColor: theme.colors.surfaceAlt,
          borderRadius: theme.radius.md,
          paddingHorizontal: theme.spacing(3.5),
          paddingVertical: theme.spacing(3),
          color: theme.colors.text,
          fontSize: 16,
          marginBottom: theme.spacing(2),
        },
        customRow: { paddingVertical: theme.spacing(3.5) },
      }),
    [theme, error],
  );

  const commit = (next: string) => {
    onChange(next);
    setOpen(false);
    setQuery('');
  };

  return (
    <View style={styles.container}>
      {label ? (
        <Text variant="label" tone="secondary">
          {label}
        </Text>
      ) : null}
      <Pressable
        testID={testID}
        accessibilityRole="button"
        accessibilityLabel={label ? `${label}: ${selected?.label ?? placeholder}` : placeholder}
        onPress={() => setOpen(true)}
        style={styles.trigger}
      >
        <Text tone={selected ? 'default' : 'muted'} numberOfLines={1} style={{ flex: 1 }}>
          {selected?.label ?? (value.length > 0 ? value : placeholder)}
        </Text>
        <Icon name="chevron-down" size={18} color={theme.colors.textMuted} />
      </Pressable>
      {error ? (
        <Text variant="caption" tone="negative">
          {error}
        </Text>
      ) : null}

      <Sheet visible={open} onClose={() => setOpen(false)} title={title ?? label ?? 'Select'}>
        {searchable || allowCustom ? (
          <TextInput
            value={query}
            onChangeText={setQuery}
            placeholder={allowCustom ? 'Search or type your own…' : 'Search…'}
            placeholderTextColor={theme.colors.textMuted}
            style={styles.search}
            autoCorrect={false}
            returnKeyType="done"
            onSubmitEditing={() => {
              if (allowCustom && query.trim().length > 0) {
                commit(query.trim());
              }
            }}
          />
        ) : null}

        {allowCustom && query.trim().length > 0 && !filtered.some((o) => o.label === query.trim()) ? (
          <Pressable style={styles.customRow} onPress={() => commit(query.trim())}>
            <Text tone="primary">Use “{query.trim()}”</Text>
          </Pressable>
        ) : null}

        {filtered.map((option) => (
          <Pressable
            key={option.value}
            style={styles.option}
            accessibilityRole="button"
            accessibilityState={{ selected: option.value === value }}
            onPress={() => commit(option.value)}
          >
            <View style={{ flex: 1 }}>
              <Text>{option.label}</Text>
              {option.description ? (
                <Text variant="caption" tone="muted">
                  {option.description}
                </Text>
              ) : null}
            </View>
            {option.value === value ? (
              <Icon name="checkmark" size={18} color={theme.colors.primary} />
            ) : null}
          </Pressable>
        ))}

        {filtered.length === 0 && !allowCustom ? (
          <Text tone="muted" style={{ paddingVertical: theme.spacing(4) }}>
            Nothing matches that search.
          </Text>
        ) : null}
      </Sheet>
    </View>
  );
}
