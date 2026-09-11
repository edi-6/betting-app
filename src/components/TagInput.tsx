import React, { useMemo, useState } from 'react';
import { Pressable, StyleSheet, TextInput, View } from 'react-native';

import { SUGGESTED_TAGS } from '../domain/catalog';
import { useTheme } from '../theme';
import { Icon } from './Icon';
import { Text } from './Text';

export interface TagInputProps {
  label?: string;
  value: string[];
  onChange: (tags: string[]) => void;
  /** Extra suggestions on top of the built-in list, e.g. tags already used. */
  suggestions?: string[];
  placeholder?: string;
}

export function TagInput({
  label,
  value,
  onChange,
  suggestions = [],
  placeholder = 'Add a tag…',
}: TagInputProps) {
  const theme = useTheme();
  const [draft, setDraft] = useState('');

  const available = useMemo(() => {
    const all = Array.from(new Set([...suggestions, ...SUGGESTED_TAGS]));
    return all.filter((tag) => !value.includes(tag)).slice(0, 12);
  }, [suggestions, value]);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        container: { gap: theme.spacing(2) },
        chips: { flexDirection: 'row', flexWrap: 'wrap', gap: theme.spacing(2) },
        chip: {
          flexDirection: 'row',
          alignItems: 'center',
          gap: theme.spacing(1.5),
          borderRadius: theme.radius.pill,
          backgroundColor: theme.colors.primarySoft,
          borderWidth: StyleSheet.hairlineWidth,
          borderColor: theme.colors.primary,
          paddingVertical: theme.spacing(1.5),
          paddingHorizontal: theme.spacing(3),
        },
        suggestion: {
          borderRadius: theme.radius.pill,
          backgroundColor: theme.colors.surfaceAlt,
          borderWidth: StyleSheet.hairlineWidth,
          borderColor: theme.colors.border,
          paddingVertical: theme.spacing(1.5),
          paddingHorizontal: theme.spacing(3),
        },
        inputRow: {
          flexDirection: 'row',
          alignItems: 'center',
          backgroundColor: theme.colors.surfaceAlt,
          borderRadius: theme.radius.md,
          borderWidth: 1,
          borderColor: theme.colors.border,
          paddingHorizontal: theme.spacing(3.5),
          minHeight: 48,
        },
        input: { flex: 1, color: theme.colors.text, fontSize: 16, paddingVertical: theme.spacing(3) },
      }),
    [theme],
  );

  const add = (tag: string) => {
    const normalized = tag.trim().toLowerCase();
    if (normalized.length === 0 || value.includes(normalized)) {
      setDraft('');
      return;
    }
    onChange([...value, normalized]);
    setDraft('');
  };

  const remove = (tag: string) => onChange(value.filter((item) => item !== tag));

  return (
    <View style={styles.container}>
      {label ? (
        <Text variant="label" tone="secondary">
          {label}
        </Text>
      ) : null}

      {value.length > 0 ? (
        <View style={styles.chips}>
          {value.map((tag) => (
            <Pressable
              key={tag}
              style={styles.chip}
              accessibilityRole="button"
              accessibilityLabel={`Remove tag ${tag}`}
              onPress={() => remove(tag)}
            >
              <Text variant="caption" tone="primary">
                {tag}
              </Text>
              <Icon name="close" size={12} color={theme.colors.primary} />
            </Pressable>
          ))}
        </View>
      ) : null}

      <View style={styles.inputRow}>
        <TextInput
          value={draft}
          onChangeText={setDraft}
          placeholder={placeholder}
          placeholderTextColor={theme.colors.textMuted}
          style={styles.input}
          autoCapitalize="none"
          autoCorrect={false}
          returnKeyType="done"
          onSubmitEditing={() => add(draft)}
        />
        {draft.trim().length > 0 ? (
          <Pressable onPress={() => add(draft)} accessibilityRole="button" accessibilityLabel="Add tag">
            <Icon name="add-circle" size={22} color={theme.colors.primary} />
          </Pressable>
        ) : null}
      </View>

      {available.length > 0 ? (
        <View style={styles.chips}>
          {available.map((tag) => (
            <Pressable
              key={tag}
              style={styles.suggestion}
              accessibilityRole="button"
              onPress={() => add(tag)}
            >
              <Text variant="caption" tone="muted">
                + {tag}
              </Text>
            </Pressable>
          ))}
        </View>
      ) : null}
    </View>
  );
}
