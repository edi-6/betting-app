import React, { useMemo, useState } from 'react';
import {
  StyleSheet,
  TextInput,
  View,
  type KeyboardTypeOptions,
  type StyleProp,
  type TextInputProps,
  type ViewStyle,
} from 'react-native';

import { useTheme } from '../theme';
import { Text } from './Text';

export interface FieldProps extends Omit<TextInputProps, 'style'> {
  label?: string;
  hint?: string;
  error?: string;
  /** Rendered inside the input on the right, e.g. a unit or a small action. */
  accessory?: React.ReactNode;
  /** Rendered inside the input on the left, e.g. a currency symbol. */
  prefix?: string;
  containerStyle?: StyleProp<ViewStyle>;
  keyboardType?: KeyboardTypeOptions;
}

export function Field({
  label,
  hint,
  error,
  accessory,
  prefix,
  containerStyle,
  onFocus,
  onBlur,
  ...rest
}: FieldProps) {
  const theme = useTheme();
  const [focused, setFocused] = useState(false);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        container: { gap: theme.spacing(1.5) },
        inputRow: {
          flexDirection: 'row',
          alignItems: 'center',
          gap: theme.spacing(2),
          backgroundColor: theme.colors.surfaceAlt,
          borderRadius: theme.radius.md,
          borderWidth: 1,
          borderColor: error
            ? theme.colors.negative
            : focused
              ? theme.colors.primary
              : theme.colors.border,
          paddingHorizontal: theme.spacing(3.5),
          minHeight: 48,
        },
        input: {
          flex: 1,
          color: theme.colors.text,
          fontSize: 16,
          paddingVertical: theme.spacing(3),
        },
      }),
    [theme, focused, error],
  );

  return (
    <View style={[styles.container, containerStyle]}>
      {label ? (
        <Text variant="label" tone="secondary">
          {label}
        </Text>
      ) : null}
      <View style={styles.inputRow}>
        {prefix ? (
          <Text variant="body" tone="muted">
            {prefix}
          </Text>
        ) : null}
        <TextInput
          {...rest}
          style={styles.input}
          placeholderTextColor={theme.colors.textMuted}
          selectionColor={theme.colors.primary}
          onFocus={(event) => {
            setFocused(true);
            onFocus?.(event);
          }}
          onBlur={(event) => {
            setFocused(false);
            onBlur?.(event);
          }}
        />
        {accessory}
      </View>
      {error ? (
        <Text variant="caption" tone="negative">
          {error}
        </Text>
      ) : hint ? (
        <Text variant="caption" tone="muted">
          {hint}
        </Text>
      ) : null}
    </View>
  );
}
