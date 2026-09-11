import React, { useMemo } from 'react';
import { Text as RNText, type TextProps as RNTextProps, type TextStyle } from 'react-native';

import { useTheme } from '../theme';
import type { Theme } from '../theme/tokens';

export type TextVariant = keyof Theme['typography'];
export type TextTone =
  | 'default'
  | 'secondary'
  | 'muted'
  | 'primary'
  | 'positive'
  | 'negative'
  | 'warning'
  | 'inverted';

export interface TextProps extends RNTextProps {
  variant?: TextVariant;
  tone?: TextTone;
  /** Convenience override; wins over `tone`. */
  color?: string;
  align?: TextStyle['textAlign'];
  weight?: TextStyle['fontWeight'];
}

export function Text({
  variant = 'body',
  tone = 'default',
  color,
  align,
  weight,
  style,
  ...rest
}: TextProps) {
  const theme = useTheme();

  const resolved = useMemo<TextStyle>(() => {
    const toneColors: Record<TextTone, string> = {
      default: theme.colors.text,
      secondary: theme.colors.textSecondary,
      muted: theme.colors.textMuted,
      primary: theme.colors.primary,
      positive: theme.colors.positive,
      negative: theme.colors.negative,
      warning: theme.colors.warning,
      inverted: theme.colors.textInverted,
    };
    return {
      ...theme.typography[variant],
      color: color ?? toneColors[tone],
      textAlign: align,
      ...(weight ? { fontWeight: weight } : null),
    };
  }, [theme, variant, tone, color, align, weight]);

  return <RNText {...rest} style={[resolved, style]} />;
}
