import React, { useMemo } from 'react';
import { Pressable, ScrollView, StyleSheet, View, type StyleProp, type ViewStyle } from 'react-native';

import { useTheme } from '../theme';
import { Text } from './Text';

export interface Segment<T extends string> {
  value: T;
  label: string;
}

export interface SegmentedControlProps<T extends string> {
  segments: Segment<T>[];
  value: T;
  onChange: (value: T) => void;
  /** Let the control scroll horizontally when the segments do not fit. */
  scrollable?: boolean;
  style?: StyleProp<ViewStyle>;
  testID?: string;
}

export function SegmentedControl<T extends string>({
  segments,
  value,
  onChange,
  scrollable = false,
  style,
  testID,
}: SegmentedControlProps<T>) {
  const theme = useTheme();

  const styles = useMemo(
    () =>
      StyleSheet.create({
        track: {
          flexDirection: 'row',
          backgroundColor: theme.colors.surfaceAlt,
          borderRadius: theme.radius.md,
          padding: 3,
          gap: 2,
        },
        segment: {
          flex: scrollable ? 0 : 1,
          alignItems: 'center',
          justifyContent: 'center',
          paddingVertical: theme.spacing(2),
          paddingHorizontal: theme.spacing(3),
          borderRadius: theme.radius.sm,
          minHeight: 36,
        },
        active: {
          backgroundColor: theme.colors.surface,
          ...theme.shadow.card,
          shadowOpacity: theme.scheme === 'dark' ? 0.25 : 0.08,
        },
      }),
    [theme, scrollable],
  );

  const content = segments.map((segment) => {
    const active = segment.value === value;
    return (
      <Pressable
        key={segment.value}
        accessibilityRole="button"
        accessibilityState={{ selected: active }}
        onPress={() => onChange(segment.value)}
        style={[styles.segment, active ? styles.active : null]}
      >
        <Text variant="label" tone={active ? 'default' : 'muted'} numberOfLines={1}>
          {segment.label}
        </Text>
      </Pressable>
    );
  });

  if (scrollable) {
    return (
      <ScrollView
        testID={testID}
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={[styles.track, style]}
      >
        {content}
      </ScrollView>
    );
  }

  return (
    <View testID={testID} style={[styles.track, style]}>
      {content}
    </View>
  );
}
