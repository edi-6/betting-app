import { Ionicons } from '@expo/vector-icons';
import React from 'react';
import type { StyleProp, TextStyle } from 'react-native';

export type IconName = React.ComponentProps<typeof Ionicons>['name'];

export interface IconProps {
  /** An Ionicons glyph name. Accepts a plain string so catalog data can be passed through. */
  name: IconName | string;
  size?: number;
  color: string;
  style?: StyleProp<TextStyle>;
}

/**
 * Thin wrapper around Ionicons. Keeping the cast in one place means catalog data
 * (which stores icon names as strings) can be rendered without sprinkling casts around.
 */
export function Icon({ name, size = 18, color, style }: IconProps) {
  return <Ionicons name={name as IconName} size={size} color={color} style={style} />;
}
