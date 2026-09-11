import { BlurView } from 'expo-blur';
import { GlassContainer, GlassView, type GlassStyle } from 'expo-glass-effect';
import React, { useMemo } from 'react';
import { Platform, StyleSheet, View, type StyleProp, type ViewStyle } from 'react-native';

import { ANDROID_BLUR_METHOD, useGlass, useTheme } from '../theme';

export interface GlassSurfaceProps {
  children?: React.ReactNode;
  /** Corner radius. Glass refracts along the rounded edge, so this is load-bearing. */
  radius?: number;
  /**
   * `regular` is Apple's default, adaptive material. `clear` is thinner and lets more
   * of the content behind show through — right for chrome over a busy scroll view.
   */
  glassStyle?: GlassStyle;
  /** Subtle brand wash over the glass. Keep the alpha low or it turns to plastic. */
  tintColor?: string;
  /** Lets the glass react to touch with Apple's built-in press animation. */
  interactive?: boolean;
  /** Blur strength for the non-Liquid-Glass path, 1–100. */
  intensity?: number;
  /** Painted under every variant so text keeps its contrast. */
  fallbackColor?: string;
  /** Hairline border. Glass already has an edge, so it is skipped on the liquid path. */
  bordered?: boolean;
  style?: StyleProp<ViewStyle>;
  testID?: string;
}

/**
 * A translucent surface that uses the best material the device offers:
 * iOS 26 Liquid Glass, else a platform blur, else an opaque fill.
 *
 * Only ever used for chrome that floats *over* scrolling content — a tab bar, a sheet,
 * the action button. Glass over a flat background has nothing to refract and just looks
 * like a muddy rectangle, so content surfaces stay opaque on purpose.
 */
export function GlassSurface({
  children,
  radius,
  glassStyle = 'regular',
  tintColor,
  interactive = false,
  intensity = 60,
  fallbackColor,
  bordered = true,
  style,
  testID,
}: GlassSurfaceProps) {
  const theme = useTheme();
  const glass = useGlass();

  const borderRadius = radius ?? theme.radius.lg;
  const solidColor = fallbackColor ?? theme.colors.backgroundElevated;

  const shared = useMemo<ViewStyle>(
    () => ({
      borderRadius,
      overflow: 'hidden',
      borderWidth: bordered ? StyleSheet.hairlineWidth : 0,
      borderColor: theme.colors.border,
    }),
    [borderRadius, bordered, theme.colors.border],
  );

  if (glass === 'liquid') {
    return (
      <GlassView
        testID={testID}
        glassEffectStyle={glassStyle}
        tintColor={tintColor}
        isInteractive={interactive}
        // The app owns its light/dark switch, so don't let the system override it.
        colorScheme={theme.scheme}
        style={[shared, { borderWidth: 0 }, style]}
      >
        {children}
      </GlassView>
    );
  }

  if (glass === 'blur') {
    return (
      <View testID={testID} style={[shared, style]}>
        <BlurView
          tint={theme.scheme === 'dark' ? 'systemChromeMaterialDark' : 'systemChromeMaterialLight'}
          intensity={intensity}
          blurMethod={ANDROID_BLUR_METHOD}
          style={StyleSheet.absoluteFill}
        />
        {/* Android below SDK 31 and some web engines ignore the blur, so keep a wash. */}
        <View
          style={[
            StyleSheet.absoluteFill,
            { backgroundColor: tintColor ?? withAlpha(solidColor, Platform.OS === 'ios' ? 0.12 : 0.55) },
          ]}
        />
        {children}
      </View>
    );
  }

  return (
    <View testID={testID} style={[shared, { backgroundColor: solidColor }, style]}>
      {children}
    </View>
  );
}

/** Apply an alpha to a #rrggbb colour. Non-hex input is returned unchanged. */
export function withAlpha(color: string, alpha: number): string {
  const match = /^#([0-9a-f]{6})$/i.exec(color.trim());
  if (!match) {
    return color;
  }
  const hex = match[1] as string;
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${Math.min(Math.max(alpha, 0), 1)})`;
}

/**
 * Groups glass elements so iOS 26 can merge them when they come close together.
 * A plain passthrough everywhere else.
 */
export function GlassGroup({
  children,
  spacing = 12,
  style,
}: {
  children: React.ReactNode;
  spacing?: number;
  style?: StyleProp<ViewStyle>;
}) {
  const glass = useGlass();

  if (glass !== 'liquid') {
    return <View style={style}>{children}</View>;
  }
  return (
    <GlassContainer spacing={spacing} style={style}>
      {children}
    </GlassContainer>
  );
}
