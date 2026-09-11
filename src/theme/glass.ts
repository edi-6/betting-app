import { isGlassEffectAPIAvailable, isLiquidGlassAvailable } from 'expo-glass-effect';
import { useEffect, useState } from 'react';
import { AccessibilityInfo, Platform } from 'react-native';

/**
 * How a translucent surface can be rendered on this device, best first.
 *
 * - `liquid` — iOS 26's Liquid Glass, via `expo-glass-effect`. Real-time refraction,
 *   specular highlights, and elements that merge when they come close together.
 * - `blur`   — a platform blur (UIVisualEffectView on older iOS, Dimezis BlurView on
 *   Android 12+, `backdrop-filter` on web). Translucent, but flat.
 * - `solid`  — an opaque surface. Used when the platform cannot blur, and whenever the
 *   user has asked for reduced transparency.
 */
export type GlassCapability = 'liquid' | 'blur' | 'solid';

/**
 * What the device could do, ignoring user preference.
 *
 * `isGlassEffectAPIAvailable` is checked first and separately: some iOS 26 betas ship
 * the Liquid Glass design without the API behind it, and touching `GlassView` there
 * crashes the app. Both checks reach for a native module, so they are also wrapped —
 * a runtime missing the module entirely must fall back, not fail.
 */
export function deviceGlassCapability(): GlassCapability {
  if (Platform.OS === 'ios') {
    try {
      if (isGlassEffectAPIAvailable() && isLiquidGlassAvailable()) {
        return 'liquid';
      }
    } catch {
      // `requireNativeModule` throws when the runtime does not bundle
      // expo-glass-effect at all — a sandbox client, say, or a build predating the
      // dependency. Treat that as "no glass" rather than taking down the app.
    }
  }
  if (Platform.OS === 'ios' || Platform.OS === 'android' || Platform.OS === 'web') {
    return 'blur';
  }
  return 'solid';
}

/** Android needs an explicit blur implementation; below SDK 31 it degrades to a tint. */
export const ANDROID_BLUR_METHOD = 'dimezisBlurViewSdk31Plus' as const;

/**
 * Tracks the iOS/Android "reduce transparency" accessibility setting.
 *
 * Apple's guidance is explicit that glass must collapse to a solid surface when this is
 * on, so the effect is a progressive enhancement rather than a legibility risk.
 */
export function useReduceTransparency(): boolean {
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    // `isReduceTransparencyEnabled` is an iOS API. It is absent on react-native-web
    // and not guaranteed on Android, so feature-detect rather than assume.
    if (typeof AccessibilityInfo.isReduceTransparencyEnabled !== 'function') {
      return;
    }

    let cancelled = false;

    try {
      void AccessibilityInfo.isReduceTransparencyEnabled()
        .then((value) => {
          if (!cancelled) {
            setReduced(value);
          }
        })
        .catch(() => undefined);
    } catch {
      return;
    }

    let subscription: { remove: () => void } | undefined;
    try {
      subscription = AccessibilityInfo.addEventListener('reduceTransparencyChanged', (value) =>
        setReduced(value),
      );
    } catch {
      subscription = undefined;
    }

    return () => {
      cancelled = true;
      subscription?.remove();
    };
  }, []);

  return reduced;
}
