import * as Haptics from 'expo-haptics';
import { useCallback } from 'react';

import { useSettings } from '../store/AppStore';

export type HapticKind = 'light' | 'medium' | 'success' | 'warning' | 'error' | 'selection';

/** Haptic feedback that respects the user's setting and never throws on unsupported devices. */
export function useHaptics() {
  const { hapticsEnabled } = useSettings();

  return useCallback(
    (kind: HapticKind = 'light') => {
      if (!hapticsEnabled) {
        return;
      }
      const run = async () => {
        switch (kind) {
          case 'medium':
            return Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
          case 'success':
            return Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
          case 'warning':
            return Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
          case 'error':
            return Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
          case 'selection':
            return Haptics.selectionAsync();
          case 'light':
          default:
            return Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        }
      };
      void run().catch(() => undefined);
    },
    [hapticsEnabled],
  );
}
