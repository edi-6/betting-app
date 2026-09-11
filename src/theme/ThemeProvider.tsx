import React, { createContext, useContext, useEffect, useMemo } from 'react';
import { useColorScheme } from 'react-native';
import * as SystemUI from 'expo-system-ui';

import { useApp } from '../store/AppStore';
import { deviceGlassCapability, useReduceTransparency, type GlassCapability } from './glass';
import { darkTheme, lightTheme, type Theme } from './tokens';

const ThemeContext = createContext<Theme>(darkTheme);
const GlassContext = createContext<GlassCapability>('solid');

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const systemScheme = useColorScheme();
  const { settings } = useApp();

  const theme = useMemo(() => {
    const mode =
      settings.themeMode === 'system' ? (systemScheme ?? 'dark') : settings.themeMode;
    return mode === 'light' ? lightTheme : darkTheme;
  }, [settings.themeMode, systemScheme]);

  const reduceTransparency = useReduceTransparency();
  const glass = useMemo<GlassCapability>(() => {
    if (!settings.glassEnabled || reduceTransparency) {
      return 'solid';
    }
    return deviceGlassCapability();
  }, [settings.glassEnabled, reduceTransparency]);

  useEffect(() => {
    // Keeps the area behind the root view in sync, avoiding a white flash on rotation.
    void SystemUI.setBackgroundColorAsync(theme.colors.background).catch(() => undefined);
  }, [theme.colors.background]);

  return (
    <ThemeContext.Provider value={theme}>
      <GlassContext.Provider value={glass}>{children}</GlassContext.Provider>
    </ThemeContext.Provider>
  );
}

/** How translucent chrome should render, after platform, preference and accessibility. */
export function useGlass(): GlassCapability {
  return useContext(GlassContext);
}

export function useTheme(): Theme {
  return useContext(ThemeContext);
}

/**
 * Build memoised styles that depend on the theme.
 *
 *   const styles = useThemedStyles((theme) => ({ card: { backgroundColor: theme.colors.surface } }));
 */
export function useThemedStyles<T>(factory: (theme: Theme) => T): T {
  const theme = useTheme();
  return useMemo(() => factory(theme), [theme, factory]);
}
