import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import type { KeyValueStore } from './data/storage';
import type { AppData } from './domain/types';
import { RootNavigator } from './navigation';
import { AppProvider } from './store/AppStore';
import { ThemeProvider, useTheme } from './theme';

function Shell() {
  const theme = useTheme();
  return (
    <>
      <StatusBar style={theme.scheme === 'dark' ? 'light' : 'dark'} />
      <RootNavigator />
    </>
  );
}

export interface AppRootProps {
  /** Injected by tests; defaults to AsyncStorage. */
  store?: KeyValueStore;
  /** Injected by tests to skip disk hydration. */
  initialData?: AppData;
}

/**
 * The whole application tree. Kept separate from `App.tsx` so tests can mount it with
 * an in-memory store and a known dataset.
 */
export function AppRoot({ store, initialData }: AppRootProps) {
  return (
    <SafeAreaProvider
      initialMetrics={{
        frame: { x: 0, y: 0, width: 390, height: 844 },
        insets: { top: 47, left: 0, right: 0, bottom: 34 },
      }}
    >
      <AppProvider store={store} initialData={initialData}>
        <ThemeProvider>
          <Shell />
        </ThemeProvider>
      </AppProvider>
    </SafeAreaProvider>
  );
}
