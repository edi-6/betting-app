import { render, screen } from '@testing-library/react-native';
import * as GlassEffect from 'expo-glass-effect';
import React from 'react';
import { AccessibilityInfo, Platform, Text as RNText } from 'react-native';

import { createEmptyData } from '../../domain/defaults';
import { createMemoryStore } from '../../data/storage';
import { AppProvider } from '../../store/AppStore';
import { deviceGlassCapability, ThemeProvider } from '../../theme';
import { GlassGroup, GlassSurface, withAlpha } from '../GlassSurface';

// The availability checks are native calls. Drive them explicitly so both the
// Liquid Glass path and the blur fallback are exercised on purpose rather than by
// whatever the test renderer happens to report.
jest.mock('expo-glass-effect', () => ({
  ...jest.requireActual('expo-glass-effect'),
  isGlassEffectAPIAvailable: jest.fn(() => true),
  isLiquidGlassAvailable: jest.fn(() => true),
}));

const mockAvailability = (available: boolean) => {
  jest.mocked(GlassEffect.isGlassEffectAPIAvailable).mockReturnValue(available);
  jest.mocked(GlassEffect.isLiquidGlassAvailable).mockReturnValue(available);
};

const setPlatform = (os: 'ios' | 'android' | 'web') => {
  Object.defineProperty(Platform, 'OS', { value: os, configurable: true });
};

const originalPlatform = Platform.OS;

afterEach(() => {
  setPlatform(originalPlatform as 'ios');
  mockAvailability(true);
});

function wrap(ui: React.ReactElement, glassEnabled = true) {
  const data = createEmptyData();
  data.settings.glassEnabled = glassEnabled;
  return render(
    <AppProvider store={createMemoryStore()} initialData={data}>
      <ThemeProvider>{ui}</ThemeProvider>
    </AppProvider>,
  );
}

describe('withAlpha', () => {
  it('converts a hex colour to rgba', () => {
    expect(withAlpha('#25D3A0', 0.5)).toBe('rgba(37, 211, 160, 0.5)');
    expect(withAlpha('#000000', 1)).toBe('rgba(0, 0, 0, 1)');
  });

  it('clamps the alpha', () => {
    expect(withAlpha('#FFFFFF', 5)).toBe('rgba(255, 255, 255, 1)');
    expect(withAlpha('#FFFFFF', -2)).toBe('rgba(255, 255, 255, 0)');
  });

  it('passes through colours it cannot parse', () => {
    expect(withAlpha('rgba(1, 2, 3, 0.5)', 0.2)).toBe('rgba(1, 2, 3, 0.5)');
    expect(withAlpha('red', 0.2)).toBe('red');
  });
});

describe('deviceGlassCapability', () => {
  it('uses Liquid Glass on iOS when the API is there', () => {
    setPlatform('ios');
    mockAvailability(true);
    expect(deviceGlassCapability()).toBe('liquid');
  });

  it('falls back to a blur on an iOS build without the glass API', () => {
    // Some iOS 26 betas ship the design without the API; touching GlassView crashes.
    setPlatform('ios');
    mockAvailability(false);
    expect(deviceGlassCapability()).toBe('blur');
  });

  it('never claims Liquid Glass off iOS', () => {
    mockAvailability(true);
    for (const os of ['android', 'web'] as const) {
      setPlatform(os);
      expect(deviceGlassCapability()).toBe('blur');
    }
  });
});

describe('reduce transparency detection', () => {
  it('does not throw when the platform has no accessibility API', () => {
    // react-native-web has no `isReduceTransparencyEnabled`; reading it used to crash
    // the whole app on launch.
    const original = AccessibilityInfo.isReduceTransparencyEnabled;
    // @ts-expect-error deliberately removing the API to mimic web
    AccessibilityInfo.isReduceTransparencyEnabled = undefined;
    try {
      expect(() =>
        wrap(
          <GlassSurface testID="surface">
            <RNText>Still here</RNText>
          </GlassSurface>,
        ),
      ).not.toThrow();
      expect(screen.getByText('Still here')).toBeTruthy();
    } finally {
      AccessibilityInfo.isReduceTransparencyEnabled = original;
    }
  });
});

describe('GlassSurface', () => {
  it('renders its children on the Liquid Glass path', () => {
    setPlatform('ios');
    mockAvailability(true);
    wrap(
      <GlassSurface testID="surface">
        <RNText>Floating chrome</RNText>
      </GlassSurface>,
    );
    expect(screen.getByTestId('surface')).toBeTruthy();
    expect(screen.getByText('Floating chrome')).toBeTruthy();
  });

  it('renders its children on the blur fallback path', () => {
    setPlatform('android');
    mockAvailability(false);
    wrap(
      <GlassSurface testID="surface">
        <RNText>Floating chrome</RNText>
      </GlassSurface>,
    );
    expect(screen.getByTestId('surface')).toBeTruthy();
    expect(screen.getByText('Floating chrome')).toBeTruthy();
  });

  it('falls back to an opaque surface when the user turns glass off', () => {
    wrap(
      <GlassSurface testID="surface" fallbackColor="#101725">
        <RNText>Solid</RNText>
      </GlassSurface>,
      false,
    );
    const styles = screen.getByTestId('surface').props.style as Record<string, unknown>[];
    const flattened = Object.assign({}, ...styles.filter(Boolean));
    expect(flattened.backgroundColor).toBe('#101725');
    expect(screen.getByText('Solid')).toBeTruthy();
  });

  it('groups children without dropping them when glass is unavailable', () => {
    wrap(
      <GlassGroup>
        <RNText>Grouped</RNText>
      </GlassGroup>,
      false,
    );
    expect(screen.getByText('Grouped')).toBeTruthy();
  });
});
