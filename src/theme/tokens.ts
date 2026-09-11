/**
 * Design tokens. A dark-first palette with a light counterpart; every colour has the
 * same role in both so components never branch on the scheme.
 */

export interface ThemeColors {
  background: string;
  backgroundElevated: string;
  surface: string;
  surfaceAlt: string;
  surfaceSunken: string;
  border: string;
  borderStrong: string;
  text: string;
  textSecondary: string;
  textMuted: string;
  textInverted: string;
  primary: string;
  primarySoft: string;
  primaryText: string;
  accent: string;
  accentSoft: string;
  positive: string;
  positiveSoft: string;
  negative: string;
  negativeSoft: string;
  warning: string;
  warningSoft: string;
  info: string;
  infoSoft: string;
  neutral: string;
  neutralSoft: string;
  overlay: string;
  tabBar: string;
  chartGrid: string;
}

export interface Theme {
  scheme: 'light' | 'dark';
  colors: ThemeColors;
  spacing: (steps: number) => number;
  radius: { sm: number; md: number; lg: number; xl: number; pill: number };
  typography: {
    display: TextStyleToken;
    title: TextStyleToken;
    heading: TextStyleToken;
    subheading: TextStyleToken;
    body: TextStyleToken;
    label: TextStyleToken;
    caption: TextStyleToken;
    mono: TextStyleToken;
  };
  shadow: {
    card: ShadowToken;
    floating: ShadowToken;
  };
}

export interface TextStyleToken {
  fontSize: number;
  lineHeight: number;
  fontWeight:
    | '100'
    | '200'
    | '300'
    | '400'
    | '500'
    | '600'
    | '700'
    | '800'
    | '900';
  letterSpacing?: number;
}

export interface ShadowToken {
  shadowColor: string;
  shadowOffset: { width: number; height: number };
  shadowOpacity: number;
  shadowRadius: number;
  elevation: number;
}

const BASE_UNIT = 4;

const typography: Theme['typography'] = {
  display: { fontSize: 34, lineHeight: 40, fontWeight: '700', letterSpacing: -0.8 },
  title: { fontSize: 26, lineHeight: 32, fontWeight: '700', letterSpacing: -0.5 },
  heading: { fontSize: 20, lineHeight: 26, fontWeight: '700', letterSpacing: -0.3 },
  subheading: { fontSize: 16, lineHeight: 22, fontWeight: '600' },
  body: { fontSize: 15, lineHeight: 21, fontWeight: '400' },
  label: { fontSize: 13, lineHeight: 18, fontWeight: '600' },
  caption: { fontSize: 12, lineHeight: 16, fontWeight: '500' },
  mono: { fontSize: 15, lineHeight: 20, fontWeight: '600' },
};

const radius = { sm: 8, md: 12, lg: 18, xl: 26, pill: 999 };

export const darkColors: ThemeColors = {
  background: '#0B0F17',
  backgroundElevated: '#101725',
  surface: '#141C2B',
  surfaceAlt: '#1B2536',
  surfaceSunken: '#0D131F',
  border: '#243044',
  borderStrong: '#334155',
  text: '#F1F5F9',
  textSecondary: '#CBD5E1',
  textMuted: '#8296B0',
  textInverted: '#06111A',
  primary: '#25D3A0',
  primarySoft: 'rgba(37, 211, 160, 0.14)',
  primaryText: '#052B22',
  accent: '#7C8CFF',
  accentSoft: 'rgba(124, 140, 255, 0.16)',
  positive: '#34D399',
  positiveSoft: 'rgba(52, 211, 153, 0.15)',
  negative: '#FB7185',
  negativeSoft: 'rgba(251, 113, 133, 0.15)',
  warning: '#FBBF24',
  warningSoft: 'rgba(251, 191, 36, 0.16)',
  info: '#38BDF8',
  infoSoft: 'rgba(56, 189, 248, 0.16)',
  neutral: '#94A3B8',
  neutralSoft: 'rgba(148, 163, 184, 0.14)',
  overlay: 'rgba(4, 8, 14, 0.72)',
  tabBar: 'rgba(16, 23, 37, 0.96)',
  chartGrid: 'rgba(148, 163, 184, 0.14)',
};

export const lightColors: ThemeColors = {
  background: '#F4F6FB',
  backgroundElevated: '#FFFFFF',
  surface: '#FFFFFF',
  surfaceAlt: '#EEF2F9',
  surfaceSunken: '#E7ECF5',
  border: '#DCE3EE',
  borderStrong: '#C2CDDE',
  text: '#0E1726',
  textSecondary: '#334155',
  textMuted: '#64748B',
  textInverted: '#FFFFFF',
  primary: '#0BA97C',
  primarySoft: 'rgba(11, 169, 124, 0.12)',
  primaryText: '#FFFFFF',
  accent: '#4F5BD5',
  accentSoft: 'rgba(79, 91, 213, 0.12)',
  positive: '#0F9D58',
  positiveSoft: 'rgba(15, 157, 88, 0.12)',
  negative: '#DC2E4F',
  negativeSoft: 'rgba(220, 46, 79, 0.12)',
  warning: '#B45309',
  warningSoft: 'rgba(180, 83, 9, 0.12)',
  info: '#0284C7',
  infoSoft: 'rgba(2, 132, 199, 0.12)',
  neutral: '#64748B',
  neutralSoft: 'rgba(100, 116, 139, 0.12)',
  overlay: 'rgba(15, 23, 42, 0.45)',
  tabBar: 'rgba(255, 255, 255, 0.97)',
  chartGrid: 'rgba(100, 116, 139, 0.16)',
};

export const darkTheme: Theme = {
  scheme: 'dark',
  colors: darkColors,
  spacing: (steps: number) => steps * BASE_UNIT,
  radius,
  typography,
  shadow: {
    card: {
      shadowColor: '#000000',
      shadowOffset: { width: 0, height: 6 },
      shadowOpacity: 0.3,
      shadowRadius: 16,
      elevation: 4,
    },
    floating: {
      shadowColor: '#000000',
      shadowOffset: { width: 0, height: 10 },
      shadowOpacity: 0.4,
      shadowRadius: 24,
      elevation: 10,
    },
  },
};

export const lightTheme: Theme = {
  ...darkTheme,
  scheme: 'light',
  colors: lightColors,
  shadow: {
    card: {
      shadowColor: '#0F172A',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.07,
      shadowRadius: 14,
      elevation: 2,
    },
    floating: {
      shadowColor: '#0F172A',
      shadowOffset: { width: 0, height: 10 },
      shadowOpacity: 0.14,
      shadowRadius: 22,
      elevation: 8,
    },
  },
};

/** Colour for a profit figure: green above zero, red below, muted at break-even. */
export function profitColor(value: number, colors: ThemeColors): string {
  if (value > 0) return colors.positive;
  if (value < 0) return colors.negative;
  return colors.textMuted;
}
