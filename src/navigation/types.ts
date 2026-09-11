import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { BottomTabScreenProps } from '@react-navigation/bottom-tabs';
import type { CompositeScreenProps } from '@react-navigation/native';

import type { BetStatus } from '../domain/types';

export type RootStackParamList = {
  Tabs: undefined;
  BetForm: { betId?: string; duplicateOf?: string } | undefined;
  BetDetail: { betId: string };
  Settings: undefined;
};

export type TabParamList = {
  Dashboard: undefined;
  Bets:
    | {
        /** Pre-select a status filter, e.g. when tapping "open bets" on the dashboard. */
        presetStatus?: BetStatus;
        /** Bumped on every navigation so repeat taps re-apply the preset. */
        requestedAt?: number;
      }
    | undefined;
  Analytics: undefined;
  Bankroll: undefined;
  Tools: undefined;
};

export type RootStackScreenProps<T extends keyof RootStackParamList> = NativeStackScreenProps<
  RootStackParamList,
  T
>;

export type TabScreenProps<T extends keyof TabParamList> = CompositeScreenProps<
  BottomTabScreenProps<TabParamList, T>,
  NativeStackScreenProps<RootStackParamList>
>;

declare global {
  namespace ReactNavigation {
    // eslint-disable-next-line @typescript-eslint/no-empty-object-type
    interface RootParamList extends RootStackParamList {}
  }
}
