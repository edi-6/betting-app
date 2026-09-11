import { fireEvent, render, screen, waitFor } from '@testing-library/react-native';
import React from 'react';

import { AppRoot } from '../AppRoot';
import { createMemoryStore } from '../data/storage';
import { createEmptyData } from '../domain/defaults';
import { createDemoData } from '../domain/demo';
import type { AppData, Bet } from '../domain/types';

/**
 * An empty ledger with the first-run notice already acknowledged. Every test except
 * the disclaimer suite itself starts here — otherwise the gate covers the app and the
 * assertions below would be checking a screen the user cannot actually see.
 */
function freshData() {
  const data = createEmptyData();
  data.settings.disclaimerAcceptedAt = '2026-01-01T00:00:00.000Z';
  return data;
}

const demo = createDemoData(new Date('2026-09-11T12:00:00.000Z'));

function renderApp(initialData: AppData = demo) {
  return render(<AppRoot store={createMemoryStore()} initialData={initialData} />);
}

async function waitForDashboard() {
  await waitFor(() => expect(screen.getByText('Recent activity')).toBeTruthy());
}

function openBet(overrides: Partial<Bet> = {}): Bet {
  return {
    id: 'bet-open',
    placedAt: '2026-09-10T12:00:00.000Z',
    createdAt: '2026-09-10T12:00:00.000Z',
    updatedAt: '2026-09-10T12:00:00.000Z',
    stake: 100,
    legs: [
      {
        id: 'leg-1',
        sport: 'Football',
        league: 'Premier League',
        event: 'Arsenal vs Spurs',
        market: '1X2',
        selection: 'Arsenal',
        odds: 2.5,
        status: 'pending',
      },
    ],
    bookmaker: 'Pinnacle',
    tags: [],
    isFreeBet: false,
    ...overrides,
  };
}

describe('settings', () => {
  it('opens from the dashboard and renders every section', async () => {
    renderApp();
    await waitForDashboard();

    fireEvent.press(screen.getByLabelText('Settings'));
    await waitFor(() => expect(screen.getByText('Preferences, limits and your data')).toBeTruthy());

    expect(screen.getByText('Display')).toBeTruthy();
    expect(screen.getByText('Betting defaults')).toBeTruthy();
    expect(screen.getByText('Limits')).toBeTruthy();
    expect(screen.getByText('Your data')).toBeTruthy();
    expect(screen.getByText('Gamble responsibly')).toBeTruthy();
  });

  it('changes the odds format and it takes effect across the app', async () => {
    renderApp(withSingleBet());
    await waitForDashboard();

    // Decimal by default.
    expect(screen.getAllByText('2.50').length).toBeGreaterThan(0);

    fireEvent.press(screen.getByLabelText('Settings'));
    await waitFor(() => expect(screen.getByText('Odds format')).toBeTruthy());
    fireEvent.press(screen.getByText('American'));
    fireEvent.press(screen.getByText('Done'));

    await waitForDashboard();
    expect(screen.getAllByText('+150').length).toBeGreaterThan(0);
  });

  it('toggles glass effects and reports what the device supports', async () => {
    renderApp();
    await waitForDashboard();

    fireEvent.press(screen.getByLabelText('Settings'));
    await waitFor(() => expect(screen.getByText('Glass effects')).toBeTruthy());

    // The caption names the material actually in use, so the setting is never a
    // silent no-op on a device that cannot render Liquid Glass.
    expect(
      screen.queryByText(/Liquid Glass|reduce transparency|Translucent tab bar/),
    ).toBeTruthy();

    fireEvent(screen.getByText('Glass effects'), 'press');
    fireEvent.press(screen.getByText('Done'));
    await waitForDashboard();
  });

  it('switches to the light theme without crashing', async () => {
    renderApp();
    await waitForDashboard();

    fireEvent.press(screen.getByLabelText('Settings'));
    await waitFor(() => expect(screen.getByText('Appearance')).toBeTruthy());
    fireEvent.press(screen.getByText('Light'));
    fireEvent.press(screen.getByText('Dark'));
    fireEvent.press(screen.getByText('System'));
    expect(screen.getByText('Appearance')).toBeTruthy();
  });

  it('changes the currency', async () => {
    renderApp(withSingleBet());
    await waitForDashboard();

    fireEvent.press(screen.getByLabelText('Settings'));
    await waitFor(() => expect(screen.getByText('Display')).toBeTruthy());

    fireEvent.press(screen.getByLabelText('Currency: USD · $'));
    await waitFor(() => expect(screen.getByText('Account currency')).toBeTruthy());
    fireEvent.press(screen.getByText('EUR · €'));

    fireEvent.press(screen.getByText('Done'));
    await waitForDashboard();
    expect(screen.getAllByText(/€/).length).toBeGreaterThan(0);
  });
});

describe('bets list', () => {
  it('filters by search text', async () => {
    renderApp(demo);
    await waitForDashboard();
    fireEvent.press(screen.getByTestId('tab-Bets'));

    const search = await screen.findByPlaceholderText('Search team, market, tag…');
    fireEvent.changeText(search, 'zzzzz-no-such-team');

    await waitFor(() => expect(screen.getByText('Nothing matches those filters')).toBeTruthy());

    fireEvent.press(screen.getByText('Clear filters'));
    await waitFor(() => expect(screen.queryByText('Nothing matches those filters')).toBeNull());
  });

  it('filters by status chip', async () => {
    renderApp(demo);
    await waitForDashboard();
    fireEvent.press(screen.getByTestId('tab-Bets'));
    await screen.findByPlaceholderText('Search team, market, tag…');

    // The first "Pending" on screen is the quick-filter chip.
    fireEvent.press(screen.getAllByText('Pending')[0]!);
    await waitFor(() =>
      expect(screen.getByLabelText('Filters, 1 active')).toBeTruthy(),
    );
  });

  it('opens the filter sheet and resets it', async () => {
    renderApp(demo);
    await waitForDashboard();
    fireEvent.press(screen.getByTestId('tab-Bets'));
    await screen.findByPlaceholderText('Search team, market, tag…');

    fireEvent.press(screen.getByLabelText(/^Filters,/));
    await waitFor(() => expect(screen.getByText('Filter & sort')).toBeTruthy());

    fireEvent.press(screen.getByText('Biggest stake'));
    fireEvent.press(screen.getByText('Reset'));
    fireEvent.press(screen.getByText('Show results'));
    await waitFor(() => expect(screen.queryByText('Filter & sort')).toBeNull());
  });

  it('jumps to the pending filter from the dashboard shortcut', async () => {
    renderApp(withSingleBet());
    await waitForDashboard();

    fireEvent.press(screen.getByText('View all'));
    await waitFor(() =>
      expect(screen.getByPlaceholderText('Search team, market, tag…')).toBeTruthy(),
    );
    // The status preset arrived with the navigation, so one filter is active.
    await waitFor(() => expect(screen.getByLabelText('Filters, 1 active')).toBeTruthy());
  });
});

describe('filter ranges', () => {
  async function openFilters() {
    renderApp(demo);
    await waitForDashboard();
    fireEvent.press(screen.getByTestId('tab-Bets'));
    await screen.findByPlaceholderText('Search team, market, tag…');
    fireEvent.press(screen.getByLabelText(/^Filters,/));
    await waitFor(() => expect(screen.getByText('Filter & sort')).toBeTruthy());
  }

  it('narrows the ledger by odds range', async () => {
    await openFilters();

    // A floor no slip can meet must empty the list — proof the filter is wired
    // through to `applyFilter` and not just held in component state.
    const [minOdds] = screen.getAllByPlaceholderText('Min');
    fireEvent.changeText(minOdds!, '9999');
    fireEvent.press(screen.getByText('Show results'));

    await waitFor(() => expect(screen.getByText('Nothing matches those filters')).toBeTruthy());

    fireEvent.press(screen.getByText('Clear filters'));
    await waitFor(() => expect(screen.queryByText('Nothing matches those filters')).toBeNull());
  });

  it('narrows the ledger by stake range', async () => {
    await openFilters();

    // Two "Max" inputs in the sheet: odds first, then stake.
    const maxFields = screen.getAllByPlaceholderText('Max');
    fireEvent.changeText(maxFields[1]!, '0.01');
    fireEvent.press(screen.getByText('Show results'));

    await waitFor(() => expect(screen.getByText('Nothing matches those filters')).toBeTruthy());
  });

  it('accepts a custom date window', async () => {
    await openFilters();

    fireEvent.press(screen.getByText('Custom'));
    await waitFor(() => expect(screen.getByText('From')).toBeTruthy());
    expect(screen.getByText('To')).toBeTruthy();
    expect(screen.getByLabelText('Filters, 1 active')).toBeTruthy();
  });

  it('clears the range inputs on reset', async () => {
    await openFilters();

    const [minOdds] = screen.getAllByPlaceholderText('Min');
    fireEvent.changeText(minOdds!, '5');
    await waitFor(() => expect(screen.getByLabelText('Filters, 1 active')).toBeTruthy());

    fireEvent.press(screen.getByText('Reset'));
    await waitFor(() => expect(screen.getByLabelText('Filters, 0 active')).toBeTruthy());
    expect(screen.getAllByPlaceholderText('Min')[0]!.props.value).toBe('');
  });
});

describe('building a parlay', () => {
  it('adds a second leg and multiplies the odds', async () => {
    renderApp(freshData());
    await waitFor(() => expect(screen.getByText('Add your first bet')).toBeTruthy());
    fireEvent.press(screen.getByText('Add your first bet'));
    await waitFor(() => expect(screen.getByText('New bet')).toBeTruthy());

    fireEvent.press(screen.getByText('Parlay'));
    await waitFor(() => expect(screen.getByText('Leg 2')).toBeTruthy());

    const oddsFields = screen.getAllByPlaceholderText('1.91');
    expect(oddsFields).toHaveLength(2);
    fireEvent.changeText(oddsFields[0]!, '2.00');
    fireEvent.changeText(oddsFields[1]!, '3.00');
    fireEvent.changeText(screen.getByPlaceholderText('25'), '50');

    await waitFor(() => expect(screen.getByText('6.00')).toBeTruthy());
    expect(screen.getByText('$300.00')).toBeTruthy();
    expect(screen.getByText('+$250.00')).toBeTruthy();

    // Dropping back to a single keeps only the first leg.
    fireEvent.press(screen.getByText('Single'));
    await waitFor(() => expect(screen.queryByText('Leg 2')).toBeNull());
    expect(screen.getByText('2.00')).toBeTruthy();
  });
});

describe('bet detail', () => {
  it('shows closing line value when a closing price is recorded', async () => {
    const data = freshData();
    data.bets = [
      openBet({
        legs: [
          {
            id: 'leg-1',
            sport: 'Football',
            league: 'Premier League',
            event: 'Arsenal vs Spurs',
            market: '1X2',
            selection: 'Arsenal',
            odds: 2.5,
            closingOdds: 2.2,
            status: 'won',
          },
        ],
        settledAt: '2026-09-10T20:00:00.000Z',
      }),
    ];

    renderApp(data);
    await waitForDashboard();

    fireEvent.press(screen.getAllByLabelText(/^Arsenal, won$/)[0]!);
    await waitFor(() => expect(screen.getByText('Closing line value')).toBeTruthy());
    expect(screen.getByText('+13.64%')).toBeTruthy();
  });

  it('records a cash out and lets you undo it', async () => {
    const data = freshData();
    data.bets = [openBet()];

    renderApp(data);
    await waitForDashboard();

    fireEvent.press(screen.getAllByLabelText(/^Arsenal, pending$/)[0]!);
    await waitFor(() => expect(screen.getByText('Grade this bet')).toBeTruthy());

    fireEvent.press(screen.getByText('Cash out'));
    await waitFor(() => expect(screen.getByText('Record what the bookmaker actually paid you')).toBeTruthy());

    fireEvent.changeText(screen.getByPlaceholderText('250.00'), '160');
    fireEvent.press(screen.getByText('Save cash out'));

    await waitFor(() => expect(screen.getByText('Settled by cash out')).toBeTruthy());
    expect(screen.getByText('+$60.00')).toBeTruthy();

    fireEvent.press(screen.getByText('Undo cash out'));
    await waitFor(() => expect(screen.getByText('Grade this bet')).toBeTruthy());
  });

  it('grades individual legs of a parlay', async () => {
    const data = freshData();
    data.bets = [
      openBet({
        legs: [
          {
            id: 'leg-1',
            sport: 'Football',
            league: 'Premier League',
            event: 'Arsenal vs Spurs',
            market: '1X2',
            selection: 'Arsenal',
            odds: 2,
            status: 'pending',
          },
          {
            id: 'leg-2',
            sport: 'Basketball',
            league: 'NBA',
            event: 'Lakers vs Celtics',
            market: 'Moneyline',
            selection: 'Lakers',
            odds: 3,
            status: 'pending',
          },
        ],
      }),
    ];

    renderApp(data);
    await waitForDashboard();

    fireEvent.press(screen.getAllByLabelText(/2-leg parlay/)[0]!);
    await waitFor(() => expect(screen.getByText('Selections')).toBeTruthy());

    // Grading one leg leaves the slip open.
    const legRows = screen.getAllByText('Won');
    fireEvent.press(legRows[legRows.length - 2]!);
    await waitFor(() => expect(screen.getByText('Grade this bet')).toBeTruthy());

    // Grading the second settles it at the combined price.
    fireEvent.press(screen.getByTestId('grade-won'));
    await waitFor(() => expect(screen.getByText('Profit / loss')).toBeTruthy());
    expect(screen.getByText('+$500.00')).toBeTruthy();
  });

  it('edits a bet and persists the change', async () => {
    const data = freshData();
    data.bets = [openBet()];

    renderApp(data);
    await waitForDashboard();

    fireEvent.press(screen.getAllByLabelText(/^Arsenal, pending$/)[0]!);
    await waitFor(() => expect(screen.getByLabelText('Edit bet')).toBeTruthy());
    fireEvent.press(screen.getByLabelText('Edit bet'));

    await waitFor(() => expect(screen.getByText('Edit bet')).toBeTruthy());
    fireEvent.changeText(screen.getByDisplayValue('100'), '250');
    fireEvent.press(screen.getByTestId('save-bet'));

    await waitFor(() => expect(screen.getByText('At risk')).toBeTruthy());
    expect(screen.getAllByText('$250.00').length).toBeGreaterThan(0);
  });
});

describe('bankroll transactions', () => {
  it('adds a deposit and updates the balance', async () => {
    const data = freshData();
    data.settings.startingBankroll = 1000;

    renderApp(data);
    await waitFor(() => expect(screen.getByText("Let's get your ledger started")).toBeTruthy());

    fireEvent.press(screen.getByTestId('tab-Bankroll'));
    await waitFor(() => expect(screen.getByText('Current balance')).toBeTruthy());
    expect(screen.getAllByText('$1,000.00').length).toBeGreaterThan(0);

    fireEvent.press(screen.getByTestId('bankroll-fab'));
    await waitFor(() => expect(screen.getByText('Record a transaction')).toBeTruthy());

    fireEvent.changeText(screen.getByPlaceholderText('100'), '500');
    fireEvent.press(screen.getByTestId('save-transaction'));

    await waitFor(() => expect(screen.getByText('Transactions (1)')).toBeTruthy());
    expect(screen.getAllByText('$1,500.00').length).toBeGreaterThan(0);
  });
});

describe('editing a transaction', () => {
  it('corrects an amount without deleting and re-adding', async () => {
    const data = freshData();
    data.settings.startingBankroll = 1000;

    renderApp(data);
    await waitFor(() => expect(screen.getByText("Let's get your ledger started")).toBeTruthy());
    fireEvent.press(screen.getByTestId('tab-Bankroll'));
    await waitFor(() => expect(screen.getByText('Current balance')).toBeTruthy());

    // Add a deposit with the wrong amount.
    fireEvent.press(screen.getByTestId('bankroll-fab'));
    await waitFor(() => expect(screen.getByText('Record a transaction')).toBeTruthy());
    fireEvent.changeText(screen.getByPlaceholderText('100'), '500');
    fireEvent.press(screen.getByTestId('save-transaction'));
    await waitFor(() => expect(screen.getByText('Transactions (1)')).toBeTruthy());
    expect(screen.getAllByText('$1,500.00').length).toBeGreaterThan(0);

    // Tap the row to correct it.
    fireEvent.press(screen.getByLabelText(/^Edit deposit of/));
    await waitFor(() => expect(screen.getByText('Edit transaction')).toBeTruthy());
    fireEvent.changeText(screen.getByDisplayValue('500'), '250');
    fireEvent.press(screen.getByTestId('save-transaction'));

    await waitFor(() => expect(screen.getAllByText('$1,250.00').length).toBeGreaterThan(0));
    // Still a single transaction — it was edited, not duplicated.
    expect(screen.getByText('Transactions (1)')).toBeTruthy();
  });
});

describe('responsible gambling limits', () => {
  it('warns on the dashboard once a limit is breached', async () => {
    const today = new Date();
    const data = freshData();
    data.settings.startingBankroll = 1000;
    data.settings.limits = { dailyStakeLimit: 50 };
    data.bets = [
      openBet({ id: 'today-1', placedAt: today.toISOString(), stake: 200 }),
    ];

    renderApp(data);
    await waitForDashboard();

    expect(screen.getByText('Daily stake limit reached')).toBeTruthy();
  });
});

function withSingleBet(): AppData {
  const data = freshData();
  data.bets = [openBet()];
  return data;
}
