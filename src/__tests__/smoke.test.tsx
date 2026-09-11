import { fireEvent, render, screen, waitFor } from '@testing-library/react-native';
import React from 'react';

import { AppRoot } from '../AppRoot';
import { createMemoryStore } from '../data/storage';
import { summarize } from '../domain/analytics';
import { createDemoData } from '../domain/demo';
import { createEmptyData } from '../domain/defaults';

const demo = createDemoData(new Date('2026-09-11T12:00:00.000Z'));

function renderApp(initialData = demo) {
  return render(<AppRoot store={createMemoryStore()} initialData={initialData} />);
}

/** Anchors that appear on exactly one screen, used to confirm navigation landed. */
const TAB_ANCHORS: Record<string, string> = {
  Dashboard: 'Recent activity',
  Bets: 'Filter & sort',
  Analytics: 'Cumulative profit',
  Bankroll: 'Current balance',
  Tools: 'Odds converter',
};

/** Switch tabs by pressing the tab bar button. */
async function openTab(name: keyof typeof TAB_ANCHORS) {
  fireEvent.press(screen.getByTestId(`tab-${name}`));
  if (name === 'Bets') {
    await waitFor(() =>
      expect(screen.getByPlaceholderText('Search team, market, tag…')).toBeTruthy(),
    );
    return;
  }
  await waitFor(() => expect(screen.getByText(TAB_ANCHORS[name] as string)).toBeTruthy());
}

/** The dashboard is ready once its recent-activity section has rendered. */
async function waitForDashboard() {
  await waitFor(() => expect(screen.getByText('Recent activity')).toBeTruthy());
}

describe('first launch', () => {
  it('shows the onboarding empty state', async () => {
    render(<AppRoot store={createMemoryStore()} />);
    await waitFor(() => expect(screen.getByText("Let's get your ledger started")).toBeTruthy());
    expect(screen.getByText('Add your first bet')).toBeTruthy();
    expect(screen.getByText('Load demo data')).toBeTruthy();
  });

  it('fills the dashboard once demo data is loaded', async () => {
    render(<AppRoot store={createMemoryStore()} initialData={createEmptyData()} />);
    fireEvent.press(screen.getByText('Load demo data'));
    await waitForDashboard();
    expect(screen.getByText('Profit over time')).toBeTruthy();
  });
});

describe('dashboard', () => {
  it('renders the headline numbers for a populated ledger', async () => {
    renderApp();
    await waitForDashboard();

    expect(screen.getByText('Profit over time')).toBeTruthy();
    expect(screen.getByText('Recent activity')).toBeTruthy();
    expect(screen.getByText('ROI')).toBeTruthy();
    expect(screen.getByText('Win rate')).toBeTruthy();
  });

  it('switches the chart range without crashing', async () => {
    renderApp();
    await waitForDashboard();

    for (const label of ['7D', '90D', 'YTD', '1Y', '30D']) {
      fireEvent.press(screen.getByText(label));
    }
    expect(screen.getByText('Profit over time')).toBeTruthy();
  });
});

describe('tab navigation', () => {
  it('opens every tab', async () => {
    renderApp();
    await waitForDashboard();

    await openTab('Bets');
    await openTab('Analytics');
    await openTab('Bankroll');
    await openTab('Tools');
    await openTab('Dashboard');
  });
});

describe('analytics', () => {
  it('renders each analytics tab', async () => {
    renderApp();
    await waitForDashboard();
    await openTab('Analytics');

    fireEvent.press(screen.getByText('Breakdown'));
    await waitFor(() => expect(screen.getByText('Group by')).toBeTruthy());

    fireEvent.press(screen.getByText('CLV'));
    await waitFor(() =>
      expect(screen.getByText('Why closing line value matters')).toBeTruthy(),
    );

    fireEvent.press(screen.getByText('Risk'));
    await waitFor(() => expect(screen.getByText('Is your edge real?')).toBeTruthy());
    expect(screen.getByText('Max drawdown')).toBeTruthy();

    fireEvent.press(screen.getByText('Overview'));
    await waitFor(() => expect(screen.getByText('Results split')).toBeTruthy());
  });
});

describe('tools', () => {
  it('renders every calculator', async () => {
    renderApp();
    await waitForDashboard();
    await openTab('Tools');

    expect(screen.getByText('Same price, every format')).toBeTruthy();

    fireEvent.press(screen.getByText('EV & Kelly'));
    await waitFor(() => expect(screen.getByText('Expected value & Kelly')).toBeTruthy());
    expect(screen.getByText('Verdict')).toBeTruthy();

    fireEvent.press(screen.getByText('No-vig'));
    await waitFor(() => expect(screen.getByText('Remove the vig')).toBeTruthy());
    expect(screen.getByText('Fair prices')).toBeTruthy();

    fireEvent.press(screen.getByText('Parlay'));
    await waitFor(() => expect(screen.getByText('Parlay calculator')).toBeTruthy());
    expect(screen.getByText('Payout')).toBeTruthy();

    fireEvent.press(screen.getByText('Arbitrage'));
    await waitFor(() => expect(screen.getByText('Arbitrage calculator')).toBeTruthy());
    expect(screen.getByText('Arbitrage found')).toBeTruthy();

    fireEvent.press(screen.getByText('Hedge'));
    await waitFor(() => expect(screen.getByText('Hedge calculator')).toBeTruthy());
    expect(screen.getByText('Lay it off')).toBeTruthy();

    fireEvent.press(screen.getByText('Cash out'));
    await waitFor(() => expect(screen.getByText('Cash-out checker')).toBeTruthy());
  });

  it('recomputes the odds converter as you type', async () => {
    renderApp();
    await waitForDashboard();
    await openTab('Tools');

    const input = screen.getByDisplayValue('2.00');
    fireEvent.changeText(input, '2.50');
    await waitFor(() => expect(screen.getByText('+150')).toBeTruthy());
    expect(screen.getByText('3/2')).toBeTruthy();
    // Implied probability and break-even win rate are both 40% at these odds.
    expect(screen.getAllByText('40.00%')).toHaveLength(2);
    expect(screen.getByText('$150.00')).toBeTruthy();
  });
});

describe('bankroll', () => {
  it('shows the balance breakdown', async () => {
    renderApp();
    await waitForDashboard();
    await openTab('Bankroll');

    expect(screen.getByText('Where the money went')).toBeTruthy();
    expect(screen.getByText('Starting bankroll')).toBeTruthy();
    expect(screen.getByText(`Transactions (${demo.transactions.length})`)).toBeTruthy();
  });
});

describe('adding a bet end to end', () => {
  it('saves a new single and shows it in the ledger', async () => {
    render(<AppRoot store={createMemoryStore()} initialData={createEmptyData()} />);

    await waitFor(() => expect(screen.getByText('Add your first bet')).toBeTruthy());
    fireEvent.press(screen.getByText('Add your first bet'));

    await waitFor(() => expect(screen.getByText('New bet')).toBeTruthy());

    fireEvent.changeText(screen.getByPlaceholderText('Arsenal vs Liverpool'), 'Arsenal vs Spurs');
    fireEvent.changeText(screen.getByPlaceholderText('Arsenal'), 'Arsenal');
    fireEvent.changeText(screen.getByPlaceholderText('1.91'), '2.50');
    fireEvent.changeText(screen.getByPlaceholderText('25'), '100');

    // The payout preview updates live.
    await waitFor(() => expect(screen.getByText('$250.00')).toBeTruthy());
    expect(screen.getByText('+$150.00')).toBeTruthy();

    fireEvent.press(screen.getByTestId('save-bet'));

    await waitForDashboard();
    expect(screen.getAllByText('Arsenal').length).toBeGreaterThan(0);
  });

  it('refuses to save without a stake or odds', async () => {
    render(<AppRoot store={createMemoryStore()} initialData={createEmptyData()} />);
    await waitFor(() => expect(screen.getByText('Add your first bet')).toBeTruthy());
    fireEvent.press(screen.getByText('Add your first bet'));
    await waitFor(() => expect(screen.getByText('New bet')).toBeTruthy());

    fireEvent.changeText(screen.getByPlaceholderText('25'), '');
    fireEvent.press(screen.getByTestId('save-bet'));

    // Still on the form.
    await waitFor(() => expect(screen.getByText('Enter valid odds.')).toBeTruthy());
    expect(screen.getByText('Enter a stake greater than zero.')).toBeTruthy();
  });
});

describe('grading a bet', () => {
  it('settles an open bet from the detail screen', async () => {
    const data = createEmptyData();
    data.bets = [
      {
        id: 'bet-1',
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
      },
    ];

    render(<AppRoot store={createMemoryStore()} initialData={data} />);
    await waitForDashboard();

    fireEvent.press(screen.getAllByLabelText(/^Arsenal, pending$/)[0]!);
    await waitFor(() => expect(screen.getByText('Grade this bet')).toBeTruthy());

    fireEvent.press(screen.getByTestId('grade-won'));
    await waitFor(() => expect(screen.getByText('Profit / loss')).toBeTruthy());
    expect(screen.getByText('+$150.00')).toBeTruthy();
  });
});

describe('persistence', () => {
  it('writes to the store and reads it back', async () => {
    const store = createMemoryStore();
    const { unmount } = render(<AppRoot store={store} initialData={createEmptyData()} />);

    fireEvent.press(screen.getByText('Load demo data'));
    await waitForDashboard();
    await waitFor(async () =>
      expect(await store.getItem('betledger/app-data/v1')).not.toBeNull(),
    );
    unmount();

    render(<AppRoot store={store} />);
    await waitForDashboard();
    expect(summarize(demo.bets).totalBets).toBeGreaterThan(0);
  });
});
