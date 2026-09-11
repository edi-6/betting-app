jest.mock('react-native-safe-area-context', () =>
  require('react-native-safe-area-context/jest/mock').default,
);

jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock'),
);

jest.mock('expo-haptics', () => ({
  impactAsync: jest.fn(async () => undefined),
  notificationAsync: jest.fn(async () => undefined),
  selectionAsync: jest.fn(async () => undefined),
  ImpactFeedbackStyle: { Light: 'light', Medium: 'medium', Heavy: 'heavy' },
  NotificationFeedbackType: { Success: 'success', Warning: 'warning', Error: 'error' },
}));

jest.mock('expo-sharing', () => ({
  isAvailableAsync: jest.fn(async () => true),
  shareAsync: jest.fn(async () => undefined),
}));

// An in-memory stand-in for the native filesystem so export/import can be tested for real.
const mockVirtualFiles = new Map();

jest.mock('expo-file-system', () => {
  class File {
    constructor(...segments) {
      this.uri = segments
        .map((segment) => (typeof segment === 'string' ? segment : segment.uri))
        .join('');
    }
    get exists() {
      return mockVirtualFiles.has(this.uri);
    }
    create() {
      mockVirtualFiles.set(this.uri, '');
    }
    write(content) {
      mockVirtualFiles.set(this.uri, String(content));
    }
    async text() {
      return mockVirtualFiles.get(this.uri) ?? '';
    }
    textSync() {
      return mockVirtualFiles.get(this.uri) ?? '';
    }
    delete() {
      mockVirtualFiles.delete(this.uri);
    }
  }

  return {
    File,
    Paths: { cache: { uri: 'file:///cache/' }, document: { uri: 'file:///documents/' } },
    __virtualFiles: mockVirtualFiles,
  };
});

jest.mock('expo-document-picker', () => ({
  getDocumentAsync: jest.fn(async () => ({ canceled: true, assets: null })),
}));

jest.mock('expo-system-ui', () => ({
  setBackgroundColorAsync: jest.fn(async () => undefined),
}));

// Silence the animated helper warning that RN emits in the test environment.
jest.spyOn(console, 'warn').mockImplementation((...args) => {
  const message = String(args[0] ?? '');
  if (message.includes('useNativeDriver') || message.includes('EventEmitter.removeListener')) {
    return;
  }
  console.info(...args);
});
