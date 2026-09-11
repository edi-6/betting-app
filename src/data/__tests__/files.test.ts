import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';

import { pickTextFile, shareTextFile, timestampedFilename } from '../files';

const virtualFiles = (FileSystem as unknown as { __virtualFiles: Map<string, string> })
  .__virtualFiles;

describe('shareTextFile', () => {
  beforeEach(() => {
    virtualFiles.clear();
    jest.clearAllMocks();
  });

  it('writes the content and hands the file to the share sheet', async () => {
    const result = await shareTextFile('bets.csv', 'id,stake\n1,10', 'text/csv');

    expect(result.uri).toBe('file:///cache/bets.csv');
    expect(result.shared).toBe(true);
    expect(virtualFiles.get('file:///cache/bets.csv')).toBe('id,stake\n1,10');
    expect(Sharing.shareAsync).toHaveBeenCalledWith(
      'file:///cache/bets.csv',
      expect.objectContaining({ mimeType: 'text/csv' }),
    );
  });

  it('overwrites a stale file from an earlier export', async () => {
    virtualFiles.set('file:///cache/bets.csv', 'old content');
    await shareTextFile('bets.csv', 'new content', 'text/csv');
    expect(virtualFiles.get('file:///cache/bets.csv')).toBe('new content');
  });

  it('still returns the path when sharing is unavailable', async () => {
    jest.mocked(Sharing.isAvailableAsync).mockResolvedValueOnce(false);
    const result = await shareTextFile('backup.json', '{}', 'application/json');

    expect(result.shared).toBe(false);
    expect(result.uri).toBe('file:///cache/backup.json');
    expect(Sharing.shareAsync).not.toHaveBeenCalled();
  });
});

describe('pickTextFile', () => {
  beforeEach(() => {
    virtualFiles.clear();
    jest.clearAllMocks();
  });

  it('returns null when the user cancels', async () => {
    jest.mocked(DocumentPicker.getDocumentAsync).mockResolvedValueOnce({
      canceled: true,
      assets: null,
    });
    await expect(pickTextFile(['text/csv'])).resolves.toBeNull();
  });

  it('reads the selected file', async () => {
    virtualFiles.set('file:///picked/import.csv', 'id,stake\nabc,10');
    jest.mocked(DocumentPicker.getDocumentAsync).mockResolvedValueOnce({
      canceled: false,
      assets: [
        {
          uri: 'file:///picked/import.csv',
          name: 'import.csv',
          mimeType: 'text/csv',
          size: 20,
          lastModified: 0,
        },
      ],
    });

    await expect(pickTextFile(['text/csv'])).resolves.toEqual({
      name: 'import.csv',
      content: 'id,stake\nabc,10',
    });
  });
});

describe('timestampedFilename', () => {
  it('builds a sortable, unique-ish filename', () => {
    const name = timestampedFilename('betledger-bets', 'csv');
    expect(name).toMatch(/^betledger-bets-\d{8}-\d{4}\.csv$/);
  });
});
