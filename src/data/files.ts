import * as DocumentPicker from 'expo-document-picker';
import { File, Paths } from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import { Platform } from 'react-native';

/**
 * Write `content` to a file in the cache directory and hand it to the OS share sheet.
 * Returns the file URI so callers can surface it if sharing is unavailable.
 */
export async function shareTextFile(
  filename: string,
  content: string,
  mimeType: string,
): Promise<{ uri: string; shared: boolean }> {
  const file = new File(Paths.cache, filename);

  if (file.exists) {
    file.delete();
  }
  file.create({ intermediates: true, overwrite: true });
  file.write(content);

  if (Platform.OS === 'web') {
    return { uri: file.uri, shared: false };
  }

  const canShare = await Sharing.isAvailableAsync();
  if (!canShare) {
    return { uri: file.uri, shared: false };
  }

  await Sharing.shareAsync(file.uri, {
    mimeType,
    dialogTitle: filename,
    UTI: mimeType === 'text/csv' ? 'public.comma-separated-values-text' : 'public.json',
  });

  return { uri: file.uri, shared: true };
}

export interface PickedFile {
  name: string;
  content: string;
}

/** Open the system file picker and read the selected text file. */
export async function pickTextFile(mimeTypes: string[]): Promise<PickedFile | null> {
  const result = await DocumentPicker.getDocumentAsync({
    type: mimeTypes,
    copyToCacheDirectory: true,
    multiple: false,
  });

  if (result.canceled) {
    return null;
  }

  const asset = result.assets?.[0];
  if (!asset) {
    return null;
  }

  const file = new File(asset.uri);
  const content = await file.text();
  return { name: asset.name, content };
}

export function timestampedFilename(base: string, extension: string): string {
  const now = new Date();
  const pad = (value: number) => `${value}`.padStart(2, '0');
  const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}`;
  return `${base}-${stamp}.${extension}`;
}
