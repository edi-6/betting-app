/**
 * Monotonic, collision-resistant identifier generator.
 *
 * Avoids a `uuid` dependency (and its `crypto.getRandomValues` polyfill requirements)
 * while still being safe for a local-first, single-device dataset: the timestamp prefix
 * keeps ids sortable and a per-process counter removes same-millisecond collisions.
 */
let counter = 0;

export function createId(prefix = 'id'): string {
  counter = (counter + 1) % 0xffffff;
  const time = Date.now().toString(36);
  const seq = counter.toString(36).padStart(4, '0');
  const random = Math.floor(Math.random() * 0xffffff)
    .toString(36)
    .padStart(4, '0');
  return `${prefix}_${time}${seq}${random}`;
}
