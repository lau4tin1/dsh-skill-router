/**
 * src/similarity.ts — vector math: normalization and cosine similarity.
 *
 * The single, canonical home for the two numeric operations the router uses:
 *
 *   - normalize(v) : scale a vector to unit length (L2 norm = 1).
 *   - cosine(a, b) : similarity in [-1, 1]; 1 = same direction, 0 = orthogonal,
 *                    -1 = opposite direction.
 *
 * Both are PURE (no I/O, no DSH, no dependencies) so they stay trivial to test.
 * embedding.ts now reuses normalize() from here instead of keeping its own
 * copy, so there is exactly one source of truth for this math.
 *
 * Design reference: the "score" step in DESIGN.md §3 and §6.
 */

export type Vector = number[];

/**
 * L2-normalize a vector to unit length. Returns a NEW array; the input is
 * never mutated. An all-zero vector has no direction, so it is passed through
 * unchanged (its cosine against anything is 0, handled by cosine()).
 */
export function normalize(vector: readonly number[]): Vector {
  let sum = 0;
  for (const value of vector) sum += value * value;
  const norm = Math.sqrt(sum);
  if (norm === 0) return [...vector];
  return vector.map((value) => value / norm);
}

/**
 * Cosine similarity between two same-length vectors.
 *
 * Computes the FULL cosine — dot product divided by the product of the two
 * magnitudes — so it is correct for both normalized and unnormalized inputs.
 * When both vectors are already unit-length, this simplifies to the dot
 * product (which is why the local embedding backend can rely on it).
 */
export function cosine(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length) {
    throw new Error(`cosine: dimension mismatch (${a.length} vs ${b.length})`);
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dot / denominator;
}

// ---------------------------------------------------------------------------
// Demo — runs only when executed directly:  node src/similarity.ts
// ---------------------------------------------------------------------------

function main(): void {
  const v = normalize([3, 4]);
  console.log(`normalize([3, 4])   = [${v.join(', ')}]`);
  console.log(`  resulting length  = ${Math.hypot(...v).toFixed(3)}   (unit length)`);

  console.log('\ncosine:');
  console.log(`  [1,0] vs [1,0]   = ${cosine([1, 0], [1, 0])}   (same direction)`);
  console.log(`  [1,0] vs [0,1]   = ${cosine([1, 0], [0, 1])}   (orthogonal)`);
  // These two are NOT normalized; full cosine still gives the right answer.
  console.log(`  [3,4] vs [0,5]   = ${cosine([3, 4], [0, 5]).toFixed(3)}   (unnormalized inputs, full cosine)`);
}

import { pathToFileURL } from 'node:url';
if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  main();
}
