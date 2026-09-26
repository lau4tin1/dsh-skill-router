/**
 * src/similarity.ts — vector math: cosine similarity and vector helpers.
 *
 * The single, canonical home for the numeric operations the router uses:
 *
 *   - cosine(a, b) : similarity in [-1, 1]; 1 = same direction, 0 = orthogonal,
 *                    -1 = opposite direction.
 *   - meanVector(vs) : component-wise mean of several vectors.
 *   - subtract(a, b) : component-wise subtraction.
 *
 * These are PURE (no I/O, no DSH, no dependencies) so they stay trivial to test.
 *
 * Design reference: the "score" step in DESIGN.md §3 and §6.
 */

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

/** Component-wise mean of several same-length vectors (undefined for none). */
export function meanVector(vectors: readonly (readonly number[])[]): number[] | undefined {
  if (vectors.length === 0) return undefined;
  const dim = vectors[0].length;
  const mean = new Array<number>(dim).fill(0);
  for (const vector of vectors) {
    for (let i = 0; i < dim; i++) mean[i] += vector[i];
  }
  for (let i = 0; i < dim; i++) mean[i] /= vectors.length;
  return mean;
}

/** Component-wise subtraction: a - b (new array). */
export function subtract(a: readonly number[], b: readonly number[]): number[] {
  return a.map((value, i) => value - b[i]);
}

// ---------------------------------------------------------------------------
// Demo — runs only when executed directly:  node src/similarity.ts
// ---------------------------------------------------------------------------

function main(): void {
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
