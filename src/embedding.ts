/**
 * src/embedding.ts — the embedding backend.
 *
 * Turns text into fixed-length numeric vectors so we can measure similarity
 * (close vectors = close meaning). This module owns ONLY the "text -> vector"
 * step; it knows nothing about skills, the registry, or injection.
 *
 * Two interchangeable implementations sit behind one interface:
 *
 *   - LocalEmbeddingBackend : a deterministic, zero-dependency "hashing
 *     vectorizer". No network, no API key. It measures *lexical* similarity
 *     (shared words). We use it for tests and for the first end-to-end demo,
 *     then swap in the HTTP backend for real semantic embeddings.
 *
 *   - HttpEmbeddingBackend  : an OpenAI-compatible `POST /v1/embeddings`
 *     client for real semantic embeddings.
 *
 * Design reference: DESIGN.md §7 ("Embedding backend").
 */

import { normalize } from './similarity.ts';

export type Embedding = number[];

/**
 * The single contract every embedding source satisfies. The rest of the plugin
 * only ever talks to this interface, so swapping "local" for "http" is a
 * config change, not a code change.
 */
export interface EmbeddingBackend {
  /** Human-readable label for logs. */
  readonly name: string;
  /**
   * Vector length. Known up front for local; for http it is taken from config
   * when set, otherwise inferred from the first response.
   */
  readonly dimensions: number | undefined;
  /** Embed a batch of texts. Returns one vector per text, in the same order. */
  embed(texts: readonly string[]): Promise<Embedding[]>;
}

// ---------------------------------------------------------------------------
// Configuration shapes (imported later by src/config.ts)
// ---------------------------------------------------------------------------

export interface LocalEmbeddingConfig {
  provider: 'local';
  /** Vector length. Bigger = fewer hash collisions, more memory. Default 384. */
  dimensions?: number;
}

export interface HttpEmbeddingConfig {
  provider: 'http';
  /** Endpoint base, e.g. 'https://api.openai.com' (trailing slash tolerated). */
  baseURL: string;
  /** Model id, e.g. 'text-embedding-3-small'. */
  model: string;
  /** Name of the environment variable holding the API key. */
  apiKeyEnv: string;
  /** Optional; validated against the response when set. */
  dimensions?: number;
}

export type EmbeddingConfig = LocalEmbeddingConfig | HttpEmbeddingConfig;

/** Pick a backend from config. */
export function createEmbeddingBackend(config: EmbeddingConfig): EmbeddingBackend {
  if (config.provider === 'local') {
    return new LocalEmbeddingBackend(config.dimensions ?? DEFAULT_LOCAL_DIMENSIONS);
  }
  return new HttpEmbeddingBackend(config);
}

// ---------------------------------------------------------------------------
// Local: hashing vectorizer
// ---------------------------------------------------------------------------

const DEFAULT_LOCAL_DIMENSIONS = 4096;

/**
 * A deterministic, dependency-free vectorizer using the "hashing trick",
 * tuned to be a usable offline stand-in:
 *
 *   - each word is hashed into one of `dimensions` buckets with a SIGN, so
 *     accidental collisions from different words partially cancel out;
 *   - bucket weights are sublinear (1 + log(count)), so a word repeated many
 *     times in one text cannot dominate;
 *   - common English stopwords are dropped (every skill description shares
 *     "use this skill when the user wants to...", which would otherwise
 *     create a spurious similarity baseline);
 *   - plural forms also add their singular stem ("presentations" matches
 *     "presentation").
 *
 * The result is L2-normalized, so cosine equals a plain dot product. It is
 * still LEXICAL (shared words), not semantic — a real embedding API via the
 * http backend remains the upgrade path for paraphrases and synonyms.
 */
export class LocalEmbeddingBackend implements EmbeddingBackend {
  readonly name = 'local';
  readonly dimensions: number;

  constructor(dimensions: number = DEFAULT_LOCAL_DIMENSIONS) {
    if (!Number.isInteger(dimensions) || dimensions <= 0) {
      throw new Error(`local embedding dimensions must be a positive integer, got ${String(dimensions)}`);
    }
    this.dimensions = dimensions;
  }

  async embed(texts: readonly string[]): Promise<Embedding[]> {
    // Deliberately sequential and side-effect free; fine for a handful of skills.
    return texts.map((text) => this.embedOne(text));
  }

  private embedOne(text: string): Embedding {
    // Accumulate signed token counts per bucket first, then apply sublinear
    // weights — the count map avoids touching the vector per token.
    const buckets = new Map<number, number>();
    for (const token of tokenize(text)) {
      const hash = fnv1a(token);
      const index = hash % this.dimensions;
      const sign = (hash & 0x80000000) !== 0 ? 1 : -1;
      buckets.set(index, (buckets.get(index) ?? 0) + sign);
    }

    const vector = new Array<number>(this.dimensions).fill(0);
    for (const [index, raw] of buckets) {
      // Signed hashing can cancel a bucket to exactly 0 (two colliding words
      // with opposite signs). Skip it: Math.log(0) is -Infinity and
      // 0 * -Infinity is NaN, which would poison the whole normalized vector.
      if (raw === 0) continue;
      vector[index] = Math.sign(raw) * (1 + Math.log(Math.abs(raw)));
    }
    return normalize(vector);
  }
}

// ---------------------------------------------------------------------------
// Remote: OpenAI-compatible HTTP client
// ---------------------------------------------------------------------------

export class HttpEmbeddingBackend implements EmbeddingBackend {
  readonly name = 'http';
  private readonly baseURL: string;
  private readonly model: string;
  private readonly apiKeyEnv: string;
  private _dimensions: number | undefined;

  constructor(config: HttpEmbeddingConfig) {
    this.baseURL = config.baseURL.replace(/\/+$/, ''); // drop trailing slashes
    this.model = config.model;
    this.apiKeyEnv = config.apiKeyEnv;
    this._dimensions = config.dimensions;
  }

  get dimensions(): number | undefined {
    return this._dimensions;
  }

  async embed(texts: readonly string[]): Promise<Embedding[]> {
    const key = process.env[this.apiKeyEnv];
    if (!key) {
      throw new Error(`embedding: environment variable ${this.apiKeyEnv} is not set`);
    }

    const response = await fetch(`${this.baseURL}/v1/embeddings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${key}`,
      },
      body: JSON.stringify({ model: this.model, input: texts }),
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`embedding: request failed (HTTP ${response.status}): ${body}`);
    }

    const payload = (await response.json()) as { data?: { embedding?: number[] }[] };
    const rows = payload.data;
    if (!Array.isArray(rows) || rows.length !== texts.length) {
      throw new Error(`embedding: expected ${texts.length} vectors, got ${rows?.length ?? 0}`);
    }
    const vectors: Embedding[] = rows.map((row) => row.embedding ?? []);
    this.assertDimension(vectors);
    return vectors;
  }

  /** Ensure every vector has the same length; record it on first use. */
  private assertDimension(vectors: Embedding[]): void {
    const expected = this._dimensions ?? vectors[0]?.length;
    if (expected === undefined) return; // empty input, nothing to check
    for (const vector of vectors) {
      if (vector.length !== expected) {
        throw new Error(`embedding: dimension mismatch (expected ${expected}, got ${vector.length})`);
      }
    }
    this._dimensions = expected;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Lowercase, split on non-alphanumerics, drop stopwords and empties, and add
 * the de-pluralized stem as an extra token ("files" -> "files" + "file") so a
 * prompt's singular matches a description's plural.
 */
function tokenize(text: string): string[] {
  const tokens: string[] = [];
  for (const raw of text.toLowerCase().split(/[^a-z0-9]+/)) {
    if (raw === '' || STOPWORDS.has(raw)) continue;
    tokens.push(raw);
    if (raw.length > 3 && raw.endsWith('s') && !raw.endsWith('ss')) {
      tokens.push(raw.slice(0, -1));
    }
  }
  return tokens;
}

/**
 * Common English stopwords. Skill descriptions are full of them ("Use this
 * skill whenever the user wants to..."), and without this filter every skill
 * shares a spurious similarity baseline.
 */
const STOPWORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'can', 'could',
  'do', 'does', 'for', 'from', 'has', 'have', 'how', 'if', 'in', 'into', 'is',
  'it', 'its', 'like', 'may', 'me', 'more', 'most', 'my', 'not', 'of', 'on', 'or', 'our',
  'out', 'so', 'such', 'than', 'that', 'the', 'their', 'them', 'then', 'there',
  'these', 'they', 'this', 'those', 'through', 'to', 'too', 'up', 'use', 'used',
  'uses', 'using', 'want', 'wants', 'was', 'we', 'what', 'when', 'where',
  'which', 'who', 'will', 'with', 'would', 'you', 'your',
]);

/** FNV-1a 32-bit hash — deterministic across runs and platforms. */
function fnv1a(str: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0; // keep it an unsigned 32-bit integer
}

// ---------------------------------------------------------------------------
// Demo — runs only when executed directly:  node src/embedding.ts
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const backend = createEmbeddingBackend({ provider: 'local' });
  console.log(`backend: ${backend.name}, dimensions: ${backend.dimensions}`);

  const texts = [
    'merge and combine pdf files',
    'combine multiple pdf files into one document',
    'deploy an application to vercel',
    'create a slide deck from a pptx file',
  ];

  const vectors = await backend.embed(texts);

  for (let i = 0; i < texts.length; i++) {
    const v = vectors[i];
    // The local backend is SPARSE: each word lands in exactly one of the
    // 4096 slots, so a short sentence leaves almost every slot at 0. Print
    // the non-zero slots instead of the first five (which are usually 0).
    let nonZero = 0;
    const hits: string[] = [];
    for (let j = 0; j < v.length; j++) {
      if (v[j] !== 0) {
        nonZero += 1;
        if (hits.length < 5) hits.push(`${j}:${v[j].toFixed(3)}`);
      }
    }
    console.log(`\n"${texts[i]}"`);
    console.log(`  dims=${v.length}  non-zero=${nonZero}/${v.length}  slots=[${hits.join(', ')}]`);
  }

  console.log('\ncosine similarity (vectors are unit-length, so cosine == dot):');
  console.log(`  pdf pair        : ${cosine(vectors[0], vectors[1]).toFixed(3)}  ("merge pdf" vs "combine pdfs")`);
  console.log(`  pdf vs vercel   : ${cosine(vectors[0], vectors[2]).toFixed(3)}`);
  console.log(`  pdf vs pptx     : ${cosine(vectors[0], vectors[3]).toFixed(3)}`);
}

/**
 * Demo-only convenience. The canonical cosine() lives in src/similarity.ts;
 * this inline copy keeps the demo self-contained.
 */
function cosine(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // unit-length inputs => dot product == cosine
}

import { pathToFileURL } from 'node:url';
if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
