/**
 * src/embedding.ts — the embedding backend: one REAL local model.
 *
 * A single implementation: a multilingual semantic embedding model running
 * in-process via transformers.js (ONNX). It produces DENSE vectors (384 dims
 * by default, every dimension non-zero) where MEANING matters — synonyms,
 * paraphrases, word order, and LANGUAGE all participate: a Chinese prompt
 * matches an English skill description when they mean the same thing.
 * Shared words are NOT required for a match.
 *
 * The model is downloaded from huggingface.co on FIRST use and cached under
 * ~/.dsh/skill-router/models ($DSH_HOME when set) — outside the working
 * directory and surviving reinstalls. Everything after that download runs
 * fully offline. The import is dynamic, so the heavy ONNX runtime only loads
 * when this module is actually used.
 *
 * Design reference: DESIGN.md §7.
 */

import { homedir } from 'node:os';
import { join } from 'node:path';
import type { FeatureExtractionPipeline } from '@huggingface/transformers';

export type Embedding = number[];

/**
 * The contract the rest of the plugin depends on. Kept as an interface even
 * though only one implementation exists today, so the index/selection logic
 * stays decoupled from the model runtime (and a second backend can be added
 * without touching anything else).
 */
export interface EmbeddingBackend {
  /** Human-readable label for logs. */
  readonly name: string;
  /** Stable identity of the model — tags the persisted index on disk. */
  readonly modelId: string;
  /** Vector length; known from config or after the first embedding. */
  readonly dimensions: number | undefined;
  /**
   * Embed a batch of texts. Returns one vector per text, in order.
   * `options.query` marks the text as a user query rather than a document
   * (skill routing text); some models (the e5 family) want a different
   * prefix for each and the backend applies it automatically.
   */
  embed(texts: readonly string[], options?: { query?: boolean }): Promise<Embedding[]>;
}

export interface EmbeddingConfig {
  /**
   * Hugging Face ONNX model id.
   * Default: Xenova/bge-m3 — the flagship multilingual model (Chinese,
   * English, and ~100 more), 1024-dim dense vectors, [CLS] pooling.
   * Alternatives: Xenova/multilingual-e5-small (fast), intfloat
   * multilingual-e5-large, Xenova/bge-large-zh-v1.5, BAAI/bge-large-en-v1.5.
   */
  model?: string;
  /** Weight precision: 'q8' (default, ~4x smaller download) or 'fp32'. */
  dtype?: 'fp32' | 'q8';
  /** Optional expected output dimension; validated after the first embedding. */
  dimensions?: number;
  /** Directory for the downloaded model. Default: $DSH_HOME/skill-router/models. */
  cacheDir?: string;
  /**
   * Prefixes for asymmetric models (the e5 family). Defaults to
   * 'query: '/'passage: ' when the model id contains 'e5', otherwise none.
   * Only set these if you know the model expects them.
   */
  queryPrefix?: string;
  documentPrefix?: string;
  /**
   * Pooling strategy. Defaults automatically: 'cls' for BGE-family models
   * (they use the [CLS] token), 'mean' for everything else (e5, MiniLM).
   * Only set this if you know the model expects a specific pooling.
   */
  pooling?: 'mean' | 'cls';
}

const DEFAULT_MODEL = 'Xenova/bge-m3';

/** Default model cache: $DSH_HOME (or ~/.dsh) + /skill-router/models. */
export function defaultModelCacheDir(): string {
  const dshHome = process.env.DSH_HOME || join(homedir(), '.dsh');
  return join(dshHome, 'skill-router', 'models');
}

/** The single supported backend today; config picks the model, not the engine. */
export function createEmbeddingBackend(config: EmbeddingConfig = {}): EmbeddingBackend {
  return new TransformersEmbeddingBackend(config);
}

export class TransformersEmbeddingBackend implements EmbeddingBackend {
  readonly name = 'transformers';
  readonly modelId: string;
  private readonly dtype: 'fp32' | 'q8';
  private readonly cacheDir: string;
  private readonly queryPrefix: string;
  private readonly documentPrefix: string;
  private readonly pooling: 'mean' | 'cls';
  private _dimensions: number | undefined;
  private extractorPromise: Promise<FeatureExtractionPipeline> | undefined;

  constructor(config: EmbeddingConfig = {}) {
    this.modelId = config.model ?? DEFAULT_MODEL;
    // q8 default: bge-m3's fp32 weights are a 2.1GB external-data pair;
    // the quantized file is one 543MB download with negligible quality loss.
    this.dtype = config.dtype ?? 'q8';
    this.cacheDir = config.cacheDir ?? defaultModelCacheDir();
    this._dimensions = config.dimensions;
    // The e5 family expects asymmetric prefixes; everything else gets none.
    const isE5 = this.modelId.toLowerCase().includes('e5');
    this.queryPrefix = config.queryPrefix ?? (isE5 ? 'query: ' : '');
    this.documentPrefix = config.documentPrefix ?? (isE5 ? 'passage: ' : '');
    // BGE models use the [CLS] token as the sentence vector; others use mean.
    const isBge = this.modelId.toLowerCase().includes('bge');
    this.pooling = config.pooling ?? (isBge ? 'cls' : 'mean');
  }

  get dimensions(): number | undefined {
    return this._dimensions;
  }

  async embed(texts: readonly string[], options?: { query?: boolean }): Promise<Embedding[]> {
    const prefix = options?.query === true ? this.queryPrefix : this.documentPrefix;
    const extractor = await this.getExtractor();

    // One batched call for many texts (the skill index): the extractor pads
    // internally, and a single forward pass avoids any per-call state issues.
    // A single text stays a single-element call.
    const input = texts.length === 1
      ? `${prefix}${texts[0]}`
      : texts.map((text) => `${prefix}${text}`);
    const tensor = await extractor(input, { pooling: this.pooling, normalize: true });

    const dim = tensor.dims[tensor.dims.length - 1];
    this.assertDimension(dim);
    const data = tensor.data as Float32Array;

    const vectors: Embedding[] = [];
    for (let i = 0; i < texts.length; i++) {
      vectors.push(Array.from(data.subarray(i * dim, (i + 1) * dim)));
    }
    return vectors;
  }

  private getExtractor(): Promise<FeatureExtractionPipeline> {
    if (this.extractorPromise === undefined) {
      this.extractorPromise = (async () => {
        const { env, pipeline } = await import('@huggingface/transformers');
        // Point the cache at ~/.dsh BEFORE building the pipeline, so the
        // first download lands outside the working directory.
        env.cacheDir = this.cacheDir;
        env.allowRemoteModels = true;
        return pipeline('feature-extraction', this.modelId, {
          dtype: this.dtype,
          // Single-threaded inference: the e5 family's vectors concentrate
          // most energy in one shared direction, so the discriminative part
          // lives in tiny residual differences that multi-threaded reduction
          // order can shuffle run-to-run. Pinning one thread makes results
          // deterministic; the workload (tens of short texts) is tiny.
          session_options: { intraOpNumThreads: 1 },
        }) as Promise<FeatureExtractionPipeline>;
      })();
    }
    return this.extractorPromise;
  }

  private assertDimension(dim: number): void {
    if (this._dimensions === undefined) {
      this._dimensions = dim;
    } else if (dim !== this._dimensions) {
      throw new Error(`transformers: dimension mismatch (expected ${this._dimensions}, got ${dim})`);
    }
  }
}

// ---------------------------------------------------------------------------
// Demo — runs only when executed directly:  node src/embedding.ts
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const backend = createEmbeddingBackend();
  console.log(`backend: ${backend.name}, model: ${backend.modelId}`);

  const texts = [
    'merge and combine pdf files',
    'combine multiple pdf files into one document',
    'deploy an application to vercel',
    'create a slide deck from a pptx file',
  ];

  const vectors = await backend.embed(texts);

  for (let i = 0; i < texts.length; i++) {
    const v = vectors[i];
    console.log(`\n"${texts[i]}"`);
    console.log(`  dims=${v.length}  head=[${v.slice(0, 5).map((x) => x.toFixed(3)).join(', ')}]`);
  }

  console.log('\ncosine similarity (vectors are unit-length, so cosine == dot):');
  console.log(`  pdf pair      : ${cosine(vectors[0], vectors[1]).toFixed(3)}  ("merge pdf" vs "combine pdfs")`);
  console.log(`  pdf vs vercel : ${cosine(vectors[0], vectors[2]).toFixed(3)}`);
  console.log(`  pdf vs pptx   : ${cosine(vectors[0], vectors[3]).toFixed(3)}`);
}

/** Demo-only convenience; the canonical cosine() lives in src/similarity.ts. */
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
