/**
 * src/embedding.ts — the embedding backend: one REAL local model.
 *
 * A single implementation: a semantic embedding model running in-process via
 * transformers.js (ONNX). It produces DENSE vectors (384 dims by default,
 * every dimension non-zero) where MEANING matters — synonyms, paraphrases,
 * and word order all participate; shared words are NOT required for a match.
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
  /** Embed a batch of texts. Returns one vector per text, in order. */
  embed(texts: readonly string[]): Promise<Embedding[]>;
}

export interface EmbeddingConfig {
  /**
   * Hugging Face ONNX model id.
   * Default: onnx-community/all-MiniLM-L6-v2-ONNX (22M params, 384 dims —
   * fast and good for short routing text).
   * More quality: Xenova/bge-base-en-v1.5 (~110MB) or
   * Xenova/bge-large-en-v1.5 (~1.3GB, near-SOTA retrieval, ~5-8x slower).
   */
  model?: string;
  /** Weight precision: 'fp32' (default) or 'q8' (~4x smaller download, tiny quality loss). */
  dtype?: 'fp32' | 'q8';
  /** Optional expected output dimension; validated after the first embedding. */
  dimensions?: number;
  /** Directory for the downloaded model. Default: $DSH_HOME/skill-router/models. */
  cacheDir?: string;
}

const DEFAULT_MODEL = 'onnx-community/all-MiniLM-L6-v2-ONNX';

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
  private _dimensions: number | undefined;
  private extractorPromise: Promise<FeatureExtractionPipeline> | undefined;

  constructor(config: EmbeddingConfig = {}) {
    this.modelId = config.model ?? DEFAULT_MODEL;
    this.dtype = config.dtype ?? 'fp32';
    this.cacheDir = config.cacheDir ?? defaultModelCacheDir();
    this._dimensions = config.dimensions;
  }

  get dimensions(): number | undefined {
    return this._dimensions;
  }

  async embed(texts: readonly string[]): Promise<Embedding[]> {
    const extractor = await this.getExtractor();
    const vectors: Embedding[] = [];
    for (const text of texts) {
      // Sequential: the index embeds tens of short routing texts, and
      // batching only pads to the longest one without saving compute.
      const tensor = await extractor(text, { pooling: 'mean', normalize: true });
      const vector = Array.from(tensor.data as Float32Array);
      this.assertDimension(vector.length);
      vectors.push(vector);
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
