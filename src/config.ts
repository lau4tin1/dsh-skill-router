/**
 * src/config.ts — configuration: the schema plus the helpers that interpret it.
 *
 * DeepSeek Harness validates each plugin's YAML config against a schemastery
 * schema and passes the NORMALIZED object (every default filled in) to
 * apply(). This file owns three things:
 *
 *   1. The TypeScript `Config` interface — the normalized shape apply() sees.
 *   2. The schemastery `Config` schema — what DSH validates the YAML against.
 *   3. Two mappers that turn `Config` into the shapes the rest of the plugin
 *      already consumes: EmbeddingConfig (embedding.ts) and SelectionConfig
 *      (selection.ts).
 *
 * The default provider is 'local' (the offline hashing vectorizer) — no API
 * key and no network. Switch to a real semantic embedder later by setting
 * embedding.provider = 'http' plus baseURL/model.
 *
 * Design reference: DESIGN.md §10 ("Proposed configuration").
 */

import z from '@deepseek-ai/schemastery';
import type { EmbeddingConfig } from './embedding.ts';
import type { SelectionConfig } from './selection.ts';

export type EmbeddingProvider = 'local' | 'http';
export type SelectionRule = 'largest-gap' | 'ratio-to-max';

const DEFAULT_API_KEY_ENV = 'EMBEDDING_API_KEY';

/** Raw embedding settings as read from YAML, before backend-specific mapping. */
export interface EmbeddingSettings {
  provider: EmbeddingProvider;
  /** Vector length. Optional: local defaults to 384, http infers from the API. */
  dimensions?: number;
  baseURL?: string;
  model?: string;
  apiKeyEnv?: string;
}

/** The normalized config object apply() receives from DSH. */
export interface Config {
  enabled: boolean;
  embedding: EmbeddingSettings;
  rule: SelectionRule;
  minScore: number;
  ratioThreshold: number;
  maxSkills: number;
  maxInjectedBytes: number;
}

/**
 * The schemastery schema. Note that `interface Config` (a type) and
 * `const Config` (a value) may share one name — TypeScript keeps types and
 * values in separate namespaces, so they never collide.
 */
export const Config = z.object({
  enabled: z.boolean().default(true),
  embedding: z.object({
    provider: z.union(['local', 'http']).default('local'),
    // No default on purpose: each backend applies its own (local -> 384,
    // http -> whatever the API returns, inferred on first call).
    dimensions: z.natural().min(1),
    baseURL: z.string(),
    model: z.string(),
    apiKeyEnv: z.string().default(DEFAULT_API_KEY_ENV),
  }),
  rule: z.union(['largest-gap', 'ratio-to-max']).default('largest-gap'),
  // Weak floor on the TOP score ("is anything relevant at all?").
  // Calibrated for the LOCAL lexical backend against the demo corpus:
  // strong matches land ~0.28-0.53, collision noise ~0.00-0.07.
  // A real embedding API (http) produces a different score distribution —
  // re-calibrate this value when switching providers.
  minScore: z.number().default(0.08),
  ratioThreshold: z.number().default(0.75),
  maxSkills: z.natural().min(1).default(4),
  maxInjectedBytes: z.natural().min(1).default(65536),
});

/**
 * Map normalized Config -> the discriminated EmbeddingConfig the backend
 * factory (embedding.ts) expects. Throws a clear error for an http config
 * that is missing its required endpoint facts.
 */
export function toEmbeddingConfig(config: Config): EmbeddingConfig {
  const e = config.embedding;
  if (e.provider === 'local') {
    return e.dimensions === undefined
      ? { provider: 'local' }
      : { provider: 'local', dimensions: e.dimensions };
  }
  if (!e.baseURL || !e.model) {
    throw new Error('config: embedding.provider "http" requires embedding.baseURL and embedding.model');
  }
  return {
    provider: 'http',
    baseURL: e.baseURL,
    model: e.model,
    apiKeyEnv: e.apiKeyEnv ?? DEFAULT_API_KEY_ENV,
    dimensions: e.dimensions,
  };
}

/** Map normalized Config -> SelectionConfig (selection.ts). */
export function toSelectionConfig(config: Config): SelectionConfig {
  return {
    rule: config.rule,
    minScore: config.minScore,
    ratioThreshold: config.ratioThreshold,
    maxSkills: config.maxSkills,
  };
}

// ---------------------------------------------------------------------------
// Demo — runs only when executed directly:  node src/config.ts
// ---------------------------------------------------------------------------

function main(): void {
  console.log('--- schemastery turns raw YAML into the Config apply() receives ---\n');

  console.log('1. empty config          ->', JSON.stringify(Config({})));
  console.log('2. local, custom dims    ->', JSON.stringify(Config({ embedding: { provider: 'local', dimensions: 256 } })));
  console.log('3. http                   ->', JSON.stringify(Config({
    embedding: { provider: 'http', baseURL: 'https://api.openai.com', model: 'text-embedding-3-small' },
  })));

  console.log('\n--- mappers ---');
  const local = Config({}) as Config;
  console.log('toEmbeddingConfig(local) :', JSON.stringify(toEmbeddingConfig(local)));
  console.log('toSelectionConfig        :', JSON.stringify(toSelectionConfig(local)));
}

import { pathToFileURL } from 'node:url';
if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  main();
}
