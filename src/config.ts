/**
 * src/config.ts — configuration: the schema plus the helpers that interpret it.
 *
 * DeepSeek Harness validates each plugin's YAML config against a schemastery
 * schema and passes the NORMALIZED object (every default filled in) to
 * apply(). This file owns three things:
 *
 *   1. The TypeScript `Config` interface — the normalized shape apply() sees.
 *   2. The schemastery `Config` schema — what DSH validates the YAML against.
 *   3. Mappers that turn `Config` into the shapes the rest of the plugin
 *      consumes: EmbeddingConfig (embedding.ts) and SelectionConfig
 *      (selection.ts), plus the cache root for persisted data.
 *
 * There is exactly ONE embedding engine — a real local model — so there is no
 * provider field; `embedding.model` only picks WHICH model.
 *
 * Design reference: DESIGN.md §7 and §10.
 */

import { homedir } from 'node:os';
import { join } from 'node:path';
import z from '@deepseek-ai/schemastery';
import type { EmbeddingConfig } from './embedding.ts';
import type { SelectionConfig } from './selection.ts';

export type SelectionRule = 'largest-gap' | 'ratio-to-max';

/** Raw embedding settings as read from YAML. */
export interface EmbeddingSettings {
  /** HF ONNX model id; default Xenova/bge-m3 (multilingual, zh+en+100 langs). */
  model?: string;
  /** 'q8' (default, ~4x smaller weights) | 'fp32'. */
  dtype?: 'fp32' | 'q8';
  /** Optional expected output dimension; inferred from the model when omitted. */
  dimensions?: number;
}

/** The normalized config object apply() receives from DSH. */
export interface Config {
  enabled: boolean;
  embedding: EmbeddingSettings;
  /**
   * Root for ALL persisted data — the downloaded model weights and the
   * skill-embedding index. Default: $DSH_HOME (or ~/.dsh) + /skill-router.
   * Deliberately OUTSIDE the working directory.
   */
  cacheDir?: string;
  rule: SelectionRule;
  minScore: number;
  ratioThreshold: number;
  maxSkills: number;
  maxInjectedBytes: number;
  /**
   * Optional z-score gate. When set, selection switches to scale-invariant
   * mode: keep skills whose score is >= zThreshold std-devs above the current
   * query's mean AND >= minScore. Unset = classic gap/ratio rule.
   */
  zThreshold?: number;
}

/**
 * The schemastery schema. Note that `interface Config` (a type) and
 * `const Config` (a value) may share one name — TypeScript keeps types and
 * values in separate namespaces, so they never collide.
 */
export const Config = z.object({
  enabled: z.boolean().default(true),
  embedding: z.object({
    model: z.string(),
    dtype: z.union(['fp32', 'q8']).default('q8'),
    dimensions: z.natural().min(1),
  }),
  cacheDir: z.string(),
  rule: z.union(['largest-gap', 'ratio-to-max']).default('largest-gap'),
  // Confidence floor on the TOP score ("is anything relevant at all?").
  // TWO modes:
  //   - classic (zThreshold unset): the only gate before the gap/ratio cut.
  //     Calibrated for bge-m3 centered cosine on the 24-skill corpus:
  //     confident matches >= 0.18, diffuse noise <= 0.13; 0.2 sits above the
  //     borderline band (measured false positives at 0.1495/0.1681).
  //   - z-score mode (zThreshold set): a WEAK absolute AND-floor — the z-gate
  //     does the discriminating, this only rejects near-garbage scores.
  //     Recommended 0.05 (the deployed cordis.patch.yml uses this).
  // Known trade-off in both modes: weak true matches like zh "项目进度报告"
  // (~0.10-0.13 raw, z ~1.4) do not inject. Re-calibrate when the model or
  // the corpus changes.
  minScore: z.number().default(0.2),
  ratioThreshold: z.number().default(0.75),
  maxSkills: z.natural().min(1).default(4),
  maxInjectedBytes: z.natural().min(1).default(65536),
  // Optional z-score gate. When set, selection switches to scale-invariant
  // mode: keep skills whose score is >= zThreshold std-devs above the current
  // query's mean AND >= minScore (weak floor). Measured on the 24-skill
  // corpus: false positives z ~1.8-2.3, true positives z ~2.9-3.4 -> 2.5
  // splits cleanly. Unset = classic gap/ratio rule.
  zThreshold: z.number(),
});

/** Map normalized Config -> the EmbeddingConfig the backend factory expects. */
export function toEmbeddingConfig(config: Config): EmbeddingConfig {
  const e = config.embedding;
  return {
    ...(e.model !== undefined ? { model: e.model } : {}),
    ...(e.dtype !== undefined ? { dtype: e.dtype } : {}),
    ...(e.dimensions !== undefined ? { dimensions: e.dimensions } : {}),
  };
}

/** Root for all persisted data; the model cache and the index file live under it. */
export function resolveCacheDir(config: Config): string {
  if (config.cacheDir !== undefined) return config.cacheDir;
  const dshHome = process.env.DSH_HOME || join(homedir(), '.dsh');
  return join(dshHome, 'skill-router');
}

/** Map normalized Config -> SelectionConfig (selection.ts). */
export function toSelectionConfig(config: Config): SelectionConfig {
  return {
    rule: config.rule,
    minScore: config.minScore,
    ratioThreshold: config.ratioThreshold,
    maxSkills: config.maxSkills,
    zThreshold: config.zThreshold,
  };
}

// ---------------------------------------------------------------------------
// Demo — runs only when executed directly:  node src/config.ts
// ---------------------------------------------------------------------------

function main(): void {
  console.log('--- schemastery turns raw YAML into the Config apply() receives ---\n');

  console.log('1. empty config ->', JSON.stringify(Config({})));
  console.log('2. bigger model ->', JSON.stringify(Config({
    embedding: { model: 'Xenova/bge-large-en-v1.5', dtype: 'q8' },
    cacheDir: '/custom/cache',
  })));

  console.log('\n--- mappers ---');
  const normalized = Config({}) as Config;
  console.log('toEmbeddingConfig :', JSON.stringify(toEmbeddingConfig(normalized)));
  console.log('toSelectionConfig  :', JSON.stringify(toSelectionConfig(normalized)));
  console.log('resolveCacheDir    :', resolveCacheDir(normalized));
}

import { pathToFileURL } from 'node:url';
if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  main();
}
