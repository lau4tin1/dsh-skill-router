/**
 * src/skillIndex.ts — the embedding index and its change-tracking lifecycle.
 *
 * This is the "load → embed → watch" core. It owns ONE responsibility:
 * keep an up-to-date map of  skill name  ->  { vector, routingDigest }.
 *
 *   - build(skills): embed everything once (used at startup).
 *   - sync(skills) : diff a NEW skill list against the current index and
 *                    re-embed only what changed (used on every skills/change).
 *
 * "Changed" is decided by a digest of the ROUTING TEXT (name + description +
 * whenToUse) — the exact text we embed. A body-only edit does not change that
 * digest, so it correctly does NOT trigger a re-embed; the fresh body is
 * fetched later at injection time (that is src/index.ts's job, not here).
 *
 * Like selection.ts and embedding.ts, this file is DSH-free: it only needs a
 * list of { name, description, whenToUse } objects and an EmbeddingBackend.
 * The plugin entry will map the registry's SkillSummary[] straight into it.
 *
 * Design reference: DESIGN.md §8.1 ("Building the index").
 */

import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { type Embedding, type EmbeddingBackend } from './embedding.ts';
import { cosine, meanVector, subtract } from './similarity.ts';
import type { ScoredSkill } from './selection.ts';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * The minimum a skill must expose for routing. `SkillSummary` from the DSH
 * skill registry satisfies this structurally, so no conversion is needed.
 */
export interface RoutingSkill {
  name: string;
  description: string;
  whenToUse?: string;
}

/** One indexed skill: the routing vector plus the digest used for change detection. */
export interface IndexEntry {
  name: string;
  vector: Embedding;
  /** SHA-256 of the routing text; changes when name/description/whenToUse change. */
  routingDigest: string;
}

/** What changed since the previous snapshot — useful for logging and debugging. */
export interface SyncReport {
  added: string[];
  removed: string[];
  changed: string[];
  /** Count of skills present before AND after with an identical routing digest. */
  unchanged: number;
}

// ---------------------------------------------------------------------------
// The index
// ---------------------------------------------------------------------------

export class SkillIndex {
  private readonly backend: EmbeddingBackend;
  private readonly entries = new Map<string, IndexEntry>();

  constructor(backend: EmbeddingBackend) {
    this.backend = backend;
  }

  get size(): number {
    return this.entries.size;
  }

  /** Sorted names, for stable logging. */
  names(): string[] {
    return [...this.entries.keys()].sort();
  }

  get(name: string): IndexEntry | undefined {
    return this.entries.get(name);
  }

  /**
   * Full rebuild from scratch (startup path). Empties the index and embeds
   * every skill. If embedding throws, the index is left untouched (we compute
   * all vectors first, then commit).
   */
  async build(skills: readonly RoutingSkill[]): Promise<SyncReport> {
    const fresh = await this.embedAll(skills);

    this.entries.clear();
    const added: string[] = [];
    for (const entry of fresh) {
      this.entries.set(entry.name, entry);
      added.push(entry.name);
    }
    return { added, removed: [], changed: [], unchanged: 0 };
  }

  /**
   * Incremental update (watch path). Diffs the incoming list against the
   * current index:
   *   - name present only in incoming        -> added   (embed it)
   *   - name present only in index           -> removed (drop it)
   *   - same name, different routing digest  -> changed (re-embed it)
   *   - same name, same digest               -> unchanged
   *
   * All needed vectors are computed in ONE batch before any mutation, so a
   * backend failure cannot leave the index half-updated.
   */
  async sync(skills: readonly RoutingSkill[]): Promise<SyncReport> {
    const incoming = new Map(skills.map((skill) => [skill.name, skill]));

    const added: string[] = [];
    const changed: string[] = [];
    const removed: string[] = [];
    const toEmbed: RoutingSkill[] = [];
    let unchanged = 0;

    for (const [name, skill] of incoming) {
      const existing = this.entries.get(name);
      if (existing === undefined) {
        added.push(name);
        toEmbed.push(skill);
      } else if (existing.routingDigest !== routingDigest(skill)) {
        changed.push(name);
        toEmbed.push(skill);
      } else {
        unchanged += 1;
      }
    }

    for (const name of this.entries.keys()) {
      if (!incoming.has(name)) removed.push(name);
    }

    // Embed the union of added+changed in one batch, then commit everything.
    const fresh = toEmbed.length > 0 ? await this.embedAll(toEmbed) : [];
    const freshByName = new Map(fresh.map((entry) => [entry.name, entry]));

    for (const name of [...added, ...changed]) {
      const entry = freshByName.get(name);
      if (entry === undefined) {
        throw new Error(`skill index: backend did not return a vector for "${name}"`);
      }
      this.entries.set(name, entry);
    }
    for (const name of removed) {
      this.entries.delete(name);
    }

    return { added, removed, changed, unchanged };
  }

  // -------------------------------------------------------------------------
  // Persistence: the index survives restarts, so skills are only re-embedded
  // when their routing text (or the model) actually changes.
  // -------------------------------------------------------------------------

  /**
   * Save every entry to disk, atomically (write a temp file, then rename over
   * the target). The payload records which backend/model produced the
   * vectors, so a model switch invalidates the file instead of mixing
   * incompatible vector spaces.
   */
  async saveToDisk(file: string): Promise<void> {
    const payload = {
      backend: {
        name: this.backend.name,
        modelId: this.backend.modelId,
        dimensions: this.backend.dimensions ?? 0,
      },
      entries: [...this.entries.values()],
    };
    await mkdir(dirname(file), { recursive: true });
    const tmp = `${file}.tmp`;
    await writeFile(tmp, JSON.stringify(payload), 'utf8');
    await rename(tmp, file);
  }

  /**
   * Restore entries from a previous run. Returns false when the file is
   * missing/unreadable or was written by a different backend/model — those
   * vectors live in another vector space and must not be mixed in.
   *
   * Restored entries are still checked by sync(): any skill whose routing
   * digest changed since the save is simply re-embedded, and skills that no
   * longer exist are dropped.
   */
  async loadFromDisk(file: string): Promise<boolean> {
    let text: string;
    try {
      text = await readFile(file, 'utf8');
    } catch {
      return false; // missing/unreadable -> cold start
    }

    let data: { backend?: { name?: string; modelId?: string; dimensions?: number }; entries?: unknown };
    try {
      data = JSON.parse(text);
    } catch {
      return false; // corrupt -> ignore and rebuild fresh
    }

    if (data.backend?.name !== this.backend.name || data.backend?.modelId !== this.backend.modelId) {
      return false; // different model -> its vectors are not comparable to ours
    }

    const expected = this.backend.dimensions ?? data.backend?.dimensions;
    if (!Array.isArray(data.entries)) return false;

    for (const raw of data.entries) {
      if (typeof raw !== 'object' || raw === null) continue;
      const entry = raw as { name?: unknown; vector?: unknown; routingDigest?: unknown };
      if (
        typeof entry.name !== 'string' ||
        !Array.isArray(entry.vector) ||
        typeof entry.routingDigest !== 'string'
      ) continue;
      if (expected !== undefined && entry.vector.length !== expected) continue;
      this.entries.set(entry.name, {
        name: entry.name,
        vector: entry.vector as number[],
        routingDigest: entry.routingDigest,
      });
    }
    return true;
  }

  /**
   * The corpus center: component-wise mean of every stored vector.
   * Used to de-bias anisotropic embeddings before scoring (see score()).
   */
  centerVector(): number[] | undefined {
    return meanVector([...this.entries.values()].map((entry) => entry.vector));
  }

  /**
   * Score every stored skill against a query vector.
   *
   * Both sides are CENTERED first (corpus mean subtracted): some embedding
   * models (the e5 family especially) concentrate most of their energy in one
   * shared direction, which inflates raw cosine so even unrelated texts score
   * ~0.8. Centering removes that common direction and exposes the
   * discriminative part — unrelated drops toward 0 and true matches stay
   * clearly positive.
   */
  score(query: number[]): ScoredSkill[] {
    const center = this.centerVector();
    const q = center === undefined ? query : subtract(query, center);
    const scored: ScoredSkill[] = [];
    for (const entry of this.entries.values()) {
      const v = center === undefined ? entry.vector : subtract(entry.vector, center);
      scored.push({ name: entry.name, score: cosine(q, v) });
    }
    return scored;
  }

  /** Embed a batch of skills into IndexEntry objects (order preserved). */
  private async embedAll(skills: readonly RoutingSkill[]): Promise<IndexEntry[]> {
    if (skills.length === 0) return [];
    const texts = skills.map(routingText);
    const vectors = await this.backend.embed(texts);
    if (vectors.length !== skills.length) {
      throw new Error(
        `skill index: backend returned ${vectors.length} vectors for ${skills.length} skills`,
      );
    }
    return skills.map((skill, i) => ({
      name: skill.name,
      vector: vectors[i],
      routingDigest: routingDigest(skill),
    }));
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** The exact text we embed, and the source of the change-detection digest. */
export function routingText(skill: RoutingSkill): string {
  return [skill.name, skill.description.trim(), (skill.whenToUse ?? '').trim()].join('\n');
}

/** SHA-256 of the routing text — stable identity for "did the routing surface change?". */
export function routingDigest(skill: RoutingSkill): string {
  return createHash('sha256').update(routingText(skill)).digest('hex');
}

// ---------------------------------------------------------------------------
// Demo — runs only when executed directly:  node src/skillIndex.ts
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const { createEmbeddingBackend } = await import('./embedding.ts');
  const backend = createEmbeddingBackend();
  const index = new SkillIndex(backend);

  const initial: RoutingSkill[] = [
    { name: 'pdf', description: 'merge and combine pdf files' },
    { name: 'xlsx', description: 'create and edit spreadsheets' },
    { name: 'vercel', description: 'deploy applications to vercel' },
  ];

  const report1 = await index.build(initial);
  console.log('after build:', report1, '| size =', index.size, '| names =', index.names().join(', '));

  // Simulate a change event: xlsx description edited, pptx added, vercel removed.
  const next: RoutingSkill[] = [
    { name: 'pdf', description: 'merge and combine pdf files' },                   // unchanged
    { name: 'xlsx', description: 'create, edit, and analyze spreadsheets' },       // changed
    { name: 'pptx', description: 'create slide decks and presentations' },         // added
  ];

  const report2 = await index.sync(next);
  console.log('after sync :', report2, '| size =', index.size, '| names =', index.names().join(', '));

  const entry = index.get('pptx');
  console.log('\nsample entry "pptx":');
  console.log('  name          :', entry?.name);
  console.log('  routingDigest :', entry ? `${entry.routingDigest.slice(0, 16)}…` : '(none)');
  console.log('  vector length :', entry?.vector.length);
}

import { pathToFileURL } from 'node:url';
if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
