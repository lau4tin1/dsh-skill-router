/**
 * src/index.ts — the plugin entry (a Cordis plugin).
 *
 * This is the ONLY file that talks to DeepSeek Harness's context (`ctx`).
 * Every other module is pure or nearly pure. Here we wire the full pipeline:
 *
 *   ctx.skills.snapshot()  ──▶  SkillIndex.sync()      (load + embed + watch)
 *   skills/change event    ──▶  mark index dirty       (watch)
 *   agent/pre-step         ──▶  embed latest user message
 *                             ▶ cosine against the index
 *                             ▶ selectSkills()
 *                             ▶ ctx.skills.get() for full bodies
 *                             ▶ renderSelection()
 *                             ▶ inject via createUserMessage()
 *
 * A Cordis plugin exports four things: `name`, `inject` (required services),
 * `Config` (the schemastery schema), and `apply(ctx, config)`. This mirrors
 * the shipped @deepseek-ai/dsh-tool-skill package exactly.
 *
 * Design reference: DESIGN.md §8 (lifecycle) and §10 (configuration).
 */

import { join } from 'node:path';
import type { Context } from '@deepseek-ai/cordis';
import type { Agent } from '@deepseek-ai/dsh-agent';
import { createUserMessage } from '@deepseek-ai/dsh-llm';
import type { UserMessage } from '@deepseek-ai/dsh-llm';
import { isModelInvocable } from '@deepseek-ai/dsh-skill';

import { createEmbeddingBackend } from './embedding.ts';
import { SkillIndex } from './skillIndex.ts';
import { selectSkills } from './selection.ts';
import { Config, resolveCacheDir, toEmbeddingConfig, toSelectionConfig } from './config.ts';
import { renderSelection } from './render.ts';
import type { SelectedSkill } from './render.ts';

export { Config };
export const name = 'skill-router';
export const inject = ['skills', 'agents'];

export function apply(ctx: Context, config: Config): void {
  if (!config.enabled) return;

  // All persisted data lives OUTSIDE the working directory:
  //   <cacheRoot>/models           — the downloaded embedding model
  //   <cacheRoot>/skill-index.json — the persisted skill embeddings
  const cacheRoot = resolveCacheDir(config);
  const backend = createEmbeddingBackend({
    ...toEmbeddingConfig(config),
    cacheDir: join(cacheRoot, 'models'),
  });
  const index = new SkillIndex(backend);
  const selection = toSelectionConfig(config);
  const indexFile = join(cacheRoot, 'skill-index.json');
  // Restore the persisted index exactly once (first pre-step), so unchanged
  // skills are not re-embedded after a restart.
  let indexRestored = false;
  // NOTE: config.maxInjectedBytes is reserved for a body-truncation pass (v1.1);
  // the selection cap (maxSkills) already bounds injected volume today.

  // Per-agent set of skill names already injected this session, so a skill whose
  // full body is already in the conversation is not re-injected on later turns.
  const injectedByAgent = new WeakMap<Agent, Set<string>>();

  // "watch": the filesystem provider already watches skill files and emits
  // `skills/change` on add/rename/delete/frontmatter edits. We only mark the
  // index dirty; the next pre-step re-snapshots and syncs (cheap, digest-diff).
  let dirty = true;
  ctx.on('skills/change', () => {
    dirty = true;
  });

  // "load + route + inject" on every proposed step (waterfall middleware).
  ctx.on('agent/pre-step', async ({ agent, messages, signal }, next) => {
    const decision = await next();
    if (decision.kind === 'reject') return decision;
    signal.throwIfAborted();

    const lookup = { cwd: agent.session.header.cwd, scope: agent, signal };

    // 1. Keep the index current. The initial build happens here, lazily, on the
    //    first step (apply() has no agent/cwd yet, so it cannot snapshot there).
    if (dirty || index.size === 0) {
      if (!indexRestored) {
        indexRestored = true;
        await index.loadFromDisk(indexFile);
      }
      const snapshot = await ctx.skills.snapshot(lookup);
      signal.throwIfAborted();
      if (snapshot.complete) {
        await index.sync(snapshot.skills.filter(isModelInvocable));
        dirty = false;
        // Persistence is best-effort: a failed save must never fail the turn.
        index.saveToDisk(indexFile).catch((error) => {
          console.warn(`skill-router: could not persist index to ${indexFile}:`, error);
        });
      }
      // Incomplete snapshot: keep last-good index, leave dirty set, retry later.
    }

    // 2. The current task is the text of the user-authored messages in this step.
    const task = userTaskText(messages);
    if (task === undefined || index.size === 0) return decision;

    // 3. Embed the task, score every skill, keep the standouts.
    const [query] = await backend.embed([task], { query: true });
    signal.throwIfAborted();
    const result = selectSkills(index.score(query), selection);
    if (result.selected.length === 0) return decision;

    // 4. Skip skills already injected this session (in-memory dedup).
    const injected = injectedByAgent.get(agent) ?? new Set<string>();
    const fresh = result.selected.filter((scored) => !injected.has(scored.name));
    if (fresh.length === 0) return decision;

    // 5. Load full bodies for the fresh skills only.
    const loaded: SelectedSkill[] = [];
    for (const scored of fresh) {
      const definition = await ctx.skills.get(scored.name, lookup);
      signal.throwIfAborted();
      if (definition !== undefined) {
        loaded.push({
          name: definition.name,
          provider: definition.provider,
          content: definition.content,
          resourceBase: definition.resourceBase,
        });
      }
    }
    if (loaded.length === 0) return decision;

    // 6. Remember what we injected, render, and append the injection.
    for (const skill of loaded) injected.add(skill.name);
    injectedByAgent.set(agent, injected);

    const message = createUserMessage({
      content: [{ type: 'text', text: renderSelection(loaded) }],
      source: { kind: 'plugin', plugin: name, form: 'instructions' },
    });

    return { ...decision, messages: [...decision.messages, message] };
  });
}

/** The task text: all text blocks of user-authored messages entering this step. */
function userTaskText(messages: readonly UserMessage[]): string | undefined {
  const parts: string[] = [];
  for (const message of messages) {
    if (message.source.kind !== 'user') continue;
    for (const block of message.content) {
      if (block.type === 'text') parts.push(block.text);
    }
  }
  const text = parts.join('\n').trim();
  return text.length > 0 ? text : undefined;
}
