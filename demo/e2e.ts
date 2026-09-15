/**
 * demo/e2e.ts — end-to-end demo on the REAL downloaded skills.
 *
 * Full pipeline, using the exact modules the plugin uses:
 *
 *   parse each skill's SKILL.md frontmatter (name + description + whenToUse)
 *     -> SkillIndex.build() with the local hashing vectorizer
 *     -> for each test prompt: embed -> cosine score every skill
 *     -> selectSkills() with the plugin's real config path
 *     -> print the scoreboard and the FINAL RETURNED SKILLS.
 *
 * Usage:
 *   node demo/e2e.ts            # plugin defaults (Config({}) -> toSelectionConfig)
 *   node demo/e2e.ts 0.05       # override minScore for calibration experiments
 *
 * This file is dev-only: it is not bundled into the plugin and not part of
 * `npm run typecheck` (tsconfig includes src/ only).
 */

import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { load } from 'js-yaml';

import { createEmbeddingBackend } from '../src/embedding.ts';
import { SkillIndex, type RoutingSkill } from '../src/skillIndex.ts';
import { cosine } from '../src/similarity.ts';
import { selectSkills } from '../src/selection.ts';
import { Config, toSelectionConfig } from '../src/config.ts';

// Test cases: prompt -> skills we EXPECT the router to return.
// Two empty-expected cases check the "nothing is relevant" path.
const CASES: { prompt: string; expected: string[] }[] = [
  { prompt: 'merge several pdf files into one document', expected: ['pdf'] },
  { prompt: 'create an xlsx spreadsheet from this csv data', expected: ['xlsx'] },
  { prompt: 'make a powerpoint presentation for our quarterly review', expected: ['pptx'] },
  { prompt: 'turn this data into a spreadsheet and a word document', expected: ['xlsx', 'docx'] },
  { prompt: 'help me write a project status update for my team', expected: ['internal-comms'] },
  { prompt: 'create an animated gif for slack of a dancing cat', expected: ['slack-gif-creator'] },
  { prompt: 'generate algorithmic art using p5.js flow fields', expected: ['algorithmic-art'] },
  { prompt: 'how do I learn to use claude', expected: ['academy-guide'] },
  { prompt: 'debug this failing playwright test for my web app', expected: ['webapp-testing'] },
  { prompt: 'build an MCP server in python that calls an api', expected: ['mcp-builder'] },
  { prompt: 'deploy my app to production', expected: ['deploy-to-vercel'] },
  { prompt: 'my app has a bug, help me fix it', expected: ['systematic-debugging'] },
  { prompt: 'build a chat app with a node backend', expected: ['fullstack-dev'] },
  { prompt: 'how do I use the openai api', expected: ['openai-docs'] },
  { prompt: 'make a landing page with animations for my startup', expected: ['frontend-dev'] },
  { prompt: 'hello, how are you today?', expected: [] },
];

/** Read every <skill>/SKILL.md and extract its routing surface (frontmatter). */
function loadSkills(): RoutingSkill[] {
  // .agents/skills is a real DSH scanned root (rank 200, project-agents);
  // the plugin itself never reads files — it gets summaries from ctx.skills.
  const root = join(import.meta.dirname, '..', '.agents', 'skills');
  const files = readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => join(root, entry.name, 'SKILL.md'));

  const skills: RoutingSkill[] = [];
  for (const file of files) {
    try {
      const text = readFileSync(file, 'utf8');
      const match = text.match(/^---\r?\n([\s\S]*?)\r?\n---/);
      if (match === null) continue;
      // js-yaml `load` is used on trusted local files; demo only.
      const data = load(match[1]) as { name?: string; description?: string; whenToUse?: string };
      if (typeof data?.name !== 'string' || typeof data?.description !== 'string') continue;
      skills.push({ name: data.name, description: data.description, whenToUse: data.whenToUse });
    } catch (error) {
      console.error(`skipping ${file}:`, error);
    }
  }
  return skills;
}

async function main(): Promise<void> {
  const skills = loadSkills();
  console.log(`loaded ${skills.length} skills from .agents/skills/`);

  // EMBEDDING_PROVIDER=transformers node demo/e2e.ts — use the real local model.
  const provider = process.env.EMBEDDING_PROVIDER ?? 'local';
  const backend = provider === 'transformers'
    ? createEmbeddingBackend({ provider: 'transformers' })
    : createEmbeddingBackend({ provider: 'local' });
  const index = new SkillIndex(backend);
  await index.build(skills);

  // The plugin's real config path: YAML -> schema -> toSelectionConfig.
  const selection = toSelectionConfig(Config({}));
  if (process.argv[2] !== undefined) {
    selection.minScore = Number(process.argv[2]);
  } else if (provider === 'transformers') {
    // Real models give even unrelated texts a moderate baseline similarity,
    // so the weak floor sits higher than the hashing vectorizer's 0.08.
    // Calibrated on this corpus: correct matches >= 0.207, noise <= 0.094.
    selection.minScore = 0.12;
  }
  console.log(`selection config: rule=${selection.rule} minScore=${selection.minScore} ratioThreshold=${selection.ratioThreshold} maxSkills=${selection.maxSkills}\n`);

  let passed = 0;
  for (const { prompt, expected } of CASES) {
    const [query] = await backend.embed([prompt]);
    const scored = index.names()
      .map((name) => {
        const entry = index.get(name);
        return { name, score: entry === undefined ? 0 : cosine(query, entry.vector) };
      })
      .sort((a, b) => b.score - a.score);
    const result = selectSkills(scored, selection);
    const selected = result.selected.map((s) => s.name);

    const missing = expected.filter((n) => !selected.includes(n));
    const ok = missing.length === 0 && (expected.length > 0 || selected.length === 0);
    if (ok) passed += 1;

    console.log(`── "${prompt}"`);
    console.log(`   expect: ${expected.length > 0 ? expected.join(', ') : '(none)'}`);
    console.log(`   top   : ${scored.slice(0, 6).map((s) => `${s.name}:${s.score.toFixed(3)}`).join('  ')}`);
    console.log(`   got   : ${selected.length > 0 ? selected.join(', ') : '(none)'}   ${ok ? '✓' : '✗'}`);
    console.log('');
  }
  console.log(`passed ${passed}/${CASES.length}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
