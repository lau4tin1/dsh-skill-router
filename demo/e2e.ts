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
  // Chinese prompts against the English skill catalog: cross-lingual routing.
  { prompt: '把几个pdf文件合并成一个文档', expected: ['pdf'] },
  { prompt: '用这个csv数据创建一个电子表格', expected: ['xlsx'] },
  { prompt: '帮我写一份项目进度报告发给团队', expected: ['internal-comms'] },
  { prompt: '做一个演示文稿展示季度总结', expected: ['pptx'] },
  { prompt: '部署我的应用到生产环境', expected: ['deploy-to-vercel'] },
  { prompt: '我的应用有个bug，帮我修复一下', expected: ['systematic-debugging'] },
  { prompt: '你好，今天天气怎么样？', expected: [] },
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

  // The real semantic model (first run downloads it to ~/.dsh/skill-router/models).
  // MODEL=<hf-id> [DTYPE=fp32|q8] node demo/e2e.ts — A/B test models without code edits.
  const backend = process.env.MODEL === undefined
    ? createEmbeddingBackend()
    : createEmbeddingBackend({
        model: process.env.MODEL,
        ...(process.env.DTYPE === undefined ? {} : { dtype: process.env.DTYPE as 'fp32' | 'q8' }),
      });
  const index = new SkillIndex(backend);
  await index.build(skills);

  // The plugin's real config path: YAML -> schema -> toSelectionConfig.
  // minScore defaults to 0.12 (calibrated for the MiniLM model).
  const selection = toSelectionConfig(Config({}));
  if (process.argv[2] !== undefined) {
    selection.minScore = Number(process.argv[2]);
  }
  console.log(`selection config: rule=${selection.rule} minScore=${selection.minScore} ratioThreshold=${selection.ratioThreshold} maxSkills=${selection.maxSkills}\n`);

  let passed = 0;
  for (const { prompt, expected } of CASES) {
    const [query] = await backend.embed([prompt], { query: true });
    // index.score() applies corpus centering, exactly like the plugin.
    const scored = index.score(query).sort((a, b) => b.score - a.score);
    if (process.env.DEBUG_SCORES === '1') {
      const top = scored.slice(0, 4).map((s) => `${s.name}:${s.score.toFixed(3)}`).join(' ');
      console.log(`DEBUG "${prompt}" -> ${top}`);
    }
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
