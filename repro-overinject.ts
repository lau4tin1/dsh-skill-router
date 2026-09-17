/**
 * Diagnostic reproduction: score the four real questions from the session
 * against the persisted skill index, using the CURRENT code paths
 * (transformers backend + centered cosine + selectSkills), and print what
 * would be injected at minScore 0.14 (current) and 0.12 (pre-fix).
 *
 * Run:  node repro-overinject.ts   (delete when done)
 */
import { join } from 'node:path';
import { homedir } from 'node:os';
import { TransformersEmbeddingBackend } from './src/embedding.ts';
import { SkillIndex } from './src/skillIndex.ts';
import { selectSkills } from './src/selection.ts';
import type { SelectionConfig } from './src/selection.ts';

const questions: Record<string, string> = {
  'Q1 (injected: systematic-debugging, academy-guide, algorithmic-art)':
    '哦你的意思是我们目前只使用了相同词匹配?我记得我已经修改过这个逻辑转而投入使用embedder模型了,你能再确认一下吗',
  'Q2 (injected: claude-api)':
    '为什么说“默认 local 后端只做词面相似,换成 http 语义模型后才能减少这类误判。”',
  'Q3 (injected: skill-creator)':
    '目前我们使用的什么算法来挑选合适的skills?',
  'Q4 (injected: nothing — control)':
    '请开始吧',
  'Q5 (positive control — should select pdf)':
    '帮我把这两个 PDF 文件合并成一个,顺便转成图片',
  'Q6 (positive control — should select xlsx)':
    '用 Python 把这个 xlsx 里的公式批量修一下',
  'Q7 (known weak positive — zh project report, raw ~0.126)':
    '帮我写一份项目进度报告',
};

const backend = new TransformersEmbeddingBackend({
  cacheDir: join(homedir(), '.dsh', 'skill-router', 'models'),
});
const index = new SkillIndex(backend);
const indexFile = join(homedir(), '.dsh', 'skill-router', 'skill-index.json');

const loaded = await index.loadFromDisk(indexFile);
console.log(`index loaded: ${loaded}, entries: ${index.size}, backend: ${backend.name}/${backend.modelId}\n`);

const configs: Record<string, SelectionConfig> = {
  'z>=2.5 AND s>=0.05': { rule: 'largest-gap', minScore: 0.05, ratioThreshold: 0.75, maxSkills: 4, zThreshold: 2.5 },
  'z>=2.5 AND s>=0.2': { rule: 'largest-gap', minScore: 0.2, ratioThreshold: 0.75, maxSkills: 4, zThreshold: 2.5 },
  'z>=2.0 AND s>=0.05': { rule: 'largest-gap', minScore: 0.05, ratioThreshold: 0.75, maxSkills: 4, zThreshold: 2 },
  '0.2 only (no z)': { rule: 'largest-gap', minScore: 0.2, ratioThreshold: 0.75, maxSkills: 4 },
};

for (const [label, q] of Object.entries(questions)) {
  const [vec] = await backend.embed([q], { query: true });
  const scored = index.score(vec);
  const top = [...scored].sort((a, b) => b.score - a.score).slice(0, 6);

  console.log(`=== ${label} ===`);
  console.log(`query: ${q}`);
  for (const s of top) console.log(`   ${s.name.padEnd(22)} ${s.score.toFixed(4)}`);
  const sortedAll = [...scored].sort((a, b) => b.score - a.score);
  const tail = sortedAll.slice(-3);
  console.log('   ... tail:', tail.map((s) => `${s.name}:${s.score.toFixed(4)}`).join('  '));
  let first = true;
  for (const [clabel, cfg] of Object.entries(configs)) {
    const r = selectSkills(scored, cfg);
    if (first) {
      const zTop = r.stdDev > 0 ? ((r.max - r.mean) / r.stdDev).toFixed(2) : 'n/a';
      console.log(`  stats: mean=${r.mean.toFixed(4)} std=${r.stdDev.toFixed(4)} z(top)=${zTop}`);
      first = false;
    }
    console.log(`  select @${clabel}: max=${r.max.toFixed(4)} cut=${r.cutIndex} -> ${r.selected.map((s) => s.name).join(', ') || '(none)'}`);
  }
  console.log('');
}
