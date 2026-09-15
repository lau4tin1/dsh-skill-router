# Skill Router for DeepSeek Harness

RAG-style skill routing plugin for DeepSeek Harness: embeds skills (their
`name + description + whenToUse` routing surface), embeds each user task, keeps
only the clearly-relevant skills with a gap-based selection rule, and
auto-injects their full bodies into the prompt — no model round-trip to load a
skill.

- **Design**: see [DESIGN.md](DESIGN.md).
- **Status**: V1 code complete — type-checks against the real DSH `0.1.2-rc.1`
  types, builds, and passes 20/23 bilingual routing prompts on real skills
  (7/7 Chinese, 13/16 English; see `demo/e2e.ts` for the case list)
  (`node demo/e2e.ts`). **Not yet mounted into a live DSH profile** — that
  integration step is intentionally deferred.

## Layout

- `src/index.ts` — the Cordis plugin entry (`name` / `inject` / `Config` / `apply`).
- `src/selection.ts` — the selection rule (largest-gap / ratio-to-max, weak floor).
- `src/similarity.ts` — normalization + cosine similarity (pure).
- `src/embedding.ts` — embedding backends: `local` (offline hashing vectorizer) and `http` (OpenAI-compatible).
- `src/skillIndex.ts` — the index: build, embed, digest-diff sync on change.
- `src/config.ts` — schemastery config schema + mappers.
- `src/render.ts` — renders the injected `<system-reminder>` block (reuses DSH's `renderSkillContent`).
- `demo/e2e.ts` — end-to-end routing test over downloaded skills.

## Build

```sh
npm install        # toolchain (typescript, esbuild); DSH types come from a local checkout symlink
npm run typecheck  # tsc --noEmit against the real DSH types
npm run build      # esbuild -> lib/index.js (@deepseek-ai/* kept external)
```

## Demo

Skills live in `.agents/skills/` — a real DSH scanned root (rank 200, see
DESIGN.md §14). The demo parses them the same way the registry would surface
them:

```sh
node demo/e2e.ts           # plugin defaults
node demo/e2e.ts 0.2       # experiment with the minScore floor
```

## Config (defaults)

```yaml
enabled: true
embedding:
  model: Xenova/bge-m3                          # optional; any HF ONNX embedding model
  dtype: q8                                     # or fp32
cacheDir: ~/.dsh/skill-router                   # optional; models + index live here
rule: largest-gap                               # or ratio-to-max
minScore: 0.12                                  # weak floor, calibrated for the default model
ratioThreshold: 0.75
maxSkills: 4
maxInjectedBytes: 65536
```

### Languages

Chinese and English both work out of the box — the default model is
multilingual, and a Chinese prompt matches an English skill description (and
vice versa) by meaning, cross-lingual. One DSH registry rule to respect:
skill **names** must stay kebab-case ASCII (`pdf-tools`), but `description`,
`whenToUse`, and the body can be Chinese. The injected instruction block is
bilingual (中文/English).

### Embedding model

One engine: a real local model via transformers.js (ONNX) — dense semantic
vectors; synonyms and paraphrases match, word sharing is not required.
Default `Xenova/bge-m3` (q8, ~543MB download, 1024-dim, [CLS] pooling) —
flagship multilingual quality for Chinese + English. Alternatives via
`embedding.model`: `multilingual-e5-small` (fast, needs query/passage
prefixes — handled automatically), `bge-large-en-v1.5`, `bge-large-zh-v1.5`.

Scoring de-biases the vectors by subtracting the corpus mean ("centering"),
which removes the shared direction that would otherwise make unrelated texts
score ~0.8. Inference is pinned to a single thread so results are
deterministic.

All heavy data lives OUTSIDE the working directory: the model downloads once
from huggingface.co into `<cacheDir>/models`, and the skill-embedding index
persists at `<cacheDir>/skill-index.json` (atomic writes, tagged with the
model id), so unchanged skills are never re-embedded after a restart.

## Dev note

`node_modules/@deepseek-ai` and `node_modules/js-yaml` are local symlinks into
a DSH checkout for type-checking and demos; recreate them after a fresh
install. At runtime a real DSH profile provides the `@deepseek-ai/*` packages
(they are declared as optional peers and kept external by the build).
