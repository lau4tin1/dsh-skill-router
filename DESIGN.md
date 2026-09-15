# Skill Router — Design

> A plugin for DeepSeek Harness that embeds skills, embeds the user's task, and
> auto-injects only the *relevant* skill(s) into the prompt — RAG over skills.
>
> **Status:** design only. No code yet. This document is the v1 proposal and the
> decisions we still need to confirm.

---

## 1. Goal (v1)

When a user sends a task, automatically:

1. Compare the task against every available skill **without an LLM round-trip**.
2. Select the skill(s) that are *clearly relevant* — zero, one, or many.
3. Inject the full instruction body of only those skills into the prompt.
4. Let the model work from those injected instructions.

The measure of success is simple: the right skill(s) reach the model, the wrong
ones and the irrelevant ones don't, and the agent never has to *choose* to load
a skill.

---

## 2. What already exists in DeepSeek Harness

This matters, because most of the plumbing is already shipped. Our plugin should
**build on it, not replace it**.

| Concern | Existing package | What it gives us |
|---|---|---|
| Skill registry | `@deepseek-ai/dsh-skill` | `ctx.skills` — one merged catalog from all providers, `list()`/`get(name)`, `register()`. |
| Local skills | `@deepseek-ai/dsh-skill-filesystem` | Discovers `<root>/<name>/SKILL.md` and `<root>/<name>.md`; parses frontmatter; watches for changes. |
| Model-facing consumer | `@deepseek-ai/dsh-tool-skill` | Renders a name+description catalog and a `skill` loader tool; handles `/name`. |
| Context injection | `dsh-agent-instructions` (pattern) | `agent/pre-step` listener → inject durable `user`-role `<system-reminder>` blocks. |
| Shared rendering | `@deepseek-ai/dsh-skill` | `renderSkillContent(...)` → the canonical `<skill_content>` block. |
| LLM service | `@deepseek-ai/dsh-llm` | `ctx.llm` — **chat completions only; no embeddings seam exists.** |
| Durable non-session data | `@deepseek-ai/dsh-storage` + `dsh-storage-json` | `ctx.storage` for the persistent embedding index. |

### 2.1 How skills are used today (the "catalog + tool" pattern)

Skills are **not** fed to the model all at once, and they are **not** "prompt
injection" in the security sense — they are *context/instruction injection*:
trusted content deliberately placed into the model's context. The standard
pattern (Claude Code, and DSH's `dsh-tool-skill`) is two-stage:

1. **Catalog — all skills, summaries only.** Before the first request the model
   receives a durable message listing every skill's `name` and a capped
   `description`. (The `<available_skills>` block visible in this conversation
   is exactly this.) Full bodies are *not* included.
2. **Lazy load — full body, on demand.** The model calls the `skill` tool with
   a name (or the user types `/name`) and receives that one skill's full body
   as a `<skill_content>` block, retained as ordinary tool history.

So: **all summaries are fed at once; full bodies are pulled one at a time, only
when chosen.** Our router replaces step 2's model *choice* with a deterministic
embedding match that auto-injects the bodies of the chosen skills.

Key facts we build on:

- A skill's **routing surface already exists** in its YAML frontmatter:
  `name` (required), `description` (required), and `whenToUse` (optional).
  These are the fields the current model-driven catalog already routes on.
- The **injection seam is proven**: a plugin listens on `agent/pre-step` and
  hands the "enter" decision a durable `user`-role message framed in
  `<system-reminder>`, escaping any literal `</system-reminder>` in content.
- There is **no vector store / embeddings capability** in DSH today, so the
  embedding backend is ours to design (see §7).

**Consequence:** our plugin is a *second consumer* of the skill registry,
sibling to `dsh-tool-skill`. It does **not** reimplement discovery, watching,
frontmatter parsing, or the registry.

---

## 3. The core idea, restated

```
skills ──(embed once, cache)──▶ skill index (name → vector + body)

user task ──(embed)──▶ query vector
                          │
                          ▼
            cosine similarity against every skill
                          │
                          ▼
        selection rule (gap / clustering cut)
                          │
              ┌───────────┴───────────┐
              │  selected (0..N)      │
              ▼                       ▼
      inject full bodies          inject nothing
      (ordered by score)          (no skill matched)
                          │
                          ▼
               combined prompt → LLM
```

This is exactly RAG, with skills as the corpus and the *whole skill body* (not a
snippet) as the retrieved document.

---

## 4. Verdict on the proposed approach

**It's a good idea, and the instinct is the right one.** Specifically:

1. **Automation over delegation.** Today the model must (a) read the catalog,
   (b) decide a skill applies, (c) call the `skill` tool, (d) read the result.
   Embedding collapses that into a deterministic, local step with no model
   round-trip and no chance the model forgets to load the right skill.

2. **"Not top-k" is correct.** A task can need zero, one, or several skills.
   A fixed `k` forces either noise (too many) or omissions (too few). A
   threshold/relative rule yields *variable cardinality*, which is the right
   primitive for this problem.

3. **It composes with what exists.** We reuse the registry, the provider, the
   frontmatter routing fields, and the injection seam. The new code is: an
   embedding backend, a similarity scorer, a selection rule, and an index cache.

**The honest risks** (each has a mitigation in §9):

- **Threshold calibration.** Cosine similarity is *not* on a universal 0–1
  scale you can threshold once — different embedding models produce different
  absolute score distributions. A hard-coded "significantly beyond a value"
  will silently misbehave. → Use a *relative / gap-based* rule, with only a
  weak absolute floor to reject the "nothing is relevant" case, and expose
  scores for tuning (see §6).
- **Vocabulary mismatch.** A task and a skill can mean the same thing with
  different words; pure embedding similarity can miss it. → Include `name` and
  `whenToUse` in the embedded text (not just `description`); keep a cheap
  lexical (BM25) baseline as a fallback/comparison in v2.
- **Over-injection.** Several near-duplicate or generic skills ("helps you
  code") can all clear the bar and blow up context. → Cap skills *and* total
  injected bytes; order by score.
- **Embedding availability.** If the embedding endpoint is down or unconfigured,
  the router must **fail open** (inject nothing) and never block the user's turn.
- **The "select beyond others" rule needs a definition.** That's §6.

### When it's worth it (practical value)

The value is **not** only at "thousands of skills" scale. Three wins exist at
any scale, and one grows with scale:

1. **One less model round-trip (any scale).** The catalog+tool pattern makes
   the model call `skill` and *wait* for the body before acting. Auto-injection
   removes that turn — a latency/cost win and a reliability win (the model can't
   forget to load a skill, or load the wrong one).
2. **Determinism and auditability (any scale).** You get a numeric relevance
   score per skill, so "why was this chosen?" is answerable and testable — hard
   to get from "the model decided".
3. **Bounded context (any scale).** You inject exactly the selected bodies
   instead of *every* summary plus whatever the model loads.
4. **Scale (the case you're picturing).** A catalog of hundreds or thousands of
   summaries costs tokens every session and dilutes the model's attention and
   selection accuracy. Embedding retrieval stays near-constant cost and degrades
   gracefully.

It is **not** worth it when: the skill set is tiny (roughly < 10) with short
descriptions, skills are highly generic and overlapping (those belong as
always-on instructions, not routed skills), or the agent is single-purpose.
The sweet spot is tens to thousands of skills, where catalog cost and selection
accuracy start to bite.

---

## 5. What to embed

v1 embeds the **routing text**, not the whole body:

```
routing_text = "<name>\n<description>\n<whenToUse>"
```

Why:

- These are the exact fields the current model-driven routing already uses, so
  we are automating a decision the system already trusts.
- Short and cheap to embed; the index builds fast.
- Embedding the full body is noisier for matching (bodies mix "how to do it"
  with "when to use it") and more expensive.

`whenToUse` is optional; omit it when absent. If `whenToUse` is absent *and* the
description is very short, optionally fold in the first heading's text as a
fallback — a v2 refinement, not v1.

The **full body** is what we *inject*, not what we *embed*. Retrieved skills are
injected via the shared `<skill_content>` rendering (see §8).

### 5.1 Why not chunk the body (v1)

Chunking the body for *routing* is usually counterproductive, and the skills in
this repo confirm it:

- **The description is already a hand-tuned trigger spec.** In the Anthropic
  convention, `description` carries explicit TRIGGER/SKIP logic (the `claude-api`
  description spells out when to read and when to skip), so it is the
  highest-signal routing text. A missing `whenToUse` loses nothing here — that
  field is folded into the mandatory `description`.
- **Bodies are execution-oriented noise for matching.** They are dominated by
  code, tables, script paths, and examples ("how to do it"), not "when to use
  it". A task "merge PDFs" matches the `pdf` *description* far more reliably
  than a random `PdfWriter` code chunk.
- **Bodies are often pointers, not content.** `claude-api` says "read the
  `{lang}/` files", `pdf` says "see REFERENCE.md / FORMS.md" — the real content
  lives across hundreds of files, not the `SKILL.md` bodies. Chunking would
  either miss it or balloon into embedding the whole tree.
- **Chunk→skill attribution is lossy.** A matched chunk still has to be mapped
  back to a skill; one spurious generic chunk (a shared "pip install" line in
  `docx`/`pptx`/`xlsx`) can pull in the wrong or several skills.

Routing is a *small-label classification* problem with hand-written labels; body
chunking is a tool for *content retrieval*, a different problem. Chunk only when
the description is genuinely thin — and even then, prefer enriching the routing
text (or auto-generating a `whenToUse`) over chunking, or use **max-pool
aggregation** (score body chunks, take the max per skill) instead of mapping a
chunk to a skill.

A cheaper recall boost that needs no chunking: fuse the embedding score with a
whole-body **BM25** lexical score (catches exact terms like `openpyxl`, `.pptx`,
`recalc.py` that embeddings can miss). Candidate for v1.5, not v1.

---

## 6. Selection rule (the heart of the plugin)

Inputs: normalized query vector `q`, skill vectors `s_i` (unit length), so
`sim_i = q · s_i` is cosine similarity.

**The principle:** the absolute value of cosine similarity is model-dependent
and unstable, so selection should be **relative / gap-based** — "which scores
form the same cluster as the top?" — with only a *weak* absolute floor to
answer "is anything relevant at all?".

### 6.1 Two questions, in order

1. **Is anything relevant?** If even the top score is below a weak sanity floor
   (`max < minScore`), select nothing. This is the *only* absolute number, and
   it guards the degenerate "best of a bad bunch" case — for `[0.05, 0.06,
   0.07]` a pure relative rule would keep the top cluster even though nothing
   is relevant.
2. **Which scores belong to the top cluster?** Split the sorted scores at the
   biggest drop (a 1-D clustering / outlier-separation cut) and keep the top
   group.

### 6.2 v1 rule — "largest gap" cut (parameter-free)

Sort descending `s1 >= s2 >= ... >= sn`; compute adjacent gaps
`g_i = s_i - s_(i+1)`; keep the prefix up to the largest gap:

```
if max < minScore:  select nothing
else:               keep s1..s_k  where k = argmax_i(g_i)
```

For your example `[0.9, 0.8, 0.25, 0.2, 0.1]` the gaps are `0.1, 0.55, 0.05,
0.1`; the largest drop sits between `0.8` and `0.25`, so it keeps exactly the
two standouts — no fixed per-skill threshold needed.

- `minScore` is now a **weak floor on the max**, not a per-skill cutoff, so it
  is far less sensitive to the embedding model than a hard per-skill threshold.
- A single skill has no gap: keep it iff it clears the floor.
- If more than `maxSkills` survive, keep the top `maxSkills`; cap total bytes
  with `maxInjectedBytes`; empty selection → inject nothing.

### 6.3 Simpler alternative — ratio-to-max

A one-parameter version of the same idea: keep `sim_i >= ratioThreshold * max`
(e.g. `0.75`). Easier to reason about, slightly less adaptive than the largest
gap; a good fallback when score lists are short or noisy.

### 6.4 The clustering view (what you're reaching for)

"Standouts" = the top connected component of a 1-D score distribution. The
largest-gap rule is a degenerate single-split clustering. Other cheap,
deterministic 1-D separators to consider in v2:

- **Otsu's method** — maximize between-class variance between kept and dropped
  groups (the classic image-binarization threshold).
- **Knee / gap statistic** — find where the sorted score curve stops falling
  steeply.
- **Jenks natural breaks** — minimize within-group variance for a small k.

All are `O(n log n)` or less on the score list and need no training. Caution:
with only a handful of skills and noisy similarities, a fancier clustering can
overfit — the largest-gap rule (plus the weak floor) is the right v1, and Otsu
is the natural upgrade if real sessions show it misfiring.

### 6.5 Defaults (placeholders — calibrate per embedding model)

| Param | Default | Meaning |
|---|---|---|
| `rule` | `largest-gap` | `largest-gap` \| `ratio-to-max` |
| `minScore` | `0.12` | Weak floor on the *top* score: "is anything relevant at all?". Calibrated for the default MiniLM model on the demo corpus (matches ≥ 0.21, noise ≤ 0.09); re-calibrate when the model or corpus changes. |
| `ratioThreshold` | `0.75` | Used only when `rule: ratio-to-max`. |
| `maxSkills` | `4` | Hard cardinality cap. |
| `maxInjectedBytes` | `65536` | Hard cap on total injected body bytes. |

`rule` and `minScore` must be first-class config and **logged** with the
per-query scores so they can be tuned against real sessions.

---

## 7. Embedding backend (implemented: one real local model)

DSH has no embeddings service, so the plugin owns the embedding call behind a
small interface (`EmbeddingBackend.embed(texts)`), with one implementation:

- **transformers.js (ONNX)** — a real semantic model running in-process.
  Default `onnx-community/all-MiniLM-L6-v2-ONNX` (384-dim dense vectors).
  The model downloads from huggingface.co on first use (~90MB fp32; `q8` is
  ~4x smaller with tiny quality loss) into `<cacheDir>/models` under
  `~/.dsh` — outside the working directory — and then runs fully offline.
  Bigger models are a config change (`embedding.model`): `bge-base-en-v1.5`
  (~110MB) or `bge-large-en-v1.5` (~1.3GB, near-SOTA retrieval).

The interface stays so the index/selection logic is decoupled from the model
runtime (and a different engine could be added without touching the pipeline).

**Calibration** (weak `minScore` floor, measured on the 24-skill demo corpus
with the default model): `0.12` — correct matches ≥ 0.21, unrelated noise
≤ 0.09. Re-tune whenever the model or corpus changes; the floor is never
portable as-is.

---

## 8. Index lifecycle and injection

### 8.1 Building the index

- On first observation of the skill catalog, embed each skill's routing text
  once and store `{ name, vector, routingDigest, body }`.
- Keyed by a **digest of the routing text** (not the name) so edits re-embed
  only changed skills. Name changes are handled by digest mismatch + old-entry
  removal.
- Persist the index to `<cacheDir>/skill-index.json` (default
  `~/.dsh/skill-router/`), tagged with the backend/model id. On startup the
  file is restored, so unchanged skills are never re-embedded; `sync()`
  re-embeds only entries whose routing digest changed. Writes are atomic
  (temp file + rename), a model switch invalidates the file, and a failed
  save never fails the turn. Watch `skills/change` (the registry emits it on
  invalidation) and re-embed only changed entries.

### 8.2 Query time

- The **query text** is the latest claimed user message (the current task).
  Simple v1: the raw latest user text; strip a leading `/name` gesture token if
  present.
- Embed it, score against the index, apply §6, and cache the query→embedding
  result in memory keyed by exact text (avoids re-embedding on retries/resume).

### 8.3 Injection

On each eligible `agent/pre-step` where the selection changed (digest over the
selected skill names), inject a durable `user`-role block:

```markdown
<system-reminder>
The following skills were automatically selected for this task. Follow their
instructions.
<selected_skills>
<skill_content name="...">...full body...</skill_content>
</selected_skills>
</system-reminder>
```

- Reuse `renderSkillContent` from `dsh-skill` so injected bodies match what the
  `skill` tool returns today (one canonical shape).
- **Escape** literal `</system-reminder>` in any skill body.
- Re-inject only when the *selected set* changes, mirroring `dsh-tool-skill`'s
  replacement-catalog digest behavior, so we don't re-spam identical blocks
  every turn.
- Empty selection → inject nothing, and emit an empty retirement block only if
  a previous non-empty block is still visible.

### 8.4 Relationship to `dsh-tool-skill`

Two workable setups:

- **(v1, recommended) Replace.** Mount the router instead of `dsh-tool-skill`.
  No catalog message, no `skill` tool — routing is fully automatic. Simplest to
  reason about.
- **(optional) Coexist.** Keep `dsh-tool-skill` mounted. The router auto-injects
  the strongly-relevant skills; the catalog+tool remains as a manual escape
  hatch for the long tail. Costs more tokens (catalog is always present).

Keep both as config choices; the router itself is orthogonal to the tool.

---

## 9. Failure modes and mitigations

| Failure | Behavior | Mitigation |
|---|---|---|
| Embedding endpoint down / not configured | **Fail open**: inject nothing, log a warning. | Router must never block a turn; catch at the embed boundary. |
| Index not built yet (cold start) | First turn injects nothing. | Lazy build + warm-up; optionally build synchronously when the skill count is small. |
| Selection too strict | Skills that should load are missed. | Gap rule is scale-free; tune the weak `minScore` floor + `maxSkills`; logged scores. |
| Selection too loose | Irrelevant or too many skills injected. | Weak `minScore` floor on the max + `maxSkills` + `maxInjectedBytes`. |
| Overlapping skills | Several near-duplicates all inject. | Same-name dedup is the registry's job; accept near-duplicate injection in v1, revisit with a diversity pass in v2. |
| Body too large | Context blowup. | `maxInjectedBytes` truncates whole-skill bodies (like `agent-instructions` budget). |
| Catalog changes mid-session | Stale vectors. | Digest-keyed re-embed on `skills/change`. |
| Query embedding latency | Added per-turn delay. | In-memory query cache; async warm; document the cost. |

---

## 10. Proposed configuration (v1, small)

```yaml
- name: '@deepseek-ai/dsh-skill'              # already required
- name: '@deepseek-ai/dsh-skill-filesystem'   # already required
- name: '<skill-router>'                      # our plugin
  config:
    enabled: true
    embedding:
      model: onnx-community/all-MiniLM-L6-v2-ONNX   # optional; any HF ONNX model
      dtype: fp32                                    # or q8 (~4x smaller)
      dimensions: 384                                # optional; validated
    cacheDir: ~/.dsh/skill-router     # optional; model weights + index file live here
    rule: largest-gap          # 'largest-gap' | 'ratio-to-max'
    minScore: 0.12             # weak floor, calibrated for the default model
    ratioThreshold: 0.75       # only used when rule: ratio-to-max
    maxSkills: 4
    maxInjectedBytes: 65536
    indexStorage: {}                  # route to ctx.storage json backend
```

Exact field names/shapes are to be finalized at implementation against the
real config schema (schemastery) — this is the shape, not the contract.

---

## 11. Non-goals for v1

- No chunking of skill bodies into sub-embeddings (embed one routing text per
  skill).
- No hybrid lexical+semantic retrieval (BM25) and no re-ranking / cross-encoder
  — worth a v2 comparison, not v1.
- No ML "router" model trained on past tasks (this is *not* the same thing as
  the "LLM skill router" literature; we do pure retrieval).
- No remote/shared vector store or multi-tenant index — one JSON sidecar.
- No per-section or per-resource embedding of a skill's referenced files.

---

## 12. Open decisions (confirm before code)

1. **Embedding provider/model.** External OpenAI-compatible endpoint vs local
   model. Determines dimensions, cost, latency, and `minScore` calibration.
2. **Replace vs coexist** with `dsh-tool-skill` (see §8.4).
3. **Selection rule + weak `minScore` floor** — confirm `largest-gap` vs
   `ratio-to-max`, and calibrate the floor against a seed set of tasks × skills
   before trusting any number.
4. **Exact injection framing** — reuse `<skill_content>` verbatim vs a leaner
   router-specific block; and whether to include a one-line "why selected"
   (relevance score) annotation for debuggability.
5. **Package name** — working name `skill-router` / `dsh-skill-router`; confirm.

---

## 13. Suggested build order (when we start coding)

1. Stub embedding backend + index in memory; wire a debug endpoint/log that
   prints the per-query ranked scores (no injection yet).
2. Selection rule (§6) as a pure function + unit tests against synthetic score
   lists (0 / 1 / many / gap / floor cases).
3. Real embedding backend (OpenAI-compatible).
4. Index persistence + `skills/change` invalidation.
5. `agent/pre-step` injection with digest-based replace + escaping.
6. Config surface + `maxInjectedBytes` budget + logging of scores.
7. Calibrate thresholds on a seed set of skills and tasks.

---

## 14. Note on demo skills

For local testing, skills must land where the filesystem provider scans. The
default project roots are `<projectRoot>/.dsh/skills` (rank 100) and
`<projectRoot>/.agents/skills` (rank 200), plus the user root `~/.dsh/skills`
(rank 400). A bare `skills/` directory at the workspace root is **not** a
default root.

The demo corpus lives in `.agents/skills/` (24 skills — Anthropic's set plus
vercel/community skills), so at runtime the mounted plugin discovers and
watches them automatically; `demo/e2e.ts` parses the same files standalone.
The provider watches these roots, so adding/renaming/deleting a skill or
editing its frontmatter mid-session triggers `skills/change` → the router
re-syncs and re-embeds only what changed.
