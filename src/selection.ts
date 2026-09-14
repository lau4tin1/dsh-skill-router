/**
 * src/selection.ts — the selection rule.
 *
 * This is the "heart" of the skill router and the only part with real logic
 * worth getting right. It answers ONE question:
 *
 *     Given every skill's similarity score for the current task,
 *     which skills should we inject into the prompt?
 *
 * It is deliberately PURE: no filesystem, no network, no DeepSeek Harness
 * imports. It takes a list of (name, score) and returns a decision, so it is
 * trivial to test and reason about in isolation.
 *
 * Design reference: DESIGN.md §6 ("Selection rule").
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** One skill together with its similarity score for the current task. */
export interface ScoredSkill {
  /** Kebab-case skill name (as reported by the skill registry). */
  name: string;
  /**
   * Cosine similarity in [-1, 1]; higher = more relevant.
   *
   * Important: we never trust the *absolute* value here. Different embedding
   * models produce different score distributions, so "0.5" means nothing on
   * its own. What is portable is the *ordering* and the *gaps* between scores.
   */
  score: number;
}

/** The tunable knobs of the selection rule. */
export interface SelectionConfig {
  /**
   * Which algorithm decides where the "top cluster" ends:
   *   - 'largest-gap'  : cut at the biggest drop between two sorted scores.
   *   - 'ratio-to-max' : keep every score >= ratioThreshold * topScore.
   */
  rule: 'largest-gap' | 'ratio-to-max';
  /**
   * The weak ABSOLUTE floor, applied to the TOP score only. If even the best
   * score is below this, nothing is relevant and we select nothing.
   *
   * This is the only absolute number, and it answers "is anything relevant at
   * all?" — NOT "is THIS skill relevant?". That is what makes it much more
   * robust than a per-skill threshold.
   */
  minScore: number;
  /**
   * Only used when rule === 'ratio-to-max'. A score is kept when it is at
   * least this fraction of the top score (e.g. 0.75 => "within 75% of best").
   */
  ratioThreshold: number;
  /**
   * Hard upper bound on how many skills may be injected. This protects the
   * context window from a burst of near-duplicate matches.
   */
  maxSkills: number;
}

/** Everything the caller needs: the decision plus numbers for logging/tuning. */
export interface SelectionResult {
  /** The chosen skills, sorted best-first. Empty when nothing is relevant. */
  selected: ScoredSkill[];
  /** The top similarity score (0 when the input was empty). */
  max: number;
  /** Adjacent gaps between sorted scores — useful for debugging the cut. */
  gaps: number[];
  /** How many skills the rule *wanted* to keep, BEFORE the maxSkills cap. */
  cutIndex: number;
  /** True when nothing was relevant (an empty selection is the correct answer). */
  empty: boolean;
}

// ---------------------------------------------------------------------------
// The rule
// ---------------------------------------------------------------------------

/**
 * Decide which skills to inject.
 *
 * Cosine similarity's absolute scale is not portable across embedding models,
 * so we do NOT threshold each score against a fixed number. Instead we ask two
 * questions, in order:
 *
 *   1. "Is anything relevant at all?"  -> keep going only if max >= minScore.
 *   2. "Which scores form the top cluster?" -> cut at the biggest gap (or the
 *      ratio-to-max line) and keep the top group.
 *
 * Motivating example (scores sorted): [0.9, 0.8, 0.25, 0.2, 0.1]
 * The gaps are [0.1, 0.55, 0.05, 0.1]. The biggest drop (0.55) sits between
 * 0.8 and 0.25, so we keep exactly the two standouts {0.9, 0.8} — no fixed
 * per-skill threshold required.
 */
export function selectSkills(
  scored: readonly ScoredSkill[],
  config: SelectionConfig,
): SelectionResult {
  // Sort a copy, best first. We never mutate the caller's array.
  const sorted = [...scored].sort((a, b) => b.score - a.score);

  // Question 1: is anything relevant at all?
  if (sorted.length === 0) {
    return { selected: [], max: 0, gaps: [], cutIndex: 0, empty: true };
  }
  const max = sorted[0].score;
  if (max < config.minScore) {
    // Best-of-a-bad-bunch: even the top score is too weak, so select nothing.
    // A pure relative rule would wrongly keep the top cluster here.
    return {
      selected: [],
      max,
      gaps: gapsOf(sorted),
      cutIndex: 0,
      empty: true,
    };
  }

  // Question 2: where does the top cluster end?
  const cutIndex = cutAt(sorted, config);

  // Hard cap so a burst of near-duplicate matches can't blow up the context.
  const keep = Math.min(cutIndex, config.maxSkills);

  return {
    selected: sorted.slice(0, keep),
    max,
    gaps: gapsOf(sorted),
    cutIndex,
    empty: false,
  };
}

/** Adjacent gaps between sorted scores: gaps[i] = score[i] - score[i+1]. */
function gapsOf(sorted: readonly ScoredSkill[]): number[] {
  const gaps: number[] = [];
  for (let i = 0; i < sorted.length - 1; i++) {
    gaps.push(sorted[i].score - sorted[i + 1].score);
  }
  return gaps;
}

/** How many sorted skills the rule keeps (before the maxSkills cap). */
function cutAt(sorted: readonly ScoredSkill[], config: SelectionConfig): number {
  // With zero or one skill there is nothing to split; keep whatever is there.
  if (sorted.length <= 1) return sorted.length;

  if (config.rule === 'ratio-to-max') {
    // Keep every score that is at least ratioThreshold of the top score.
    // Example: [0.9, 0.8, 0.25, ...] with ratio 0.75 keeps scores >= 0.675
    //          => {0.9, 0.8}.
    const floor = config.ratioThreshold * sorted[0].score;
    let i = 0;
    while (i < sorted.length && sorted[i].score >= floor) i++;
    return Math.max(1, i);
  }

  // 'largest-gap': find the biggest drop between two adjacent sorted scores
  // and cut there. This is a one-dimensional clustering / "find the standouts"
  // step — the standouts are everything above the widest gap.
  let bestIndex = 0;
  let bestGap = -Infinity;
  for (let i = 0; i < sorted.length - 1; i++) {
    const gap = sorted[i].score - sorted[i + 1].score;
    // On a tie we keep the EARLIER (leftmost) gap, i.e. keep MORE skills —
    // more recall is the safer default for routing.
    if (gap > bestGap) {
      bestGap = gap;
      bestIndex = i;
    }
  }
  // bestIndex is the index of the last score BEFORE the gap, so we keep
  // bestIndex + 1 skills.
  return bestIndex + 1;
}

// ---------------------------------------------------------------------------
// Optional demo — runs only when this file is executed directly:
//
//     node src/selection.ts
//
// It is skipped when the module is imported by the plugin. You can delete this
// section once the real plugin is wired up.
// ---------------------------------------------------------------------------

function main(): void {
  const gapRule: SelectionConfig = {
    rule: 'largest-gap',
    minScore: 0.25,
    ratioThreshold: 0.75,
    maxSkills: 4,
  };
  const ratioRule: SelectionConfig = { ...gapRule, rule: 'ratio-to-max' };

  const show = (label: string, scored: ScoredSkill[], config: SelectionConfig) => {
    const result = selectSkills(scored, config);
    const names = scored.map((s) => `${s.name}:${s.score}`).join('  ');
    const picked = result.empty
      ? '(empty)'
      : result.selected.map((s) => s.name).join(', ');
    console.log(`${label}`);
    console.log(`  rule   : ${config.rule}`);
    console.log(`  scores : ${names}`);
    // Round gaps for display only — floating-point subtraction leaves tiny
  // noise (0.1 -> 0.099999...). The decision itself uses the unrounded values.
  const gaps = result.gaps.map((g) => Math.round(g * 1000) / 1000);
  console.log(`  gaps   : [${gaps.join(', ')}]  max=${result.max}  cutIndex=${result.cutIndex}`);
    console.log(`  picked : ${picked}`);
    console.log('');
  };

  show('1. Standouts (keep the top two)', [
    { name: 'a', score: 0.1 },
    { name: 'b', score: 0.2 },
    { name: 'c', score: 0.25 },
    { name: 'd', score: 0.8 },
    { name: 'e', score: 0.9 },
  ], gapRule);

  show('2. Nothing relevant (all scores weak)', [
    { name: 'a', score: 0.05 },
    { name: 'b', score: 0.06 },
    { name: 'c', score: 0.07 },
  ], gapRule);

  show('3. Single skill', [{ name: 'only', score: 0.9 }], gapRule);

  // The two rules genuinely differ here: [0.9, 0.6, 0.29].
  // largest-gap sees {0.9, 0.6} as the top cluster (the biggest drop is after
  // 0.6); ratio-to-max (0.75) requires a score to be within 75% of 0.9
  // (= 0.675), so it keeps only {0.9}.
  show('4a. largest-gap keeps the top cluster {0.9, 0.6}', [
    { name: 'a', score: 0.9 },
    { name: 'b', score: 0.6 },
    { name: 'c', score: 0.29 },
  ], gapRule);

  show('4b. ratio-to-max (0.75) keeps only {0.9}', [
    { name: 'a', score: 0.9 },
    { name: 'b', score: 0.6 },
    { name: 'c', score: 0.29 },
  ], ratioRule);
}

// `process.argv[1]` is the path this file was launched with; `import.meta.url`
// is the module's own URL. When they match, we were run directly (not imported).
import { pathToFileURL } from 'node:url';
if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  main();
}
