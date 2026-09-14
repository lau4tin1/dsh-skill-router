/**
 * src/render.ts — render the injected skill block.
 *
 * Turns the SELECTED skills (full bodies already loaded from the registry)
 * into one durable model-facing `<system-reminder>` block:
 *
 *   <system-reminder>
 *   The following skills were automatically selected for this task...
 *   <selected_skills>
 *   <skill_content name="...">...body...</skill_content>
 *   ...
 *   </selected_skills>
 *   </system-reminder>
 *
 * Each skill body is rendered by DSH's own `renderSkillContent()` (from
 * @deepseek-ai/dsh-skill), so an auto-injected body looks EXACTLY like what the
 * `skill` tool returns — one canonical shape, no divergence.
 *
 * Design reference: DESIGN.md §8.3 ("Injection").
 */

import { renderSkillContent } from '@deepseek-ai/dsh-skill';
import type { SkillResourceBase } from '@deepseek-ai/dsh-skill';

/** The minimal shape of a loaded skill this renderer needs. */
export interface SelectedSkill {
  name: string;
  provider: string;
  content: string;
  resourceBase?: SkillResourceBase;
}

const FRAME_CLOSE = '</system-reminder>';
const FRAME_CLOSE_ESCAPED = '<\\/system-reminder>';

/**
 * Escape a skill body so it cannot close the plugin-owned `<system-reminder>`
 * frame. renderSkillContent() embeds the body verbatim (skills are trusted
 * local content), but repository-controlled text must still not be able to
 * terminate our frame — this mirrors the convention used by
 * dsh-agent-instructions.
 */
export function escapeFrame(text: string): string {
  return text.replaceAll(FRAME_CLOSE, FRAME_CLOSE_ESCAPED);
}

/**
 * Render the complete injected block for a list of selected skills.
 * Caller passes skills already in the desired (score-descending) order.
 */
export function renderSelection(skills: readonly SelectedSkill[]): string {
  const bodies = skills
    .map((skill) => escapeFrame(renderSkillContent(skill)))
    .join('\n');

  return [
    '<system-reminder>',
    'The following skills were automatically selected for this task. Follow their instructions.',
    '',
    '<selected_skills>',
    bodies,
    '</selected_skills>',
    '</system-reminder>',
  ].join('\n');
}

// ---------------------------------------------------------------------------
// Demo — runs only when executed directly:  node src/render.ts
// ---------------------------------------------------------------------------

function main(): void {
  const skills: SelectedSkill[] = [
    {
      name: 'pdf',
      provider: 'filesystem',
      // Note the attempted frame-close inside the body — it must be escaped.
      content: 'Use pypdf to merge PDFs.\n\nA body must not close this frame: </system-reminder>',
    },
    {
      name: 'xlsx',
      provider: 'filesystem',
      content: 'Use openpyxl for spreadsheets.',
    },
  ];

  console.log(renderSelection(skills));
}

import { pathToFileURL } from 'node:url';
if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  main();
}
