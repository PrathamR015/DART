import { plural } from '../brand/format'
import type { QuestionInput } from '../api'

/** Rough token count shown under the context box (the model limit is 256 tokens per question with context). */
const TOKENS_PER_WORD = 1.33

export const SEED_CONTEXT =
  'Maria is 34 and works as a nurse in Lisbon. She cycles to the hospital every day, has two cats, and is learning ' +
  'Japanese because she plans to visit Tokyo next spring. She dislikes coffee and drinks green tea instead.'

export const SEED_QUESTIONS: QuestionInput[] = [
  { query: 'Where does Maria work?', options: 'hospital, airport, school, bank' },
  { query: 'Which city does Maria live in?', options: 'Lisbon, Madrid, Tokyo, Rome' },
  { query: 'How does she get to work?', options: 'bicycle, bus, car, train' },
  { query: 'Why is she learning Japanese?', options: 'for her job, to visit Tokyo, her cats, school exam' },
]

export function estimateTokens(text: string): number {
  const words = text.trim().split(/\s+/).filter(Boolean).length
  return Math.round(words * TOKENS_PER_WORD)
}

/**
 * How the server will run the questions: the shared-context pass reads the context once, but on a GPU it only pays
 * off from `parallelMin` questions (Health.parallel_min_questions); below that each question runs on the fast path.
 * `parallelMin` is null while the server's setting is unknown.
 */
export type RunPlan = 'parallel' | 'separate' | 'unknown'

export function planRun(context: string, count: number, parallelMin: number | null): RunPlan {
  if (parallelMin === null) return 'unknown'
  return context.trim() !== '' && count >= parallelMin ? 'parallel' : 'separate'
}

/** The caption next to the Decide button. */
export function planCaption(plan: RunPlan, count: number): string {
  const questions = plural(count, 'question')
  if (plan === 'parallel') return `One parallel pass · ${questions}`
  if (plan === 'separate') return `${plural(count, 'fast pass', 'fast passes')} · ${questions}`
  return `One request · ${questions}`
}

/** The status line while a request runs. */
export function runningCaption(plan: RunPlan, count: number): string {
  if (plan === 'parallel') return `Prefilling shared context · scoring ${count} questions in parallel`
  return `Scoring ${plural(count, 'question')}`
}
