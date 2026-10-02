import type { OptionsKind } from './options'

export type RankedOption = { option: string; probability: number }

export type Decision = {
  id: string
  ranked: RankedOption[]
  decision: string
  confidence: number
  options: string[]
  options_kind: OptionsKind
  record_id: string | null
}

export type DecideResponse = {
  model: string
  decisions: Decision[]
  latency_ms: number
  /** True when the shared context was read once for every question. */
  parallel: boolean
  stored: boolean
}

export type Health = {
  status: string
  model_loaded: boolean
  device: string
  model_version: string
  /** From this many questions over a shared context, the server reads the context once (parallel pass). */
  parallel_min_questions: number
}

/** One question as the user typed it: `options` is a list ("yes, no") or a range ("0-5"). */
export interface QuestionInput {
  query: string
  options: string
}

export type Outcome = { result: DecideResponse; roundTripMs: number }

export const MAX_QUESTIONS = 10

const BASE_URL: string = import.meta.env.VITE_API_URL ?? ''
const UNREACHABLE = 'Cannot reach the D.A.R.T. server. Is the backend running?'

/** Turns an API error body (FastAPI sends `detail` as text, or as a list for validation errors) into a message. */
export function errorMessage(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((item) => (item as { msg?: string }).msg ?? 'Invalid input').join('; ')
  }
  return `The server returned an error (${status}).`
}

async function request(path: string, init?: RequestInit): Promise<unknown> {
  let response: Response
  try {
    response = await fetch(`${BASE_URL}${path}`, init)
  } catch {
    throw new Error(UNREACHABLE)
  }
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) throw new Error(errorMessage(response.status, body))
  return body
}

/** Decides every question in one request; the server picks the faster way to run them. */
export async function decide(context: string, questions: QuestionInput[]): Promise<Outcome> {
  const start = performance.now()
  const body = await request('/api/decide', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ context, questions }),
  })
  return { result: body as DecideResponse, roundTripMs: performance.now() - start }
}

export async function health(): Promise<Health> {
  return (await request('/api/health')) as Health
}
