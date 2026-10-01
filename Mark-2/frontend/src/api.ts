export type RankedOption = { option: string; probability: number }

export type Decision = {
  id: string
  ranked: RankedOption[]
  decision: string
  confidence: number
}

export type DecideResponse = {
  id: string | null
  model: string
  decisions: Decision[]
  latency_ms: number
  options: string[]
  options_kind: 'list' | 'integer_range' | 'decimal_range'
  stored: boolean
}

export type Outcome = { result: DecideResponse; roundTripMs: number }

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

export async function decide(query: string, options: string): Promise<Outcome> {
  const start = performance.now()
  let response: Response
  try {
    response = await fetch(`${BASE_URL}/api/decide`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, options }),
    })
  } catch {
    throw new Error(UNREACHABLE)
  }
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) throw new Error(errorMessage(response.status, body))
  return { result: body as DecideResponse, roundTripMs: performance.now() - start }
}
