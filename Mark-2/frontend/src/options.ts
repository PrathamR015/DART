/**
 * Live preview of the options field, so the playground can show the option chips while the user types.
 * A port of Mark-2/backend/dartweb/options.py; keep the two in step. The server stays the authority: whatever it
 * answers on Decide wins.
 */
export const MAX_OPTIONS = 12
export const MAX_OPTION_LENGTH = 100

export type OptionsKind = 'list' | 'integer_range' | 'decimal_range'

export type ParsedOptions = { ok: true; options: string[]; kind: OptionsKind } | { ok: false; error: string }

const NUMBER = String.raw`-?\d+(?:\.\d+)?`
const RANGE = new RegExp(
  String.raw`^\s*(${NUMBER})\s*(?:to|\.\.|[-–—])\s*(${NUMBER})(?:\s+step\s+(${NUMBER}))?\s*$`,
  'i',
)
const SUGGESTED_STEPS = [0.25, 0.5, 1, 2, 5, 10, 50]

const fail = (error: string): ParsedOptions => ({ ok: false, error })

function decimalPlaces(text: string): number {
  const dot = text.indexOf('.')
  return dot < 0 ? 0 : text.length - dot - 1
}

function countError(count: number): string | null {
  if (count < 2) return 'need at least 2 options'
  if (count > MAX_OPTIONS) return `at most ${MAX_OPTIONS} options are supported, got ${count}`
  return null
}

function parseList(text: string): ParsedOptions {
  const separator = text.includes('|') ? '|' : ','
  const options = text
    .split(separator)
    .map((part) => part.trim())
    .filter(Boolean)
  const error = countError(options.length)
  if (error) return fail(error)
  if (options.some((option) => option.length > MAX_OPTION_LENGTH)) {
    return fail(`each option must be at most ${MAX_OPTION_LENGTH} characters`)
  }
  if (new Set(options.map((option) => option.toLowerCase())).size !== options.length) {
    return fail('options must be distinct')
  }
  return { ok: true, options, kind: 'list' }
}

function suggestStep(span: number): string {
  const step = SUGGESTED_STEPS.find((candidate) => span / candidate + 1 <= MAX_OPTIONS)
  return String(step ?? span)
}

/** Works in whole units of the finest decimal place, so 0.1 + 0.2 never shows up as 0.30000000000000004. */
function parseRange(match: RegExpMatchArray): ParsedOptions {
  const [, startText, endText, stepText = ''] = match
  const places = Math.max(decimalPlaces(startText), decimalPlaces(endText), decimalPlaces(stepText))
  const unit = 10 ** places
  const start = Math.round(Number(startText) * unit)
  const end = Math.round(Number(endText) * unit)
  const step = stepText ? Math.round(Number(stepText) * unit) : 1
  if (step <= 0) return fail('step must be greater than 0')
  if (start >= end) return fail('the range must start below where it ends')
  const count = Math.floor((end - start) / step) + 1
  if (count > MAX_OPTIONS) {
    const suggestion = suggestStep((end - start) / unit)
    return fail(
      `that range has ${count} values but at most ${MAX_OPTIONS} are supported; ` +
        `add a larger step, for example '${startText}-${endText} step ${suggestion}'`,
    )
  }
  const error = countError(count)
  if (error) return fail(error)
  const options = Array.from({ length: count }, (_, i) => ((start + i * step) / unit).toFixed(places))
  return { ok: true, options, kind: places ? 'decimal_range' : 'integer_range' }
}

export function parseOptions(input: string): ParsedOptions {
  const text = input.trim()
  if (!text) return fail("enter the options, for example 'yes, no' or a range like '0-5'")
  const match = text.match(RANGE)
  return match ? parseRange(match) : parseList(text)
}

/** The short label next to the options field: "4 options" or "range · 6 values". */
export function describeOptions(parsed: ParsedOptions): string {
  if (!parsed.ok) return ''
  const count = parsed.options.length
  return parsed.kind === 'list' ? `${count} options` : `range · ${count} values`
}
