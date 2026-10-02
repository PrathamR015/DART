import type { ReactElement } from 'react'

const OUTER_RING = 'M56.95 19.29 A28 28 0 1 1 44.71 7.05'
const INNER_RING = 'M47.15 24.28 A17 17 0 1 1 39.72 16.85'
/** Below this size the inner ring drops and the strokes thicken (brand rule). */
const SMALL_SIZE = 24

interface MarkProps {
  size?: number
  /** Colour of the rings. */
  ink?: string
  /** Colour of the strike and the bullseye. `null` hides the strike (the empty-state mark). */
  accent?: string | null
  strokeWidth?: number
  label?: string
}

/** The D.A.R.T. mark: two open rings, one strike through the gaps, landing on the bullseye. */
export default function Mark({
  size = 32,
  ink = 'var(--paper)',
  accent = 'var(--signal)',
  strokeWidth,
  label,
}: MarkProps): ReactElement {
  const small = size < SMALL_SIZE
  const stroke = strokeWidth ?? (small ? 7 : 4.5)
  const a11y = label ? { role: 'img', 'aria-label': label } : { 'aria-hidden': true }
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none" {...a11y}>
      <path d={OUTER_RING} stroke={ink} strokeWidth={stroke} strokeLinecap="round" />
      {!small && <path d={INNER_RING} stroke={ink} strokeWidth={stroke} strokeLinecap="round" />}
      {accent !== null && (
        <line x1="59" y1="5" x2="36.5" y2="27.5" stroke={accent} strokeWidth={stroke} strokeLinecap="round" />
      )}
      <circle cx="32" cy="32" r={small ? 9 : 6.5} fill={accent ?? ink} />
    </svg>
  )
}
