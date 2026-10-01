import type { ReactElement } from 'react'

interface LogoProps {
  size?: number
}

/** The D.A.R.T. mark: a dart in a target. */
export default function Logo({ size = 40 }: LogoProps): ReactElement {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" role="img" aria-label="D.A.R.T. logo">
      <circle cx="28" cy="36" r="24" fill="none" stroke="var(--accent)" strokeWidth="4" />
      <circle cx="28" cy="36" r="14" fill="none" stroke="var(--accent)" strokeWidth="4" />
      <circle cx="28" cy="36" r="4" fill="var(--accent)" />
      <path d="M28 36 L58 6" stroke="var(--ink)" strokeWidth="4" strokeLinecap="round" />
      <path
        d="M50 6 H58 V14"
        fill="none"
        stroke="var(--ink)"
        strokeWidth="4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}
