import type { ReactElement } from 'react'
import Mark from './Mark'

interface WordmarkProps {
  fontSize?: number
}

/** "D.A.R.T." with its four Signal dots. Screen readers hear the name once, not letter by letter. */
export function Wordmark({ fontSize = 18 }: WordmarkProps): ReactElement {
  return (
    <span className="wordmark" style={{ fontSize }}>
      <span className="visually-hidden">D.A.R.T.</span>
      <span aria-hidden="true">
        {['D', 'A', 'R', 'T'].map((letter) => (
          <span key={letter}>
            {letter}
            <span className="dot">.</span>
          </span>
        ))}
      </span>
    </span>
  )
}

interface BrandLinkProps {
  href: string
  markSize?: number
  fontSize?: number
}

/** The lockup (mark + wordmark) as the home link in a header. */
export function BrandLink({ href, markSize = 32, fontSize = 18 }: BrandLinkProps): ReactElement {
  return (
    <a href={href} className="brand-link" aria-label="D.A.R.T. home">
      <Mark size={markSize} />
      <Wordmark fontSize={fontSize} />
    </a>
  )
}
