import type { ReactElement } from 'react'
import { pad2, percent } from '../brand/format'
import Mark from '../brand/Mark'
import type { Decision, Outcome } from '../api'

/** Options shown per decision; the rest are summarised as "+N more options". */
const SHOWN_OPTIONS = 6
/** The thinnest bar drawn, so a near-zero option is still visible. */
const MIN_BAR_PERCENT = 1.2

interface RunStatsProps {
  outcome: Outcome
}

export function RunStats({ outcome }: RunStatsProps): ReactElement {
  const { result, roundTripMs } = outcome
  const count = result.decisions.length
  const stats = [
    { label: 'Questions', value: String(count) },
    { label: 'Forward passes', value: String(result.parallel ? 1 : count) },
    { label: 'Model', value: `${result.latency_ms.toFixed(1)} ms`, isSignal: true },
    { label: 'Round trip', value: `${Math.round(roundTripMs)} ms` },
  ]
  return (
    <dl className="run-stats">
      {stats.map((stat) => (
        <div key={stat.label} className="run-stat">
          <dt>{stat.label}</dt>
          <dd className={stat.isSignal ? 'mono is-signal' : 'mono'}>{stat.value}</dd>
        </div>
      ))}
    </dl>
  )
}

interface ConfidenceRingProps {
  confidence: number
}

function ConfidenceRing({ confidence }: ConfidenceRingProps): ReactElement {
  return (
    <div
      className="conf-ring"
      role="img"
      aria-label={`Confidence ${percent(confidence)}`}
      style={{ background: `conic-gradient(var(--signal) ${(confidence * 360).toFixed(1)}deg, var(--edge) 0)` }}
    >
      <div className="conf-ring-inner">
        <span className="mono conf-value">{Math.round(confidence * 100)}%</span>
        <span className="mono conf-label">CONF</span>
      </div>
    </div>
  )
}

interface DecisionCardProps {
  number: number
  query: string
  decision: Decision
}

export function DecisionCard({ number, query, decision }: DecisionCardProps): ReactElement {
  const shown = decision.ranked.slice(0, SHOWN_OPTIONS)
  const more = decision.ranked.length - shown.length
  return (
    <article className="decision-card card-in" aria-label={`Result ${number}`}>
      <div className="decision-head">
        <div className="decision-copy">
          <span className="mono decision-num">Q{pad2(number)}</span>
          <h3 className="decision-query">{query}</h3>
          <div className="decision-line">
            <span className="mono decision-tag">DECISION</span>
            <strong className="display decision-value">{decision.decision}</strong>
          </div>
        </div>
        <ConfidenceRing confidence={decision.confidence} />
      </div>
      <ol className="ranked">
        {shown.map((item, index) => (
          <li key={item.option} className={index === 0 ? 'ranked-row is-top' : 'ranked-row'}>
            <span className="mono ranked-rank">{pad2(index + 1)}</span>
            <span className="ranked-label">{item.option}</span>
            <span className="bar-track">
              <span
                className={index === 0 ? 'bar-fill bar-grow is-top' : 'bar-fill bar-grow'}
                style={{ width: `${Math.max(item.probability * 100, MIN_BAR_PERCENT)}%` }}
              />
            </span>
            <span className="mono ranked-pct">{percent(item.probability)}</span>
          </li>
        ))}
        {more > 0 && <li className="mono ranked-more">+{more} more options</li>}
      </ol>
      {decision.options_kind !== 'list' && (
        <p className="mono range-values">Range values: {decision.options.join(', ')}</p>
      )}
    </article>
  )
}

interface RunningProps {
  caption: string
}

export function Running({ caption }: RunningProps): ReactElement {
  return (
    <div className="panel running" role="status">
      <div className="scan-track">
        <div className="scan" />
      </div>
      <div className="mono running-line">
        <span className="blink" aria-hidden="true" />
        {caption}
      </div>
    </div>
  )
}

export function Idle(): ReactElement {
  return (
    <div className="idle">
      <Mark size={44} ink="var(--mark-dim)" accent={null} strokeWidth={4} />
      <span className="idle-title">Every option, ranked, in one pass.</span>
      <span className="idle-text">Write your questions, then press Decide.</span>
    </div>
  )
}

interface FailedProps {
  message: string
}

export function Failed({ message }: FailedProps): ReactElement {
  return (
    <div className="panel failed" role="alert">
      <span className="mono failed-tag">NO DECISION</span>
      <span className="failed-text">{message}</span>
    </div>
  )
}
