import { useState } from 'react'
import type { FormEvent, KeyboardEvent, ReactElement } from 'react'
import { decide } from './api'
import type { Outcome } from './api'
import Logo from './Logo'

const QUERY_HINT = 'Put the question last, and name the scale if there is one, e.g. "Rate this review from 0 to 5: ..."'
const OPTIONS_HINT = 'A comma-separated list (yes, no, maybe) or a range: 0-5, 1-10, 0.0-1.0, 0-5 step 0.5'

interface ResultProps {
  outcome: Outcome
}

function Result({ outcome }: ResultProps): ReactElement {
  const { result, roundTripMs } = outcome
  const decision = result.decisions[0]
  return (
    <section className="result" aria-live="polite">
      <p className="decision">
        Decision: <strong>{decision.decision}</strong>
        <span className="confidence"> confidence {(decision.confidence * 100).toFixed(1)}%</span>
      </p>
      <p className="latency">
        Model: {result.latency_ms.toFixed(1)} ms · Round trip: {Math.round(roundTripMs)} ms
      </p>
      <ol className="ranked">
        {decision.ranked.map((item) => (
          <li key={item.option}>
            <span className="option">{item.option}</span>
            <span className="bar">
              <span style={{ width: `${item.probability * 100}%` }} />
            </span>
            <span className="percent">{(item.probability * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ol>
      {result.options_kind !== 'list' && (
        <p className="range-values">Range values: {result.options.join(', ')}</p>
      )}
    </section>
  )
}

export default function App(): ReactElement {
  const [query, setQuery] = useState('')
  const [options, setOptions] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [outcome, setOutcome] = useState<Outcome | null>(null)
  const canSubmit = query.trim() !== '' && options.trim() !== '' && !loading

  async function submit(event?: FormEvent) {
    event?.preventDefault()
    if (!canSubmit) return
    setLoading(true)
    setError(null)
    try {
      setOutcome(await decide(query, options))
    } catch (failure) {
      setOutcome(null)
      setError(failure instanceof Error ? failure.message : 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  function onQueryKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault()
      void submit()
    }
  }

  return (
    <main>
      <header>
        <Logo />
        <div>
          <h1>D.A.R.T.</h1>
          <p className="tagline">Decision Already Reached, Thanks.</p>
        </div>
      </header>

      <form onSubmit={submit}>
        <label htmlFor="query">Your query</label>
        <textarea
          id="query"
          rows={5}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={onQueryKeyDown}
          placeholder="Type your question, including any context it needs"
          maxLength={2000}
        />
        <p className="hint">{QUERY_HINT}</p>

        <label htmlFor="options">Options</label>
        <input
          id="options"
          value={options}
          onChange={(event) => setOptions(event.target.value)}
          placeholder="yes, no, maybe   or   0-5"
          maxLength={500}
        />
        <p className="hint">{OPTIONS_HINT}</p>

        <button type="submit" disabled={!canSubmit}>
          {loading ? 'Deciding...' : 'Decide'}
        </button>
      </form>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {outcome && <Result outcome={outcome} />}
    </main>
  )
}
