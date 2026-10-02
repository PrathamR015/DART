import type { KeyboardEvent, ReactElement } from 'react'
import { pad2 } from '../brand/format'
import { CloseIcon } from '../brand/icons'
import type { QuestionInput } from '../api'
import { describeOptions, parseOptions } from '../options'

const MAX_CHIPS = 10
const INCOMPLETE = 'Add a query and at least two options.'

export interface QuestionDraft extends QuestionInput {
  key: number
}

/** A question is ready when it has a query and its options parse. Returns the message to show, or null. */
export function questionProblem(question: QuestionInput): string | null {
  if (!question.query.trim() || !question.options.trim()) return INCOMPLETE
  const parsed = parseOptions(question.options)
  if (parsed.ok) return null
  return parsed.error === 'need at least 2 options' ? INCOMPLETE : `${parsed.error[0].toUpperCase()}${parsed.error.slice(1)}.`
}

interface QuestionCardProps {
  number: number
  question: QuestionDraft
  removable: boolean
  onChange: (field: keyof QuestionInput, value: string) => void
  onRemove: () => void
  onKeyDown: (event: KeyboardEvent<HTMLElement>) => void
}

export default function QuestionCard({
  number,
  question,
  removable,
  onChange,
  onRemove,
  onKeyDown,
}: QuestionCardProps): ReactElement {
  const queryId = `q-${question.key}-query`
  const optionsId = `q-${question.key}-opts`
  const parsed = parseOptions(question.options)
  const chips = parsed.ok ? parsed.options.slice(0, MAX_CHIPS) : []
  const hiddenChips = parsed.ok ? parsed.options.length - chips.length : 0
  const problem = questionProblem(question)
  return (
    <div className="question-card" role="group" aria-label={`Question ${number}`}>
      <div className="question-head">
        <span className="mono question-num">Q{pad2(number)}</span>
        {removable && (
          <button type="button" className="btn-icon" onClick={onRemove} aria-label={`Remove question ${number}`}>
            <CloseIcon />
          </button>
        )}
      </div>
      <div className="field">
        <label htmlFor={queryId} className="field-label">
          Query
        </label>
        <textarea
          id={queryId}
          className="f-in"
          rows={2}
          value={question.query}
          onChange={(event) => onChange('query', event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="What should be decided?"
          maxLength={500}
        />
      </div>
      <div className="field">
        <div className="field-row">
          <label htmlFor={optionsId} className="field-label">
            Options
          </label>
          <span className="mono field-meta">{describeOptions(parsed)}</span>
        </div>
        <input
          id={optionsId}
          className="f-in"
          type="text"
          value={question.options}
          onChange={(event) => onChange('options', event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="yes, no, maybe  ·  0-5  ·  0-5 step 0.5"
          maxLength={500}
        />
        {chips.length > 0 && (
          <ul className="chips" aria-label={`Options for question ${number}`}>
            {chips.map((chip) => (
              <li key={chip} className="chip">
                {chip}
              </li>
            ))}
            {hiddenChips > 0 && <li className="chip">+{hiddenChips}</li>}
          </ul>
        )}
        {problem && <span className="field-warn">{problem}</span>}
      </div>
    </div>
  )
}
