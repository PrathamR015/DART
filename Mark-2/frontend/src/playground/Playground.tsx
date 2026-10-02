import { useEffect, useRef, useState } from 'react'
import type { KeyboardEvent, ReactElement } from 'react'
import { ArrowIcon, GitHubIcon, PlusIcon } from '../brand/icons'
import { BrandLink } from '../brand/Wordmark'
import { decide, health, MAX_QUESTIONS } from '../api'
import type { Health, Outcome, QuestionInput } from '../api'
import { API_DOCS_URL, GITHUB_URL, HOME_URL } from '../links'
import { estimateTokens, planCaption, planRun, runningCaption, SEED_CONTEXT, SEED_QUESTIONS } from './plan'
import QuestionCard, { questionProblem } from './QuestionCard'
import type { QuestionDraft } from './QuestionCard'
import { DecisionCard, Failed, Idle, RunStats, Running } from './Results'

type Run =
  | { phase: 'idle' }
  | { phase: 'running'; count: number }
  | { phase: 'done'; outcome: Outcome; questions: QuestionInput[] }
  | { phase: 'failed'; message: string }

/** The server's status: unknown until /api/health answers, then the answer or "offline". */
type ServerState = { kind: 'checking' } | { kind: 'online'; health: Health } | { kind: 'offline' }

function StatusPill({ server }: { server: ServerState }): ReactElement {
  const online = server.kind === 'online'
  const label = server.kind === 'offline' ? 'server offline' : 'dart_v1.0 active'
  const title = online ? `${server.health.model_version} on ${server.health.device}` : undefined
  return (
    <span className="pill" title={title}>
      <span className={online ? 'pill-dot' : 'pill-dot is-off'} aria-hidden="true" />
      {label}
    </span>
  )
}

function useServer(): ServerState {
  const [server, setServer] = useState<ServerState>({ kind: 'checking' })
  useEffect(() => {
    let active = true
    health()
      .then((status) => active && setServer({ kind: 'online', health: status }))
      .catch(() => active && setServer({ kind: 'offline' }))
    return () => {
      active = false
    }
  }, [])
  return server
}

export default function Playground(): ReactElement {
  const nextKey = useRef(1)
  const draft = (question: QuestionInput): QuestionDraft => ({ ...question, key: nextKey.current++ })
  const [context, setContext] = useState(SEED_CONTEXT)
  const [questions, setQuestions] = useState<QuestionDraft[]>(() => SEED_QUESTIONS.map(draft))
  const [run, setRun] = useState<Run>({ phase: 'idle' })
  const server = useServer()

  const running = run.phase === 'running'
  const ready = questions.every((question) => questionProblem(question) === null)
  const parallelMin = server.kind === 'online' ? server.health.parallel_min_questions : null
  const plan = planRun(context, questions.length, parallelMin)

  function updateQuestion(key: number, field: keyof QuestionInput, value: string) {
    setQuestions((current) => current.map((q) => (q.key === key ? { ...q, [field]: value } : q)))
  }

  async function submit() {
    if (!ready || running) return
    const sent = questions.map(({ query, options }) => ({ query, options }))
    setRun({ phase: 'running', count: sent.length })
    try {
      setRun({ phase: 'done', outcome: await decide(context, sent), questions: sent })
    } catch (failure) {
      setRun({ phase: 'failed', message: failure instanceof Error ? failure.message : 'Something went wrong.' })
    }
  }

  function onKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault()
      void submit()
    }
  }

  return (
    <div className="playground">
      <header className="app-header">
        <BrandLink href={HOME_URL} markSize={30} fontSize={17} />
        <div className="app-header-side">
          <StatusPill server={server} />
          <a className="app-header-link" href={API_DOCS_URL}>
            API docs
          </a>
          <a className="app-header-link repo-link" href={GITHUB_URL} target="_blank" rel="noopener noreferrer">
            <GitHubIcon size={16} />
            GitHub
          </a>
        </div>
      </header>

      <main className="workspace">
        <section aria-labelledby="in-h" className="input-column">
          <div className="column-head">
            <span className="eyebrow">01 — INPUT</span>
            <h1 id="in-h" className="display column-title">
              Ask once. Decide everything.
            </h1>
          </div>

          <div className="field">
            <div className="field-row">
              <label htmlFor="ctx" className="field-label is-strong">
                Context
              </label>
              <span className="mono field-meta">~{estimateTokens(context)} tokens · read once</span>
            </div>
            <textarea
              id="ctx"
              className="f-in"
              rows={5}
              value={context}
              onChange={(event) => setContext(event.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Paste the text every question is about"
              maxLength={2000}
            />
            <span className="field-hint">Optional. Shared by every question below.</span>
          </div>

          <div className="questions-head">
            <h2 className="questions-title">Questions</h2>
            <span className="mono count-badge" aria-label={`${questions.length} questions`}>
              {questions.length}
            </span>
          </div>

          <div className="question-list">
            {questions.map((question, index) => (
              <QuestionCard
                key={question.key}
                number={index + 1}
                question={question}
                removable={questions.length > 1}
                onChange={(field, value) => updateQuestion(question.key, field, value)}
                onRemove={() => setQuestions((current) => current.filter((q) => q.key !== question.key))}
                onKeyDown={onKeyDown}
              />
            ))}
          </div>

          <button
            type="button"
            className="btn-add"
            onClick={() => setQuestions((current) => [...current, draft({ query: '', options: '' })])}
            disabled={questions.length >= MAX_QUESTIONS}
          >
            <PlusIcon />
            {questions.length >= MAX_QUESTIONS ? `Up to ${MAX_QUESTIONS} questions` : 'Add question'}
          </button>

          <div className="decide-row">
            <button type="button" className="btn-primary" onClick={() => void submit()} disabled={!ready || running}>
              {running ? 'Deciding…' : 'Decide'}
              <ArrowIcon />
            </button>
            <span className="decide-caption">{planCaption(plan, questions.length)}</span>
          </div>
        </section>

        <section aria-labelledby="out-h" className="output-column">
          <div className="output-head">
            <div className="column-head">
              <span className="eyebrow">02 — DECISIONS</span>
              <h2 id="out-h" className="display column-title">
                Decision already reached.
              </h2>
            </div>
            {server.kind === 'online' && (
              <span className="mono model-badge">
                {server.health.model_version} · {server.health.device}
              </span>
            )}
          </div>

          {run.phase === 'idle' && <Idle />}
          {run.phase === 'running' && <Running caption={runningCaption(plan, run.count)} />}
          {run.phase === 'failed' && <Failed message={run.message} />}
          {run.phase === 'done' && (
            <>
              <RunStats outcome={run.outcome} />
              {run.outcome.result.decisions.map((decision, index) => (
                <DecisionCard
                  key={decision.id}
                  number={index + 1}
                  query={run.questions[index]?.query ?? ''}
                  decision={decision}
                />
              ))}
            </>
          )}
        </section>
      </main>
    </div>
  )
}
