import type { ReactElement, ReactNode } from 'react'
import { PLAYGROUND_URL } from '../links'

/** A JSON key in the code samples (dimmed, as in the design). */
function K({ children }: { children: ReactNode }): ReactElement {
  return <span className="code-key">"{children}"</span>
}

/** A probability in the code samples (in Signal). */
function P({ children }: { children: ReactNode }): ReactElement {
  return <span className="code-signal">{children}</span>
}

interface CodePanelProps {
  method: string
  path: string
  caption: string
  children: ReactNode
}

function CodePanel({ method, path, caption, children }: CodePanelProps): ReactElement {
  return (
    <div className="code-panel">
      <div className="code-panel-head mono">
        <span className="code-signal">
          {method} <span className="code-path">{path}</span>
        </span>
        <span className="code-caption">{caption}</span>
      </div>
      <pre className="code-body">{children}</pre>
    </div>
  )
}

/** The API section. The samples match the real Mark-2 endpoint (Mark-2/backend/dartweb/app.py). */
export function ApiSection(): ReactElement {
  return (
    <section id="api" className="container section api">
      <div className="api-copy">
        <span className="eyebrow eyebrow-lg">API</span>
        <h2 className="display api-title">One request. Every decision.</h2>
        <p className="api-text">
          Send a context and as many questions as you need. Each comes back with its decision, its confidence and the
          full ranking.
        </p>
        <a className="btn-ghost api-try" href={PLAYGROUND_URL}>
          Try it in the playground
        </a>
      </div>
      <div className="api-code">
        <CodePanel method="POST" path="/api/decide" caption="request">
          {'{\n  '}
          <K>context</K>
          {': "The parcel arrived three weeks late, the box was crushed…",\n  '}
          <K>questions</K>
          {': [\n    { '}
          <K>query</K>
          {": \"What is the customer's overall sentiment?\",\n      "}
          <K>options</K>
          {': ["negative", "neutral", "positive"] },\n    { '}
          <K>query</K>
          {': "Rate the delivery from 0 to 5",\n      '}
          <K>options</K>
          {': "0-5" }\n  ]\n}'}
        </CodePanel>
        <CodePanel method="200" path="OK" caption="response">
          {'{\n  '}
          <K>decisions</K>
          {': [\n    { '}
          <K>decision</K>
          {': "negative", '}
          <K>confidence</K>
          {': '}
          <P>0.856</P>
          {',\n      '}
          <K>ranked</K>
          {': [{ '}
          <K>option</K>
          {': "negative", '}
          <K>probability</K>
          {': '}
          <P>0.856</P>
          {' }, …] },\n    …\n  ],\n  '}
          <K>latency_ms</K>
          {': 36.8\n}'}
        </CodePanel>
      </div>
    </section>
  )
}
