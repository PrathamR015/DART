import type { ReactElement } from 'react'
import { percent } from '../brand/format'
import { ArrowIcon, GitHubIcon } from '../brand/icons'
import Mark from '../brand/Mark'
import { BrandLink } from '../brand/Wordmark'
import { GITHUB_URL, PLAYGROUND_URL } from '../links'
import { ApiSection } from './ApiSection'

const CONTACT_EMAIL: string | undefined = import.meta.env.VITE_CONTACT_EMAIL || undefined

/** Real output of the exported model (v1.0, RTX 3050 Laptop GPU): standalone questions that need no context. */
const HERO = {
  query: 'My debit card was stolen this morning. What should I do?',
  latencyMs: 14.0,
  ranked: [
    { option: 'report lost card', probability: 0.91 },
    { option: 'change PIN', probability: 0.053 },
    { option: 'check balance', probability: 0.023 },
    { option: 'order checkbook', probability: 0.013 },
  ],
  others: [
    { query: 'Which of these animals is a mammal?', decision: 'dolphin', probability: 0.961 },
    { query: 'Is water made of hydrogen and oxygen?', decision: 'yes', probability: 0.934 },
    { query: 'The server is down for every customer. How urgent?', decision: 'high', probability: 0.76 },
  ],
}

const PROPERTIES = [
  { title: 'No decoding', text: 'One forward pass. The answer is read off a classification head, not generated.' },
  { title: 'Parallel', text: 'Many questions share one context and resolve in the same pass.' },
  { title: 'Ranked', text: 'Every option comes back with its probability, highest first.' },
]

const STEPS = [
  { tag: '01 / CONTEXT', title: 'Read once', text: 'Your text is prefilled a single time and shared by every question that follows.' },
  { tag: '02 / FAN OUT', title: 'Ask in parallel', text: 'Each question and its options attend to that shared context in the same forward pass.' },
  { tag: '03 / SCORE', title: 'Head, not decoder', text: 'A custom answer head assigns a probability to every option. No tokens are generated.' },
  { tag: '04 / DECIDE', title: 'Ranked and final', text: 'All options return in descending order. The top one is the decision, with its confidence.' },
]

function SiteHeader(): ReactElement {
  return (
    <header className="site-header">
      <BrandLink href="#top" />
      <nav aria-label="Main" className="site-nav">
        <a className="nav-link" href="#how">How it works</a>
        <a className="nav-link" href="#api">API</a>
        <a className="nav-link" href={PLAYGROUND_URL}>Playground</a>
        <a className="nav-link repo-link" href={GITHUB_URL} target="_blank" rel="noopener noreferrer">
          <GitHubIcon />
          GitHub
        </a>
        <a className="nav-cta" href="#access">Try it Now!</a>
      </nav>
    </header>
  )
}

function HeroCard(): ReactElement {
  const [top] = HERO.ranked
  return (
    <div className="hero-card-wrap">
      <div className="hero-card-dots dot-grid" aria-hidden="true" />
      <figure className="hero-card" aria-label="Example decision">
        <div className="hero-card-meta mono">
          <span>Q01</span>
          <span>1 pass · {HERO.latencyMs} ms</span>
        </div>
        <span className="hero-card-query">{HERO.query}</span>
        <div className="hero-card-decision">
          <span className="display">{top.option}</span>
          <span className="confidence-badge mono">{percent(top.probability)}</span>
        </div>
        <div className="hero-ranked">
          {HERO.ranked.map((item, index) => (
            <div key={item.option} className="hero-ranked-row">
              <span className={index === 0 ? undefined : 'is-muted'}>{item.option}</span>
              <span className="bar-track">
                <span
                  className={index === 0 ? 'bar-fill is-top' : 'bar-fill'}
                  style={{ width: `${Math.max(item.probability * 100, 1.2)}%` }}
                />
              </span>
              <span className={index === 0 ? 'mono pct' : 'mono pct is-muted'}>{percent(item.probability)}</span>
            </div>
          ))}
        </div>
        <div className="hero-card-rule" />
        <ul className="hero-card-more" aria-label="More decisions">
          {HERO.others.map((other) => (
            <li key={other.query} className="hero-card-other">
              <span className="hero-card-other-query">{other.query}</span>
              <span className="chip">
                {other.decision} · {percent(other.probability)}
              </span>
            </li>
          ))}
        </ul>
      </figure>
    </div>
  )
}

function Hero(): ReactElement {
  return (
    <section id="top" className="hero container">
      <div className="hero-copy">
        <span className="eyebrow eyebrow-lg">SYSTEM 1 THINKING DECISION  MODEL</span>
        <h1 className="display hero-title">
          Decision Already Reached.<span className="hero-title-thanks">Thanks.</span>
        </h1>
        <p className="hero-lede">
          D.A.R.T. reads your context once and answers every question in a single parallel pass. Faster and Cheaper than LLMs. The next generation model is based on system one thinking for decion making.
        </p>
        <div className="hero-actions">
          <a className="btn-primary btn-lg" href={PLAYGROUND_URL}>
            Open the playground
            <ArrowIcon />
          </a>
          <a className="btn-ghost btn-lg" href="#api">See the API</a>
        </div>
      </div>
      <HeroCard />
    </section>
  )
}

function Properties(): ReactElement {
  return (
    <section aria-label="Core properties" className="properties">
      <div className="container properties-grid">
        {PROPERTIES.map((property) => (
          <div key={property.title} className="property">
            <span className="display property-title">{property.title}</span>
            <span className="property-text">{property.text}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

function HowItWorks(): ReactElement {
  return (
    <section id="how" className="container section how">
      <div className="section-head">
        <span className="eyebrow eyebrow-lg">HOW IT WORKS</span>
        <h2 className="display section-title">From context to decision in one strike.</h2>
      </div>
      <ol className="steps">
        {STEPS.map((step, index) => (
          <li key={step.tag} className={index === STEPS.length - 1 ? 'step is-final' : 'step'}>
            <span className="mono step-tag">{step.tag}</span>
            <span className="step-title">{step.title}</span>
            <span className="step-text">{step.text}</span>
          </li>
        ))}
      </ol>
      <div className="mono how-options">
        <span>Options can be a list</span>
        <span className="is-bright">yes, no, maybe</span>
        <span>or a scale</span>
        <span className="is-bright">0-5 · 1-10 · 0-5 step 0.5</span>
      </div>
    </section>
  )
}

function Statement(): ReactElement {
  return (
    <section aria-labelledby="statement-title" className="statement">
      <div className="container section statement-inner">
        <h2 id="statement-title" className="display statement-title">
          No thresholds.<span className="block">No escalation.</span>
          <span className="block is-signal">No second pass.</span>
        </h2>
        <p className="statement-text">
          D.A.R.T. is pure System 1 Thinking Desicion Model. It answers at prefill speed and reports its confidence exactly as the model sees it.
          The answer it returns is the answer. Build whatever guardrails you need around it; the model stays fast.
        </p>
      </div>
    </section>
  )
}

function Access(): ReactElement {
  return (
    <section id="access" className="container access-wrap">
      <div className="access">
        <div className="access-copy">
          <h2 className="display access-title">Ship decisions with System One Thinking.</h2>
          <p className="access-text">
            D.A.R.T. runs on your own GPU today. Hosted API access will be available soon .
          </p>
        </div>
        {CONTACT_EMAIL ? (
          <a className="access-button" href={`mailto:${CONTACT_EMAIL}?subject=D.A.R.T.%20API%20access`}>
            Request API access
            <ArrowIcon />
          </a>
        ) : (
          <a className="access-button" href={PLAYGROUND_URL}>
            Open the playground
            <ArrowIcon />
          </a>
        )}
      </div>
    </section>
  )
}

function SiteFooter(): ReactElement {
  return (
    <footer className="site-footer">
      <div className="container footer-inner">
        <div className="footer-brand">
          <Mark size={24} strokeWidth={5} />
          <span className="mono">Decision Already Reached, Thanks.</span>
        </div>
        <span className="mono footer-copy">© {new Date().getFullYear()} D.A.R.T.</span>
      </div>
    </footer>
  )
}

export default function Landing(): ReactElement {
  return (
    <div className="landing">
      <SiteHeader />
      <main>
        <Hero />
        <Properties />
        <HowItWorks />
        <Statement />
        <ApiSection />
        <Access />
      </main>
      <SiteFooter />
    </div>
  )
}
