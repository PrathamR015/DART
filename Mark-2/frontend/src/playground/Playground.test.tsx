import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../api'
import Playground from './Playground'
import { SEED_CONTEXT, SEED_QUESTIONS } from './plan'

// Forwarded through plain functions (not spies) so a rejected call is handled only by the page itself.
let decideImpl: typeof api.decide
let healthImpl: typeof api.health
vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof api>()),
  decide: (...args: Parameters<typeof api.decide>) => decideImpl(...args),
  health: () => healthImpl(),
}))
const decide = vi.fn<typeof api.decide>()

const GPU: api.Health = {
  status: 'ok',
  model_loaded: true,
  device: 'cuda',
  model_version: 'v1.0',
  parallel_min_questions: 6,
}

const decision = (id: string, ranked: [string, number][], kind: api.Decision['options_kind'] = 'list'): api.Decision => ({
  id,
  decision: ranked[0][0],
  confidence: ranked[0][1],
  ranked: ranked.map(([option, probability]) => ({ option, probability })),
  options: ranked.map(([option]) => option),
  options_kind: kind,
  record_id: `record-${id}`,
})

const outcome = (decisions: api.Decision[], parallel = false): api.Outcome => ({
  roundTripMs: 120.6,
  result: { model: 'D.A.R.T.', decisions, latency_ms: 98.44, parallel, stored: true },
})

const field = (question: number, name: 'Query' | 'Options') =>
  within(screen.getByRole('group', { name: `Question ${question}` })).getByLabelText(name)

async function startEmpty() {
  const user = userEvent.setup()
  render(<Playground />)
  for (let i = SEED_QUESTIONS.length; i > 1; i--) {
    await user.click(screen.getByRole('button', { name: `Remove question ${i}` }))
  }
  await user.clear(field(1, 'Query'))
  await user.clear(field(1, 'Options'))
  await user.clear(screen.getByLabelText('Context'))
  return user
}

beforeEach(() => {
  decide.mockReset()
  decideImpl = decide
  healthImpl = vi.fn().mockResolvedValue(GPU)
})
afterEach(() => vi.unstubAllGlobals())

describe('Playground', () => {
  it('opens with the example ready to decide and an empty result area', async () => {
    render(<Playground />)
    expect(screen.getByLabelText('Context')).toHaveValue(SEED_CONTEXT)
    expect(screen.getAllByRole('group')).toHaveLength(SEED_QUESTIONS.length)
    expect(field(1, 'Query')).toHaveValue('Where does Maria work?')
    expect(screen.getByRole('button', { name: /Decide/ })).toBeEnabled()
    expect(screen.getByText('Every option, ranked, in one pass.')).toBeInTheDocument()
    expect(await screen.findByText('v1.0 · cuda')).toBeInTheDocument()
  })

  it('links home and to the API docs on the landing page', () => {
    render(<Playground />)
    expect(screen.getByRole('link', { name: 'D.A.R.T. home' })).toHaveAttribute('href', '/')
    expect(screen.getByRole('link', { name: 'API docs' })).toHaveAttribute('href', '/#api')
    expect(screen.getByRole('link', { name: 'GitHub' })).toHaveAttribute('href', 'https://github.com/PrathamR015/DART')
  })

  it('previews the options as chips while typing, ranges included', async () => {
    const user = await startEmpty()
    await user.type(field(1, 'Options'), '0-5')
    const chips = within(screen.getByRole('list', { name: 'Options for question 1' })).getAllByRole('listitem')
    expect(chips.map((chip) => chip.textContent)).toEqual(['0', '1', '2', '3', '4', '5'])
    expect(screen.getByText('range · 6 values')).toBeInTheDocument()
  })

  it('explains an unusable range and keeps Decide disabled', async () => {
    const user = await startEmpty()
    await user.type(field(1, 'Query'), 'Rate it')
    await user.type(field(1, 'Options'), '0-100')
    expect(screen.getByText(/add a larger step, for example '0-100 step 10'/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Decide/ })).toBeDisabled()
  })

  it('asks for a query and two options before deciding', async () => {
    const user = await startEmpty()
    expect(screen.getByText('Add a query and at least two options.')).toBeInTheDocument()
    await user.type(field(1, 'Query'), 'Is it good?')
    await user.type(field(1, 'Options'), 'yes')
    expect(screen.getByRole('button', { name: /Decide/ })).toBeDisabled()
    await user.type(field(1, 'Options'), ', no')
    expect(screen.queryByText('Add a query and at least two options.')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Decide/ })).toBeEnabled()
  })

  it('adds and removes questions, up to the limit', async () => {
    const user = userEvent.setup()
    render(<Playground />)
    const add = screen.getByRole('button', { name: 'Add question' })
    for (let i = SEED_QUESTIONS.length; i < api.MAX_QUESTIONS; i++) await user.click(add)
    expect(screen.getAllByRole('group')).toHaveLength(api.MAX_QUESTIONS)
    expect(screen.getByRole('button', { name: `Up to ${api.MAX_QUESTIONS} questions` })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Remove question 1' }))
    expect(field(1, 'Query')).toHaveValue('Which city does Maria live in?')
  })

  it('predicts how the server will run the questions', async () => {
    const user = userEvent.setup()
    render(<Playground />)
    expect(await screen.findByText('4 fast passes · 4 questions')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Add question' }))
    await user.click(screen.getByRole('button', { name: 'Add question' }))
    expect(screen.getByText('One parallel pass · 6 questions')).toBeInTheDocument()
    await user.clear(screen.getByLabelText('Context'))
    expect(screen.getByText('6 fast passes · 6 questions')).toBeInTheDocument()
  })

  it('sends the context and every question, then shows each ranked decision and the run stats', async () => {
    decide.mockResolvedValue(
      outcome([
        decision('q1', [['hospital', 0.974], ['airport', 0.011], ['school', 0.008], ['bank', 0.007]]),
        decision('q2', [['Lisbon', 0.602], ['Madrid', 0.268], ['Tokyo', 0.087], ['Rome', 0.043]]),
        decision('q3', [['bicycle', 0.946], ['bus', 0.022], ['train', 0.019], ['car', 0.013]]),
        decision('q4', [['to visit Tokyo', 0.973], ['her cats', 0.011], ['for her job', 0.01], ['school exam', 0.006]]),
      ]),
    )
    const user = userEvent.setup()
    render(<Playground />)
    await user.click(screen.getByRole('button', { name: /Decide/ }))
    expect(decide).toHaveBeenCalledWith(SEED_CONTEXT, SEED_QUESTIONS)
    const first = await screen.findByRole('article', { name: 'Result 1' })
    expect(within(first).getByText('hospital', { selector: 'strong' })).toBeInTheDocument()
    expect(within(first).getByRole('img', { name: 'Confidence 97.4%' })).toBeInTheDocument()
    expect(within(first).getAllByRole('listitem')).toHaveLength(4)
    const second = screen.getByRole('article', { name: 'Result 2' })
    expect(within(second).getByText('Which city does Maria live in?')).toBeInTheDocument()
    const stats = screen.getByText('Forward passes').closest('dl')!
    expect(within(stats).getByText('98.4 ms')).toBeInTheDocument()
    expect(within(stats).getByText('121 ms')).toBeInTheDocument()
    expect(within(stats).getAllByText('4')).toHaveLength(2) // 4 questions, 4 forward passes (not parallel)
  })

  it('reports one forward pass when the server used the parallel path', async () => {
    decide.mockResolvedValue(outcome([decision('q1', [['a', 0.6], ['b', 0.4]]), decision('q2', [['c', 0.7], ['d', 0.3]])], true))
    const user = userEvent.setup()
    render(<Playground />)
    await user.click(screen.getByRole('button', { name: /Decide/ }))
    const stats = (await screen.findByText('Forward passes')).closest('div')!
    expect(within(stats).getByText('1')).toBeInTheDocument()
  })

  it('summarises long rankings and shows resolved range values', async () => {
    const ranked: [string, number][] = Array.from({ length: 11 }, (_, i) => [String(i), i === 7 ? 0.5 : 0.05])
    decide.mockResolvedValue(outcome([decision('q1', [ranked[7], ...ranked.filter((_, i) => i !== 7)], 'integer_range')]))
    const user = await startEmpty()
    await user.type(field(1, 'Query'), 'Rate from 0 to 10')
    await user.type(field(1, 'Options'), '0-10')
    await user.click(screen.getByRole('button', { name: /Decide/ }))
    const card = await screen.findByRole('article', { name: 'Result 1' })
    expect(within(card).getByText('+5 more options')).toBeInTheDocument()
    expect(within(card).getByText(/Range values: 7, 0, 1/)).toBeInTheDocument()
  })

  it('shows the server message when the request fails', async () => {
    const real = await vi.importActual<typeof api>('../api')
    decideImpl = real.decide
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: false, status: 422, json: async () => ({ detail: 'Question 2: options must be distinct' }) }),
    )
    const user = userEvent.setup()
    render(<Playground />)
    await user.click(screen.getByRole('button', { name: /Decide/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Question 2: options must be distinct')
    expect(screen.queryByRole('article')).not.toBeInTheDocument()
  })

  it('marks the server offline when the health check fails', async () => {
    healthImpl = vi.fn().mockRejectedValue(new Error('Cannot reach the D.A.R.T. server.'))
    render(<Playground />)
    expect(await screen.findByText('server offline')).toBeInTheDocument()
    expect(screen.getByText('One request · 4 questions')).toBeInTheDocument()
  })

  it('submits with Ctrl+Enter', async () => {
    decide.mockResolvedValue(outcome([decision('q1', [['a', 0.6], ['b', 0.4]])]))
    const user = userEvent.setup()
    render(<Playground />)
    await user.type(field(1, 'Query'), '{Control>}{Enter}{/Control}')
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(1))
  })
})
