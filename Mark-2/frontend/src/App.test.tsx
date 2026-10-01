import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import * as api from './api'

// `decide` is forwarded through a plain function (not a spy) so a rejected call is handled only by the app itself.
let implementation: typeof api.decide
vi.mock('./api', async (importOriginal) => ({
  ...(await importOriginal<typeof api>()),
  decide: (...args: Parameters<typeof api.decide>) => implementation(...args),
}))
const decide = vi.fn<typeof api.decide>()

const outcome = (kind: api.DecideResponse['options_kind'] = 'list'): api.Outcome => ({
  roundTripMs: 250.4,
  result: {
    id: 'abc',
    model: 'D.A.R.T.',
    decisions: [
      {
        id: 'q1',
        decision: 'England',
        confidence: 0.973,
        ranked: [
          { option: 'England', probability: 0.973 },
          { option: 'Japan', probability: 0.027 },
        ],
      },
    ],
    latency_ms: 11.42,
    options: ['England', 'Japan'],
    options_kind: kind,
    stored: true,
  },
})

async function fill(query = 'Where is Tony from?', options = 'England, Japan') {
  const user = userEvent.setup()
  await user.type(screen.getByLabelText('Your query'), query)
  await user.type(screen.getByLabelText('Options'), options)
  return user
}

beforeEach(() => {
  decide.mockReset()
  implementation = decide
})
afterEach(() => vi.unstubAllGlobals())

describe('App', () => {
  it('shows the D.A.R.T. branding and a single query box', () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: 'D.A.R.T.' })).toBeInTheDocument()
    expect(screen.getByText('Decision Already Reached, Thanks.')).toBeInTheDocument()
    expect(screen.getAllByRole('textbox')).toHaveLength(2) // the query and the options, no separate context box
  })

  it('keeps the button disabled until both fields are filled', async () => {
    render(<App />)
    const button = screen.getByRole('button', { name: 'Decide' })
    expect(button).toBeDisabled()
    await fill()
    expect(button).toBeEnabled()
  })

  it('sends the query and options and shows the decision, confidence, probabilities and latency', async () => {
    decide.mockResolvedValue(outcome())
    render(<App />)
    const user = await fill()
    await user.click(screen.getByRole('button', { name: 'Decide' }))
    expect(decide).toHaveBeenCalledWith('Where is Tony from?', 'England, Japan')
    expect(await screen.findByText('England', { selector: 'strong' })).toBeInTheDocument()
    expect(screen.getByText(/confidence 97\.3%/)).toBeInTheDocument()
    expect(screen.getByText(/Model: 11\.4 ms · Round trip: 250 ms/)).toBeInTheDocument()
    expect(screen.getByText('2.7%')).toBeInTheDocument()
  })

  it('shows the resolved values for a range', async () => {
    decide.mockResolvedValue({ ...outcome('integer_range') })
    render(<App />)
    const user = await fill('Rate this review from 0 to 5: great', '0-5')
    await user.click(screen.getByRole('button', { name: 'Decide' }))
    expect(await screen.findByText(/Range values: England, Japan/)).toBeInTheDocument()
  })

  it('shows an error message and no result when the request fails', async () => {
    // Run the real client against a fake 422 response, so the whole error path (server message to screen) is tested.
    const real = await vi.importActual<typeof api>('./api')
    implementation = real.decide
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: false, status: 422, json: async () => ({ detail: 'need at least 2 options' }) }),
    )
    render(<App />)
    const user = await fill('q', 'only')
    await user.click(screen.getByRole('button', { name: 'Decide' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('need at least 2 options')
    expect(screen.queryByText(/Decision:/)).not.toBeInTheDocument()
  })

  it('submits with Ctrl+Enter from the query box', async () => {
    decide.mockResolvedValue(outcome())
    render(<App />)
    const user = await fill()
    await user.type(screen.getByLabelText('Your query'), '{Control>}{Enter}{/Control}')
    expect(decide).toHaveBeenCalledTimes(1)
  })
})
