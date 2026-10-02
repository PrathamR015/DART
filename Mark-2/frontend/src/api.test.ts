import { afterEach, describe, expect, it, vi } from 'vitest'
import { decide, errorMessage, health } from './api'

const okBody = {
  model: 'D.A.R.T.',
  decisions: [
    {
      id: 'q1',
      ranked: [{ option: 'yes', probability: 0.9 }],
      decision: 'yes',
      confidence: 0.9,
      options: ['yes', 'no'],
      options_kind: 'list',
      record_id: 'abc',
    },
  ],
  latency_ms: 12.3,
  parallel: false,
  stored: true,
}

function mockFetch(response: Partial<Response> | Error) {
  const fetchMock = response instanceof Error ? vi.fn().mockRejectedValue(response) : vi.fn().mockResolvedValue(response)
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

const one = [{ query: 'Is it good?', options: 'yes, no' }]

afterEach(() => vi.unstubAllGlobals())

describe('decide', () => {
  it('posts the context and questions and returns the result with the round trip time', async () => {
    const fetchMock = mockFetch({ ok: true, status: 200, json: async () => okBody })
    const questions = [...one, { query: 'How good?', options: '1-5' }]
    const { result, roundTripMs } = await decide('A review: great.', questions)
    expect(result.decisions[0].decision).toBe('yes')
    expect(roundTripMs).toBeGreaterThanOrEqual(0)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/decide')
    expect(JSON.parse(init.body)).toEqual({ context: 'A review: great.', questions })
  })

  it('throws the server message for a rejected request', async () => {
    mockFetch({ ok: false, status: 422, json: async () => ({ detail: 'need at least 2 options' }) })
    await expect(decide('', [{ query: 'q', options: 'only' }])).rejects.toThrow('need at least 2 options')
  })

  it('explains when the server cannot be reached', async () => {
    mockFetch(new TypeError('Failed to fetch'))
    await expect(decide('', one)).rejects.toThrow('Cannot reach the D.A.R.T. server')
  })
})

describe('health', () => {
  it('reads the model status', async () => {
    const status = { status: 'ok', model_loaded: true, device: 'cuda', model_version: 'v1.0', parallel_min_questions: 6 }
    const fetchMock = mockFetch({ ok: true, status: 200, json: async () => status })
    expect(await health()).toEqual(status)
    expect(fetchMock.mock.calls[0][0]).toBe('/api/health')
  })

  it('explains when the server cannot be reached', async () => {
    mockFetch(new TypeError('Failed to fetch'))
    await expect(health()).rejects.toThrow('Cannot reach the D.A.R.T. server')
  })
})

describe('errorMessage', () => {
  it('reads a text detail', () => expect(errorMessage(422, { detail: 'bad' })).toBe('bad'))
  it('joins validation error messages', () => {
    expect(errorMessage(422, { detail: [{ msg: 'too short' }, { msg: 'too long' }] })).toBe('too short; too long')
  })
  it('falls back to the status code', () => expect(errorMessage(500, null)).toContain('500'))
})
