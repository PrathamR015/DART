import { afterEach, describe, expect, it, vi } from 'vitest'
import { decide, errorMessage } from './api'

const okBody = {
  id: 'abc',
  model: 'D.A.R.T.',
  decisions: [{ id: 'q1', ranked: [{ option: 'yes', probability: 0.9 }], decision: 'yes', confidence: 0.9 }],
  latency_ms: 12.3,
  options: ['yes', 'no'],
  options_kind: 'list',
  stored: true,
}

function mockFetch(response: Partial<Response> | Error) {
  const fetchMock = response instanceof Error ? vi.fn().mockRejectedValue(response) : vi.fn().mockResolvedValue(response)
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => vi.unstubAllGlobals())

describe('decide', () => {
  it('posts the query and options and returns the result with the round trip time', async () => {
    const fetchMock = mockFetch({ ok: true, status: 200, json: async () => okBody })
    const { result, roundTripMs } = await decide('Is it good?', 'yes, no')
    expect(result.decisions[0].decision).toBe('yes')
    expect(roundTripMs).toBeGreaterThanOrEqual(0)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/decide')
    expect(JSON.parse(init.body)).toEqual({ query: 'Is it good?', options: 'yes, no' })
  })

  it('throws the server message for a rejected request', async () => {
    mockFetch({ ok: false, status: 422, json: async () => ({ detail: 'need at least 2 options' }) })
    await expect(decide('q', 'only')).rejects.toThrow('need at least 2 options')
  })

  it('explains when the server cannot be reached', async () => {
    mockFetch(new TypeError('Failed to fetch'))
    await expect(decide('q', 'a, b')).rejects.toThrow('Cannot reach the D.A.R.T. server')
  })
})

describe('errorMessage', () => {
  it('reads a text detail', () => expect(errorMessage(422, { detail: 'bad' })).toBe('bad'))
  it('joins validation error messages', () => {
    expect(errorMessage(422, { detail: [{ msg: 'too short' }, { msg: 'too long' }] })).toBe('too short; too long')
  })
  it('falls back to the status code', () => expect(errorMessage(500, null)).toContain('500'))
})
