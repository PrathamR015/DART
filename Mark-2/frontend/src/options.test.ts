import { describe, expect, it } from 'vitest'
import { describeOptions, parseOptions } from './options'

// The same cases as Mark-2/backend/tests/test_mark2_options.py, so the preview agrees with the server.
const values = (text: string) => {
  const parsed = parseOptions(text)
  if (!parsed.ok) throw new Error(parsed.error)
  return parsed.options
}
const kind = (text: string) => {
  const parsed = parseOptions(text)
  return parsed.ok ? parsed.kind : null
}
const error = (text: string) => {
  const parsed = parseOptions(text)
  return parsed.ok ? null : parsed.error
}

describe('parseOptions', () => {
  it('reads a comma-separated list', () => {
    expect(values('yes, no, maybe')).toEqual(['yes', 'no', 'maybe'])
    expect(kind('yes, no, maybe')).toBe('list')
  })

  it('uses pipes when options contain commas', () => {
    expect(values('Paris, France | London, UK | Rome, Italy')).toEqual(['Paris, France', 'London, UK', 'Rome, Italy'])
  })

  it('ignores extra spaces and empty items', () => expect(values(' a , , b ,')).toEqual(['a', 'b']))

  it.each(['only', 'a, A', '', '  ', ',', ', ,'])('rejects the bad list %j', (text) => {
    expect(error(text)).not.toBeNull()
  })

  it('rejects more than twelve options and overlong options', () => {
    expect(error(Array.from({ length: 13 }, (_, i) => i).join(', '))).toContain('12')
    expect(error(`a, ${'x'.repeat(200)}`)).toContain('100 characters')
  })

  it.each(['0-5', '0 to 5', '0..5', '0–5', '  0 - 5  ', '0 TO 5'])('reads the integer range %j', (text) => {
    expect(values(text)).toEqual(['0', '1', '2', '3', '4', '5'])
    expect(kind(text)).toBe('integer_range')
  })

  it('reads integer steps', () => expect(values('0 to 10 step 2')).toEqual(['0', '2', '4', '6', '8', '10']))

  it('steps decimal ranges by their finest decimal place', () => {
    expect(values('0.0-1.0')).toEqual([...Array.from({ length: 10 }, (_, i) => `0.${i}`), '1.0'])
    expect(kind('0.0-1.0')).toBe('decimal_range')
  })

  it('formats decimal steps consistently and exactly', () => {
    expect(values('0-5 step 0.5')).toEqual(['0.0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0'])
    expect(values('0.1-0.4 step 0.1')).toEqual(['0.1', '0.2', '0.3', '0.4'])
  })

  it('handles negative ranges written with "to"', () => {
    expect(values('-1 to 1')).toEqual(['-1', '0', '1'])
    expect(values('-1 to 1 step 0.5')).toEqual(['-1.0', '-0.5', '0.0', '0.5', '1.0'])
  })

  it('stops before the end when the step does not divide the range', () => {
    expect(values('0-5 step 2')).toEqual(['0', '2', '4'])
  })

  it.each([
    ['0-100', '10'],
    ['0.0-5.0', '0.5'],
    ['0-20', '2'],
  ])('suggests a plain step for the too-large range %j', (text, step) => {
    expect(error(text)).toContain(`'${text} step ${step}'`)
  })

  it.each(['5-0', '3-3', '0-5 step 0', '0-5 step 10'])('rejects the invalid range %j', (text) => {
    expect(error(text)).not.toBeNull()
  })

  it('treats words that only look like a range as list items', () => {
    expect(error('well-known')).toBe('need at least 2 options')
    expect(values('well-known, lesser-known')).toEqual(['well-known', 'lesser-known'])
  })
})

describe('describeOptions', () => {
  it('counts list options and range values', () => {
    expect(describeOptions(parseOptions('a, b, c, d'))).toBe('4 options')
    expect(describeOptions(parseOptions('0-5'))).toBe('range · 6 values')
    expect(describeOptions(parseOptions('only'))).toBe('')
  })
})
