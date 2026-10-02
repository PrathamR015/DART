import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import Landing from './Landing'

describe('Landing', () => {
  it('leads with the brand promise and links to the playground page', () => {
    render(<Landing />)
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(/^Decision already reached\.Thanks\.$/i)
    const ctas = screen.getAllByRole('link', { name: /Open the playground/ })
    expect(ctas.map((link) => link.getAttribute('href'))).toEqual(['/playground/', '/playground/'])
    const nav = screen.getByRole('navigation', { name: 'Main' })
    expect(within(nav).getByRole('link', { name: 'Playground' })).toHaveAttribute('href', '/playground/')
    expect(within(nav).getByRole('link', { name: 'API' })).toHaveAttribute('href', '#api')
    const repo = within(nav).getByRole('link', { name: 'GitHub' })
    expect(repo).toHaveAttribute('href', 'https://github.com/PrathamR015/DART')
    expect(repo).toHaveAttribute('target', '_blank')
    expect(repo).toHaveAttribute('rel', 'noopener noreferrer')
    expect(repo.querySelector('svg')).not.toBeNull()
  })

  it('names the brand once for screen readers', () => {
    render(<Landing />)
    expect(screen.getByRole('link', { name: 'D.A.R.T. home' })).toHaveAttribute('href', '#top')
  })

  it('shows a real example decision with every option ranked', () => {
    render(<Landing />)
    const card = screen.getByRole('figure', { name: 'Example decision' })
    expect(within(card).getByText('My debit card was stolen this morning. What should I do?')).toBeInTheDocument()
    expect(within(card).getAllByText('91.0%')).toHaveLength(2) // the badge and the top bar
    // every other decision sits next to its own question, so the card reads without any context
    const others = within(within(card).getByRole('list', { name: 'More decisions' })).getAllByRole('listitem')
    expect(others[0]).toHaveTextContent('Which of these animals is a mammal?dolphin · 96.1%')
    expect(card.textContent).not.toContain('Maria')
  })

  it('documents the endpoint the backend actually serves', () => {
    render(<Landing />)
    const api = document.getElementById('api')!
    expect(within(api).getByText('/api/decide')).toBeInTheDocument()
    expect(api.textContent).toContain('"options": "0-5"')
    expect(api.textContent).toContain('"latency_ms"')
    expect(api.textContent).not.toContain('Maria')
  })

  it('sends people to the playground when no contact email is configured', () => {
    render(<Landing />)
    const access = document.getElementById('access')!
    expect(within(access).getByRole('link')).toHaveAttribute('href', '/playground/')
    expect(access.textContent).not.toContain('[')  // no unfilled placeholders from the mockup
  })

  it('has the four how-it-works steps', () => {
    render(<Landing />)
    const steps = within(document.getElementById('how')!).getAllByRole('listitem')
    expect(steps.map((step) => step.querySelector('.step-title')?.textContent)).toEqual([
      'Read once',
      'Ask in parallel',
      'Head, not decoder',
      'Ranked and final',
    ])
  })
})
