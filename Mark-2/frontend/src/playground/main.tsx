import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '../brand/brand.css'
import './playground.css'
import Playground from './Playground'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Playground />
  </StrictMode>,
)
