/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In development the dev server forwards /api to the FastAPI backend, so no CORS setup is needed.
// For a deployed build, set VITE_API_URL to the backend's address.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': 'http://localhost:8000' } },
  test: { environment: 'jsdom', globals: true, setupFiles: './src/setupTests.ts' },
})
