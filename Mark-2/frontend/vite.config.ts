/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Two standalone pages: the landing page at / and the playground at /playground/.
// In development the dev server forwards /api to the FastAPI backend, so no CORS setup is needed.
// For a deployed build, set VITE_API_URL to the backend's address.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': 'http://localhost:8000' } },
  build: {
    rollupOptions: {
      input: {
        landing: 'index.html', // paths are relative to the project root
        playground: 'playground/index.html',
      },
    },
  },
  test: { environment: 'jsdom', globals: true, setupFiles: './src/setupTests.ts' },
})
