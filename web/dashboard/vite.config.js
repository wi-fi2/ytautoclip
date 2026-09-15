import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        // Defaults to plain local dev (`npm run dev` outside Docker). Docker
        // Compose overrides this to the 'backend' service name via the
        // VITE_API_TARGET env var (see docker-compose.yml).
        target: process.env.VITE_API_TARGET || 'http://localhost:8000',
        changeOrigin: true,
      },
    },
    watch: {
      usePolling: true,
    },
  },
})
