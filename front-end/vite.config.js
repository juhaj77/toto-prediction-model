import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api-veikkaus': {
        target: 'https://www.veikkaus.fi',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api-veikkaus/, '')
      }
    }
  }
})
