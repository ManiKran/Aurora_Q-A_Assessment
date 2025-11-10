import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  preview: {
    allowedHosts: [
      'gentle-recreation-production.up.railway.app', // ✅ your Railway domain
    ],
    port: 4173, // optional, just to be explicit
  },
})