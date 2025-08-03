import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@components': resolve(__dirname, 'src/components'),
      '@pages': resolve(__dirname, 'src/pages'),
      '@hooks': resolve(__dirname, 'src/hooks'),
      '@utils': resolve(__dirname, 'src/utils'),
      '@store': resolve(__dirname, 'src/store'),
      '@types': resolve(__dirname, 'src/types'),
      '@api': resolve(__dirname, 'src/api'),
      '@assets': resolve(__dirname, 'src/assets'),
    },
  },
  server: {
    port: 3000,
    host: true,
    proxy: {
      '/api': {
        target: 'http://172.16.1.3:8080',
        changeOrigin: true,
        secure: false,
      },
      // WebSocket 代理配置
      '^/ws/.*': {
        target: 'ws://172.16.1.3:8080',
        ws: true,
        changeOrigin: true,
        rewrite: (path) => {
          console.log('WebSocket proxy rewrite:', path);
          return path; // 保持原路径
        }
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          antd: ['antd'],
          router: ['react-router-dom'],
          redux: ['@reduxjs/toolkit', 'react-redux'],
          charts: ['recharts'],
          map: ['react-map-gl', 'mapbox-gl'],
        },
      },
    },
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'antd', 'lodash'],
  },
}) 