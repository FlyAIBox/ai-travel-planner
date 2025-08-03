/// <reference types="vite/client" />

interface ImportMetaEnv {
  // API 配置
  readonly VITE_API_BASE_URL: string
  readonly VITE_API_TIMEOUT: string
  
  // WebSocket 配置
  readonly VITE_WS_URL: string
  readonly VITE_WS_RECONNECT_ATTEMPTS: string
  readonly VITE_WS_RECONNECT_INTERVAL: string
  
  // 聊天服务配置
  readonly VITE_CHAT_WS_URL: string
  
  // 日志配置
  readonly VITE_LOG_LEVEL: string
  readonly VITE_LOG_ENABLE_CONSOLE: string
  readonly VITE_LOG_ENABLE_REMOTE: string
  
  // 地图配置
  readonly VITE_MAPBOX_ACCESS_TOKEN: string
  
  // 应用配置
  readonly VITE_APP_TITLE: string
  readonly VITE_APP_VERSION: string
  
  // Vite 代理配置
  readonly VITE_PROXY_API_TARGET: string
  readonly VITE_PROXY_WS_TARGET: string
  
  // 环境变量
  readonly NODE_ENV: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
