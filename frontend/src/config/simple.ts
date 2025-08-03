/**
 * 简化的配置管理
 * 避免复杂的类型问题，直接提供配置对象
 */

// 环境类型
export type Environment = 'development' | 'production' | 'test'

// 日志级别
export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARN = 'warn',
  ERROR = 'error'
}

// 获取环境变量的辅助函数
const getEnvVar = (key: string, defaultValue: string = ''): string => {
  // 在开发环境下，直接返回配置的值（避免 import.meta 类型问题）
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      // 开发环境的配置（基于 .env.development 文件）
      const devConfig: Record<string, string> = {
        'VITE_API_BASE_URL': 'http://172.16.1.3:8080/api/v1',
        'VITE_WS_URL': 'ws://172.16.1.3:8080/ws',
        'VITE_CHAT_WS_URL': 'ws://172.16.1.3:8000',
        'VITE_API_TIMEOUT': '30000',
        'VITE_WS_RECONNECT_ATTEMPTS': '5',
        'VITE_WS_RECONNECT_INTERVAL': '1000',
        'VITE_LOG_LEVEL': 'debug',
        'VITE_LOG_ENABLE_CONSOLE': 'true',
        'VITE_LOG_ENABLE_REMOTE': 'false',
        'VITE_MAPBOX_ACCESS_TOKEN': '',
        'VITE_APP_TITLE': 'AI智能旅行规划助手',
        'VITE_APP_VERSION': '1.0.0'
      }

      if (devConfig[key]) {
        return devConfig[key]
      }
    }
  }

  // 尝试从 process.env 获取（Node.js环境变量）
  if (typeof process !== 'undefined' && process.env) {
    return process.env[key] || defaultValue
  }

  return defaultValue
}

// 检测环境
const isDevelopment = typeof window !== 'undefined' &&
  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')

// 配置对象
export const config = {
  // 环境信息
  env: (isDevelopment ? 'development' : 'production') as Environment,
  isDevelopment,
  isProduction: !isDevelopment,

  // API 配置
  api: {
    baseUrl: getEnvVar('VITE_API_BASE_URL', '/api/v1'),
    timeout: parseInt(getEnvVar('VITE_API_TIMEOUT', '30000'), 10),
  },

  // WebSocket 配置
  websocket: {
    url: getEnvVar('VITE_WS_URL', isDevelopment ? 'ws://localhost:8080/ws' : `ws://${typeof window !== 'undefined' ? window.location.host : 'localhost'}/ws`),
    reconnectAttempts: parseInt(getEnvVar('VITE_WS_RECONNECT_ATTEMPTS', '5'), 10),
    reconnectInterval: parseInt(getEnvVar('VITE_WS_RECONNECT_INTERVAL', '1000'), 10),
  },

  // 聊天服务配置
  chat: {
    wsUrl: getEnvVar('VITE_CHAT_WS_URL', isDevelopment ? 'ws://localhost:8000' : `ws://${typeof window !== 'undefined' ? window.location.host : 'localhost'}`),
  },

  // 日志配置
  logging: {
    level: getEnvVar('VITE_LOG_LEVEL', isDevelopment ? LogLevel.DEBUG : LogLevel.ERROR) as LogLevel,
    enableConsole: getEnvVar('VITE_LOG_ENABLE_CONSOLE', isDevelopment ? 'true' : 'false') === 'true',
    enableRemote: getEnvVar('VITE_LOG_ENABLE_REMOTE', isDevelopment ? 'false' : 'true') === 'true',
  },

  // 地图配置
  map: {
    mapboxAccessToken: getEnvVar('VITE_MAPBOX_ACCESS_TOKEN', ''),
  },

  // 应用信息
  app: {
    title: getEnvVar('VITE_APP_TITLE', 'AI智能旅行规划助手'),
    version: getEnvVar('VITE_APP_VERSION', '1.0.0'),
  },
}

// 导出默认配置
export default config

// 开发环境下打印配置信息
if (isDevelopment && typeof console !== 'undefined') {
  console.log('App Config:', config)
}
