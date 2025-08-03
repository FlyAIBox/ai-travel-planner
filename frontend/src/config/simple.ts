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
    baseUrl: '/api/v1',
    timeout: 30000,
  },
  
  // WebSocket 配置
  websocket: {
    url: isDevelopment ? 'ws://localhost:8080/ws' : `ws://${typeof window !== 'undefined' ? window.location.host : 'localhost'}/ws`,
    reconnectAttempts: 5,
    reconnectInterval: 1000,
  },
  
  // 聊天服务配置
  chat: {
    wsUrl: isDevelopment ? 'ws://localhost:8000' : `ws://${typeof window !== 'undefined' ? window.location.host : 'localhost'}`,
  },
  
  // 日志配置
  logging: {
    level: isDevelopment ? LogLevel.DEBUG : LogLevel.ERROR,
    enableConsole: isDevelopment,
    enableRemote: !isDevelopment,
  },
  
  // 地图配置
  map: {
    mapboxAccessToken: '',
  },
  
  // 应用信息
  app: {
    title: 'AI智能旅行规划助手',
    version: '1.0.0',
  },
}

// 导出默认配置
export default config

// 开发环境下打印配置信息
if (isDevelopment && typeof console !== 'undefined') {
  console.log('App Config:', config)
}
