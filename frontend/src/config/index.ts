/**
 * 应用配置管理
 * 统一管理所有环境变量和配置项
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

// 配置接口
export interface AppConfig {
  // 环境信息
  env: Environment
  isDevelopment: boolean
  isProduction: boolean
  
  // API 配置
  api: {
    baseUrl: string
    timeout: number
  }
  
  // WebSocket 配置
  websocket: {
    url: string
    reconnectAttempts: number
    reconnectInterval: number
  }
  
  // 聊天服务配置
  chat: {
    wsUrl: string
  }
  
  // 日志配置
  logging: {
    level: LogLevel
    enableConsole: boolean
    enableRemote: boolean
  }
  
  // 地图配置
  map: {
    mapboxAccessToken: string
  }
  
  // 应用信息
  app: {
    title: string
    version: string
  }
}

/**
 * 获取环境变量值，支持默认值
 * 使用简化的方法避免 import.meta 类型问题
 */
function getEnvVar(key: string, defaultValue: string = ''): string {
  // 在开发环境下，直接返回配置的值
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      // 开发环境的硬编码配置（使用前端代理）
      const devConfig: Record<string, string> = {
        'VITE_API_BASE_URL': '/api/v1',
        'VITE_WS_URL': '/ws',
        'VITE_CHAT_WS_URL': '/ws',
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

  // 兼容不同环境的环境变量获取方式
  if (typeof window !== 'undefined' && (window as any).__ENV__) {
    return (window as any).__ENV__[key] || defaultValue
  }

  // Node.js 环境变量
  if (typeof process !== 'undefined' && process.env) {
    return process.env[key] || defaultValue
  }

  return defaultValue
}

/**
 * 获取布尔类型环境变量
 */
function getBooleanEnvVar(key: string, defaultValue: boolean = false): boolean {
  const value = getEnvVar(key)
  if (!value) return defaultValue
  return value.toLowerCase() === 'true'
}

/**
 * 获取数字类型环境变量
 */
function getNumberEnvVar(key: string, defaultValue: number = 0): number {
  const value = getEnvVar(key)
  if (!value) return defaultValue
  const parsed = parseInt(value, 10)
  return isNaN(parsed) ? defaultValue : parsed
}

/**
 * 创建应用配置
 */
function createConfig(): AppConfig {
  // 简化环境检测
  const isDev = typeof window !== 'undefined' && window.location.hostname === 'localhost'
  const env: Environment = isDev ? 'development' : 'production'

  return {
    // 环境信息
    env,
    isDevelopment: env === 'development',
    isProduction: env === 'production',

    // API 配置
    api: {
      baseUrl: getEnvVar('VITE_API_BASE_URL', '/api/v1'),
      timeout: getNumberEnvVar('VITE_API_TIMEOUT', 30000),
    },

    // WebSocket 配置
    websocket: {
      url: getEnvVar('VITE_WS_URL', env === 'development' ? 'ws://localhost:8080/ws' : `ws://${typeof window !== 'undefined' ? window.location.host : 'localhost'}/ws`),
      reconnectAttempts: getNumberEnvVar('VITE_WS_RECONNECT_ATTEMPTS', 5),
      reconnectInterval: getNumberEnvVar('VITE_WS_RECONNECT_INTERVAL', 1000),
    },

    // 聊天服务配置
    chat: {
      wsUrl: getEnvVar('VITE_CHAT_WS_URL', env === 'development' ? 'ws://localhost:8000' : `ws://${typeof window !== 'undefined' ? window.location.host : 'localhost'}`),
    },

    // 日志配置
    logging: {
      level: getEnvVar('VITE_LOG_LEVEL', env === 'development' ? LogLevel.DEBUG : LogLevel.ERROR) as LogLevel,
      enableConsole: getBooleanEnvVar('VITE_LOG_ENABLE_CONSOLE', env === 'development'),
      enableRemote: getBooleanEnvVar('VITE_LOG_ENABLE_REMOTE', env === 'production'),
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
}

// 导出配置实例
export const config = createConfig()

// 导出默认配置
export default config

// 开发环境下打印配置信息
if (config.isDevelopment) {
  console.log('App Config:', config)
}
