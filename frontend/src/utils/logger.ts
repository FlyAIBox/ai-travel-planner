// 日志级别
export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
}

// 日志配置
interface LoggerConfig {
  level: LogLevel
  enableConsole: boolean
  enableRemote: boolean
  remoteEndpoint?: string
}

import config from '@/config'

// 默认配置
const defaultConfig: LoggerConfig = {
  level: config.logging.level === 'debug' ? LogLevel.DEBUG :
        config.logging.level === 'info' ? LogLevel.INFO :
        config.logging.level === 'warn' ? LogLevel.WARN : LogLevel.ERROR,
  enableConsole: config.logging.enableConsole,
  enableRemote: config.logging.enableRemote,
  remoteEndpoint: `${config.api.baseUrl}/logs`,
}

class Logger {
  private config: LoggerConfig
  private logQueue: any[] = []
  private isFlushingLogs = false

  constructor(config: Partial<LoggerConfig> = {}) {
    this.config = { ...defaultConfig, ...config }
  }

  // 格式化时间戳
  private formatTimestamp(): string {
    return new Date().toISOString()
  }

  // 格式化日志消息
  private formatMessage(level: string, message: string, data?: any): string {
    const timestamp = this.formatTimestamp()
    const dataStr = data ? ` | ${JSON.stringify(data)}` : ''
    return `[${timestamp}] [${level}] ${message}${dataStr}`
  }

  // 输出到控制台
  private logToConsole(level: LogLevel, message: string, data?: any) {
    if (!this.config.enableConsole) return

    const levelName = LogLevel[level]
    const formattedMessage = this.formatMessage(levelName, message, data)

    switch (level) {
      case LogLevel.DEBUG:
        console.debug(formattedMessage)
        break
      case LogLevel.INFO:
        console.info(formattedMessage)
        break
      case LogLevel.WARN:
        console.warn(formattedMessage)
        break
      case LogLevel.ERROR:
        console.error(formattedMessage)
        break
    }
  }

  // 发送到远程服务器
  private async logToRemote(level: LogLevel, message: string, data?: any) {
    if (!this.config.enableRemote || !this.config.remoteEndpoint) return

    const logEntry = {
      timestamp: this.formatTimestamp(),
      level: LogLevel[level],
      message,
      data,
      userAgent: navigator.userAgent,
      url: window.location.href,
      userId: localStorage.getItem('userId') || 'anonymous',
    }

    this.logQueue.push(logEntry)
    
    // 批量发送日志
    if (!this.isFlushingLogs) {
      setTimeout(() => this.flushLogs(), 1000)
    }
  }

  // 批量发送日志到服务器
  private async flushLogs() {
    if (this.isFlushingLogs || this.logQueue.length === 0) return

    this.isFlushingLogs = true
    const logsToSend = [...this.logQueue]
    this.logQueue = []

    try {
      await fetch(this.config.remoteEndpoint!, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token') || ''}`,
        },
        body: JSON.stringify({ logs: logsToSend }),
      })
    } catch (error) {
      // 如果发送失败，重新加入队列
      this.logQueue.unshift(...logsToSend)
      console.error('Failed to send logs to remote server:', error)
    } finally {
      this.isFlushingLogs = false
    }
  }

  // 通用日志方法
  private log(level: LogLevel, message: string, data?: any) {
    if (level < this.config.level) return

    this.logToConsole(level, message, data)
    this.logToRemote(level, message, data)
  }

  // 公共方法
  debug(message: string, data?: any) {
    this.log(LogLevel.DEBUG, message, data)
  }

  info(message: string, data?: any) {
    this.log(LogLevel.INFO, message, data)
  }

  warn(message: string, data?: any) {
    this.log(LogLevel.WARN, message, data)
  }

  error(message: string, data?: any) {
    this.log(LogLevel.ERROR, message, data)
  }

  // 性能监控
  time(label: string) {
    console.time(label)
  }

  timeEnd(label: string) {
    console.timeEnd(label)
  }

  // 用户行为追踪
  trackEvent(event: string, properties?: Record<string, any>) {
    this.info(`User Event: ${event}`, properties)
  }

  // API请求追踪
  trackApiCall(method: string, url: string, status: number, duration: number, error?: any) {
    const logData = {
      method,
      url,
      status,
      duration,
      error: error ? error.message : undefined,
    }

    if (status >= 400) {
      this.error(`API Error: ${method} ${url}`, logData)
    } else {
      this.info(`API Call: ${method} ${url}`, logData)
    }
  }

  // 更新配置
  updateConfig(config: Partial<LoggerConfig>) {
    this.config = { ...this.config, ...config }
  }
}

// 创建全局日志器实例
export const logger = new Logger()

// 导出工具函数
export const createLogger = (config?: Partial<LoggerConfig>) => new Logger(config)

// 错误边界日志记录
export const logError = (error: Error, errorInfo?: any) => {
  logger.error('Application Error', {
    error: {
      name: error.name,
      message: error.message,
      stack: error.stack,
    },
    errorInfo,
  })
}

// 性能监控装饰器
export function performanceMonitor(target: any, propertyName: string, descriptor: PropertyDescriptor) {
  const method = descriptor.value

  descriptor.value = function (...args: any[]) {
    const label = `${target.constructor.name}.${propertyName}`
    logger.time(label)
    
    try {
      const result = method.apply(this, args)
      
      if (result instanceof Promise) {
        return result.finally(() => logger.timeEnd(label))
      } else {
        logger.timeEnd(label)
        return result
      }
    } catch (error) {
      logger.timeEnd(label)
      logger.error(`Method ${label} failed`, error)
      throw error
    }
  }

  return descriptor
}

export default logger 