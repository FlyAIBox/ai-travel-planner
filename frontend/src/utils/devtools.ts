import { logger } from './logger'

// 开发工具类
class DevTools {
  private isEnabled: boolean
  
  constructor() {
    this.isEnabled = process.env.NODE_ENV === 'development'
    
    if (this.isEnabled) {
      this.initializeDevTools()
    }
  }

  // 初始化开发工具
  private initializeDevTools() {
    // 在window对象上添加调试工具
    (window as any).__AI_TRAVEL_DEVTOOLS__ = {
      logger,
      clearLogs: () => console.clear(),
      logLevel: logger.debug.bind(logger),
      
      // Redux状态检查
      getState: () => {
        const store = (window as any).__REDUX_STORE__
        return store ? store.getState() : 'Redux store not found'
      },
      
      // WebSocket状态检查
      getWebSocketState: () => {
        const ws = (window as any).__WEBSOCKET_SERVICE__
        return ws ? {
          readyState: ws.getReadyState(),
          isConnected: ws.isConnected()
        } : 'WebSocket service not found'
      },
      
      // 本地存储检查
      getStorageInfo: () => ({
        localStorage: { ...localStorage },
        sessionStorage: { ...sessionStorage },
        storageUsage: this.getStorageUsage()
      }),
      
      // 性能信息
      getPerformanceInfo: () => ({
        memory: (performance as any).memory,
        timing: performance.timing,
        navigation: performance.navigation
      }),
      
      // API调用历史
      apiCallHistory: [],
      
      // 组件渲染追踪
      componentRenderCount: new Map(),
      
      // 工具方法
      utils: {
        formatBytes: this.formatBytes,
        formatDuration: this.formatDuration,
        copyToClipboard: this.copyToClipboard,
        exportLogs: this.exportLogs.bind(this),
        simulateError: this.simulateError,
        testApiEndpoint: this.testApiEndpoint,
      }
    }

    // 打印欢迎信息
    console.log(
      '%c🚀 AI Travel Planner DevTools Loaded',
      'color: #1890ff; font-size: 16px; font-weight: bold;'
    )
    console.log('Use __AI_TRAVEL_DEVTOOLS__ to access debugging tools')
    
    // 监听未捕获的错误
    this.setupErrorHandling()
    
    // 监听性能相关事件
    this.setupPerformanceMonitoring()
  }

  // 设置错误处理
  private setupErrorHandling() {
    window.addEventListener('error', (event) => {
      logger.error('Global Error', {
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        error: event.error
      })
    })

    window.addEventListener('unhandledrejection', (event) => {
      logger.error('Unhandled Promise Rejection', {
        reason: event.reason
      })
    })
  }

  // 设置性能监控
  private setupPerformanceMonitoring() {
    // 监听资源加载
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.entryType === 'resource') {
          const resourceEntry = entry as PerformanceResourceTiming
          if (resourceEntry.duration > 1000) { // 只记录超过1秒的资源
            logger.warn('Slow Resource Loading', {
              name: resourceEntry.name,
              duration: resourceEntry.duration,
              size: resourceEntry.transferSize
            })
          }
        }
      }
    })

    try {
      observer.observe({ entryTypes: ['resource', 'navigation'] })
    } catch (error) {
      console.warn('Performance Observer not supported')
    }
  }

  // 获取存储使用情况
  private getStorageUsage() {
    let localStorageSize = 0
    let sessionStorageSize = 0

    for (const key in localStorage) {
      if (localStorage.hasOwnProperty(key)) {
        localStorageSize += localStorage[key].length + key.length
      }
    }

    for (const key in sessionStorage) {
      if (sessionStorage.hasOwnProperty(key)) {
        sessionStorageSize += sessionStorage[key].length + key.length
      }
    }

    return {
      localStorage: this.formatBytes(localStorageSize),
      sessionStorage: this.formatBytes(sessionStorageSize)
    }
  }

  // 格式化字节数
  private formatBytes(bytes: number): string {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  // 格式化持续时间
  private formatDuration(ms: number): string {
    if (ms < 1000) return `${ms.toFixed(2)}ms`
    if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`
    return `${(ms / 60000).toFixed(2)}m`
  }

  // 复制到剪贴板
  private copyToClipboard(text: string): Promise<void> {
    return navigator.clipboard.writeText(text).then(() => {
      console.log('Copied to clipboard:', text)
    })
  }

  // 导出日志
  private exportLogs() {
    const logs = (window as any).__AI_TRAVEL_DEVTOOLS__.apiCallHistory
    const dataStr = JSON.stringify(logs, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `ai-travel-logs-${new Date().toISOString().split('T')[0]}.json`
    link.click()
    
    URL.revokeObjectURL(url)
  }

  // 模拟错误（用于测试错误处理）
  private simulateError() {
    throw new Error('Simulated error for testing')
  }

  // 测试API端点
  private async testApiEndpoint(url: string, method: string = 'GET') {
    const startTime = performance.now()
    
    try {
      const response = await fetch(url, { method })
      const duration = performance.now() - startTime
      
      const result = {
        url,
        method,
        status: response.status,
        statusText: response.statusText,
        duration: this.formatDuration(duration),
        headers: Object.fromEntries(response.headers.entries())
      }
      
      console.log('API Test Result:', result)
      return result
    } catch (error) {
      const duration = performance.now() - startTime
      const result = {
        url,
        method,
        error: (error as Error).message,
        duration: this.formatDuration(duration)
      }
      
      console.error('API Test Failed:', result)
      return result
    }
  }

  // API调用追踪器
  trackApiCall(config: any, response: any, error?: any) {
    if (!this.isEnabled) return

    const devtools = (window as any).__AI_TRAVEL_DEVTOOLS__
    if (devtools) {
      devtools.apiCallHistory.push({
        timestamp: new Date().toISOString(),
        url: config.url,
        method: config.method?.toUpperCase() || 'GET',
        status: response?.status,
        duration: response?.config?.metadata?.endTime - response?.config?.metadata?.startTime,
        error: error?.message
      })

      // 只保留最近100条记录
      if (devtools.apiCallHistory.length > 100) {
        devtools.apiCallHistory = devtools.apiCallHistory.slice(-100)
      }
    }
  }

  // 组件渲染追踪
  trackComponentRender(componentName: string) {
    if (!this.isEnabled) return

    const devtools = (window as any).__AI_TRAVEL_DEVTOOLS__
    if (devtools) {
      const count = devtools.componentRenderCount.get(componentName) || 0
      devtools.componentRenderCount.set(componentName, count + 1)
    }
  }
}

// 创建全局开发工具实例
export const devtools = new DevTools()

// React组件渲染追踪Hook
export const useRenderTracker = (componentName: string) => {
  if (process.env.NODE_ENV === 'development') {
    devtools.trackComponentRender(componentName)
  }
}

export default devtools 