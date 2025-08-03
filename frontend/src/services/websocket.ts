// WebSocket 服务类
export class WebSocketService {
  private ws: WebSocket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectInterval = 1000
  private listeners: Map<string, Function[]> = new Map()
  private isConnecting = false

  constructor(private url: string) {
    this.connect()
  }

  // 连接WebSocket
  private connect() {
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.CONNECTING)) {
      return
    }

    this.isConnecting = true
    
    try {
      this.ws = new WebSocket(this.url)
      
      this.ws.onopen = () => {
        console.log('WebSocket connected')
        this.isConnecting = false
        this.reconnectAttempts = 0
        this.emit('connected', null)
      }

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          this.emit('message', data)
          
          // 根据消息类型触发特定事件
          if (data.type) {
            this.emit(data.type, data)
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      this.ws.onclose = () => {
        console.log('WebSocket disconnected')
        this.isConnecting = false
        this.emit('disconnected', null)
        this.handleReconnect()
      }

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        this.isConnecting = false
        this.emit('error', error)
      }
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      this.isConnecting = false
      this.handleReconnect()
    }
  }

  // 处理重连
  private handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      const delay = this.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1)
      
      console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`)
      
      setTimeout(() => {
        this.connect()
      }, delay)
    } else {
      console.error('Max reconnection attempts reached')
      this.emit('maxReconnectAttemptsReached', null)
    }
  }

  // 发送消息
  send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
      return true
    } else {
      console.warn('WebSocket is not connected')
      return false
    }
  }

  // 添加事件监听器
  on(event: string, callback: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, [])
    }
    this.listeners.get(event)!.push(callback)
  }

  // 移除事件监听器
  off(event: string, callback?: Function) {
    const callbacks = this.listeners.get(event)
    if (callbacks) {
      if (callback) {
        const index = callbacks.indexOf(callback)
        if (index > -1) {
          callbacks.splice(index, 1)
        }
      } else {
        this.listeners.set(event, [])
      }
    }
  }

  // 触发事件
  private emit(event: string, data: any) {
    const callbacks = this.listeners.get(event)
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(data)
        } catch (error) {
          console.error('Error in WebSocket event callback:', error)
        }
      })
    }
  }

  // 获取连接状态
  getReadyState() {
    return this.ws ? this.ws.readyState : WebSocket.CLOSED
  }

  // 是否已连接
  isConnected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN
  }

  // 关闭连接
  close() {
    this.maxReconnectAttempts = 0 // 阻止自动重连
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }
}

import config from '@/config'

// 创建全局WebSocket服务实例
let wsService: WebSocketService | null = null

export const getWebSocketService = () => {
  if (!wsService) {
    wsService = new WebSocketService(config.websocket.url)
  }
  return wsService
}

export default WebSocketService 