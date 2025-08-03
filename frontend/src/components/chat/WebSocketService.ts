/**
 * WebSocket服务
 * 用于实现前后端实时通信
 */

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'system' | 'error';
  content: string;
  timestamp: string;
  metadata?: {
    conversationId?: string;
    userId?: string;
    sessionId?: string;
    [key: string]: any;
  };
}

export interface WebSocketMessage {
  type: 'message' | 'typing' | 'heartbeat' | 'error' | 'stream_start' | 'stream_chunk' | 'stream_end';
  data: any;
  timestamp: string;
  messageId?: string;
}

export interface ConnectionOptions {
  url: string;
  protocols?: string[];
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  userId?: string;
  sessionId?: string;
}

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error' | 'reconnecting';

export interface WebSocketServiceEvents {
  onMessage: (message: ChatMessage) => void;
  onConnectionChange: (status: ConnectionStatus) => void;
  onError: (error: Error) => void;
  onTyping: (isTyping: boolean) => void;
  onStreamStart: () => void;
  onStreamChunk: (chunk: string) => void;
  onStreamEnd: () => void;
}

export class WebSocketService {
  private ws: WebSocket | null = null;
  private connectionStatus: ConnectionStatus = 'disconnected';
  private reconnectAttempts = 0;
  private reconnectTimer: number | null = null;
  private heartbeatTimer: number | null = null;
  private options: ConnectionOptions;
  private events: Partial<WebSocketServiceEvents> = {};
  private messageQueue: WebSocketMessage[] = [];
  private currentStreamMessage = '';

  constructor(options: ConnectionOptions) {
    this.options = {
      reconnectInterval: 3000,
      maxReconnectAttempts: 5,
      heartbeatInterval: 30000,
      ...options
    };
  }

  /**
   * 设置事件监听器
   */
  on<K extends keyof WebSocketServiceEvents>(event: K, callback: WebSocketServiceEvents[K]): void {
    this.events[event] = callback;
  }

  /**
   * 移除事件监听器
   */
  off<K extends keyof WebSocketServiceEvents>(event: K): void {
    delete this.events[event];
  }

  /**
   * 连接WebSocket
   */
  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.setConnectionStatus('connecting');

        // 设置连接超时（10秒）
        const connectionTimeout = setTimeout(() => {
          if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
            this.ws.close();
            this.setConnectionStatus('error');
            reject(new Error('WebSocket连接超时'));
          }
        }, 10000);

        // 构建WebSocket URL
        // 注意：后端期望路径参数格式 /ws/{user_id}，而不是查询参数
        let wsUrl = this.options.url;

        // 如果 URL 已经包含了 user_id 路径参数，直接使用
        // 否则，可以添加查询参数作为备选（但后端应该支持路径参数）
        if (this.options.sessionId && !wsUrl.includes('conversation_id')) {
          const url = new URL(wsUrl);
          url.searchParams.set('conversation_id', this.options.sessionId);
          wsUrl = url.toString();
        }

        console.log('WebSocketService 连接 URL:', wsUrl);
        console.log('等待连接建立...');
        this.ws = new WebSocket(wsUrl, this.options.protocols);

        this.ws.onopen = () => {
          clearTimeout(connectionTimeout);
          console.log('✅ WebSocket连接已成功建立');
          this.setConnectionStatus('connected');
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          this.flushMessageQueue();
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event);
        };

        this.ws.onclose = (event) => {
          clearTimeout(connectionTimeout);
          console.log('WebSocket连接已关闭', event.code, event.reason);
          this.setConnectionStatus('disconnected');
          this.stopHeartbeat();

          // 如果不是手动关闭，尝试重连
          if (event.code !== 1000) {
            this.attemptReconnect();
          }
        };

        this.ws.onerror = (error) => {
          clearTimeout(connectionTimeout);
          console.error('❌ WebSocket连接错误:', error);
          this.setConnectionStatus('error');
          this.events.onError?.(new Error('WebSocket连接错误'));
          reject(error);
        };

      } catch (error) {
        console.error('创建WebSocket连接失败:', error);
        this.setConnectionStatus('error');
        this.events.onError?.(error as Error);
        reject(error);
      }
    });
  }

  /**
   * 断开连接
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, '手动关闭');
      this.ws = null;
    }
    
    this.stopHeartbeat();
    this.clearReconnectTimer();
    this.setConnectionStatus('disconnected');
  }

  /**
   * 发送消息
   */
  sendMessage(content: string, type: string = 'user'): void {
    const message: WebSocketMessage = {
      type: 'message',
      data: {
        id: this.generateMessageId(),
        type,
        content,
        timestamp: new Date().toISOString(),
        metadata: {
          userId: this.options.userId,
          sessionId: this.options.sessionId
        }
      },
      timestamp: new Date().toISOString(),
      messageId: this.generateMessageId()
    };

    this.send(message);
  }

  /**
   * 发送打字状态
   */
  sendTypingStatus(isTyping: boolean): void {
    const message: WebSocketMessage = {
      type: 'typing',
      data: { isTyping },
      timestamp: new Date().toISOString()
    };

    this.send(message);
  }

  /**
   * 获取连接状态
   */
  getConnectionStatus(): ConnectionStatus {
    return this.connectionStatus;
  }

  /**
   * 检查连接是否活跃
   */
  isConnected(): boolean {
    return this.connectionStatus === 'connected' && this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * 发送WebSocket消息
   */
  private send(message: WebSocketMessage): void {
    if (this.isConnected()) {
      try {
        this.ws!.send(JSON.stringify(message));
      } catch (error) {
        console.error('发送消息失败:', error);
        this.events.onError?.(error as Error);
      }
    } else {
      // 如果未连接，加入队列
      this.messageQueue.push(message);
      console.log('消息已加入队列，等待连接');
    }
  }

  /**
   * 处理接收到的消息
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const wsMessage: WebSocketMessage = JSON.parse(event.data);

      switch (wsMessage.type) {
        case 'message':
          const chatMessage: ChatMessage = wsMessage.data;
          this.events.onMessage?.(chatMessage);
          break;

        case 'typing':
          this.events.onTyping?.(wsMessage.data.isTyping);
          break;

        case 'stream_start':
          this.currentStreamMessage = '';
          this.events.onStreamStart?.();
          break;

        case 'stream_chunk':
          this.currentStreamMessage += wsMessage.data.chunk;
          this.events.onStreamChunk?.(wsMessage.data.chunk);
          break;

        case 'stream_end':
          // 创建完整的流式消息
          const streamMessage: ChatMessage = {
            id: this.generateMessageId(),
            type: 'assistant',
            content: this.currentStreamMessage,
            timestamp: new Date().toISOString(),
            metadata: wsMessage.data.metadata
          };
          this.events.onMessage?.(streamMessage);
          this.events.onStreamEnd?.();
          this.currentStreamMessage = '';
          break;

        case 'error':
          console.error('服务器错误:', wsMessage.data);
          this.events.onError?.(new Error(wsMessage.data.message || '服务器错误'));
          break;

        case 'heartbeat':
          // 心跳响应，无需处理
          break;

        default:
          console.warn('未知消息类型:', wsMessage.type);
      }
    } catch (error) {
      console.error('解析消息失败:', error);
      this.events.onError?.(error as Error);
    }
  }

  /**
   * 设置连接状态
   */
  private setConnectionStatus(status: ConnectionStatus): void {
    if (this.connectionStatus !== status) {
      this.connectionStatus = status;
      this.events.onConnectionChange?.(status);
    }
  }

  /**
   * 尝试重连
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.options.maxReconnectAttempts!) {
      console.error('达到最大重连次数，停止重连');
      this.setConnectionStatus('error');
      return;
    }

    this.reconnectAttempts++;
    this.setConnectionStatus('reconnecting');

    console.log(`尝试重连 (${this.reconnectAttempts}/${this.options.maxReconnectAttempts})`);

    this.reconnectTimer = window.setTimeout(() => {
      this.connect().catch(() => {
        // 重连失败，会触发下一次重连
      });
    }, this.options.reconnectInterval!);
  }

  /**
   * 清除重连定时器
   */
  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  /**
   * 开始心跳
   */
  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatTimer = window.setInterval(() => {
      if (this.isConnected()) {
        const heartbeatMessage: WebSocketMessage = {
          type: 'heartbeat',
          data: { timestamp: Date.now() },
          timestamp: new Date().toISOString()
        };
        this.send(heartbeatMessage);
      }
    }, this.options.heartbeatInterval!);
  }

  /**
   * 停止心跳
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * 刷新消息队列
   */
  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message) {
        this.send(message);
      }
    }
  }

  /**
   * 生成消息ID
   */
  private generateMessageId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
  }
}

/**
 * 创建WebSocket服务实例
 */
export function createWebSocketService(options: ConnectionOptions): WebSocketService {
  return new WebSocketService(options);
}

/**
 * 默认WebSocket服务实例
 */
let defaultWebSocketService: WebSocketService | null = null;

export function getDefaultWebSocketService(): WebSocketService | null {
  return defaultWebSocketService;
}

export function initializeDefaultWebSocketService(options: ConnectionOptions): WebSocketService {
  defaultWebSocketService = new WebSocketService(options);
  return defaultWebSocketService;
} 