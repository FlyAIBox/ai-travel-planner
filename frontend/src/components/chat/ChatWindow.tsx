import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Input, Button, List, Avatar, Spin, message, Upload, Tooltip } from 'antd'
import {
  SendOutlined,
  RobotOutlined,
  UserOutlined,
  PaperClipOutlined,
  PictureOutlined,
  FileTextOutlined,
  LoadingOutlined,
  WifiOutlined,
  DisconnectOutlined
} from '@ant-design/icons'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import dayjs from 'dayjs'

import './ChatWindow.css'
import { 
  WebSocketService, 
  ChatMessage, 
  ConnectionStatus,
  createWebSocketService 
} from './WebSocketService'

// 类型定义
interface Message {
  id: string
  content: string
  type: 'text' | 'image' | 'file' | 'system'
  sender: 'user' | 'assistant'
  timestamp: string
  metadata?: {
    fileName?: string
    fileSize?: number
    imageUrl?: string
    isStreaming?: boolean
    conversationId?: string
  }
}

interface ChatWindowProps {
  conversationId?: string
  onNewConversation?: (conversationId: string) => void
  className?: string
  userId?: string
  apiBaseUrl?: string
}

import config from '@/config'

const ChatWindow: React.FC<ChatWindowProps> = ({
  conversationId,
  onNewConversation,
  className,
  userId = 'user_' + Date.now(),
  apiBaseUrl = config.chat.wsUrl
}) => {
  // 状态管理
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected')
  const [currentStreamingId, setCurrentStreamingId] = useState<string | null>(null)
  const [isTyping, setIsTyping] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<any>(null)
  const webSocketService = useRef<WebSocketService | null>(null)
  const typingTimeoutRef = useRef<number | null>(null)

  // 滚动到底部
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  // 初始化WebSocket连接
  useEffect(() => {
    // 调试信息
    console.log('ChatWindow 配置调试:')
    console.log('- config.chat.wsUrl:', config.chat.wsUrl)
    console.log('- apiBaseUrl prop:', apiBaseUrl)

    // 构建正确的WebSocket URL
    let wsUrl = apiBaseUrl

    // 如果apiBaseUrl是HTTP URL，转换为WebSocket URL
    if (apiBaseUrl.startsWith('http://')) {
      wsUrl = apiBaseUrl.replace('http://', 'ws://')
      console.log('- 转换 HTTP 为 WS:', wsUrl)
    } else if (apiBaseUrl.startsWith('https://')) {
      wsUrl = apiBaseUrl.replace('https://', 'wss://')
      console.log('- 转换 HTTPS 为 WSS:', wsUrl)
    } else {
      console.log('- 已经是 WebSocket URL:', wsUrl)
    }

    // WebSocket URL 已经是完整的，不需要添加额外路径
    // 确保 URL 格式正确
    if (!wsUrl.endsWith('/')) {
      wsUrl += '/'
      console.log('- 添加尾部斜杠:', wsUrl)
    } else {
      console.log('- URL 格式正确:', wsUrl)
    }

    console.log('- 最终 WebSocket URL:', wsUrl)
    
    webSocketService.current = createWebSocketService({
      url: wsUrl,
      userId,
      sessionId: conversationId || 'session_' + Date.now(),
      reconnectInterval: 3000,
      maxReconnectAttempts: 5,
      heartbeatInterval: 30000
    })

    const ws = webSocketService.current

    // 设置事件监听器
    ws.on('onConnectionChange', (status: ConnectionStatus) => {
      setConnectionStatus(status)
      
      if (status === 'connected') {
        message.success('连接已建立')
      } else if (status === 'disconnected') {
        message.warning('连接已断开')
      } else if (status === 'error') {
        message.error('连接错误')
      } else if (status === 'reconnecting') {
        message.info('正在重连...')
      }
    })

    ws.on('onMessage', (chatMessage: ChatMessage) => {
      const newMessage: Message = {
        id: chatMessage.id,
        content: chatMessage.content,
        type: 'text',
        sender: chatMessage.type === 'user' ? 'user' : 'assistant',
        timestamp: chatMessage.timestamp,
        metadata: chatMessage.metadata
      }

      setMessages(prev => [...prev, newMessage])
      setIsLoading(false)
      setCurrentStreamingId(null)
      setStreamingContent('')
    })

    ws.on('onTyping', (typing: boolean) => {
      setIsTyping(typing)
    })

    ws.on('onStreamStart', () => {
      const streamId = 'stream_' + Date.now()
      setCurrentStreamingId(streamId)
      setStreamingContent('')
      setIsLoading(false)
    })

    ws.on('onStreamChunk', (chunk: string) => {
      setStreamingContent(prev => prev + chunk)
    })

    ws.on('onStreamEnd', () => {
      setCurrentStreamingId(null)
      setStreamingContent('')
    })

    ws.on('onError', (error: Error) => {
      console.error('WebSocket错误:', error)
      message.error('连接错误: ' + error.message)
      setIsLoading(false)
    })

    // 建立连接
    ws.connect().catch(err => {
      console.error('WebSocket连接失败:', err)
      message.error('无法连接到服务器')
    })

    // 清理函数
    return () => {
      ws.disconnect()
    }
  }, [conversationId, userId, apiBaseUrl])

  // 滚动效果
  useEffect(() => {
    scrollToBottom()
  }, [messages, streamingContent, scrollToBottom])

  // 发送消息
  const handleSendMessage = useCallback(async () => {
    if (!inputValue.trim() || !webSocketService.current?.isConnected()) {
      if (!webSocketService.current?.isConnected()) {
        message.warning('连接未建立，请稍后重试')
      }
      return
    }

    const messageContent = inputValue.trim()
    setInputValue('')
    setIsLoading(true)

    // 添加用户消息到界面
    const userMessage: Message = {
      id: 'user_' + Date.now(),
      content: messageContent,
      type: 'text',
      sender: 'user',
      timestamp: new Date().toISOString()
    }
    
    setMessages(prev => [...prev, userMessage])

    try {
      // 通过WebSocket发送消息
      webSocketService.current.sendMessage(messageContent, 'user')
    } catch (error) {
      console.error('发送消息失败:', error)
      message.error('发送消息失败')
      setIsLoading(false)
    }
  }, [inputValue])

  // 处理输入变化
  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value)
    
    // 发送打字状态
    if (webSocketService.current?.isConnected()) {
      webSocketService.current.sendTypingStatus(true)
      
      // 清除之前的定时器
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current)
      }
      
      // 设置新的定时器，3秒后停止打字状态
      typingTimeoutRef.current = window.setTimeout(() => {
        webSocketService.current?.sendTypingStatus(false)
      }, 3000)
    }
  }, [])

  // 处理回车键
  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }, [handleSendMessage])

  // 重连功能
  const handleReconnect = useCallback(() => {
    if (webSocketService.current) {
      webSocketService.current.connect().catch(err => {
        console.error('重连失败:', err)
        message.error('重连失败')
      })
    }
  }, [])

  // 获取连接状态图标
  const getConnectionIcon = () => {
    switch (connectionStatus) {
      case 'connected':
        return <WifiOutlined style={{ color: '#52c41a' }} />
      case 'connecting':
      case 'reconnecting':
        return <LoadingOutlined style={{ color: '#1890ff' }} />
      default:
        return <DisconnectOutlined style={{ color: '#ff4d4f' }} />
    }
  }

  // 获取连接状态文本
  const getConnectionText = () => {
    switch (connectionStatus) {
      case 'connected':
        return '已连接'
      case 'connecting':
        return '连接中...'
      case 'reconnecting':
        return '重连中...'
      case 'disconnected':
        return '已断开'
      case 'error':
        return '连接错误'
      default:
        return '未知状态'
    }
  }

  // 渲染消息项
  const renderMessage = (msg: Message) => {
    const isUser = msg.sender === 'user'
    
    return (
      <motion.div
        key={msg.id}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className={`message-item ${isUser ? 'user-message' : 'assistant-message'}`}
      >
        <div className="message-content">
          <Avatar 
            icon={isUser ? <UserOutlined /> : <RobotOutlined />}
            className={`message-avatar ${isUser ? 'user-avatar' : 'assistant-avatar'}`}
          />
          <div className="message-bubble">
            <div className="message-text">
              {msg.type === 'text' ? (
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              ) : (
                <span>{msg.content}</span>
              )}
            </div>
            <div className="message-time">
              {dayjs(msg.timestamp).format('HH:mm')}
            </div>
          </div>
        </div>
      </motion.div>
    )
  }

  // 渲染流式消息
  const renderStreamingMessage = () => {
    if (!currentStreamingId || !streamingContent) return null

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="message-item assistant-message streaming-message"
      >
        <div className="message-content">
          <Avatar 
            icon={<RobotOutlined />}
            className="message-avatar assistant-avatar"
          />
          <div className="message-bubble">
            <div className="message-text">
              <ReactMarkdown>{streamingContent}</ReactMarkdown>
              <span className="streaming-cursor">|</span>
            </div>
          </div>
        </div>
      </motion.div>
    )
  }

  return (
    <div className={`chat-window ${className || ''}`}>
      {/* 头部状态栏 */}
      <div className="chat-header">
        <div className="chat-title">AI 旅行助手</div>
        <div className="connection-status">
          <Tooltip title={getConnectionText()}>
            <span className="connection-indicator">
              {getConnectionIcon()}
              <span className="connection-text">{getConnectionText()}</span>
            </span>
          </Tooltip>
          {connectionStatus !== 'connected' && (
            <Button size="small" onClick={handleReconnect}>
              重连
            </Button>
          )}
        </div>
      </div>

      {/* 消息列表 */}
      <div className="chat-messages">
        <AnimatePresence>
          {messages.map(renderMessage)}
          {renderStreamingMessage()}
          {isLoading && !currentStreamingId && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="message-item assistant-message"
            >
              <div className="message-content">
                <Avatar 
                  icon={<RobotOutlined />}
                  className="message-avatar assistant-avatar"
                />
                <div className="message-bubble">
                  <div className="message-text">
                    <Spin indicator={<LoadingOutlined />} />
                    <span style={{ marginLeft: 8 }}>AI 正在思考...</span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
          {isTyping && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="message-item assistant-message"
            >
              <div className="message-content">
                <Avatar 
                  icon={<RobotOutlined />}
                  className="message-avatar assistant-avatar"
                />
                <div className="message-bubble">
                  <div className="message-text">
                    <span className="typing-indicator">正在输入...</span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>

      {/* 输入区域 */}
      <div className="chat-input">
        <div className="input-container">
          <Input
            ref={inputRef}
            value={inputValue}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
            placeholder="输入您的旅行需求..."
            disabled={connectionStatus !== 'connected'}
            suffix={
              <div className="input-actions">
                <Tooltip title="发送文件">
                  <Upload
                    showUploadList={false}
                    beforeUpload={() => false}
                  >
                    <Button
                      type="text"
                      icon={<PaperClipOutlined />}
                      size="small"
                    />
                  </Upload>
                </Tooltip>
                <Button
                  type="primary"
                  icon={<SendOutlined />}
                  onClick={handleSendMessage}
                  disabled={!inputValue.trim() || connectionStatus !== 'connected'}
                  loading={isLoading && !currentStreamingId}
                />
              </div>
            }
          />
        </div>
      </div>
    </div>
  )
}

export default ChatWindow 