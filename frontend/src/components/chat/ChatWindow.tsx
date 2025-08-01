import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Input, Button, List, Avatar, Spin, message, Upload, Tooltip } from 'antd'
import {
  SendOutlined,
  RobotOutlined,
  UserOutlined,
  PaperClipOutlined,
  PictureOutlined,
  FileTextOutlined,
  LoadingOutlined
} from '@ant-design/icons'
import { motion, AnimatePresence } from 'framer-motion'
import io, { Socket } from 'socket.io-client'
import ReactMarkdown from 'react-markdown'
import dayjs from 'dayjs'

import './ChatWindow.css'

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
  }
}

interface ChatWindowProps {
  conversationId?: string
  onNewConversation?: (conversationId: string) => void
  className?: string
}

const ChatWindow: React.FC<ChatWindowProps> = ({
  conversationId,
  onNewConversation,
  className
}) => {
  // 状态管理
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [currentStreamingId, setCurrentStreamingId] = useState<string | null>(null)

  // 引用
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const socketRef = useRef<Socket | null>(null)
  const inputRef = useRef<any>(null)

  // WebSocket连接
  useEffect(() => {
    const socket = io('/api/v1/ws', {
      transports: ['websocket'],
      query: conversationId ? { conversation_id: conversationId } : {}
    })

    socketRef.current = socket

    socket.on('connect', () => {
      setIsConnected(true)
      console.log('WebSocket连接成功')
    })

    socket.on('disconnect', () => {
      setIsConnected(false)
      console.log('WebSocket连接断开')
    })

    socket.on('message', (data: any) => {
      handleReceivedMessage(data)
    })

    socket.on('message_stream', (data: any) => {
      handleStreamMessage(data)
    })

    socket.on('error', (error: any) => {
      console.error('WebSocket错误:', error)
      message.error('连接错误，请刷新页面重试')
    })

    return () => {
      socket.disconnect()
    }
  }, [conversationId])

  // 处理接收到的消息
  const handleReceivedMessage = useCallback((data: any) => {
    const newMessage: Message = {
      id: data.message_id || Date.now().toString(),
      content: data.content,
      type: data.type || 'text',
      sender: 'assistant',
      timestamp: data.timestamp || new Date().toISOString(),
      metadata: data.metadata
    }

    setMessages(prev => [...prev, newMessage])
    setIsLoading(false)
    setCurrentStreamingId(null)
  }, [])

  // 处理流式消息
  const handleStreamMessage = useCallback((data: any) => {
    if (data.type === 'start') {
      const streamMessage: Message = {
        id: data.message_id,
        content: '',
        type: 'text',
        sender: 'assistant',
        timestamp: new Date().toISOString(),
        metadata: { isStreaming: true }
      }
      setMessages(prev => [...prev, streamMessage])
      setCurrentStreamingId(data.message_id)
      setIsLoading(false)
    } else if (data.type === 'chunk') {
      setMessages(prev => prev.map(msg => 
        msg.id === data.message_id 
          ? { ...msg, content: msg.content + data.content }
          : msg
      ))
    } else if (data.type === 'end') {
      setMessages(prev => prev.map(msg => 
        msg.id === data.message_id 
          ? { ...msg, metadata: { ...msg.metadata, isStreaming: false } }
          : msg
      ))
      setCurrentStreamingId(null)
    }
  }, [])

  // 滚动到底部
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // 发送消息
  const sendMessage = async (content: string, type: 'text' | 'image' | 'file' = 'text', metadata?: any) => {
    if (!content.trim() && type === 'text') return
    if (!socketRef.current?.connected) {
      message.error('连接已断开，请刷新页面重试')
      return
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      type,
      sender: 'user',
      timestamp: new Date().toISOString(),
      metadata
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    // 通过WebSocket发送消息
    socketRef.current.emit('send_message', {
      content,
      type,
      metadata,
      conversation_id: conversationId
    })
  }

  // 处理键盘事件
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(inputValue)
    }
  }

  // 文件上传处理
  const handleFileUpload = async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/api/v1/upload', {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        const result = await response.json()
        
        if (file.type.startsWith('image/')) {
          sendMessage(result.url, 'image', {
            fileName: file.name,
            fileSize: file.size,
            imageUrl: result.url
          })
        } else {
          sendMessage(file.name, 'file', {
            fileName: file.name,
            fileSize: file.size,
            fileUrl: result.url
          })
        }
      } else {
        message.error('文件上传失败')
      }
    } catch (error) {
      console.error('文件上传错误:', error)
      message.error('文件上传失败')
    }
  }

  // 渲染消息内容
  const renderMessageContent = (msg: Message) => {
    switch (msg.type) {
      case 'image':
        return (
          <div className="message-image">
            <img 
              src={msg.metadata?.imageUrl || msg.content} 
              alt={msg.metadata?.fileName || 'Image'}
              style={{ maxWidth: '300px', maxHeight: '200px', borderRadius: '8px' }}
            />
            {msg.metadata?.fileName && (
              <div className="image-filename">{msg.metadata.fileName}</div>
            )}
          </div>
        )
      
      case 'file':
        return (
          <div className="message-file">
            <FileTextOutlined style={{ fontSize: '24px', color: '#1890ff' }} />
            <div className="file-info">
              <div className="file-name">{msg.metadata?.fileName || msg.content}</div>
              {msg.metadata?.fileSize && (
                <div className="file-size">
                  {(msg.metadata.fileSize / 1024).toFixed(1)} KB
                </div>
              )}
            </div>
          </div>
        )
      
      case 'system':
        return (
          <div className="system-message">
            {msg.content}
          </div>
        )
      
      default:
        return (
          <div className="message-text">
            {msg.sender === 'assistant' ? (
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            ) : (
              <span>{msg.content}</span>
            )}
            {msg.metadata?.isStreaming && (
              <LoadingOutlined style={{ marginLeft: '8px', color: '#1890ff' }} />
            )}
          </div>
        )
    }
  }

  // 渲染消息项
  const renderMessage = (msg: Message, index: number) => (
    <motion.div
      key={msg.id}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
      className={`message-item ${msg.sender}`}
    >
      <div className="message-avatar">
        <Avatar 
          icon={msg.sender === 'user' ? <UserOutlined /> : <RobotOutlined />}
          style={{ 
            backgroundColor: msg.sender === 'user' ? '#1890ff' : '#52c41a' 
          }}
        />
      </div>
      
      <div className="message-content">
        <div className="message-header">
          <span className="sender-name">
            {msg.sender === 'user' ? '我' : 'AI助手'}
          </span>
          <span className="message-time">
            {dayjs(msg.timestamp).format('HH:mm')}
          </span>
        </div>
        
        <div className="message-body">
          {renderMessageContent(msg)}
        </div>
      </div>
    </motion.div>
  )

  return (
    <div className={`chat-window ${className || ''}`}>
      {/* 聊天头部 */}
      <div className="chat-header">
        <div className="chat-title">
          <RobotOutlined style={{ marginRight: '8px' }} />
          AI旅行助手
        </div>
        <div className="connection-status">
          <div className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`} />
          {isConnected ? '已连接' : '连接中...'}
        </div>
      </div>

      {/* 消息列表 */}
      <div className="messages-container">
        <AnimatePresence>
          {messages.length === 0 ? (
            <motion.div 
              className="welcome-message"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
            >
              <RobotOutlined style={{ fontSize: '48px', color: '#1890ff', marginBottom: '16px' }} />
              <h3>欢迎使用AI旅行助手</h3>
              <p>我可以帮您规划旅行路线、推荐景点、查询天气等。有什么可以帮您的吗？</p>
            </motion.div>
          ) : (
            messages.map((msg, index) => renderMessage(msg, index))
          )}
        </AnimatePresence>

        {/* 加载指示器 */}
        {isLoading && (
          <motion.div 
            className="loading-message"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#52c41a' }} />
            <div className="loading-content">
              <Spin indicator={<LoadingOutlined style={{ fontSize: 16 }} />} />
              <span style={{ marginLeft: '8px' }}>AI正在思考...</span>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* 输入区域 */}
      <div className="input-container">
        <div className="input-wrapper">
          {/* 文件上传按钮 */}
          <div className="upload-buttons">
            <Upload
              beforeUpload={handleFileUpload}
              showUploadList={false}
              accept="image/*"
            >
              <Tooltip title="上传图片">
                <Button 
                  type="text" 
                  icon={<PictureOutlined />}
                  className="upload-btn"
                />
              </Tooltip>
            </Upload>

            <Upload
              beforeUpload={handleFileUpload}
              showUploadList={false}
              accept=".pdf,.doc,.docx,.txt"
            >
              <Tooltip title="上传文件">
                <Button 
                  type="text" 
                  icon={<PaperClipOutlined />}
                  className="upload-btn"
                />
              </Tooltip>
            </Upload>
          </div>

          {/* 文本输入 */}
          <Input.TextArea
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="输入消息... (Shift+Enter换行)"
            autoSize={{ minRows: 1, maxRows: 4 }}
            disabled={!isConnected || isLoading}
            className="message-input"
          />

          {/* 发送按钮 */}
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={() => sendMessage(inputValue)}
            disabled={!isConnected || isLoading || !inputValue.trim()}
            className="send-button"
          />
        </div>
      </div>
    </div>
  )
}

export default ChatWindow 