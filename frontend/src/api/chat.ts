import axios from 'axios'
import config from '@/config'

const apiClient = axios.create({
  baseURL: config.api.baseUrl,
  timeout: config.api.timeout,
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// 聊天相关接口类型定义
export interface Message {
  id: string
  content: string
  type: 'user' | 'assistant' | 'system'
  timestamp: string
  metadata?: {
    sources?: string[]
    tools_used?: string[]
    confidence?: number
    [key: string]: any
  }
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt: string
  updatedAt: string
  metadata?: Record<string, any>
}

export interface ChatRequest {
  message: string
  conversationId?: string
  context?: Record<string, any>
  streamResponse?: boolean
}

export interface ChatResponse {
  message: Message
  conversationId: string
  suggestions?: string[]
}

export interface ConversationListResponse {
  conversations: Conversation[]
  total: number
  page: number
  pageSize: number
}

const chatAPI = {
  // 发送聊天消息
  sendMessage: (data: ChatRequest) => {
    return apiClient.post<ChatResponse>('/chat/message', data)
  },

  // 获取对话列表
  getConversations: (params?: { page?: number; pageSize?: number; search?: string }) => {
    return apiClient.get<ConversationListResponse>('/chat/conversations', { params })
  },

  // 获取特定对话
  getConversation: (conversationId: string) => {
    return apiClient.get<Conversation>(`/chat/conversations/${conversationId}`)
  },

  // 创建新对话
  createConversation: (data: { title?: string; message: string }) => {
    return apiClient.post<Conversation>('/chat/conversations', data)
  },

  // 更新对话标题
  updateConversationTitle: (conversationId: string, title: string) => {
    return apiClient.put(`/chat/conversations/${conversationId}/title`, { title })
  },

  // 删除对话
  deleteConversation: (conversationId: string) => {
    return apiClient.delete(`/chat/conversations/${conversationId}`)
  },

  // 清空对话历史
  clearConversationHistory: (conversationId: string) => {
    return apiClient.delete(`/chat/conversations/${conversationId}/messages`)
  },

  // 获取聊天建议
  getSuggestions: (context?: string) => {
    return apiClient.get<{ suggestions: string[] }>('/chat/suggestions', {
      params: { context }
    })
  },

  // 上传文件（用于聊天中的文件分享）
  uploadFile: (file: File, conversationId?: string) => {
    const formData = new FormData()
    formData.append('file', file)
    if (conversationId) {
      formData.append('conversationId', conversationId)
    }
    
    return apiClient.post('/chat/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
  },

  // 导出对话记录
  exportConversation: (conversationId: string, format: 'json' | 'txt' | 'pdf' = 'json') => {
    return apiClient.get(`/chat/conversations/${conversationId}/export`, {
      params: { format },
      responseType: 'blob',
    })
  },
}

export default chatAPI 