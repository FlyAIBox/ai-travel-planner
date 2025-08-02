// 通用响应类型
export interface ApiResponse<T = any> {
  success: boolean
  data: T
  message?: string
  code?: number
  timestamp?: string
}

// 分页相关类型
export interface PaginationParams {
  page: number
  pageSize: number
  total?: number
}

export interface PaginatedResponse<T> {
  items: T[]
  pagination: PaginationParams
}

// 错误类型
export interface ApiError {
  code: string
  message: string
  details?: any
  stack?: string
}

// 用户相关类型
export interface User {
  id: string
  username: string
  email: string
  avatar?: string
  nickname?: string
  bio?: string
  location?: string
  preferences: UserPreferences
  createdAt: string
  updatedAt: string
}

export interface UserPreferences {
  theme: 'light' | 'dark'
  language: 'zh-CN' | 'en-US'
  currency: 'CNY' | 'USD' | 'EUR'
  travelStyle: 'budget' | 'comfort' | 'luxury'
  interests: string[]
  notifications: {
    email: boolean
    push: boolean
    sms: boolean
  }
}

// 地理位置类型
export interface Location {
  name: string
  latitude: number
  longitude: number
  address?: string
  country?: string
  region?: string
  city?: string
}

// 文件上传类型
export interface UploadFile {
  id: string
  name: string
  url: string
  size: number
  type: string
  uploadedAt: string
}

// 表单状态类型
export interface FormState {
  loading: boolean
  error: string | null
  success: boolean
}

// 组件Props基础类型
export interface BaseComponentProps {
  className?: string
  style?: React.CSSProperties
  children?: React.ReactNode
}

// 路由相关类型
export interface RouteParams {
  [key: string]: string | undefined
}

// WebSocket消息类型
export interface WebSocketMessage {
  type: string
  data: any
  timestamp: string
  id?: string
}

// 搜索相关类型
export interface SearchFilters {
  keyword?: string
  category?: string
  dateRange?: {
    start: string
    end: string
  }
  priceRange?: {
    min: number
    max: number
  }
  location?: string
  tags?: string[]
}

// 导出相关类型
export type ExportFormat = 'json' | 'csv' | 'pdf' | 'excel'

// 主题相关类型
export type ThemeMode = 'light' | 'dark' | 'auto'

// 语言相关类型
export type Language = 'zh-CN' | 'en-US'

// 状态类型
export type LoadingState = 'idle' | 'loading' | 'success' | 'error'

// 权限相关类型
export interface Permission {
  id: string
  name: string
  description: string
  resource: string
  action: string
}

export interface Role {
  id: string
  name: string
  description: string
  permissions: Permission[]
}

// 统计数据类型
export interface Statistics {
  label: string
  value: number
  unit?: string
  change?: {
    value: number
    type: 'increase' | 'decrease'
    period: string
  }
}

// 通知类型
export interface Notification {
  id: string
  title: string
  message: string
  type: 'info' | 'success' | 'warning' | 'error'
  read: boolean
  createdAt: string
  actions?: {
    label: string
    action: () => void
  }[]
}

// 环境变量类型
export interface Environment {
  NODE_ENV: 'development' | 'production' | 'test'
  API_BASE_URL: string
  WS_URL: string
  VERSION: string
}

// 工具函数类型
export type Debounced<T extends (...args: any[]) => any> = T & {
  cancel: () => void
}

// 组件状态类型
export interface ComponentState {
  mounted: boolean
  visible: boolean
  disabled: boolean
  loading: boolean
} 