import axios from 'axios'
import config from '@/config/simple'

// 创建axios实例
const apiClient = axios.create({
  baseURL: config.api.baseUrl,
  timeout: config.api.timeout,
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器 - 添加认证token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器 - 处理错误
apiClient.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    if (error.response?.status === 401) {
      // 清除token并跳转到登录页
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// 认证相关接口
export interface LoginRequest {
  email: string
  password: string
}

export interface RegisterRequest {
  username: string
  email: string
  password: string
}

export interface AuthResponse {
  token: string
  user: {
    id: string
    username: string
    email: string
    avatar?: string
    nickname?: string
    preferences?: Record<string, any>
  }
}

export interface UserResponse {
  id: string
  username: string
  email: string
  avatar?: string
  nickname?: string
  preferences?: Record<string, any>
  createdAt: string
  updatedAt: string
}

const authAPI = {
  // 用户登录
  login: (data: LoginRequest) => {
    return apiClient.post<AuthResponse>('/auth/login', data)
  },

  // 用户注册
  register: (data: RegisterRequest) => {
    return apiClient.post<AuthResponse>('/auth/register', data)
  },

  // 获取当前用户信息
  getCurrentUser: () => {
    return apiClient.get<UserResponse>('/auth/me')
  },

  // 刷新token
  refreshToken: () => {
    return apiClient.post<{ token: string }>('/auth/refresh')
  },

  // 用户退出
  logout: () => {
    return apiClient.post('/auth/logout')
  },

  // 修改密码
  changePassword: (data: { oldPassword: string; newPassword: string }) => {
    return apiClient.put('/auth/password', data)
  },

  // 重置密码（发送邮件）
  resetPassword: (email: string) => {
    return apiClient.post('/auth/reset-password', { email })
  },

  // 验证邮箱
  verifyEmail: (token: string) => {
    return apiClient.post('/auth/verify-email', { token })
  },
}

export default authAPI 