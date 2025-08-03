import axios from 'axios'
import config from '@/config/simple'

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

// 旅行规划相关类型定义
export interface Activity {
  id: string
  name: string
  description: string
  location: {
    name: string
    latitude: number
    longitude: number
    address: string
  }
  startTime: string
  endTime: string
  duration: number // 分钟
  cost: {
    amount: number
    currency: string
  }
  category: string
  tags: string[]
  rating?: number
  images?: string[]
  bookingInfo?: {
    url?: string
    phone?: string
    requirements?: string[]
  }
}

export interface TravelPlan {
  id: string
  title: string
  description: string
  destination: {
    name: string
    country: string
    region: string
    coordinates: {
      latitude: number
      longitude: number
    }
  }
  startDate: string
  endDate: string
  duration: number // 天数
  budget: {
    total: number
    currency: string
    breakdown: {
      accommodation: number
      transportation: number
      food: number
      activities: number
      others: number
    }
  }
  activities: Activity[]
  status: 'draft' | 'confirmed' | 'completed' | 'cancelled'
  preferences: {
    travelStyle: 'budget' | 'comfort' | 'luxury'
    interests: string[]
    groupSize: number
    mobility: 'low' | 'medium' | 'high'
  }
  createdAt: string
  updatedAt: string
  sharedWith?: string[]
}

export interface PlanningRequest {
  destination: string
  startDate: string
  endDate: string
  budget?: number
  currency?: string
  preferences?: {
    travelStyle?: 'budget' | 'comfort' | 'luxury'
    interests?: string[]
    groupSize?: number
    mobility?: 'low' | 'medium' | 'high'
  }
  requirements?: string
}

export interface FlightSearchRequest {
  from: string
  to: string
  departDate: string
  returnDate?: string
  passengers: number
  class?: 'economy' | 'business' | 'first'
}

export interface HotelSearchRequest {
  location: string
  checkIn: string
  checkOut: string
  guests: number
  rooms: number
  category?: 'budget' | 'standard' | 'luxury'
}

const travelAPI = {
  // 旅行计划相关
  getPlans: (params?: { 
    page?: number
    pageSize?: number
    status?: string
    search?: string 
  }) => {
    return apiClient.get<{
      plans: TravelPlan[]
      total: number
      page: number
      pageSize: number
    }>('/travel/plans', { params })
  },

  getPlan: (planId: string) => {
    return apiClient.get<TravelPlan>(`/travel/plans/${planId}`)
  },

  createPlan: (data: PlanningRequest) => {
    return apiClient.post<TravelPlan>('/travel/plans', data)
  },

  updatePlan: (planId: string, data: Partial<TravelPlan>) => {
    return apiClient.put<TravelPlan>(`/travel/plans/${planId}`, data)
  },

  deletePlan: (planId: string) => {
    return apiClient.delete(`/travel/plans/${planId}`)
  },

  // 活动相关
  addActivity: (planId: string, activity: Omit<Activity, 'id'>) => {
    return apiClient.post<Activity>(`/travel/plans/${planId}/activities`, activity)
  },

  updateActivity: (planId: string, activityId: string, data: Partial<Activity>) => {
    return apiClient.put<Activity>(`/travel/plans/${planId}/activities/${activityId}`, data)
  },

  deleteActivity: (planId: string, activityId: string) => {
    return apiClient.delete(`/travel/plans/${planId}/activities/${activityId}`)
  },

  // 搜索相关
  searchFlights: (data: FlightSearchRequest) => {
    return apiClient.post('/travel/search/flights', data)
  },

  searchHotels: (data: HotelSearchRequest) => {
    return apiClient.post('/travel/search/hotels', data)
  },

  searchDestinations: (query: string) => {
    return apiClient.get('/travel/search/destinations', {
      params: { q: query }
    })
  },

  getDestinationInfo: (destination: string) => {
    return apiClient.get(`/travel/destinations/${encodeURIComponent(destination)}`)
  },

  // 智能推荐
  getRecommendations: (planId: string, type: 'activities' | 'restaurants' | 'hotels') => {
    return apiClient.get(`/travel/plans/${planId}/recommendations/${type}`)
  },

  // 天气信息
  getWeather: (location: string, date?: string) => {
    return apiClient.get('/travel/weather', {
      params: { location, date }
    })
  },

  // 分享计划
  sharePlan: (planId: string, data: { emails: string[]; message?: string }) => {
    return apiClient.post(`/travel/plans/${planId}/share`, data)
  },

  // 导出计划
  exportPlan: (planId: string, format: 'pdf' | 'json' | 'ical' = 'pdf') => {
    return apiClient.get(`/travel/plans/${planId}/export`, {
      params: { format },
      responseType: 'blob',
    })
  },

  // 获取用户偏好
  getUserPreferences: () => {
    return apiClient.get('/travel/preferences')
  },

  // 更新用户偏好
  updateUserPreferences: (preferences: Record<string, any>) => {
    return apiClient.put('/travel/preferences', preferences)
  },
}

export default travelAPI 