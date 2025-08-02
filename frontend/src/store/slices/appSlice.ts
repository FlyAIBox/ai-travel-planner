import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'

interface AppState {
  isInitialized: boolean
  loading: boolean
  theme: 'light' | 'dark'
  language: 'zh-CN' | 'en-US'
  sidebarCollapsed: boolean
  error: string | null
}

const initialState: AppState = {
  isInitialized: false,
  loading: false,
  theme: 'light',
  language: 'zh-CN',
  sidebarCollapsed: false,
  error: null,
}

// 异步操作：初始化应用
export const initializeApp = createAsyncThunk(
  'app/initialize',
  async () => {
    // 模拟初始化过程
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // 从localStorage获取用户偏好设置
    const theme = localStorage.getItem('theme') as 'light' | 'dark' || 'light'
    const language = localStorage.getItem('language') as 'zh-CN' | 'en-US' || 'zh-CN'
    const sidebarCollapsed = localStorage.getItem('sidebarCollapsed') === 'true'
    
    return { theme, language, sidebarCollapsed }
  }
)

const appSlice = createSlice({
  name: 'app',
  initialState,
  reducers: {
    setTheme: (state, action: PayloadAction<'light' | 'dark'>) => {
      state.theme = action.payload
      localStorage.setItem('theme', action.payload)
    },
    setLanguage: (state, action: PayloadAction<'zh-CN' | 'en-US'>) => {
      state.language = action.payload
      localStorage.setItem('language', action.payload)
    },
    setSidebarCollapsed: (state, action: PayloadAction<boolean>) => {
      state.sidebarCollapsed = action.payload
      localStorage.setItem('sidebarCollapsed', action.payload.toString())
    },
    clearError: (state) => {
      state.error = null
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(initializeApp.pending, (state) => {
        state.loading = true
      })
      .addCase(initializeApp.fulfilled, (state, action) => {
        state.loading = false
        state.isInitialized = true
        state.theme = action.payload.theme
        state.language = action.payload.language
        state.sidebarCollapsed = action.payload.sidebarCollapsed
      })
      .addCase(initializeApp.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || '应用初始化失败'
      })
  },
})

export const { setTheme, setLanguage, setSidebarCollapsed, clearError } = appSlice.actions
export default appSlice.reducer 