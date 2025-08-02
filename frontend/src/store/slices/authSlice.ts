import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'
import authAPI from '@/api/auth'

interface User {
  id: string
  username: string
  email: string
  avatar?: string
  nickname?: string
  preferences?: Record<string, any>
}

interface AuthState {
  isAuthenticated: boolean
  user: User | null
  token: string | null
  loading: boolean
  error: string | null
}

const initialState: AuthState = {
  isAuthenticated: false,
  user: null,
  token: localStorage.getItem('token'),
  loading: false,
  error: null,
}

// 检查认证状态
export const checkAuth = createAsyncThunk(
  'auth/checkAuth',
  async (_, { rejectWithValue }) => {
    try {
      const token = localStorage.getItem('token')
      if (!token) {
        throw new Error('No token found')
      }
      
      const response = await authAPI.getCurrentUser()
      return response.data
    } catch (error: any) {
      localStorage.removeItem('token')
      return rejectWithValue(error.response?.data?.message || '认证失败')
    }
  }
)

// 用户登录
export const login = createAsyncThunk(
  'auth/login',
  async (credentials: { email: string; password: string }, { rejectWithValue }) => {
    try {
      const response = await authAPI.login(credentials)
      const { token, user } = response.data
      
      localStorage.setItem('token', token)
      return { token, user }
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || '登录失败')
    }
  }
)

// 用户注册
export const register = createAsyncThunk(
  'auth/register',
  async (userData: { username: string; email: string; password: string }, { rejectWithValue }) => {
    try {
      const response = await authAPI.register(userData)
      const { token, user } = response.data
      
      localStorage.setItem('token', token)
      return { token, user }
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || '注册失败')
    }
  }
)

// 退出登录
export const logout = createAsyncThunk(
  'auth/logout',
  async () => {
    localStorage.removeItem('token')
    return null
  }
)

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null
    },
    updateUser: (state, action: PayloadAction<Partial<User>>) => {
      if (state.user) {
        state.user = { ...state.user, ...action.payload }
      }
    },
  },
  extraReducers: (builder) => {
    builder
      // checkAuth
      .addCase(checkAuth.pending, (state) => {
        state.loading = true
      })
      .addCase(checkAuth.fulfilled, (state, action) => {
        state.loading = false
        state.isAuthenticated = true
        state.user = action.payload
      })
      .addCase(checkAuth.rejected, (state, action) => {
        state.loading = false
        state.isAuthenticated = false
        state.user = null
        state.token = null
      })
      // login
      .addCase(login.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(login.fulfilled, (state, action) => {
        state.loading = false
        state.isAuthenticated = true
        state.user = action.payload.user
        state.token = action.payload.token
      })
      .addCase(login.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })
      // register
      .addCase(register.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(register.fulfilled, (state, action) => {
        state.loading = false
        state.isAuthenticated = true
        state.user = action.payload.user
        state.token = action.payload.token
      })
      .addCase(register.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })
      // logout
      .addCase(logout.fulfilled, (state) => {
        state.isAuthenticated = false
        state.user = null
        state.token = null
      })
  },
})

export const { clearError, updateUser } = authSlice.actions
export default authSlice.reducer 