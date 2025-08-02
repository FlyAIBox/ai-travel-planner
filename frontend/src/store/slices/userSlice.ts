import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface UserPreferences {
  theme: 'light' | 'dark'
  language: 'zh-CN' | 'en-US'
  currency: 'CNY' | 'USD' | 'EUR'
  travelStyle: 'budget' | 'comfort' | 'luxury'
  interests: string[]
}

interface UserProfile {
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

interface UserState {
  profile: UserProfile | null
  loading: boolean
  error: string | null
}

const initialState: UserState = {
  profile: null,
  loading: false,
  error: null,
}

const userSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {
    setProfile: (state, action: PayloadAction<UserProfile>) => {
      state.profile = action.payload
    },
    updateProfile: (state, action: PayloadAction<Partial<UserProfile>>) => {
      if (state.profile) {
        state.profile = { ...state.profile, ...action.payload }
      }
    },
    updatePreferences: (state, action: PayloadAction<Partial<UserPreferences>>) => {
      if (state.profile) {
        state.profile.preferences = { ...state.profile.preferences, ...action.payload }
      }
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload
    },
    clearError: (state) => {
      state.error = null
    },
  },
})

export const {
  setProfile,
  updateProfile,
  updatePreferences,
  setLoading,
  setError,
  clearError,
} = userSlice.actions

export default userSlice.reducer 