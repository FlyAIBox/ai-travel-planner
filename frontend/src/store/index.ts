import { configureStore } from '@reduxjs/toolkit'
import { combineReducers } from '@reduxjs/toolkit'

// Slice imports
import appSlice from './slices/appSlice'
import authSlice from './slices/authSlice'
import chatSlice from './slices/chatSlice'
import planSlice from './slices/planSlice'
import userSlice from './slices/userSlice'

const rootReducer = combineReducers({
  app: appSlice,
  auth: authSlice,
  chat: chatSlice,
  plan: planSlice,
  user: userSlice,
})

export const store = configureStore({
  reducer: rootReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST'],
      },
    }),
  devTools: process.env.NODE_ENV !== 'production',
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch 