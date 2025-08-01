import { configureStore } from '@reduxjs/toolkit'
import authSlice from './slices/authSlice'
import appSlice from './slices/appSlice'
import chatSlice from './slices/chatSlice'
import planSlice from './slices/planSlice'

export const store = configureStore({
  reducer: {
    auth: authSlice,
    app: appSlice,
    chat: chatSlice,
    plan: planSlice,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
      },
    }),
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch

export default store 