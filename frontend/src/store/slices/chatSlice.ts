import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface Message {
  id: string
  content: string
  type: 'user' | 'assistant' | 'system'
  timestamp: string
  metadata?: Record<string, any>
}

interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt: string
  updatedAt: string
}

interface ChatState {
  conversations: Conversation[]
  activeConversationId: string | null
  isConnected: boolean
  isTyping: boolean
  loading: boolean
  error: string | null
}

const initialState: ChatState = {
  conversations: [],
  activeConversationId: null,
  isConnected: false,
  isTyping: false,
  loading: false,
  error: null,
}

const chatSlice = createSlice({
  name: 'chat',
  initialState,
  reducers: {
    setConnected: (state, action: PayloadAction<boolean>) => {
      state.isConnected = action.payload
    },
    setTyping: (state, action: PayloadAction<boolean>) => {
      state.isTyping = action.payload
    },
    addMessage: (state, action: PayloadAction<{ conversationId: string; message: Message }>) => {
      const { conversationId, message } = action.payload
      const conversation = state.conversations.find(c => c.id === conversationId)
      if (conversation) {
        conversation.messages.push(message)
        conversation.updatedAt = new Date().toISOString()
      }
    },
    createConversation: (state, action: PayloadAction<Conversation>) => {
      state.conversations.unshift(action.payload)
      state.activeConversationId = action.payload.id
    },
    setActiveConversation: (state, action: PayloadAction<string | null>) => {
      state.activeConversationId = action.payload
    },
    updateConversationTitle: (state, action: PayloadAction<{ id: string; title: string }>) => {
      const conversation = state.conversations.find(c => c.id === action.payload.id)
      if (conversation) {
        conversation.title = action.payload.title
      }
    },
    deleteConversation: (state, action: PayloadAction<string>) => {
      state.conversations = state.conversations.filter(c => c.id !== action.payload)
      if (state.activeConversationId === action.payload) {
        state.activeConversationId = null
      }
    },
    clearError: (state) => {
      state.error = null
    },
  },
})

export const {
  setConnected,
  setTyping,
  addMessage,
  createConversation,
  setActiveConversation,
  updateConversationTitle,
  deleteConversation,
  clearError,
} = chatSlice.actions

export default chatSlice.reducer 