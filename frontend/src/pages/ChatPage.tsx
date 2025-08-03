import React from 'react'
import { Layout } from 'antd'
import { useParams } from 'react-router-dom'
import ChatWindow from '@/components/chat/ChatWindow'

const { Content } = Layout

const ChatPage: React.FC = () => {
  const { conversationId } = useParams<{ conversationId?: string }>()

  return (
    <Content style={{ height: 'calc(100vh - 64px)', overflow: 'hidden' }}>
      <ChatWindow
        conversationId={conversationId}
        userId="stable_user_123" // 使用固定的用户ID进行测试
      />
    </Content>
  )
}

export default ChatPage