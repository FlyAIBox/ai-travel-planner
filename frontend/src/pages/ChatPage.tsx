import React from 'react'
import { Layout } from 'antd'
import ChatWindow from '@/components/chat/ChatWindow'

const { Content } = Layout

const ChatPage: React.FC = () => {
  return (
    <Content style={{ height: 'calc(100vh - 64px)', overflow: 'hidden' }}>
      <ChatWindow />
    </Content>
  )
}

export default ChatPage 