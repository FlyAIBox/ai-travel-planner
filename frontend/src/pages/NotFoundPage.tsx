import React from 'react'
import { Layout, Result, Button } from 'antd'
import { useNavigate } from 'react-router-dom'

const { Content } = Layout

const NotFoundPage: React.FC = () => {
  const navigate = useNavigate()

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Content style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center' 
      }}>
        <Result
          status="404"
          title="404"
          subTitle="抱歉，您访问的页面不存在。"
          extra={
            <Button type="primary" onClick={() => navigate('/')}>
              返回首页
            </Button>
          }
        />
      </Content>
    </Layout>
  )
}

export default NotFoundPage 