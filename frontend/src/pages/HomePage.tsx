import React from 'react'
import { Layout, Typography, Button, Row, Col, Card, Space } from 'antd'
import { MessageOutlined, CalendarOutlined, UserOutlined, RocketOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'

const { Content } = Layout
const { Title, Paragraph } = Typography

const HomePage: React.FC = () => {
  const navigate = useNavigate()

  const features = [
    {
      icon: <MessageOutlined style={{ fontSize: '2rem', color: '#1890ff' }} />,
      title: '智能对话',
      description: '与AI助手对话，获得个性化的旅行建议和推荐',
      action: () => navigate('/chat')
    },
    {
      icon: <CalendarOutlined style={{ fontSize: '2rem', color: '#52c41a' }} />,
      title: '行程规划',
      description: '制定详细的旅行计划，包括景点、住宿、交通安排',
      action: () => navigate('/plans')
    },
    {
      icon: <UserOutlined style={{ fontSize: '2rem', color: '#722ed1' }} />,
      title: '个人中心',
      description: '管理您的偏好设置、历史记录和个人信息',
      action: () => navigate('/user')
    }
  ]

  return (
    <Content style={{ padding: '50px 50px', minHeight: '80vh' }}>
      <div style={{ textAlign: 'center', marginBottom: '50px' }}>
        <Title level={1}>
          <RocketOutlined style={{ marginRight: '16px', color: '#1890ff' }} />
          AI智能旅行规划助手
        </Title>
        <Paragraph style={{ fontSize: '18px', color: '#666', maxWidth: '600px', margin: '0 auto' }}>
          基于人工智能的个性化旅行规划助手，为您提供专业的旅行建议、智能行程推荐和便捷的预订服务
        </Paragraph>
      </div>

      <Row gutter={[32, 32]} justify="center">
        {features.map((feature, index) => (
          <Col xs={24} sm={12} lg={8} key={index}>
            <Card
              hoverable
              style={{ 
                height: '280px',
                borderRadius: '12px',
                border: '1px solid #f0f0f0',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
              }}
              bodyStyle={{ 
                padding: '32px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                textAlign: 'center',
                height: '100%'
              }}
              onClick={feature.action}
            >
              <div style={{ marginBottom: '20px' }}>
                {feature.icon}
              </div>
              <Title level={3} style={{ marginBottom: '16px' }}>
                {feature.title}
              </Title>
              <Paragraph style={{ color: '#666', lineHeight: '1.6', flex: 1 }}>
                {feature.description}
              </Paragraph>
            </Card>
          </Col>
        ))}
      </Row>

      <div style={{ textAlign: 'center', marginTop: '60px' }}>
        <Space size="large">
          <Button 
            type="primary" 
            size="large" 
            icon={<MessageOutlined />}
            onClick={() => navigate('/chat')}
            style={{ borderRadius: '6px', height: '48px', padding: '0 32px' }}
          >
            开始对话
          </Button>
          <Button 
            size="large" 
            onClick={() => navigate('/plans')}
            style={{ borderRadius: '6px', height: '48px', padding: '0 32px' }}
          >
            查看计划
          </Button>
        </Space>
      </div>
    </Content>
  )
}

export default HomePage 