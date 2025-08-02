import React, { useState } from 'react'
import { Layout, Card, List, Button, Typography, Empty, Space } from 'antd'
import { PlusOutlined, CalendarOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'

const { Content } = Layout
const { Title, Text } = Typography

const PlanPage: React.FC = () => {
  const navigate = useNavigate()
  const [plans] = useState([
    {
      id: '1',
      title: '北京3日游',
      description: '探索中国古都的历史文化',
      destination: '北京',
      startDate: '2024-03-15',
      endDate: '2024-03-17',
      status: 'draft'
    },
    {
      id: '2', 
      title: '上海商务行',
      description: '商务会议+城市观光',
      destination: '上海',
      startDate: '2024-04-01',
      endDate: '2024-04-03',
      status: 'confirmed'
    }
  ])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'draft': return '#faad14'
      case 'confirmed': return '#52c41a'
      case 'completed': return '#722ed1'
      default: return '#d9d9d9'
    }
  }

  const getStatusText = (status: string) => {
    switch (status) {
      case 'draft': return '草稿'
      case 'confirmed': return '已确认'
      case 'completed': return '已完成'
      default: return '未知'
    }
  }

  return (
    <Content style={{ padding: '24px', minHeight: '80vh' }}>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>
          <CalendarOutlined style={{ marginRight: '8px' }} />
          我的旅行计划
        </Title>
        <Button 
          type="primary" 
          icon={<PlusOutlined />}
          onClick={() => navigate('/chat')}
          size="large"
        >
          创建新计划
        </Button>
      </div>

      {plans.length === 0 ? (
        <Card>
          <Empty
            description="暂无旅行计划"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          >
            <Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/chat')}>
              开始规划
            </Button>
          </Empty>
        </Card>
      ) : (
        <List
          grid={{ 
            gutter: 16, 
            xs: 1, 
            sm: 2, 
            md: 2, 
            lg: 3, 
            xl: 3, 
            xxl: 4 
          }}
          dataSource={plans}
          renderItem={(plan) => (
            <List.Item>
              <Card
                hoverable
                style={{ borderRadius: '8px' }}
                bodyStyle={{ padding: '20px' }}
                onClick={() => navigate(`/plans/${plan.id}`)}
              >
                <div style={{ marginBottom: '12px' }}>
                  <Title level={4} style={{ margin: 0, marginBottom: '8px' }}>
                    {plan.title}
                  </Title>
                  <Text type="secondary" style={{ fontSize: '14px' }}>
                    {plan.description}
                  </Text>
                </div>
                
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>目的地: </Text>
                    <Text>{plan.destination}</Text>
                  </div>
                  <div>
                    <Text strong>时间: </Text>
                    <Text>{plan.startDate} 至 {plan.endDate}</Text>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px' }}>
                    <span
                      style={{
                        display: 'inline-block',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        backgroundColor: getStatusColor(plan.status),
                        color: 'white',
                        fontSize: '12px'
                      }}
                    >
                      {getStatusText(plan.status)}
                    </span>
                  </div>
                </Space>
              </Card>
            </List.Item>
          )}
        />
      )}
    </Content>
  )
}

export default PlanPage 