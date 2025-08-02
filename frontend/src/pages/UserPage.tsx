import React from 'react'
import { Layout, Card, Avatar, Typography, Space, Button, Row, Col } from 'antd'
import { UserOutlined, SettingOutlined, HistoryOutlined, HeartOutlined } from '@ant-design/icons'

const { Content } = Layout
const { Title, Text } = Typography

const UserPage: React.FC = () => {
  const user = {
    username: 'traveler',
    email: 'traveler@example.com',
    avatar: '',
    nickname: '旅行爱好者',
    location: '北京',
    joinDate: '2024-01-15'
  }

  const stats = [
    { label: '旅行计划', value: '5', icon: <HistoryOutlined /> },
    { label: '已完成', value: '3', icon: <HeartOutlined /> },
    { label: '收藏地点', value: '12', icon: <HeartOutlined /> },
    { label: '使用天数', value: '30', icon: <HistoryOutlined /> }
  ]

  return (
    <Content style={{ padding: '24px', minHeight: '80vh' }}>
      <Title level={2}>
        <UserOutlined style={{ marginRight: '8px' }} />
        个人中心
      </Title>

      <Row gutter={[24, 24]}>
        <Col xs={24} lg={8}>
          <Card style={{ textAlign: 'center' }}>
            <Avatar 
              size={80} 
              icon={<UserOutlined />} 
              src={user.avatar}
              style={{ marginBottom: '16px' }}
            />
            <Title level={4} style={{ margin: '8px 0' }}>
              {user.nickname || user.username}
            </Title>
            <Text type="secondary">{user.email}</Text>
            <div style={{ marginTop: '16px' }}>
              <Text type="secondary">
                <UserOutlined style={{ marginRight: '4px' }} />
                {user.location}
              </Text>
            </div>
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary">
                加入时间: {user.joinDate}
              </Text>
            </div>
            <Button 
              type="primary" 
              icon={<SettingOutlined />}
              style={{ marginTop: '20px', width: '100%' }}
            >
              编辑资料
            </Button>
          </Card>
        </Col>

        <Col xs={24} lg={16}>
          <Card title="统计信息" style={{ marginBottom: '24px' }}>
            <Row gutter={[16, 16]}>
              {stats.map((stat, index) => (
                <Col xs={12} sm={6} key={index}>
                  <div style={{ textAlign: 'center', padding: '16px' }}>
                    <div style={{ fontSize: '24px', color: '#1890ff', marginBottom: '8px' }}>
                      {stat.icon}
                    </div>
                    <Title level={3} style={{ margin: '8px 0 4px 0', color: '#1890ff' }}>
                      {stat.value}
                    </Title>
                    <Text type="secondary">{stat.label}</Text>
                  </div>
                </Col>
              ))}
            </Row>
          </Card>

          <Card title="偏好设置">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 0' }}>
                <div>
                  <Text strong>主题设置</Text>
                  <br />
                  <Text type="secondary">选择浅色或深色主题</Text>
                </div>
                <Button>浅色</Button>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 0' }}>
                <div>
                  <Text strong>语言设置</Text>
                  <br />
                  <Text type="secondary">选择界面显示语言</Text>
                </div>
                <Button>中文</Button>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 0' }}>
                <div>
                  <Text strong>旅行风格</Text>
                  <br />
                  <Text type="secondary">预算型、舒适型或豪华型</Text>
                </div>
                <Button>舒适型</Button>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 0' }}>
                <div>
                  <Text strong>货币单位</Text>
                  <br />
                  <Text type="secondary">价格显示的货币单位</Text>
                </div>
                <Button>人民币 (CNY)</Button>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>
    </Content>
  )
}

export default UserPage 