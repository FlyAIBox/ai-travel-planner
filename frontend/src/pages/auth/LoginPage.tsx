import React, { useState } from 'react'
import { Layout, Card, Form, Input, Button, Typography, Space, message } from 'antd'
import { UserOutlined, LockOutlined, RocketOutlined } from '@ant-design/icons'
import { Link, useNavigate } from 'react-router-dom'

const { Content } = Layout
const { Title, Text } = Typography

const LoginPage: React.FC = () => {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)

  const onFinish = async (values: any) => {
    setLoading(true)
    try {
      // 模拟登录API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // 模拟保存token
      localStorage.setItem('token', 'mock-jwt-token')
      
      message.success('登录成功！')
      navigate('/')
    } catch (error) {
      message.error('登录失败，请检查用户名和密码')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Layout style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
      <Content style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center',
        padding: '50px'
      }}>
        <Card 
          style={{ 
            width: '100%',
            maxWidth: '400px',
            borderRadius: '12px',
            boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
          }}
          bodyStyle={{ padding: '40px' }}
        >
          <div style={{ textAlign: 'center', marginBottom: '32px' }}>
            <RocketOutlined style={{ fontSize: '48px', color: '#1890ff', marginBottom: '16px' }} />
            <Title level={2} style={{ margin: 0 }}>
              AI旅行助手
            </Title>
            <Text type="secondary">
              登录开始您的智能旅行规划之旅
            </Text>
          </div>

          <Form
            name="login"
            onFinish={onFinish}
            size="large"
            autoComplete="off"
          >
            <Form.Item
              name="email"
              rules={[
                { required: true, message: '请输入邮箱地址!' },
                { type: 'email', message: '请输入有效的邮箱地址!' }
              ]}
            >
              <Input 
                prefix={<UserOutlined />} 
                placeholder="邮箱地址"
                style={{ borderRadius: '6px' }}
              />
            </Form.Item>

            <Form.Item
              name="password"
              rules={[
                { required: true, message: '请输入密码!' },
                { min: 6, message: '密码至少6位字符!' }
              ]}
            >
              <Input.Password 
                prefix={<LockOutlined />} 
                placeholder="密码"
                style={{ borderRadius: '6px' }}
              />
            </Form.Item>

            <Form.Item>
              <Button 
                type="primary" 
                htmlType="submit" 
                loading={loading}
                style={{ 
                  width: '100%', 
                  height: '44px',
                  borderRadius: '6px',
                  fontSize: '16px'
                }}
              >
                登录
              </Button>
            </Form.Item>
          </Form>

          <div style={{ textAlign: 'center', marginTop: '24px' }}>
            <Space>
              <Text type="secondary">还没有账户？</Text>
              <Link to="/register">立即注册</Link>
            </Space>
          </div>
        </Card>
      </Content>
    </Layout>
  )
}

export default LoginPage 