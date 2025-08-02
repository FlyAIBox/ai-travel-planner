import React, { useState } from 'react'
import { Layout, Card, Form, Input, Button, Typography, Space, message } from 'antd'
import { UserOutlined, LockOutlined, MailOutlined, RocketOutlined } from '@ant-design/icons'
import { Link, useNavigate } from 'react-router-dom'

const { Content } = Layout
const { Title, Text } = Typography

const RegisterPage: React.FC = () => {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)

  const onFinish = async (values: any) => {
    setLoading(true)
    try {
      // 模拟注册API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      message.success('注册成功！正在跳转到登录页面...')
      setTimeout(() => navigate('/login'), 1500)
    } catch (error) {
      message.error('注册失败，请重试')
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
              创建账户
            </Title>
            <Text type="secondary">
              注册成为AI旅行规划助手的用户
            </Text>
          </div>

          <Form
            name="register"
            onFinish={onFinish}
            size="large"
            autoComplete="off"
          >
            <Form.Item
              name="username"
              rules={[
                { required: true, message: '请输入用户名!' },
                { min: 3, message: '用户名至少3位字符!' }
              ]}
            >
              <Input 
                prefix={<UserOutlined />} 
                placeholder="用户名"
                style={{ borderRadius: '6px' }}
              />
            </Form.Item>

            <Form.Item
              name="email"
              rules={[
                { required: true, message: '请输入邮箱地址!' },
                { type: 'email', message: '请输入有效的邮箱地址!' }
              ]}
            >
              <Input 
                prefix={<MailOutlined />} 
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

            <Form.Item
              name="confirmPassword"
              dependencies={['password']}
              rules={[
                { required: true, message: '请确认密码!' },
                ({ getFieldValue }) => ({
                  validator(_, value) {
                    if (!value || getFieldValue('password') === value) {
                      return Promise.resolve()
                    }
                    return Promise.reject(new Error('两次输入的密码不一致!'))
                  },
                }),
              ]}
            >
              <Input.Password 
                prefix={<LockOutlined />} 
                placeholder="确认密码"
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
                注册
              </Button>
            </Form.Item>
          </Form>

          <div style={{ textAlign: 'center', marginTop: '24px' }}>
            <Space>
              <Text type="secondary">已有账户？</Text>
              <Link to="/login">立即登录</Link>
            </Space>
          </div>
        </Card>
      </Content>
    </Layout>
  )
}

export default RegisterPage 