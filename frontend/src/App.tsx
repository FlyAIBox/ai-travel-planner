import React, { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout, Spin } from 'antd'
import { useSelector, useDispatch } from 'react-redux'
import { Helmet } from 'react-helmet-async'

import type { RootState } from '@store/index'
import { initializeApp } from '@store/slices/appSlice'
import { checkAuth } from '@store/slices/authSlice'

// 页面组件
import MainLayout from '@components/layout/MainLayout'
import ChatPage from '@pages/ChatPage'
import PlanPage from '@pages/PlanPage'
import UserPage from '@pages/UserPage'
import LoginPage from '@pages/auth/LoginPage'
import RegisterPage from '@pages/auth/RegisterPage'
import HomePage from '@pages/HomePage'
import NotFoundPage from '@pages/NotFoundPage'

// 受保护的路由组件
import ProtectedRoute from '@components/auth/ProtectedRoute'

// 样式
import './App.css'

const { Content } = Layout

const App: React.FC = () => {
  const dispatch = useDispatch()
  const { isInitialized, loading } = useSelector((state: RootState) => state.app)
  const { isAuthenticated } = useSelector((state: RootState) => state.auth)

  useEffect(() => {
    // 初始化应用
    dispatch(initializeApp())
    // 检查用户认证状态
    dispatch(checkAuth())
  }, [dispatch])

  // 应用初始化中
  if (!isInitialized || loading) {
    return (
      <Layout style={{ minHeight: '100vh' }}>
        <Content 
          style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center' 
          }}
        >
          <Spin size="large" tip="正在加载应用..." />
        </Content>
      </Layout>
    )
  }

  return (
    <>
      <Helmet>
        <title>AI智能旅行规划助手</title>
        <meta name="description" content="基于人工智能的个性化旅行规划助手，提供智能行程推荐、实时预订和专业旅行建议" />
        <meta name="keywords" content="AI,人工智能,旅行规划,行程推荐,智能助手" />
      </Helmet>

      <Routes>
        {/* 公开路由 */}
        <Route path="/login" element={
          !isAuthenticated ? <LoginPage /> : <Navigate to="/" replace />
        } />
        <Route path="/register" element={
          !isAuthenticated ? <RegisterPage /> : <Navigate to="/" replace />
        } />

        {/* 主应用路由 */}
        <Route path="/" element={<MainLayout />}>
          <Route index element={<HomePage />} />
          
          {/* 聊天页面 */}
          <Route path="chat" element={<ChatPage />} />
          <Route path="chat/:conversationId" element={<ChatPage />} />
          
          {/* 旅行计划页面 */}
          <Route path="plans" element={
            <ProtectedRoute>
              <PlanPage />
            </ProtectedRoute>
          } />
          <Route path="plans/:planId" element={
            <ProtectedRoute>
              <PlanPage />
            </ProtectedRoute>
          } />
          
          {/* 用户中心 */}
          <Route path="user" element={
            <ProtectedRoute>
              <UserPage />
            </ProtectedRoute>
          } />
          <Route path="user/:section" element={
            <ProtectedRoute>
              <UserPage />
            </ProtectedRoute>
          } />
        </Route>

        {/* 404页面 */}
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </>
  )
}

export default App 