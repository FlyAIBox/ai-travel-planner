import React from 'react'
import { Navigate, useLocation } from 'react-router-dom'

interface ProtectedRouteProps {
  children: React.ReactNode
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const location = useLocation()
  
  // 模拟认证状态检查
  const isAuthenticated = localStorage.getItem('token') !== null
  
  if (!isAuthenticated) {
    // 重定向到登录页面，并保存当前路径
    return <Navigate to="/login" state={{ from: location }} replace />
  }
  
  return <>{children}</>
}

export default ProtectedRoute 