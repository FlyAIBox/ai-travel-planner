# AI旅行规划助手 - 前端应用

基于 React 18 + TypeScript + Vite 构建的现代化前端应用，为AI智能旅行规划助手提供用户界面。

## 🚀 快速启动

### 环境要求
- Node.js >= 16.0.0
- npm >= 8.0.0

### 安装依赖
```bash
npm install
```

### 开发模式启动
```bash
npm run dev
```
访问 [http://localhost:3000](http://localhost:3000)

### 生产构建
```bash
npm run build
```

### 预览生产版本
```bash
npm run preview
```

### 代码检查
```bash
npm run lint
npm run lint:fix
```

### 类型检查
```bash
npm run type-check
```

### 测试
```bash
npm test
npm run test:coverage
```

## 📁 项目结构

```
frontend/
├── src/
│   ├── api/                 # API服务层
│   │   ├── auth.ts         # 认证API
│   │   ├── chat.ts         # 聊天API
│   │   └── travel.ts       # 旅行规划API
│   ├── components/         # React组件
│   │   ├── chat/           # 聊天相关组件
│   │   ├── layout/         # 布局组件
│   │   ├── auth/           # 认证组件
│   │   └── TravelPlan/     # 旅行计划组件
│   ├── pages/              # 页面组件
│   │   ├── auth/           # 认证页面
│   │   ├── HomePage.tsx    # 首页
│   │   ├── ChatPage.tsx    # 聊天页面
│   │   ├── PlanPage.tsx    # 计划页面
│   │   ├── UserPage.tsx    # 用户中心
│   │   └── NotFoundPage.tsx# 404页面
│   ├── store/              # Redux状态管理
│   │   ├── slices/         # Redux切片
│   │   └── index.ts        # Store配置
│   ├── services/           # 业务服务
│   │   └── websocket.ts    # WebSocket服务
│   ├── utils/              # 工具函数
│   │   ├── logger.ts       # 日志工具
│   │   └── devtools.ts     # 开发调试工具
│   ├── types/              # TypeScript类型定义
│   ├── App.tsx             # 主应用组件
│   ├── main.tsx            # 应用入口
│   ├── index.css           # 全局样式
│   └── App.css             # 应用样式
├── public/                 # 静态资源
├── index.html              # HTML入口文件
├── package.json            # 项目配置
├── vite.config.ts          # Vite配置
├── tsconfig.json           # TypeScript配置
└── Dockerfile              # Docker镜像配置
```

## 🛠️ 技术栈

### 核心框架
- **React 18** - 前端框架
- **TypeScript** - 类型安全
- **Vite** - 构建工具

### 状态管理
- **Redux Toolkit** - 状态管理
- **React Query** - 服务端状态管理

### UI组件库
- **Ant Design** - UI组件库
- **Ant Design Icons** - 图标库

### 路由和导航
- **React Router Dom** - 路由管理

### 样式方案
- **CSS Modules** - 样式模块化
- **Styled Components** - CSS-in-JS (可选)

### 开发工具
- **ESLint** - 代码检查
- **Prettier** - 代码格式化
- **Jest** - 单元测试

## 🔧 开发工具和调试

### 开发者工具
应用内置了强大的开发者工具，在开发模式下可通过浏览器控制台访问：

```javascript
// 访问开发工具
__AI_TRAVEL_DEVTOOLS__

// 常用调试方法
__AI_TRAVEL_DEVTOOLS__.getState()           // 查看Redux状态
__AI_TRAVEL_DEVTOOLS__.getWebSocketState()  // 查看WebSocket状态
__AI_TRAVEL_DEVTOOLS__.getStorageInfo()     // 查看本地存储信息
__AI_TRAVEL_DEVTOOLS__.utils.testApiEndpoint('/api/v1/health') // 测试API端点
```

### 日志系统
```javascript
import { logger } from '@/utils/logger'

logger.info('用户登录', { userId: 123 })
logger.error('API调用失败', { error: 'Network Error' })
logger.trackEvent('button_click', { button: 'login' })
```

### WebSocket调试
```javascript
import { getWebSocketService } from '@/services/websocket'

const ws = getWebSocketService()
ws.on('message', (data) => console.log('收到消息:', data))
ws.send({ type: 'ping', data: 'hello' })
```

## 🎨 组件开发指南

### 组件命名规范
- 组件文件使用 PascalCase: `UserProfile.tsx`
- 样式文件使用相同名称: `UserProfile.css`
- 测试文件使用 `.test.tsx` 后缀

### 组件结构模板
```tsx
import React from 'react'
import { Button, Card } from 'antd'
import './ComponentName.css'

interface ComponentNameProps {
  title: string
  onAction?: () => void
}

const ComponentName: React.FC<ComponentNameProps> = ({ 
  title, 
  onAction 
}) => {
  return (
    <Card title={title}>
      <Button onClick={onAction}>
        操作按钮
      </Button>
    </Card>
  )
}

export default ComponentName
```

## 🔄 API集成

### API调用示例
```tsx
import { useQuery, useMutation } from '@tanstack/react-query'
import authAPI from '@/api/auth'

// 查询数据
const { data, loading, error } = useQuery({
  queryKey: ['user'],
  queryFn: authAPI.getCurrentUser
})

// 修改数据
const loginMutation = useMutation({
  mutationFn: authAPI.login,
  onSuccess: (data) => {
    console.log('登录成功', data)
  },
  onError: (error) => {
    console.error('登录失败', error)
  }
})
```

## 📱 响应式设计

### 断点配置
```css
/* 移动端 */
@media (max-width: 768px) {
  .mobile-hidden { display: none !important; }
}

/* 桌面端 */
@media (min-width: 769px) {
  .desktop-hidden { display: none !important; }
}
```

### Ant Design 响应式栅格
```tsx
<Row gutter={[16, 16]}>
  <Col xs={24} sm={12} md={8} lg={6}>
    <Card>响应式卡片</Card>
  </Col>
</Row>
```

## 🚀 性能优化

### 代码分割
```tsx
import { lazy, Suspense } from 'react'

const LazyComponent = lazy(() => import('./HeavyComponent'))

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <LazyComponent />
    </Suspense>
  )
}
```

### 组件优化
```tsx
import { memo, useCallback, useMemo } from 'react'

const OptimizedComponent = memo(({ data, onUpdate }) => {
  const processedData = useMemo(() => {
    return data.map(item => processItem(item))
  }, [data])
  
  const handleUpdate = useCallback((id) => {
    onUpdate(id)
  }, [onUpdate])
  
  return <div>{/* 组件内容 */}</div>
})
```

## 🔒 环境配置

### 开发环境
```bash
# .env.development
VITE_API_BASE_URL=http://localhost:8080/api/v1
VITE_WS_URL=ws://localhost:8080/ws
VITE_LOG_LEVEL=debug
```

### 生产环境
```bash
# .env.production
VITE_API_BASE_URL=/api/v1
VITE_WS_URL=wss://yourserver.com/ws
VITE_LOG_LEVEL=error
```

## 📋 常见问题

### Q: 如何添加新的API接口？
A: 在 `src/api/` 目录下对应的文件中添加新的接口方法，并更新TypeScript类型定义。

### Q: 如何添加新的页面？
A: 
1. 在 `src/pages/` 目录下创建页面组件
2. 在 `src/App.tsx` 中添加路由配置
3. 在导航菜单中添加相应链接

### Q: 如何进行状态管理？
A: 使用Redux Toolkit进行全局状态管理，创建对应的slice文件并在store中注册。

### Q: 如何处理错误？
A: 使用React Error Boundary处理组件错误，API错误通过统一的错误处理机制处理。

## 📚 相关文档

- [React 官方文档](https://reactjs.org/)
- [TypeScript 官方文档](https://www.typescriptlang.org/)
- [Ant Design 组件文档](https://ant.design/)
- [Vite 官方文档](https://vitejs.dev/)
- [Redux Toolkit 文档](https://redux-toolkit.js.org/)

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。 