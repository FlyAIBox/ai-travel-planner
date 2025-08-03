# 配置管理说明

## 概述

本项目使用统一的配置管理系统，将所有硬编码的外部地址和配置项集中管理，支持不同环境的配置。

## 配置文件

### 环境变量文件

- `.env.development` - 开发环境配置
- `.env.production` - 生产环境配置  
- `.env.example` - 配置示例文件

### 配置管理模块

- `src/config/simple.ts` - 简化的配置管理模块（推荐使用）
- `src/config/index.ts` - 完整的配置管理模块（支持环境变量）

## 配置项说明

### API 配置
- `VITE_API_BASE_URL` - API 基础地址
- `VITE_API_TIMEOUT` - API 请求超时时间

### WebSocket 配置
- `VITE_WS_URL` - WebSocket 连接地址
- `VITE_WS_RECONNECT_ATTEMPTS` - 重连尝试次数
- `VITE_WS_RECONNECT_INTERVAL` - 重连间隔时间

### 聊天服务配置
- `VITE_CHAT_WS_URL` - 聊天服务 WebSocket 地址

### 日志配置
- `VITE_LOG_LEVEL` - 日志级别 (debug/info/warn/error)
- `VITE_LOG_ENABLE_CONSOLE` - 是否启用控制台日志
- `VITE_LOG_ENABLE_REMOTE` - 是否启用远程日志

### 地图配置
- `VITE_MAPBOX_ACCESS_TOKEN` - Mapbox 访问令牌

### 应用配置
- `VITE_APP_TITLE` - 应用标题
- `VITE_APP_VERSION` - 应用版本

### Vite 代理配置
- `VITE_PROXY_API_TARGET` - API 代理目标地址
- `VITE_PROXY_WS_TARGET` - WebSocket 代理目标地址

## 使用方法

### 1. 在代码中使用配置

```typescript
// 推荐：使用简化版配置（避免类型问题）
import config from '@/config/simple'

// 或者：使用完整版配置（支持环境变量）
import config from '@/config'

// 使用 API 配置
const apiClient = axios.create({
  baseURL: config.api.baseUrl,
  timeout: config.api.timeout,
})

// 使用 WebSocket 配置
const ws = new WebSocket(config.websocket.url)

// 使用聊天服务配置
const chatWs = new WebSocket(config.chat.wsUrl)
```

### 2. 环境配置

#### 开发环境
复制 `.env.example` 为 `.env.development` 并修改相应配置：

```bash
cp .env.example .env.development
```

#### 生产环境
复制 `.env.example` 为 `.env.production` 并修改相应配置：

```bash
cp .env.example .env.production
```

### 3. Docker 环境配置

在 `docker-compose.yml` 中设置环境变量：

```yaml
frontend:
  environment:
    - VITE_API_BASE_URL=http://api-gateway:8000/api/v1
    - VITE_WS_URL=ws://api-gateway:8000/ws
    - VITE_CHAT_WS_URL=ws://api-gateway:8000
```

## 迁移说明

本次重构将以下硬编码地址改为配置管理：

1. `frontend/src/components/chat/ChatWindow.tsx` - 聊天 WebSocket 地址
2. `frontend/src/services/websocket.ts` - WebSocket 服务地址
3. `frontend/src/api/*.ts` - API 基础地址和超时配置
4. `frontend/src/utils/logger.ts` - 日志配置
5. `frontend/vite.config.ts` - 开发服务器代理配置

## 注意事项

1. 所有环境变量必须以 `VITE_` 前缀开头才能在前端代码中访问
2. 修改配置后需要重启开发服务器
3. 生产环境部署时确保设置正确的环境变量
4. 敏感信息（如 API 密钥）不要提交到版本控制系统
