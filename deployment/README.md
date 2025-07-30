# 🐳 AI Travel Planner - 部署指南

## 📋 概述

本文档提供了AI Travel Planner项目的完整部署指南，包括开发环境、生产环境和监控环境的配置。

## 🏗️ 架构概述

```
├── 负载均衡层 (Nginx)
├── 微服务层 (FastAPI)
│   ├── API网关服务 (8000)
│   ├── 聊天服务 (8001)
│   ├── 智能体服务 (8002)
│   ├── RAG服务 (8003)
│   └── 用户服务 (8004)
├── 数据存储层
│   ├── MySQL (3306) - 关系型数据库
│   ├── Redis (6379) - 缓存/会话
│   ├── Qdrant (6333) - 向量数据库
│   └── Elasticsearch (9200) - 搜索引擎
├── 工作流层
    └── n8n (5678) - 工作流引擎
```

## 🚀 快速开始

### 1. 环境要求

- **操作系统**: Ubuntu 22.04 LTS 或更高版本
- **Docker**: 20.10+ 
- **Docker Compose**: 2.0+
- **内存**: 最低 8GB，推荐 16GB+
- **存储**: 最低 50GB 可用空间
- **CPU**: 最低 4核，推荐 8核+

### 2. 克隆项目

```bash
git clone https://github.com/your-org/ai-travel-planner.git
cd ai-travel-planner
```

### 3. 环境配置

#### 开发环境

```bash
# 复制开发环境配置
cp .env.example .env

# 编辑配置文件（按需修改）
vim .env
```

#### 生产环境

```bash
# 复制生产环境配置模板
cp .env.prod.example .env.prod

# 编辑生产环境配置（必须修改）
vim .env.prod
```

**⚠️ 重要**: 生产环境必须修改以下配置项：
- `MYSQL_ROOT_PASSWORD` - MySQL root密码
- `MYSQL_PASSWORD` - MySQL应用用户密码
- `JWT_SECRET` - JWT密钥
- `OPENAI_API_KEY` - OpenAI API密钥（或配置国产大模型）
- `N8N_BASIC_AUTH_PASSWORD` - n8n管理密码
- `GRAFANA_ADMIN_PASSWORD` - Grafana管理密码

**🇨🇳 中国大陆服务配置** (根据需要选择配置)：
- 旅游API：`CTRIP_API_KEY`, `QUNAR_API_KEY`, `FLIGGY_API_KEY`, `MEITUAN_API_KEY`
- 地图API：`BAIDU_MAP_API_KEY`, `AMAP_API_KEY`, `TENCENT_MAP_API_KEY`
- 天气API：`CAIYUN_WEATHER_API_KEY`, `HEWEATHER_API_KEY`, `XINZHI_WEATHER_API_KEY`
- 国产大模型：`BAIDU_QIANFAN_API_KEY`, `ALIBABA_DASHSCOPE_API_KEY`, `TENCENT_HUNYUAN_SECRET_ID`
- 支付服务：`ALIPAY_APP_ID`, `WECHAT_PAY_APP_ID`
- 社交登录：`WECHAT_APP_ID`, `QQ_APP_ID`, `WEIBO_APP_KEY`

## 🛠️ 部署方式

### 方式一：使用管理脚本（推荐）

```bash
# 生产环境
./scripts/deployment/deploy.sh
```

### 方式二：手动Docker Compose

```bash
# 开发环境
docker-compose -f deployment/docker/docker-compose.dev.yml up -d --build

# 生产环境
docker-compose -f deployment/docker/docker-compose.prod.yml up -d --build

# 监控服务
docker-compose -f deployment/docker/docker-compose.monitoring.yml up -d
```

## 📝 环境说明


**特点**:
- 多进程worker
- 安全加固配置
- 性能优化
- 完整监控体系
- 自动故障恢复

**服务列表**:
- MySQL生产数据库
- Redis集群
- Qdrant向量数据库
- Elasticsearch搜索引擎
- 完整微服务架构
- Nginx负载均衡
- 日志收集系统

**访问地址**:
- 🌐 主入口: http://localhost
- 🔌 API网关: http://localhost/api
- 💬 聊天服务: http://localhost/chat
- 🤖 智能体: http://localhost/agent
- 📚 RAG服务: http://localhost/rag
- 👤 用户服务: http://localhost/users
- 🔧 工作流管理: http://localhost/workflow
- 📊 监控面板: http://localhost/grafana

## 🔧 管理命令

### Docker管理脚本

```bash
# 查看帮助
./scripts/docker/manage.sh help

# 启动环境
./scripts/docker/manage.sh <env> up [--build]

# 停止环境
./scripts/docker/manage.sh <env> down [--force-rm]

# 查看状态
./scripts/docker/manage.sh <env> ps

# 查看日志
./scripts/docker/manage.sh <env> logs [--follow] [--service <name>]

# 进入容器
./scripts/docker/manage.sh <env> exec --service <name>

# 清理资源
./scripts/docker/manage.sh <env> clean [--volumes]
```

### 数据库管理

```bash
# 初始化数据库
python scripts/database/init_db.py init

# 检查数据库状态
python scripts/database/init_db.py check

# 创建示例数据
python scripts/database/init_db.py sample

# 重置数据库 (危险操作)
python scripts/database/init_db.py reset --force
```

### 生产部署脚本

```bash
# 完整部署
./scripts/deployment/deploy.sh

# 仅构建镜像
./scripts/deployment/deploy.sh --build-only

# 停止所有服务
./scripts/deployment/deploy.sh --stop

# 查看服务日志
./scripts/deployment/deploy.sh --logs
```

## 🔍 故障排除

### 常见问题

1. **容器启动失败**
   ```bash
   # 查看详细日志
   ./scripts/docker/manage.sh <env> logs --service <service_name>
   
   # 检查容器状态
   docker ps -a
   ```

2. **数据库连接失败**
   ```bash
   # 检查数据库状态
   docker exec -it ai-travel-mysql-prod mysqladmin ping
   
   # 进入数据库容器
   ./scripts/docker/manage.sh prod exec mysql-prod
   ```

3. **端口冲突**
   ```bash
   # 检查端口占用
   netstat -tlnp | grep <port>
   
   # 修改配置文件中的端口映射
   ```

4. **内存不足**
   ```bash
   # 检查系统资源
   docker system df
   docker stats
   
   # 清理未使用资源
   docker system prune -f
   ```

### 日志位置

- **应用日志**: `/var/log/<service>/`
- **容器日志**: `docker logs <container_name>`
- **系统日志**: `/var/log/syslog`

## 🛡️ 安全配置

### 生产环境安全检查清单

- [ ] 修改所有默认密码
- [ ] 配置SSL/TLS证书
- [ ] 设置防火墙规则
- [ ] 启用访问日志记录
- [ ] 配置备份策略
- [ ] 设置监控告警
- [ ] 限制管理接口访问
- [ ] 定期安全更新

### 网络安全

```bash
# 防火墙配置示例 (UFW)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw deny 3306/tcp   # MySQL (仅内部访问)
sudo ufw deny 6379/tcp   # Redis (仅内部访问)
```

## 📊 性能优化

### 系统调优

```bash
# 增加文件描述符限制
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# 调整内核参数
echo "net.core.somaxconn = 65535" >> /etc/sysctl.conf
echo "vm.max_map_count = 262144" >> /etc/sysctl.conf
sysctl -p
```

### Docker优化

```bash
# 调整Docker守护进程配置
cat > /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2"
}
EOF

sudo systemctl restart docker
```

## 🔄 备份与恢复

### 数据备份

```bash
# MySQL备份
docker exec ai-travel-mysql-prod mysqldump -u root -p ai_travel_planner > backup.sql

# Redis备份
docker exec ai-travel-redis-prod redis-cli --rdb backup.rdb

# 文件备份
tar -czf data-backup.tar.gz data/
```

### 数据恢复

```bash
# MySQL恢复
docker exec -i ai-travel-mysql-prod mysql -u root -p ai_travel_planner < backup.sql

# Redis恢复
docker exec -i ai-travel-redis-prod redis-cli --pipe < backup.rdb
```