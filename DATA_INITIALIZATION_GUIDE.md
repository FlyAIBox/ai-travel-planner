# AI Travel Planner 数据初始化指南

## 📋 目录
- [概述](#概述)
- [初始化流程](#初始化流程)
- [数据库初始化](#数据库初始化)
- [向量数据库初始化](#向量数据库初始化)
- [知识库构建](#知识库构建)
- [验证步骤](#验证步骤)
- [故障排除](#故障排除)

## 🎯 概述

数据初始化是AI Travel Planner系统正常运行的关键步骤，包括：
- MySQL数据库表结构创建
- 基础数据导入
- Qdrant向量数据库集合创建
- 旅游知识库构建
- 用户权限和角色设置

## 🚀 初始化流程

### 自动初始化（推荐）
```bash
# 确保服务已启动
cd /path/to/ai-travel-planner

# 运行自动初始化脚本
python backend/scripts/init_system.py

# 或使用启动脚本（包含初始化）
./backend/scripts/start_system.sh
```

### 手动初始化
如果自动初始化失败，可以按以下步骤手动执行：

## 🗄️ 数据库初始化

### 1. 检查数据库连接
```bash
# 进入MySQL容器
docker compose -f deployment/docker/docker-compose.dev.yml exec mysql mysql -u root -p

# 或使用客户端连接
mysql -h localhost -P 3306 -u ai_travel_user -p ai_travel_db
```

### 2. 创建数据库表结构
```sql
-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    phone VARCHAR(20),
    avatar_url VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_email (email)
);

-- 用户配置表
CREATE TABLE IF NOT EXISTS user_preferences (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL,
    language VARCHAR(10) DEFAULT 'zh',
    timezone VARCHAR(50) DEFAULT 'Asia/Shanghai',
    currency VARCHAR(10) DEFAULT 'CNY',
    travel_style JSON,
    budget_range JSON,
    interests JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 对话会话表
CREATE TABLE IF NOT EXISTS chat_sessions (
    id VARCHAR(36) PRIMARY KEY,
    user_id BIGINT NOT NULL,
    title VARCHAR(200),
    status ENUM('active', 'archived', 'deleted') DEFAULT 'active',
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_status (status)
);

-- 对话消息表
CREATE TABLE IF NOT EXISTS chat_messages (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    role ENUM('user', 'assistant', 'system') NOT NULL,
    content TEXT NOT NULL,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at)
);

-- 旅行计划表
CREATE TABLE IF NOT EXISTS travel_plans (
    id VARCHAR(36) PRIMARY KEY,
    user_id BIGINT NOT NULL,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    destination VARCHAR(100),
    start_date DATE,
    end_date DATE,
    budget DECIMAL(10,2),
    status ENUM('draft', 'confirmed', 'completed', 'cancelled') DEFAULT 'draft',
    itinerary JSON,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_destination (destination),
    INDEX idx_status (status)
);

-- 知识库文档表
CREATE TABLE IF NOT EXISTS knowledge_documents (
    id VARCHAR(36) PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    source VARCHAR(100),
    category VARCHAR(50),
    language VARCHAR(10) DEFAULT 'zh',
    tags JSON,
    metadata JSON,
    checksum VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_category (category),
    INDEX idx_language (language),
    INDEX idx_checksum (checksum)
);

-- 系统配置表
CREATE TABLE IF NOT EXISTS system_config (
    id INT PRIMARY KEY AUTO_INCREMENT,
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value TEXT,
    description VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### 3. 插入基础数据
```sql
-- 插入系统配置
INSERT INTO system_config (config_key, config_value, description) VALUES
('system_version', '1.0.0', '系统版本'),
('default_language', 'zh', '默认语言'),
('max_chat_history', '100', '最大对话历史记录数'),
('vector_db_collection', 'travel_knowledge', '向量数据库集合名称'),
('embedding_model', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', '嵌入模型名称')
ON DUPLICATE KEY UPDATE 
config_value = VALUES(config_value),
updated_at = CURRENT_TIMESTAMP;

-- 创建默认管理员用户（密码: admin123）
INSERT INTO users (username, email, password_hash, full_name, is_active, is_verified) VALUES
('admin', 'admin@ai-travel-planner.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6hsxq5S/kS', '系统管理员', TRUE, TRUE)
ON DUPLICATE KEY UPDATE updated_at = CURRENT_TIMESTAMP;

-- 获取管理员用户ID并插入偏好设置
SET @admin_user_id = (SELECT id FROM users WHERE username = 'admin');
INSERT INTO user_preferences (user_id, language, timezone, currency, travel_style, budget_range, interests) VALUES
(@admin_user_id, 'zh', 'Asia/Shanghai', 'CNY', 
 '{"style": "comfortable", "pace": "moderate"}',
 '{"min": 1000, "max": 10000, "currency": "CNY"}',
 '["文化", "美食", "自然风光", "历史古迹"]')
ON DUPLICATE KEY UPDATE updated_at = CURRENT_TIMESTAMP;
```

## 🔍 向量数据库初始化

### 1. 创建Qdrant集合
```python
# 使用Python脚本创建集合
import requests
import json

# Qdrant配置
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "travel_knowledge"

# 创建集合
collection_config = {
    "vectors": {
        "size": 384,  # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 的向量维度
        "distance": "Cosine"
    },
    "optimizers_config": {
        "default_segment_number": 2
    },
    "replication_factor": 1
}

response = requests.put(
    f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
    json=collection_config
)

print(f"Collection creation status: {response.status_code}")
```

### 2. 验证集合创建
```bash
# 检查集合信息
curl -X GET "http://localhost:6333/collections/travel_knowledge"

# 检查集合统计
curl -X GET "http://localhost:6333/collections/travel_knowledge/cluster"
```

## 📚 知识库构建

### 1. 准备知识库数据
创建基础旅游知识数据文件 `data/knowledge/base_knowledge.json`：

```json
[
  {
    "title": "中国热门旅游城市介绍",
    "content": "北京是中国的首都，拥有丰富的历史文化遗产，包括故宫、天安门广场、长城等著名景点。上海是中国的经济中心，现代化程度很高，外滩、东方明珠塔是标志性景点。",
    "category": "destination",
    "tags": ["北京", "上海", "热门城市", "旅游景点"],
    "source": "system_init"
  },
  {
    "title": "旅行预算规划建议",
    "content": "制定旅行预算时需要考虑交通费、住宿费、餐饮费、景点门票和购物费用。一般来说，国内游人均日预算在200-800元不等，具体取决于目的地和旅行标准。",
    "category": "planning",
    "tags": ["预算", "规划", "费用", "建议"],
    "source": "system_init"
  },
  {
    "title": "最佳旅行时间指南",
    "content": "春季（3-5月）和秋季（9-11月）通常是大部分地区的最佳旅行时间，气候宜人，景色优美。夏季适合海滨和高原地区，冬季适合南方温暖地区和冰雪旅游。",
    "category": "timing",
    "tags": ["时间", "季节", "气候", "最佳时机"],
    "source": "system_init"
  }
]
```

### 2. 运行知识库构建脚本
```bash
# 构建知识库
python backend/scripts/build_knowledge_base.py

# 或使用RAG服务API
curl -X POST "http://localhost:8001/api/v1/knowledge/build" \
  -H "Content-Type: application/json" \
  -d '{"source_path": "data/knowledge/base_knowledge.json"}'
```

## ✅ 验证步骤

### 1. 数据库验证
```sql
-- 检查表是否创建成功
SHOW TABLES;

-- 检查用户数据
SELECT id, username, email, is_active FROM users;

-- 检查系统配置
SELECT * FROM system_config;
```

### 2. 向量数据库验证
```bash
# 检查集合状态
curl -X GET "http://localhost:6333/collections/travel_knowledge"

# 检查向量数量
curl -X GET "http://localhost:6333/collections/travel_knowledge/points/count"
```

### 3. 服务健康检查
```bash
# 检查所有服务状态
curl http://localhost:8080/gateway/health
curl http://localhost:8001/api/v1/health
curl http://localhost:8002/api/v1/health
curl http://localhost:8003/api/v1/health
curl http://localhost:8004/api/v1/health
curl http://localhost:8005/api/v1/health
```

### 4. 功能验证
```bash
# 测试用户注册
curl -X POST "http://localhost:8003/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "testpass123",
    "full_name": "测试用户"
  }'

# 测试用户登录
curl -X POST "http://localhost:8003/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123"
  }'

# 测试知识检索
curl -X POST "http://localhost:8001/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "北京旅游景点推荐",
    "limit": 5
  }'
```

## 🔧 故障排除

### 1. 数据库连接失败
```bash
# 检查MySQL容器状态
docker compose -f deployment/docker/docker-compose.dev.yml logs mysql

# 重启MySQL服务
docker compose -f deployment/docker/docker-compose.dev.yml restart mysql

# 检查网络连接
docker compose -f deployment/docker/docker-compose.dev.yml exec chat-service ping mysql
```

### 2. 向量数据库初始化失败
```bash
# 检查Qdrant容器状态
docker compose -f deployment/docker/docker-compose.dev.yml logs qdrant

# 重启Qdrant服务
docker compose -f deployment/docker/docker-compose.dev.yml restart qdrant

# 手动创建集合
curl -X PUT "http://localhost:6333/collections/travel_knowledge" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 384,
      "distance": "Cosine"
    }
  }'
```

### 3. 知识库构建失败
```bash
# 检查RAG服务日志
docker compose -f deployment/docker/docker-compose.dev.yml logs rag-service

# 检查知识库文件路径
ls -la data/knowledge/

# 手动重建知识库
python backend/scripts/build_knowledge_base.py --force-rebuild
```

### 4. 权限问题
```bash
# 检查数据目录权限
ls -la data/

# 修复权限
sudo chown -R $USER:$USER data/
chmod -R 755 data/
```

## 📊 初始化完成检查清单

- [ ] MySQL数据库表创建成功
- [ ] 基础数据插入完成
- [ ] 默认管理员用户创建
- [ ] Qdrant向量数据库集合创建
- [ ] 基础知识库数据导入
- [ ] 所有服务健康检查通过
- [ ] 用户注册/登录功能正常
- [ ] 知识检索功能正常
- [ ] 对话功能正常
- [ ] 旅行规划功能正常

## 📞 技术支持

如果初始化过程中遇到问题：
1. 检查相关服务日志
2. 验证环境变量配置
3. 确认网络连接正常
4. 查看系统资源使用情况
5. 参考故障排除章节

---

**注意**: 初始化过程可能需要几分钟时间，请耐心等待。建议在系统资源充足的环境下进行初始化。
