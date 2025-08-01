#!/bin/bash

# AI Travel Planner 部署脚本
# 支持开发、测试和生产环境的一键部署

set -e  # 遇到错误立即退出

# ==================== 配置变量 ====================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/deploy.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 默认配置
ENVIRONMENT="development"
SKIP_BUILD=false
SKIP_TESTS=false
FORCE_RECREATE=false
CLEANUP_VOLUMES=false
ENABLE_MONITORING=true
ENABLE_LOGGING=true

# ==================== 帮助函数 ====================

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$LOG_FILE"
}

# 显示帮助信息
show_help() {
    cat << EOF
AI Travel Planner 部署脚本

用法: $0 [选项]

选项:
    -e, --env ENV              设置环境 (development|staging|production) [默认: development]
    -s, --skip-build           跳过Docker镜像构建
    -t, --skip-tests           跳过测试运行
    -f, --force-recreate       强制重新创建所有容器
    -c, --cleanup-volumes      清理所有数据卷（危险操作！）
    --no-monitoring            禁用监控服务
    --no-logging               禁用日志服务
    -h, --help                 显示此帮助信息

环境说明:
    development               开发环境，启用热重载和调试功能
    staging                   测试环境，模拟生产配置
    production                生产环境，优化性能和安全性

示例:
    $0                        # 使用默认开发环境部署
    $0 -e production -f       # 生产环境强制重新部署
    $0 -s -t                  # 跳过构建和测试的快速部署

EOF
}

# ==================== 环境检查 ====================

check_prerequisites() {
    log "🔍 检查系统环境..."
    
    # 检查操作系统
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="windows"
    else
        error "不支持的操作系统: $OSTYPE"
        exit 1
    fi
    
    info "检测到操作系统: $OS"
    
    # 检查 Docker
    if ! command -v docker &> /dev/null; then
        error "Docker 未安装。请访问 https://docs.docker.com/get-docker/ 安装"
        exit 1
    fi
    
    # 检查 Docker 版本
    DOCKER_VERSION=$(docker --version | grep -oE '[0-9]+\.[0-9]+')
    REQUIRED_DOCKER_VERSION="20.10"
    
    if [ "$(printf '%s\n' "$REQUIRED_DOCKER_VERSION" "$DOCKER_VERSION" | sort -V | head -n1)" != "$REQUIRED_DOCKER_VERSION" ]; then
        warn "Docker 版本 ($DOCKER_VERSION) 可能过低，建议使用 $REQUIRED_DOCKER_VERSION 或更高版本"
    fi
    
    # 检查 Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose 未安装。请安装 Docker Compose"
        exit 1
    fi
    
    # 检查 Docker 守护进程
    if ! docker info &> /dev/null; then
        error "Docker 守护进程未运行。请启动 Docker"
        exit 1
    fi
    
    # 检查系统资源
    check_system_resources
    
    log "✅ 环境检查完成"
}

check_system_resources() {
    info "检查系统资源..."
    
    # 检查内存
    if [[ "$OS" == "linux" ]]; then
        TOTAL_MEM=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
        AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7/1024}')
    elif [[ "$OS" == "macos" ]]; then
        TOTAL_MEM=$(system_profiler SPHardwareDataType | awk '/Memory/ {print int($2)}')
        AVAILABLE_MEM=$TOTAL_MEM  # 简化处理
    fi
    
    info "系统内存: ${TOTAL_MEM}GB"
    
    if [ "$TOTAL_MEM" -lt 8 ]; then
        warn "系统内存不足 8GB，可能影响性能"
    fi
    
    # 检查磁盘空间
    AVAILABLE_DISK=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [ "$AVAILABLE_DISK" -lt 20 ]; then
        warn "磁盘可用空间不足 20GB，建议清理磁盘空间"
    fi
    
    info "可用磁盘空间: ${AVAILABLE_DISK}GB"
}

# ==================== 环境配置 ====================

setup_environment() {
    log "🔧 配置环境变量..."
    
    # 创建环境配置文件
    ENV_FILE="$PROJECT_ROOT/.env.${ENVIRONMENT}"
    
    if [ ! -f "$ENV_FILE" ]; then
        warn "环境配置文件不存在，创建默认配置: $ENV_FILE"
        create_env_file "$ENV_FILE"
    fi
    
    # 复制到主配置文件
    cp "$ENV_FILE" "$PROJECT_ROOT/.env"
    
    info "使用环境配置: $ENV_FILE"
}

create_env_file() {
    local env_file=$1
    
    cat > "$env_file" << EOF
# AI Travel Planner Environment Configuration
ENVIRONMENT=$ENVIRONMENT

# 数据库配置
DATABASE_URL=postgresql://travel_user:travel_password_2024@postgres:5432/ai_travel_planner
POSTGRES_DB=ai_travel_planner
POSTGRES_USER=travel_user
POSTGRES_PASSWORD=travel_password_2024

# Redis配置
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=redis_password_2024

# API密钥 (请在生产环境中修改)
OPENAI_API_KEY=your_openai_api_key_here
FLIGHT_API_KEY=your_flight_api_key_here
HOTEL_API_KEY=your_hotel_api_key_here
WEATHER_API_KEY=your_weather_api_key_here

# JWT配置
JWT_SECRET=your_jwt_secret_key_2024

# 应用配置
LOG_LEVEL=info
DEBUG=$( [ "$ENVIRONMENT" = "development" ] && echo "true" || echo "false" )

# 监控配置
ENABLE_MONITORING=$ENABLE_MONITORING
ENABLE_LOGGING=$ENABLE_LOGGING

# 服务端口配置
API_GATEWAY_PORT=8000
CHAT_SERVICE_PORT=8001
RAG_SERVICE_PORT=8002
AGENT_SERVICE_PORT=8003
PLANNING_SERVICE_PORT=8004
INTEGRATION_SERVICE_PORT=8005
USER_SERVICE_PORT=8006
FRONTEND_PORT=3000

EOF
    
    info "已创建环境配置文件: $env_file"
}

# ==================== 目录结构 ====================

setup_directories() {
    log "📁 创建必要的目录结构..."
    
    # 创建日志目录
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/logs/services"
    mkdir -p "$PROJECT_ROOT/logs/nginx"
    
    # 创建数据目录
    mkdir -p "$PROJECT_ROOT/data/documents"
    mkdir -p "$PROJECT_ROOT/data/uploads"
    mkdir -p "$PROJECT_ROOT/data/exports"
    
    # 创建配置目录
    mkdir -p "$PROJECT_ROOT/config/nginx"
    mkdir -p "$PROJECT_ROOT/config/redis"
    mkdir -p "$PROJECT_ROOT/config/qdrant"
    
    # 创建监控配置目录
    mkdir -p "$PROJECT_ROOT/monitoring/prometheus"
    mkdir -p "$PROJECT_ROOT/monitoring/grafana/dashboards"
    mkdir -p "$PROJECT_ROOT/monitoring/grafana/datasources"
    mkdir -p "$PROJECT_ROOT/monitoring/logstash/pipeline"
    
    # 创建SSL证书目录
    mkdir -p "$PROJECT_ROOT/ssl"
    
    # 设置权限
    chmod 755 "$PROJECT_ROOT/logs"
    chmod 755 "$PROJECT_ROOT/data"
    
    info "目录结构创建完成"
}

# ==================== 配置文件生成 ====================

generate_config_files() {
    log "📝 生成配置文件..."
    
    # Nginx配置
    generate_nginx_config
    
    # Prometheus配置
    generate_prometheus_config
    
    # Grafana数据源配置
    generate_grafana_config
    
    # Logstash配置
    generate_logstash_config
    
    info "配置文件生成完成"
}

generate_nginx_config() {
    cat > "$PROJECT_ROOT/nginx/nginx.conf" << 'EOF'
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                   '$status $body_bytes_sent "$http_referer" '
                   '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Gzip压缩
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;

    # 上游服务器配置
    upstream api_gateway {
        server api-gateway:8000;
    }

    upstream frontend {
        server frontend:3000;
    }

    # 主服务器配置
    server {
        listen 80;
        server_name localhost;

        # 前端应用
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # API路由
        location /api/ {
            proxy_pass http://api_gateway;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket支持
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # 健康检查
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
EOF
}

generate_prometheus_config() {
    cat > "$PROJECT_ROOT/monitoring/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'api-gateway'
    static_configs:
      - targets: ['api-gateway:8000']
    metrics_path: '/metrics'

  - job_name: 'chat-service'
    static_configs:
      - targets: ['chat-service:8001']
    metrics_path: '/metrics'

  - job_name: 'rag-service'
    static_configs:
      - targets: ['rag-service:8002']
    metrics_path: '/metrics'

  - job_name: 'agent-service'
    static_configs:
      - targets: ['agent-service:8003']
    metrics_path: '/metrics'

  - job_name: 'planning-service'
    static_configs:
      - targets: ['planning-service:8004']
    metrics_path: '/metrics'

  - job_name: 'integration-service'
    static_configs:
      - targets: ['integration-service:8005']
    metrics_path: '/metrics'

  - job_name: 'user-service'
    static_configs:
      - targets: ['user-service:8006']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
EOF
}

generate_grafana_config() {
    # Grafana数据源配置
    cat > "$PROJECT_ROOT/monitoring/grafana/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    # Grafana仪表板配置
    cat > "$PROJECT_ROOT/monitoring/grafana/dashboards/dashboard.yml" << 'EOF'
apiVersion: 1

providers:
  - name: 'ai-travel-planner'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /etc/grafana/provisioning/dashboards
EOF
}

generate_logstash_config() {
    cat > "$PROJECT_ROOT/monitoring/logstash/pipeline/logstash.conf" << 'EOF'
input {
  file {
    path => "/usr/share/logstash/logs/*.log"
    start_position => "beginning"
    sincedb_path => "/dev/null"
  }
}

filter {
  if [message] =~ /^\[/ {
    grok {
      match => { "message" => "\[%{TIMESTAMP_ISO8601:timestamp}\] %{WORD:level}: %{GREEDYDATA:msg}" }
    }
    
    date {
      match => [ "timestamp", "yyyy-MM-dd HH:mm:ss" ]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch-log:9200"]
    index => "ai-travel-planner-%{+YYYY.MM.dd}"
  }
  
  stdout {
    codec => rubydebug
  }
}
EOF
}

# ==================== 构建和部署 ====================

build_images() {
    if [ "$SKIP_BUILD" = true ]; then
        info "跳过Docker镜像构建"
        return
    fi
    
    log "🔨 构建Docker镜像..."
    
    # 获取Git提交哈希作为标签
    GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
    
    # 构建参数
    BUILD_ARGS="--build-arg ENVIRONMENT=$ENVIRONMENT --build-arg BUILD_VERSION=$GIT_HASH"
    
    # 构建服务镜像
    info "构建后端服务镜像..."
    docker-compose build $BUILD_ARGS
    
    # 构建前端镜像
    info "构建前端镜像..."
    cd "$PROJECT_ROOT/frontend"
    docker build -t ai-travel-frontend:$GIT_HASH .
    cd "$PROJECT_ROOT"
    
    log "✅ Docker镜像构建完成"
}

run_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        info "跳过测试运行"
        return
    fi
    
    log "🧪 运行测试..."
    
    # 运行后端测试
    info "运行后端测试..."
    python -m pytest tests/ -v --tb=short
    
    # 运行前端测试
    info "运行前端测试..."
    cd "$PROJECT_ROOT/frontend"
    npm test -- --watchAll=false --coverage
    cd "$PROJECT_ROOT"
    
    log "✅ 测试运行完成"
}

deploy_services() {
    log "🚀 部署服务..."
    
    # Docker Compose参数
    COMPOSE_ARGS=""
    
    if [ "$FORCE_RECREATE" = true ]; then
        COMPOSE_ARGS="$COMPOSE_ARGS --force-recreate"
    fi
    
    if [ "$CLEANUP_VOLUMES" = true ]; then
        warn "清理数据卷（这将删除所有数据！）"
        docker-compose down -v
        COMPOSE_ARGS="$COMPOSE_ARGS --renew-anon-volumes"
    fi
    
    # 部署核心服务
    info "启动核心服务..."
    docker-compose up -d postgres redis qdrant $COMPOSE_ARGS
    
    # 等待数据库启动
    wait_for_service "postgres" "5432"
    wait_for_service "redis" "6379"
    wait_for_service "qdrant" "6333"
    
    # 运行数据库迁移
    run_migrations
    
    # 启动后端服务
    info "启动后端服务..."
    docker-compose up -d \
        api-gateway \
        chat-service \
        rag-service \
        agent-service \
        planning-service \
        integration-service \
        user-service \
        $COMPOSE_ARGS
    
    # 启动前端服务
    info "启动前端服务..."
    docker-compose up -d frontend $COMPOSE_ARGS
    
    # 启动监控服务
    if [ "$ENABLE_MONITORING" = true ]; then
        info "启动监控服务..."
        docker-compose up -d prometheus grafana jaeger $COMPOSE_ARGS
    fi
    
    # 启动日志服务
    if [ "$ENABLE_LOGGING" = true ]; then
        info "启动日志服务..."
        docker-compose up -d elasticsearch-log logstash kibana $COMPOSE_ARGS
    fi
    
    # 启动工具服务
    info "启动工具服务..."
    docker-compose up -d nginx minio adminer redis-commander $COMPOSE_ARGS
    
    # 启动任务队列
    info "启动任务队列..."
    docker-compose up -d celery-worker celery-beat flower $COMPOSE_ARGS
    
    log "✅ 服务部署完成"
}

run_migrations() {
    log "📊 运行数据库迁移..."
    
    # 等待PostgreSQL完全启动
    sleep 10
    
    # 运行迁移脚本
    docker-compose exec -T postgres psql -U travel_user -d ai_travel_planner -c "
        CREATE SCHEMA IF NOT EXISTS travel;
        
        -- 用户表
        CREATE TABLE IF NOT EXISTS travel.users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 旅行计划表
        CREATE TABLE IF NOT EXISTS travel.plans (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES travel.users(id),
            name VARCHAR(200) NOT NULL,
            description TEXT,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            status VARCHAR(20) DEFAULT 'draft',
            total_cost DECIMAL(10,2) DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 会话表
        CREATE TABLE IF NOT EXISTS travel.conversations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES travel.users(id),
            title VARCHAR(200),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 创建索引
        CREATE INDEX IF NOT EXISTS idx_users_email ON travel.users(email);
        CREATE INDEX IF NOT EXISTS idx_plans_user_id ON travel.plans(user_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON travel.conversations(user_id);
    "
    
    info "数据库迁移完成"
}

# ==================== 健康检查 ====================

wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    info "等待服务 $service:$port 启动..."
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose exec -T "$service" nc -z localhost "$port" 2>/dev/null; then
            info "服务 $service:$port 已启动"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    error "服务 $service:$port 启动失败"
    return 1
}

health_check() {
    log "🏥 执行健康检查..."
    
    local services=(
        "http://localhost:8000/health:API网关"
        "http://localhost:8001/health:聊天服务"
        "http://localhost:8002/health:RAG服务"
        "http://localhost:8003/health:智能体服务"
        "http://localhost:8004/health:规划服务"
        "http://localhost:8005/health:集成服务"
        "http://localhost:8006/health:用户服务"
        "http://localhost:3000:前端应用"
    )
    
    local failed_services=()
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r url name <<< "$service_info"
        
        if curl -f -s --max-time 10 "$url" >/dev/null 2>&1; then
            info "✅ $name - 健康"
        else
            error "❌ $name - 不健康"
            failed_services+=("$name")
        fi
    done
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        log "🎉 所有服务健康检查通过"
        return 0
    else
        error "以下服务健康检查失败: ${failed_services[*]}"
        return 1
    fi
}

# ==================== 部署后配置 ====================

post_deploy_setup() {
    log "⚙️  执行部署后配置..."
    
    # 创建默认用户
    create_default_user
    
    # 初始化向量数据库
    initialize_vector_database
    
    # 设置监控告警
    if [ "$ENABLE_MONITORING" = true ]; then
        setup_monitoring_alerts
    fi
    
    info "部署后配置完成"
}

create_default_user() {
    info "创建默认管理员用户..."
    
    # 通过API创建默认用户
    curl -s -X POST http://localhost:8000/api/v1/users/register \
        -H "Content-Type: application/json" \
        -d '{
            "username": "admin",
            "email": "admin@example.com",
            "password": "admin123"
        }' || warn "默认用户创建失败，可能已存在"
}

initialize_vector_database() {
    info "初始化向量数据库..."
    
    # 创建默认集合
    curl -s -X POST http://localhost:6333/collections/travel_documents \
        -H "Content-Type: application/json" \
        -d '{
            "vectors": {
                "size": 768,
                "distance": "Cosine"
            }
        }' || warn "向量数据库初始化失败"
}

setup_monitoring_alerts() {
    info "设置监控告警..."
    # 这里可以添加Prometheus告警规则配置
    # 或者Grafana告警配置
}

# ==================== 显示信息 ====================

show_deployment_info() {
    log "📋 部署信息"
    
    echo ""
    echo "🌐 访问地址:"
    echo "  主应用:          http://localhost"
    echo "  API网关:         http://localhost:8000"
    echo "  前端应用:        http://localhost:3000"
    echo ""
    echo "🔧 管理工具:"
    echo "  数据库管理:      http://localhost:8080"
    echo "  Redis管理:       http://localhost:8081"
    echo "  对象存储:        http://localhost:9001"
    echo ""
    
    if [ "$ENABLE_MONITORING" = true ]; then
        echo "📊 监控服务:"
        echo "  Prometheus:      http://localhost:9090"
        echo "  Grafana:         http://localhost:3001 (admin/admin123)"
        echo "  Jaeger:          http://localhost:16686"
        echo "  Flower:          http://localhost:5555"
        echo ""
    fi
    
    if [ "$ENABLE_LOGGING" = true ]; then
        echo "📝 日志服务:"
        echo "  Kibana:          http://localhost:5601"
        echo ""
    fi
    
    echo "🔑 默认凭据:"
    echo "  管理员用户:      admin / admin123"
    echo "  数据库:          travel_user / travel_password_2024"
    echo "  Redis:           redis_password_2024"
    echo "  MinIO:           minioadmin / minioadmin123"
    echo ""
    
    echo "📁 重要路径:"
    echo "  日志目录:        $PROJECT_ROOT/logs"
    echo "  数据目录:        $PROJECT_ROOT/data"
    echo "  配置目录:        $PROJECT_ROOT/config"
    echo ""
    
    echo "🛠️  常用命令:"
    echo "  查看日志:        docker-compose logs -f [service]"
    echo "  重启服务:        docker-compose restart [service]"
    echo "  停止所有服务:    docker-compose down"
    echo "  更新服务:        $0 -f"
    echo ""
}

# ==================== 清理函数 ====================

cleanup_on_exit() {
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        error "部署过程中发生错误，退出码: $exit_code"
        echo ""
        echo "🔍 故障排除建议:"
        echo "  1. 查看日志: tail -f $LOG_FILE"
        echo "  2. 检查服务状态: docker-compose ps"
        echo "  3. 查看服务日志: docker-compose logs [service]"
        echo "  4. 重新部署: $0 -f"
        echo ""
    fi
    
    exit $exit_code
}

# ==================== 参数解析 ====================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -s|--skip-build)
                SKIP_BUILD=true
                shift
                ;;
            -t|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -f|--force-recreate)
                FORCE_RECREATE=true
                shift
                ;;
            -c|--cleanup-volumes)
                CLEANUP_VOLUMES=true
                shift
                ;;
            --no-monitoring)
                ENABLE_MONITORING=false
                shift
                ;;
            --no-logging)
                ENABLE_LOGGING=false
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 验证环境参数
    if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
        error "无效的环境: $ENVIRONMENT"
        exit 1
    fi
}

# ==================== 主函数 ====================

main() {
    # 设置信号处理
    trap cleanup_on_exit EXIT
    
    # 初始化日志
    echo "AI Travel Planner 部署开始 - $(date)" > "$LOG_FILE"
    
    log "🚀 AI Travel Planner 部署脚本启动"
    log "环境: $ENVIRONMENT"
    
    # 检查并创建必要目录
    setup_directories
    
    # 检查系统环境
    check_prerequisites
    
    # 配置环境
    setup_environment
    
    # 生成配置文件
    generate_config_files
    
    # 构建镜像
    build_images
    
    # 运行测试
    run_tests
    
    # 部署服务
    deploy_services
    
    # 健康检查
    sleep 30  # 等待服务完全启动
    health_check
    
    # 部署后配置
    post_deploy_setup
    
    # 显示部署信息
    show_deployment_info
    
    log "🎉 AI Travel Planner 部署完成！"
}

# ==================== 脚本入口 ====================

# 解析命令行参数
parse_arguments "$@"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 执行主函数
main 