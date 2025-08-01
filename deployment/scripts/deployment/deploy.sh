
#!/bin/bash

# ==================== 部署脚本 ====================
# AI Travel Planner - 一键部署

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# 检查必要工具
check_requirements() {
    log_step "检查系统要求..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    
    # 检查Docker Compose (优先使用新版本)
    if command -v docker &> /dev/null && docker compose version &> /dev/null; then
        log_info "使用 Docker Compose v2"
        DOCKER_COMPOSE_CMD="docker compose"
    elif command -v docker compose &> /dev/null; then
        log_info "使用 Docker Compose v1"
        DOCKER_COMPOSE_CMD="docker-compose"
    else
        log_error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    fi
    
    # 检查Docker是否运行
    if ! docker info &> /dev/null; then
        log_error "Docker 服务未启动，请先启动 Docker 服务"
        exit 1
    fi
    
    log_success "系统要求检查通过"
}

# 加载环境变量
load_environment() {
    log_step "加载环境变量..."

    # 检查 .env 文件是否存在
    if [[ -f ".env" ]]; then
        log_info "发现 .env 文件，正在加载环境变量..."
        # 打印 .env 文件的绝对路径，便于用户确认加载的环境文件位置
        echo "加载的 .env 文件路径: $(realpath .env)"
        # 安全地导出 .env 文件中的环境变量（忽略注释和空行）
        while IFS= read -r line; do
            # 跳过空行和注释行
            if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
                # 检查是否是有效的环境变量格式 (KEY=VALUE)
                if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
                    export "$line"
                    # echo "export $line"
                fi
            fi
        done < .env
        log_success "环境变量加载完成"
    else
        log_warning "未找到 .env 文件，请确保环境变量已通过其他方式设置"
    fi
}

# 环境变量检查
check_environment() {
    log_step "检查环境变量..."
    
    # 必需的环境变量
    REQUIRED_VARS=(
        "MYSQL_ROOT_PASSWORD"
        "MYSQL_PASSWORD"
        "JWT_SECRET"
        "OPENAI_API_KEY"
        "N8N_BASIC_AUTH_PASSWORD"
        "GRAFANA_ADMIN_PASSWORD"
    )
    
    MISSING_VARS=()
    
    for var in "${REQUIRED_VARS[@]}"; do
        if [[ -z "${!var}" ]]; then
            MISSING_VARS+=("$var")
        fi
    done
    
    if [[ ${#MISSING_VARS[@]} -gt 0 ]]; then
        log_error "缺少必需的环境变量:"
        for var in "${MISSING_VARS[@]}"; do
            echo "  - $var"
        done
        log_info "请在 .env 文件中设置这些变量，或者使用 export 命令设置"
        exit 1
    fi
    
    log_success "环境变量检查通过"
}

# 创建必要目录
create_directories() {
    log_step "创建必要目录..."
    
    # 数据目录
    mkdir -p data/{mysql,redis,qdrant,elasticsearch,n8n,prometheus,grafana}
    
    # 日志目录
    mkdir -p logs/{api,chat,agent,rag,user,nginx}
    
    # 配置目录
    mkdir -p deployment/{mysql,redis,qdrant,nginx,prometheus,grafana,filebeat}
    
    log_success "目录创建完成"
}

# 停止现有容器
stop_existing() {
    log_step "停止现有容器..."
    
    if $DOCKER_COMPOSE_CMD -f deployment/docker/docker-compose.yml ps -q | grep -q .; then
        $DOCKER_COMPOSE_CMD -f deployment/docker/docker-compose.yml down
        log_info "已停止现有容器"
    else
        log_info "没有运行中的容器"
    fi
}

# 拉取最新镜像
pull_images() {
    log_step "拉取最新镜像..."
    
    $DOCKER_COMPOSE_CMD -f deployment/docker/docker-compose.yml pull
    
    log_success "镜像拉取完成"
}

# 构建应用镜像
build_images() {
    log_step "构建应用镜像..."
    
    $DOCKER_COMPOSE_CMD -f deployment/docker/docker-compose.yml build --no-cache
    
    log_success "应用镜像构建完成"
}

# 启动数据库服务
start_databases() {
    log_step "启动数据库服务..."
    
    # 先启动数据库服务
    $DOCKER_COMPOSE_CMD -f deployment/docker/docker-compose.yml up -d \
        mysql-prod redis-prod qdrant-prod elasticsearch-prod
    
    log_info "等待数据库服务就绪..."
    
    # 等待MySQL就绪
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if $DOCKER_COMPOSE_CMD -f deployment/docker/docker-compose.yml exec -T mysql-prod mysqladmin ping -h localhost --silent; then
            log_success "MySQL 服务就绪"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "MySQL 服务启动超时"
            exit 1
        fi
        
        echo -n "."
        sleep 5
        ((attempt++))
    done
    
    # 等待Redis就绪
    attempt=1
    while [[ $attempt -le $max_attempts ]]; do
        if $DOCKER_COMPOSE_CMD -f deployment/docker/docker-compose.yml exec -T redis-prod redis-cli ping | grep -q PONG; then
            log_success "Redis 服务就绪"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Redis 服务启动超时"
            exit 1
        fi
        
        echo -n "."
        sleep 3
        ((attempt++))
    done
    
    log_success "数据库服务启动完成"
}

# 数据库初始化
init_database() {
    log_step "初始化数据库..."
    
    # 运行数据库迁移
    $DOCKER_COMPOSE_CMD -f deployment/docker/docker-compose.yml run --rm api-gateway-prod \
        python scripts/database/init_db.py init
    
    log_success "数据库初始化完成"
}

# 启动应用服务
start_applications() {
    log_step "启动应用服务..."
    
    # 启动核心应用服务
    $DOCKER_COMPOSE_CMD -f deployment/docker/docker-compose.yml up -d \
        api-gateway-prod chat-service-prod agent-service-prod \
        rag-service-prod user-service-prod
    
    log_info "等待应用服务就绪..."
    sleep 30
    
    # 检查服务健康状态
    local services=("api-gateway-prod:8080" "chat-service-prod:8001" "agent-service-prod:8002" "rag-service-prod:8003" "user-service-prod:8004")
    
    for service in "${services[@]}"; do
        local service_name=$(echo $service | cut -d: -f1)
        local port=$(echo $service | cut -d: -f2)
        
        if $DOCKER_COMPOSE_CMD -f deployment/docker/docker-compose.yml exec -T $service_name curl -f http://localhost:$port/health &> /dev/null; then
            log_success "$service_name 服务健康"
        else
            log_warning "$service_name 服务可能未就绪"
        fi
    done
    
    log_success "应用服务启动完成"
}

# 启动工作流和监控服务
start_workflow_monitoring() {
    log_step "启动工作流和监控服务..."
    
    $DOCKER_COMPOSE_CMD -f deployment/docker/docker-compose.yml up -d \
        n8n-prod prometheus-prod grafana-prod filebeat-prod
    
    log_success "工作流和监控服务启动完成"
}

# 启动负载均衡
start_loadbalancer() {
    log_step "启动负载均衡..."
    
    $DOCKER_COMPOSE_CMD -f deployment/docker/docker-compose.yml up -d nginx-prod
    
    log_success "负载均衡启动完成"
}

# 健康检查
health_check() {
    log_step "系统健康检查..."
    
    local endpoints=(
        "http://localhost/health"
        "http://localhost/api/health"
        "http://localhost/chat/health"
        "http://localhost/agent/health"
        "http://localhost/rag/health"
        "http://localhost/users/health"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -f "$endpoint" &> /dev/null; then
            log_success "✓ $endpoint"
        else
            log_warning "✗ $endpoint"
        fi
    done
    
    log_success "健康检查完成"
}

# 显示部署信息
show_deployment_info() {
    log_step "部署信息"
    
    echo -e "${CYAN}==================== 部署完成 ====================${NC}"
    echo -e "${GREEN}🎉 AI Travel Planner 部署成功！${NC}"
    echo ""
    echo -e "${YELLOW}服务访问地址:${NC}"
    echo -e "  🌐 主入口:          http://localhost"
    echo -e "  🔌 API网关:         http://localhost/api"
    echo -e "  💬 聊天服务:        http://localhost/chat"
    echo -e "  🤖 智能体服务:      http://localhost/agent"
    echo -e "  📚 RAG服务:         http://localhost/rag"
    echo -e "  👤 用户服务:        http://localhost/users"
    echo -e "  🔧 工作流管理:      http://localhost/workflow"
    echo -e "  📊 监控面板:        http://localhost/grafana"
    echo ""
    echo -e "${YELLOW}管理工具:${NC}"
    echo -e "  📈 Prometheus:      http://localhost:9090"
    echo -e "  📊 Grafana:         http://localhost:3000"
    echo -e "  🔄 n8n:             http://localhost:5678"
    echo ""
    echo -e "${YELLOW}数据库:${NC}"
    echo -e "  🗄️  MySQL:          localhost:3306"
    echo -e "  🔴 Redis:           localhost:6379"
    echo -e "  🔍 Qdrant:          localhost:6333"
    echo -e "  🔎 Elasticsearch:   localhost:9200"
    echo ""
    echo -e "${RED}⚠️  重要提醒:${NC}"
    echo -e "  • 请确保防火墙已正确配置"
    echo -e "  • 建议启用 HTTPS 证书"
    echo -e "  • 定期备份数据库和配置文件"
    echo -e "  • 监控系统资源使用情况"
    echo ""
    echo -e "${CYAN}=================================================${NC}"
}

# 主函数
main() {
    echo -e "${CYAN}==================== AI Travel Planner ====================${NC}"
    echo -e "${GREEN}🚀 开始部署${NC}"
    echo ""
    
    # 检查参数
    if [[ $# -gt 0 ]]; then
        case "$1" in
            --build-only)
                log_info "仅构建模式"
                check_requirements
                build_images
                log_success "构建完成"
                exit 0
                ;;
            --stop)
                log_info "停止所有服务"
                stop_existing
                log_success "服务已停止"
                exit 0
                ;;
            --logs)
                log_info "查看服务日志"
                $DOCKER_COMPOSE_CMD -f deployment/docker/docker-compose.yml logs -f
                exit 0
                ;;
            --help|-h)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --build-only    仅构建镜像"
                echo "  --stop          停止所有服务"
                echo "  --logs          查看服务日志"
                echo "  --help, -h      显示帮助信息"
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                echo "使用 $0 --help 查看帮助信息"
                exit 1
                ;;
        esac
    fi
    
    # 执行部署步骤
    check_requirements
    load_environment
    check_environment
    create_directories
    stop_existing
    pull_images
    build_images
    start_databases
    init_database
    start_applications
    start_workflow_monitoring
    start_loadbalancer
    health_check
    show_deployment_info
    
    log_success "🎉 部署完成！"
}

# 执行主函数
main "$@" 
