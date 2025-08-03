#!/bin/bash

# AI Travel Planner 部署脚本
# 支持开发、测试、生产环境的一键部署

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 项目信息
PROJECT_NAME="ai-travel-planner"
PROJECT_VERSION="2.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 默认配置
ENVIRONMENT="development"
SERVICES="all"
ENABLE_MONITORING=true
ENABLE_LOGGING=true
FORCE_REBUILD=false
SKIP_TESTS=false
QUICK_START=false

# 显示帮助信息
show_help() {
    cat << EOF
AI Travel Planner 部署脚本 v${PROJECT_VERSION}

用法: $0 [选项]

选项:
  -e, --environment <env>     指定环境 (development|testing|production) [默认: development]
  -s, --services <services>   指定要部署的服务 (all|core|frontend|monitoring) [默认: all]
  -m, --enable-monitoring     启用监控系统 (prometheus, grafana) [默认: true]
  -l, --enable-logging        启用日志系统 (elk stack) [默认: true]
  -f, --force-rebuild         强制重新构建所有镜像
  -t, --skip-tests           跳过测试
  -q, --quick-start          快速启动 (仅核心服务)
  -h, --help                 显示此帮助信息

示例:
  $0                                          # 开发环境完整部署
  $0 -e production -f                        # 生产环境强制重新构建
  $0 -e testing -s core --skip-tests        # 测试环境仅部署核心服务
  $0 -q                                      # 快速启动核心服务

环境说明:
  development - 开发环境，启用热重载和调试功能
  testing     - 测试环境，包含测试数据和工具
  production  - 生产环境，优化性能和安全性

服务说明:
  all        - 所有服务 (核心服务 + 前端 + 监控 + 日志)
  core       - 核心服务 (API网关, 聊天服务, RAG服务等)
  frontend   - 前端服务
  monitoring - 监控服务 (Prometheus, Grafana)
  logging    - 日志服务 (ELK Stack)

EOF
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -s|--services)
                SERVICES="$2"
                shift 2
                ;;
            -m|--enable-monitoring)
                ENABLE_MONITORING=true
                shift
                ;;
            --disable-monitoring)
                ENABLE_MONITORING=false
                shift
                ;;
            -l|--enable-logging)
                ENABLE_LOGGING=true
                shift
                ;;
            --disable-logging)
                ENABLE_LOGGING=false
                shift
                ;;
            -f|--force-rebuild)
                FORCE_REBUILD=true
                shift
                ;;
            -t|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -q|--quick-start)
                QUICK_START=true
                SERVICES="core"
                ENABLE_MONITORING=false
                ENABLE_LOGGING=false
                SKIP_TESTS=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}未知选项: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
}

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

# 显示项目信息
show_project_info() {
    cat << EOF

${CYAN}╔══════════════════════════════════════════════════════════════╗
║                  AI Travel Planner 部署系统                   ║
║                                                              ║
║  版本: ${PROJECT_VERSION}                                         ║
║  环境: ${ENVIRONMENT}                                        ║
║  服务: ${SERVICES}                                           ║
║  监控: $([ "$ENABLE_MONITORING" = true ] && echo "启用" || echo "禁用")                                            ║
║  日志: $([ "$ENABLE_LOGGING" = true ] && echo "启用" || echo "禁用")                                            ║
╚══════════════════════════════════════════════════════════════╝${NC}

EOF
}

# 检查依赖
check_dependencies() {
    log_step "检查系统依赖"
    
    local deps=("docker" "docker-compose" "git")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "缺少以下依赖: ${missing_deps[*]}"
        log_info "请安装缺少的依赖后重试"
        exit 1
    fi
    
    # 检查 Docker 服务状态
    if ! docker info &> /dev/null; then
        log_error "Docker 服务未启动"
        log_info "请启动 Docker 服务: sudo systemctl start docker"
        exit 1
    fi
    
    # 检查 Docker Compose 版本
    local compose_version
    compose_version=$(docker-compose --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    log_info "Docker Compose 版本: $compose_version"
    
    log_success "依赖检查完成"
}

# 设置环境变量
setup_environment() {
    log_step "设置环境变量"
    
    local env_file="$PROJECT_ROOT/.env"
    local env_template="$PROJECT_ROOT/.env.example"
    
    # 如果没有 .env 文件，从模板创建
    if [ ! -f "$env_file" ]; then
        if [ -f "$env_template" ]; then
            cp "$env_template" "$env_file"
            log_info "从模板创建 .env 文件"
        else
            log_warning ".env 文件不存在，使用默认配置"
        fi
    fi
    
    # 根据环境设置特定变量
    export COMPOSE_PROJECT_NAME="${PROJECT_NAME}-${ENVIRONMENT}"
    export ENVIRONMENT
    export PROJECT_VERSION
    
    case $ENVIRONMENT in
        development)
            export DEBUG=true
            export LOG_LEVEL=debug
            export RELOAD=true
            ;;
        testing)
            export DEBUG=true
            export LOG_LEVEL=info
            export RELOAD=false
            ;;
        production)
            export DEBUG=false
            export LOG_LEVEL=error
            export RELOAD=false
            ;;
    esac
    
    log_success "环境变量设置完成"
}

# 创建必要目录
create_directories() {
    log_step "创建必要目录"
    
    local dirs=(
        "$PROJECT_ROOT/logs"
        "$PROJECT_ROOT/data/postgres"
        "$PROJECT_ROOT/data/redis"
        "$PROJECT_ROOT/data/qdrant"
        "$PROJECT_ROOT/data/elasticsearch"
        "$PROJECT_ROOT/data/prometheus"
        "$PROJECT_ROOT/data/grafana"
        "$PROJECT_ROOT/deployment/monitoring"
        "$PROJECT_ROOT/deployment/logging"
        "$PROJECT_ROOT/deployment/nginx/conf.d"
        "$PROJECT_ROOT/deployment/nginx/ssl"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
    
    log_success "目录创建完成"
}

# 运行测试
run_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        log_warning "跳过测试"
        return 0
    fi
    
    log_step "运行测试"
    
    cd "$PROJECT_ROOT"
    
    # 运行单元测试
    if [ -f "backend/scripts/run_tests.py" ]; then
        python backend/scripts/run_tests.py
    else
        log_warning "测试脚本不存在，跳过测试"
    fi
    
    log_success "测试完成"
}

# 构建 Docker 镜像
build_images() {
    log_step "构建 Docker 镜像"
    
    cd "$PROJECT_ROOT/deployment"
    
    local build_args=""
    if [ "$FORCE_REBUILD" = true ]; then
        build_args="--no-cache"
        log_info "强制重新构建所有镜像"
    fi
    
    # 根据服务类型构建不同的镜像
    case $SERVICES in
        all)
            docker-compose build $build_args
            ;;
        core)
            docker-compose build $build_args api-gateway chat-service rag-service agent-service planning-service integration-service user-service
            ;;
        frontend)
            docker-compose build $build_args frontend
            ;;
        monitoring)
            log_info "监控服务使用官方镜像，无需构建"
            ;;
        logging)
            log_info "日志服务使用官方镜像，无需构建"
            ;;
    esac
    
    log_success "镜像构建完成"
}

# 启动服务
start_services() {
    log_step "启动服务"
    
    cd "$PROJECT_ROOT/deployment"
    
    # 构建服务列表
    local service_list=()
    
    case $SERVICES in
        all)
            service_list+=(
                "postgres" "redis" "qdrant"
                "api-gateway" "chat-service" "rag-service" 
                "agent-service" "planning-service" "integration-service" "user-service"
                "frontend"
            )
            
            if [ "$ENABLE_MONITORING" = true ]; then
                service_list+=("prometheus" "grafana" "node-exporter" "cadvisor")
            fi
            
            if [ "$ENABLE_LOGGING" = true ]; then
                service_list+=("elasticsearch" "kibana" "logstash" "filebeat")
            fi
            ;;
        core)
            service_list+=(
                "postgres" "redis" "qdrant"
                "api-gateway" "chat-service" "rag-service"
                "agent-service" "planning-service" "integration-service" "user-service"
            )
            ;;
        frontend)
            service_list+=("frontend")
            ;;
        monitoring)
            service_list+=("prometheus" "grafana" "node-exporter" "cadvisor")
            ;;
        logging)
            service_list+=("elasticsearch" "kibana" "logstash" "filebeat")
            ;;
    esac
    
    # 分阶段启动服务
    log_info "启动基础服务..."
    docker-compose up -d postgres redis qdrant
    
    # 等待基础服务就绪
    wait_for_service "postgres" 5432 60
    wait_for_service "redis" 6379 30
    wait_for_service "qdrant" 6333 60
    
    if [[ " ${service_list[*]} " =~ "elasticsearch" ]]; then
        log_info "启动日志服务..."
        docker-compose up -d elasticsearch
        wait_for_service "elasticsearch" 9200 120
        docker-compose up -d kibana logstash filebeat
    fi
    
    if [[ " ${service_list[*]} " =~ "prometheus" ]]; then
        log_info "启动监控服务..."
        docker-compose up -d prometheus grafana node-exporter cadvisor
    fi
    
    log_info "启动应用服务..."
    for service in api-gateway chat-service rag-service agent-service planning-service integration-service user-service; do
        if [[ " ${service_list[*]} " =~ "$service" ]]; then
            docker-compose up -d "$service"
        fi
    done
    
    if [[ " ${service_list[*]} " =~ "frontend" ]]; then
        log_info "启动前端服务..."
        docker-compose up -d frontend
    fi
    
    log_success "服务启动完成"
}

# 等待服务就绪
wait_for_service() {
    local service_name=$1
    local port=$2
    local timeout=${3:-30}
    local count=0
    
    log_info "等待 $service_name 服务就绪..."
    
    while [ $count -lt $timeout ]; do
        if docker-compose exec -T "$service_name" sh -c "nc -z localhost $port" 2>/dev/null; then
            log_success "$service_name 服务已就绪"
            return 0
        fi
        
        sleep 1
        count=$((count + 1))
        
        if [ $((count % 10)) -eq 0 ]; then
            log_info "等待 $service_name 服务... ($count/$timeout)"
        fi
    done
    
    log_warning "$service_name 服务启动超时，但继续部署"
    return 1
}

# 健康检查
health_check() {
    log_step "执行健康检查"
    
    local services
    services=$(docker-compose ps --services --filter "status=running")
    
    echo "服务状态检查:"
    echo "============================================"
    
    for service in $services; do
        local status
        status=$(docker-compose ps "$service" --format "table {{.Status}}" | tail -n 1)
        
        if [[ $status == *"Up"* ]]; then
            echo -e "${GREEN}✓${NC} $service: $status"
        else
            echo -e "${RED}✗${NC} $service: $status"
        fi
    done
    
    echo "============================================"
    
    # 检查关键端点
    log_info "检查关键端点..."
    local endpoints=(
        "http://localhost:8000/health:API网关"
        "http://localhost:3000:前端应用"
    )
    
    if [ "$ENABLE_MONITORING" = true ]; then
        endpoints+=("http://localhost:9090:Prometheus")
        endpoints+=("http://localhost:3001:Grafana")
    fi
    
    if [ "$ENABLE_LOGGING" = true ]; then
        endpoints+=("http://localhost:9200:Elasticsearch")
        endpoints+=("http://localhost:5601:Kibana")
    fi
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r url name <<< "$endpoint_info"
        
        if curl -s --max-time 5 "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} $name: $url"
        else
            echo -e "${YELLOW}⚠${NC} $name: $url (可能仍在启动中)"
        fi
    done
    
    log_success "健康检查完成"
}

# 显示访问信息
show_access_info() {
    log_step "部署完成！"
    
    cat << EOF

${GREEN}🎉 AI Travel Planner 部署成功！${NC}

${CYAN}📋 服务访问地址:${NC}
┌─────────────────────────────────────────────────────────────┐
│ 🌐 前端应用:        http://localhost:3000                    │
│ 🔌 API网关:         http://localhost:8000                    │
│ 📚 API文档:         http://localhost:8000/docs               │
│ 💬 WebSocket:       ws://localhost:8000/api/v1/chat/websocket │
└─────────────────────────────────────────────────────────────┘

EOF

    if [ "$ENABLE_MONITORING" = true ]; then
        cat << EOF
${CYAN}📊 监控系统:${NC}
┌─────────────────────────────────────────────────────────────┐
│ 📈 Grafana:         http://localhost:3001 (admin/admin123)   │
│ 🔍 Prometheus:      http://localhost:9090                    │
│ 📊 cAdvisor:        http://localhost:8081                    │
└─────────────────────────────────────────────────────────────┘

EOF
    fi

    if [ "$ENABLE_LOGGING" = true ]; then
        cat << EOF
${CYAN}📋 日志系统:${NC}
┌─────────────────────────────────────────────────────────────┐
│ 📊 Kibana:          http://localhost:5601                    │
│ 🔍 Elasticsearch:   http://localhost:9200                    │
└─────────────────────────────────────────────────────────────┘

EOF
    fi

    cat << EOF
${CYAN}🛠 管理命令:${NC}
┌─────────────────────────────────────────────────────────────┐
│ 查看日志:   docker-compose logs -f [service_name]            │
│ 停止服务:   docker-compose down                              │
│ 重启服务:   docker-compose restart [service_name]           │
│ 查看状态:   docker-compose ps                                │
│ 进入容器:   docker-compose exec [service_name] sh           │
└─────────────────────────────────────────────────────────────┘

${YELLOW}⚠ 注意事项:${NC}
• 首次启动可能需要较长时间下载模型和初始化数据库
• 如遇到问题，请查看具体服务日志进行调试
• 生产环境部署前请修改默认密码和密钥

${GREEN}🚀 开始您的AI旅行规划之旅吧！${NC}

EOF
}

# 清理函数
cleanup() {
    if [ $? -ne 0 ]; then
        log_error "部署过程中发生错误"
        log_info "正在清理..."
        
        cd "$PROJECT_ROOT/deployment" 2>/dev/null || true
        docker-compose down 2>/dev/null || true
    fi
}

# 主函数
main() {
    # 设置错误处理
    trap cleanup EXIT
    
    # 解析参数
    parse_args "$@"
    
    # 显示项目信息
    show_project_info
    
    # 执行部署步骤
    check_dependencies
    setup_environment
    create_directories
    
    if [ "$QUICK_START" != true ]; then
        run_tests
    fi
    
    build_images
    start_services
    
    # 等待服务稳定
    sleep 10
    
    health_check
    show_access_info
    
    log_success "部署完成！"
}

# 执行主函数
main "$@" 