#!/bin/bash

# 测试所有服务健康状态的脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== 测试所有服务健康状态 ===${NC}"

# 定义服务列表
declare -A services
services["RAG服务"]="http://localhost:8001/api/v1/health"
services["Chat服务"]="http://localhost:8080/api/v1/health"
services["User服务"]="http://localhost:8003/api/v1/health"
services["Planning服务"]="http://localhost:8004/api/v1/health"
services["Integration服务"]="http://localhost:8005/api/v1/health"
services["API网关"]="http://localhost:8006/health"

# 测试每个服务
for service_name in "${!services[@]}"; do
    endpoint="${services[$service_name]}"
    echo -n "测试 $service_name: "
    
    # 使用curl测试，设置超时时间
    response=$(curl -s --connect-timeout 3 --max-time 5 "$endpoint" 2>/dev/null)
    curl_exit_code=$?
    
    if [ $curl_exit_code -eq 0 ] && [ -n "$response" ]; then
        # 检查响应是否包含健康状态
        if echo "$response" | grep -q '"status".*"healthy"' || echo "$response" | grep -q '"status":"ok"' || echo "$response" | grep -q '"healthy":true'; then
            echo -e "${GREEN}✅ 健康${NC}"
        else
            echo -e "${YELLOW}⚠️  响应异常${NC}"
            echo "    响应: $response"
        fi
    else
        echo -e "${RED}❌ 不健康${NC}"
        case $curl_exit_code in
            7) echo "    错误: 无法连接到服务" ;;
            28) echo "    错误: 连接超时" ;;
            *) echo "    错误代码: $curl_exit_code" ;;
        esac
    fi
done

echo -e "\n${BLUE}=== 端口监听状态 ===${NC}"
netstat -tuln | grep -E ":(8001|8080|8003|8004|8005|8006) " | while read line; do
    port=$(echo "$line" | grep -oE ":(8001|8080|800[3-6])" | cut -d: -f2)
    echo -e "端口 $port: ${GREEN}✅ 监听中${NC}"
done
