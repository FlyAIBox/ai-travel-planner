# 多阶段构建 - 构建阶段
FROM node:18-alpine as builder

# 设置工作目录
WORKDIR /app

# 复制package.json
COPY package.json ./

# 安装依赖（处理没有package-lock.json的情况）
RUN npm install --production=false

# 复制源码
COPY . .

# 构建应用
RUN npm run build

# 生产阶段 - Nginx
FROM nginx:alpine

# 复制自定义nginx配置
COPY nginx.conf /etc/nginx/nginx.conf

# 复制构建的应用到nginx目录
COPY --from=builder /app/dist /usr/share/nginx/html

# 暴露端口
EXPOSE 80

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

# 启动nginx
CMD ["nginx", "-g", "daemon off;"] 