-- AI Travel Planner 数据库初始化脚本
-- 创建数据库和用户

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS `ai_travel_db` 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

-- 创建用户（如果不存在）
CREATE USER IF NOT EXISTS 'ai_travel_user'@'%' IDENTIFIED BY 'ai_travel_pass';

-- 授予权限
GRANT ALL PRIVILEGES ON `ai_travel_db`.* TO 'ai_travel_user'@'%';

-- 刷新权限
FLUSH PRIVILEGES;

-- 使用数据库
USE `ai_travel_db`;

-- 创建基础表结构（这些表将由ORM自动创建，这里只是确保数据库存在）
-- 实际的表结构由 SQLAlchemy ORM 模型定义

-- 插入初始化完成标记
CREATE TABLE IF NOT EXISTS `_init_status` (
    `id` INT PRIMARY KEY AUTO_INCREMENT,
    `component` VARCHAR(50) NOT NULL,
    `status` VARCHAR(20) NOT NULL DEFAULT 'completed',
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `message` TEXT
);

INSERT INTO `_init_status` (`component`, `message`) 
VALUES ('database', 'Database and user created successfully')
ON DUPLICATE KEY UPDATE 
    `status` = 'completed',
    `created_at` = CURRENT_TIMESTAMP,
    `message` = 'Database and user created successfully';
