#!/usr/bin/env python3
"""
Chat服务启动脚本
正确设置Python路径并启动服务
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 设置工作目录为项目根目录
os.chdir(project_root)

if __name__ == "__main__":
    import uvicorn
    
    # 启动服务
    uvicorn.run(
        "services.chat-service.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        reload_dirs=[str(project_root)],
        log_level="info"
    )
