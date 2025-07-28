# 🇨🇳 中国大陆第三方服务集成指南

## 📋 概述

本文档详细说明了AI Travel Planner项目中集成的中国大陆第三方服务，包括在线旅游、地图、天气、邮件、支付、社交登录等服务的配置和使用方法。

## 🎯 服务分类

### 🏨 在线旅游服务

#### 1. 携程 (Ctrip)
- **服务描述**: 中国最大的在线旅游服务提供商
- **主要功能**: 机票、酒店、火车票、门票预订
- **官方文档**: https://openapi.ctrip.com/
- **配置参数**:
  ```bash
  CTRIP_API_KEY=your_ctrip_api_key_here
  CTRIP_API_SECRET=your_ctrip_api_secret_here
  CTRIP_API_URL=https://openapi.ctrip.com
  ```

#### 2. 去哪儿 (Qunar)
- **服务描述**: 知名旅游搜索引擎和预订平台
- **主要功能**: 机票比价、酒店预订、度假产品
- **官方文档**: https://open.qunar.com/
- **配置参数**:
  ```bash
  QUNAR_API_KEY=your_qunar_api_key_here
  QUNAR_API_SECRET=your_qunar_api_secret_here
  QUNAR_API_URL=https://open.qunar.com
  ```

#### 3. 飞猪 (Fliggy)
- **服务描述**: 阿里巴巴旗下综合性旅游服务平台
- **主要功能**: 机票、酒店、度假、签证服务
- **官方文档**: https://open.taobao.com/
- **配置参数**:
  ```bash
  FLIGGY_API_KEY=your_fliggy_api_key_here
  FLIGGY_APP_SECRET=your_fliggy_app_secret_here
  FLIGGY_API_URL=https://eco.taobao.com/router/rest
  ```

#### 4. 美团 (Meituan)
- **服务描述**: 生活服务平台，提供酒店和景点服务
- **主要功能**: 酒店预订、景点门票、美食推荐
- **官方文档**: https://open.meituan.com/
- **配置参数**:
  ```bash
  MEITUAN_API_KEY=your_meituan_api_key_here
  MEITUAN_API_SECRET=your_meituan_api_secret_here
  MEITUAN_API_URL=https://api.meituan.com
  ```

### 🗺️ 地图服务

#### 1. 百度地图 (Baidu Maps)
- **服务描述**: 百度提供的地图和位置服务
- **主要功能**: 地理编码、路径规划、POI搜索、实时交通
- **官方文档**: https://lbsyun.baidu.com/
- **配置参数**:
  ```bash
  BAIDU_MAP_API_KEY=your_baidu_map_api_key_here
  BAIDU_MAP_API_URL=https://api.map.baidu.com
  ```
- **主要API**:
  - 地理编码: `/geocoding/v3/`
  - 逆地理编码: `/reverse_geocoding/v3/`
  - 路径规划: `/direction/v2/`
  - POI搜索: `/place/v2/search`

#### 2. 高德地图 (Amap)
- **服务描述**: 阿里巴巴旗下的地图和导航服务
- **主要功能**: 地理编码、导航、POI搜索、天气信息
- **官方文档**: https://lbs.amap.com/
- **配置参数**:
  ```bash
  AMAP_API_KEY=your_amap_api_key_here
  AMAP_API_URL=https://restapi.amap.com
  ```
- **主要API**:
  - 地理编码: `/v3/geocode/geo`
  - 逆地理编码: `/v3/geocode/regeo`
  - 路径规划: `/v3/direction/driving`
  - POI搜索: `/v3/place/text`

#### 3. 腾讯地图 (Tencent Maps)
- **服务描述**: 腾讯提供的地图和位置服务
- **主要功能**: 地理编码、路径规划、地点搜索
- **官方文档**: https://lbs.qq.com/
- **配置参数**:
  ```bash
  TENCENT_MAP_API_KEY=your_tencent_map_api_key_here
  TENCENT_MAP_API_URL=https://apis.map.qq.com
  ```

### 🌤️ 天气服务

#### 1. 彩云天气 (Caiyun Weather)
- **服务描述**: 精准的天气预报和气象数据服务
- **主要功能**: 实时天气、预报、分钟级降水预报
- **官方文档**: https://docs.caiyunapp.com/
- **配置参数**:
  ```bash
  CAIYUN_WEATHER_API_KEY=your_caiyun_weather_api_key_here
  CAIYUN_WEATHER_API_URL=https://api.caiyunapp.com
  ```

#### 2. 和风天气 (HeWeather)
- **服务描述**: 专业的天气数据服务提供商
- **主要功能**: 天气预报、历史天气、灾害预警
- **官方文档**: https://dev.qweather.com/
- **配置参数**:
  ```bash
  HEWEATHER_API_KEY=your_heweather_api_key_here
  HEWEATHER_API_URL=https://api.qweather.com
  ```

#### 3. 心知天气 (Xinzhi Weather)
- **服务描述**: 气象数据和天气API服务
- **主要功能**: 天气预报、空气质量、生活指数
- **官方文档**: https://docs.seniverse.com/
- **配置参数**:
  ```bash
  XINZHI_WEATHER_API_KEY=your_xinzhi_weather_api_key_here
  XINZHI_WEATHER_API_URL=https://api.seniverse.com
  ```

### 📧 邮件服务

#### 1. 163邮箱 (NetEase Mail)
- **服务描述**: 网易提供的免费邮箱服务
- **配置参数**:
  ```bash
  SMTP_HOST=smtp.163.com
  SMTP_PORT=465
  SMTP_USER=your_email@163.com
  SMTP_PASSWORD=your_email_auth_code_here
  SMTP_USE_SSL=true
  ```

#### 2. QQ邮箱 (QQ Mail)
- **服务描述**: 腾讯提供的邮箱服务
- **配置参数**:
  ```bash
  QQ_SMTP_HOST=smtp.qq.com
  QQ_SMTP_PORT=587
  QQ_SMTP_USER=your_email@qq.com
  QQ_SMTP_PASSWORD=your_qq_auth_code_here
  ```

#### 3. 阿里云邮件推送
- **服务描述**: 阿里云提供的企业级邮件发送服务
- **官方文档**: https://help.aliyun.com/product/29412.html
- **配置参数**:
  ```bash
  ALIYUN_EMAIL_ACCESS_KEY_ID=your_aliyun_email_access_key_here
  ALIYUN_EMAIL_ACCESS_KEY_SECRET=your_aliyun_email_secret_here
  ALIYUN_EMAIL_REGION=cn-hangzhou
  ```

### 💰 支付服务

#### 1. 支付宝 (Alipay)
- **服务描述**: 蚂蚁集团提供的第三方支付平台
- **官方文档**: https://opendocs.alipay.com/
- **配置参数**:
  ```bash
  ALIPAY_APP_ID=your_alipay_app_id_here
  ALIPAY_PRIVATE_KEY=your_alipay_private_key_here
  ALIPAY_PUBLIC_KEY=your_alipay_public_key_here
  ALIPAY_GATEWAY_URL=https://openapi.alipay.com/gateway.do
  ```

#### 2. 微信支付 (WeChat Pay)
- **服务描述**: 腾讯提供的移动支付服务
- **官方文档**: https://pay.weixin.qq.com/docs/
- **配置参数**:
  ```bash
  WECHAT_PAY_APP_ID=your_wechat_app_id_here
  WECHAT_PAY_MCH_ID=your_wechat_mch_id_here
  WECHAT_PAY_API_KEY=your_wechat_pay_api_key_here
  ```

### 👥 社交登录

#### 1. 微信开放平台 (WeChat Open Platform)
- **服务描述**: 微信提供的第三方登录服务
- **官方文档**: https://developers.weixin.qq.com/
- **配置参数**:
  ```bash
  WECHAT_APP_ID=your_wechat_app_id_here
  WECHAT_APP_SECRET=your_wechat_app_secret_here
  ```

#### 2. QQ登录 (QQ Connect)
- **服务描述**: 腾讯QQ提供的第三方登录服务
- **官方文档**: https://connect.qq.com/
- **配置参数**:
  ```bash
  QQ_APP_ID=your_qq_app_id_here
  QQ_APP_KEY=your_qq_app_key_here
  ```

#### 3. 微博登录 (Weibo OAuth)
- **服务描述**: 新浪微博提供的第三方登录服务
- **官方文档**: https://open.weibo.com/
- **配置参数**:
  ```bash
  WEIBO_APP_KEY=your_weibo_app_key_here
  WEIBO_APP_SECRET=your_weibo_app_secret_here
  ```

### 📱 短信服务

#### 1. 阿里云短信服务
- **服务描述**: 阿里云提供的短信发送服务
- **官方文档**: https://help.aliyun.com/product/44282.html
- **配置参数**:
  ```bash
  ALIYUN_SMS_ACCESS_KEY_ID=your_aliyun_sms_access_key_here
  ALIYUN_SMS_ACCESS_KEY_SECRET=your_aliyun_sms_secret_here
  ALIYUN_SMS_SIGN_NAME=AI旅行规划师
  ALIYUN_SMS_TEMPLATE_CODE=SMS_123456789
  ```

#### 2. 腾讯云短信服务
- **服务描述**: 腾讯云提供的短信服务
- **官方文档**: https://cloud.tencent.com/product/sms
- **配置参数**:
  ```bash
  TENCENT_SMS_SECRET_ID=your_tencent_sms_secret_id_here
  TENCENT_SMS_SECRET_KEY=your_tencent_sms_secret_key_here
  TENCENT_SMS_APP_ID=1234567890
  ```

### 💾 云存储服务

#### 1. 阿里云OSS
- **服务描述**: 阿里云对象存储服务
- **官方文档**: https://help.aliyun.com/product/31815.html
- **配置参数**:
  ```bash
  ALIYUN_OSS_ACCESS_KEY_ID=your_aliyun_oss_access_key_here
  ALIYUN_OSS_ACCESS_KEY_SECRET=your_aliyun_oss_secret_here
  ALIYUN_OSS_BUCKET_NAME=ai-travel-planner-files
  ALIYUN_OSS_ENDPOINT=https://oss-cn-hangzhou.aliyuncs.com
  ```

#### 2. 腾讯云COS
- **服务描述**: 腾讯云对象存储服务
- **官方文档**: https://cloud.tencent.com/product/cos
- **配置参数**:
  ```bash
  TENCENT_COS_SECRET_ID=your_tencent_cos_secret_id_here
  TENCENT_COS_SECRET_KEY=your_tencent_cos_secret_key_here
  TENCENT_COS_BUCKET=ai-travel-planner-files-1234567890
  TENCENT_COS_REGION=ap-beijing
  ```

#### 3. 七牛云存储
- **服务描述**: 七牛云提供的对象存储服务
- **官方文档**: https://developer.qiniu.com/
- **配置参数**:
  ```bash
  QINIU_ACCESS_KEY=your_qiniu_access_key_here
  QINIU_SECRET_KEY=your_qiniu_secret_key_here
  QINIU_BUCKET_NAME=ai-travel-planner
  QINIU_DOMAIN=your-domain.qiniucdn.com
  ```

### 🤖 国产大模型服务

#### 1. 百度千帆大模型平台
- **服务描述**: 百度提供的大模型API服务
- **官方文档**: https://cloud.baidu.com/product/wenxinworkshop
- **配置参数**:
  ```bash
  BAIDU_QIANFAN_API_KEY=your_baidu_qianfan_api_key_here
  BAIDU_QIANFAN_SECRET_KEY=your_baidu_qianfan_secret_key_here
  ```

#### 2. 阿里云通义千问
- **服务描述**: 阿里云提供的大模型服务
- **官方文档**: https://help.aliyun.com/product/2400256.html
- **配置参数**:
  ```bash
  ALIBABA_DASHSCOPE_API_KEY=your_alibaba_dashscope_api_key_here
  ```

#### 3. 腾讯混元大模型
- **服务描述**: 腾讯云提供的大模型服务
- **官方文档**: https://cloud.tencent.com/product/hunyuan
- **配置参数**:
  ```bash
  TENCENT_HUNYUAN_SECRET_ID=your_tencent_hunyuan_secret_id_here
  TENCENT_HUNYUAN_SECRET_KEY=your_tencent_hunyuan_secret_key_here
  ```

## 📝 申请和配置指南

### 🔑 API密钥申请流程

#### 携程开放平台
1. 访问 https://openapi.ctrip.com/
2. 注册企业开发者账户
3. 提交应用审核
4. 获取API Key和Secret

#### 百度地图开放平台
1. 访问 https://lbsyun.baidu.com/
2. 注册百度账户
3. 创建应用
4. 获取AK (API Key)

#### 高德开放平台
1. 访问 https://lbs.amap.com/
2. 注册高德账户
3. 创建应用
4. 获取Key

#### 支付宝开放平台
1. 访问 https://open.alipay.com/
2. 注册企业账户
3. 创建应用
4. 配置公私钥
5. 获取APPID

### 🔐 安全配置建议

#### 1. 密钥管理
```bash
# 使用环境变量存储敏感信息
export CTRIP_API_SECRET="your_secret_here"

# 在生产环境中使用密钥管理服务
# 如：阿里云KMS、腾讯云KMS
```

#### 2. 网络安全
```bash
# 配置API访问白名单
# 限制调用频率
# 启用HTTPS加密传输
```

#### 3. 审计日志
```bash
# 记录所有API调用
# 监控异常访问
# 定期轮换密钥
```

## 🚨 合规要求

### 📋 中国大陆特殊要求

#### 1. ICP备案
- 网站必须进行ICP备案
- 配置备案号显示

#### 2. 实名认证
- 用户注册需要实名认证
- 集成身份证验证服务

#### 3. 内容审核
- 启用内容审核功能
- 集成百度内容审核API
- 配置敏感词过滤

#### 4. 数据本地化
- 用户数据存储在中国大陆
- 遵循《网络安全法》
- 遵循《数据安全法》

### 📝 配置示例

```bash
# 内容审核配置
CONTENT_MODERATION_ENABLED=true
BAIDU_TEXT_CENSOR_API_KEY=your_baidu_text_censor_api_key_here
TENCENT_CMS_SECRET_ID=your_tencent_cms_secret_id_here

# 地区和语言设置
DEFAULT_COUNTRY=CN
DEFAULT_LANGUAGE=zh-CN
DEFAULT_CURRENCY=CNY
TIMEZONE=Asia/Shanghai

# ICP备案信息
ICP_LICENSE=your_icp_license_here
```

## 🔧 开发建议

### 1. API封装
- 为每个服务创建统一的SDK
- 实现重试机制和熔断器
- 添加缓存层提高性能

### 2. 错误处理
- 统一错误码规范
- 实现优雅降级
- 记录详细的错误日志

### 3. 监控告警
- 监控API调用成功率
- 设置调用量告警
- 监控服务响应时间

### 4. 成本控制
- 合理设置调用频率限制
- 使用缓存减少API调用
- 监控各服务的费用开销

## 📞 技术支持

如果在集成过程中遇到问题，可以：

1. 查阅各服务商的官方文档
2. 联系对应的技术支持
3. 参考项目中的示例代码
4. 提交Issue到项目仓库 