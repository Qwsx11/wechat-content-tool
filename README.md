# 公众号内容工坊

Deepseek V4 Pro + 豆包文生图 · 一站式公众号创作平台

## 项目结构

```
wechat-content-tool/
├── app.py          # 单体应用（前端+后端+API 全在一个文件）
├── requirements.txt  # Python 依赖
├── .env            # 环境变量（API 密钥）
├── .env.example    # 环境变量模板
├── Dockerfile      # Docker 部署
├── render.yaml     # Render.com 一键部署
├── railway.json    # Railway 部署
├── Procfile        # Heroku 兼容
└── README.md
```

## 快速部署到云端（推荐）

### Render.com（免费）

1. Fork 或上传本项目到 GitHub
2. 登录 [Render.com](https://render.com) → New → Web Service
3. 连接你的 GitHub 仓库，Render 自动识别 `render.yaml`
4. 在 Environment 中添加以下环境变量：
   - `DEEPSEEK_API_KEY`
   - `DUABAO_ARK_API_KEY`
   - `DUABAO_ARK_ENDPOINT`
   - `DUABAO_ARK_BASE_URL`
5. 点击部署，获得 `https://你的项目.onrender.com` 地址

### Railway（免费额度）

1. 上传项目到 GitHub
2. 登录 [Railway.app](https://railway.app) → New Project → Deploy from GitHub
3. 添加环境变量，部署完成

### Docker 部署

```bash
docker build -t wechat-tool .
docker run -p 8765:8765 --env-file .env wechat-tool
```

## 本地运行

```bash
cp .env.example .env    # 编辑填入 API 密钥
pip install -r requirements.txt --break-system-packages
python app.py           # 访问 http://localhost:8765
```

## 功能

- **赛道输入** — 输入赛道/领域，AI 推荐高分选题
- **选题列表** — 浏览选题，一键选中开始创作
- **图文预览** — Markdown / 微信排版双视图切换
- **一键复制** — 复制 Markdown 源码或微信排版 HTML
- **文生图** — 自动生成配图提示词，调用豆包生成配图
- **卡兹克风格** — 内置「数字生命卡兹克」公众号长文写作风格
