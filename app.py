"""
公众号内容工坊 — 单体 Web 应用
FastAPI + Deepseek V4 Pro + 豆包文生图 (火山方舟 Ark)
一键部署到 Railway / Render / 任意云服务器
"""

import os
import re
import json
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="公众号内容工坊",
    description="Deepseek + 豆包文生图，一站式公众号创作平台",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 配置 ─────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DOUBAO_ARK_API_KEY = os.getenv("DOUBAO_ARK_API_KEY", "")
DOUBAO_ARK_ENDPOINT = os.getenv("DOUBAO_ARK_ENDPOINT", "")
DOUBAO_ARK_BASE_URL = os.getenv("DOUBAO_ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")

# ── 请求模型 ─────────────────────────────────────────────
class OutlineRequest(BaseModel):
    track: str
    keywords: str = ""
    count: int = 5

class ArticleRequest(BaseModel):
    topic: str
    track: str = ""
    outline: str = ""
    style: str = "专业深度"
    length: str = "medium"

class ImageRequest(BaseModel):
    prompt: str
    style: str = "illustration"
    width: int = 900
    height: int = 500

class ConvertRequest(BaseModel):
    markdown: str
    title: str = ""

# ── System Prompts ──────────────────────────────────────

SYSTEM_PROMPT = """你是一位资深的公众号主编，擅长撰写符合微信公众号阅读习惯的高质量文章。

写作要求：
1. 标题要有吸引力，善用数字、对比、悬念
2. 开头120字内必须抓住读者，用故事/数据/金句切入
3. 正文用Markdown格式，层次分明
4. 段落简洁，每段不超过4行，多用短句
5. 关键数据用**加粗**突出
6. 每300-400字配一句引导语或小结
7. 结尾要有行动号召或思考延伸"""

KHAZIX_SYSTEM_PROMPT = """你是数字生命卡兹克，AI行业深耕三年的内容创作者，公众号「数字生命卡兹克」，slogan「永远对世界保持好奇」。

你的风格：有见识的普通人在认真聊一件打动他的事。

## 核心风格
- 讲人话，像个活人。大胆用「我觉得」「我认为」，分享真实经历和感受
- 节奏感：像跟朋友聊天，句子时长时短，经常一句话自成一段制造重点
- 知识是「聊着聊着顺手掏出来」的，不是「下面我来科普一下」
- 敢下判断有明确好恶，但姿态是「我被打动了」不是居高临下
- 亲自下场叙事，让读者感觉你真的做了这件事
- 不加小标题，全文一口气顺下来，靠口语化转场（「说到这个」「回到xxx这块」「顺着上面的再聊聊」）
- 段落短，很多时候一句话就是一段，重要观点前后留白
- 回环呼应：前面埋的细节后面要响

## 禁用词（出现就是AI味）
「说白了」「意味着什么」「这意味着」「本质上」「换句话说」「不可否认」「综上所述」「值得注意的是」「不难发现」「首先…其次…最后」「在当今…的时代」「随着…的进步」

## 禁用标点
不用冒号「：」（用逗号代替）、不用破折号「——」、不用双引号（用「」代替）

## 推荐口语（自然使用，全文至少8处）
坦率的讲、说真的、我是真的觉得、反正我觉得、怎么说呢、其实吧、你想想看、我有时候觉得、我一直觉得、说实话我也不确定、我自己也还在摸索、这种感觉太爽了、我当时就愣住了、我真的被震撼到了、太离谱了、你敢信？？？

## 情绪标点
。。。表示震惊/无语/遗憾 | ？？？表示极度惊讶 | = = 表示无语吐槽

## 开头（从具体场景切入，绝不宏大叙事）
「故事是这样的。」「最近这两天，被xxx刷屏了」「前两天在网上刷到一张图」

## 结尾固定格式
以上，既然看到这里了，如果觉得不错，随手点个赞、在看、转发三连吧，如果想第一时间收到推送，也可以给我个星标⭐～
谢谢你看我的文章，我们，下次再见。
> / 作者：卡兹克

输出格式：先给标题（# 标记），然后正文纯段落叙述，不加小标题，最后用 [IMAGE_PROMPT: ...] 标注配图提示词。"""

# ── Deepseek API ───────────────────────────────────────

def call_deepseek(prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = 4096) -> str:
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="未配置 DEEPSEEK_API_KEY")

    import httpx

    try:
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.8,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Deepseek 调用失败: {str(e)[:200]}")


# ── 豆包文生图 (火山方舟 Ark, OpenAI 兼容) ──────────────

def call_doubao_image(prompt: str, style: str = "illustration",
                      width: int = 900, height: int = 500) -> dict:
    if not DOUBAO_ARK_API_KEY:
        raise HTTPException(status_code=500, detail="未配置 DOUBAO_ARK_API_KEY")

    import httpx

    style_map = {
        "illustration": "插画风格，色彩鲜明",
        "photo": "真实摄影风格",
        "poster": "海报设计风格",
    }
    full_prompt = f"{style_map.get(style, style)}，{prompt}"

    headers = {
        "Authorization": f"Bearer {DOUBAO_ARK_API_KEY}",
        "Content-Type": "application/json",
    }

    if DOUBAO_ARK_ENDPOINT:
        url = f"{DOUBAO_ARK_BASE_URL}/endpoints/{DOUBAO_ARK_ENDPOINT}/images/generations"
    else:
        url = f"{DOUBAO_ARK_BASE_URL}/images/generations"

    payload = {
        "model": "doubao-seedream-4-0-250828",
        "prompt": full_prompt,
        "n": 1,
        "size": f"{width}x{height}",
        "response_format": "url",
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code != 200 and DOUBAO_ARK_ENDPOINT:
                alt_resp = client.post(f"{DOUBAO_ARK_BASE_URL}/images/generations", headers=headers, json=payload)
                alt_resp.raise_for_status()
                return {"success": True, "data": alt_resp.json()}
            resp.raise_for_status()
            return {"success": True, "data": resp.json()}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"豆包调用失败: {str(e)[:200]}")


# ── Markdown → 微信公众号 HTML ──────────────────────────

def md_to_wechat_html(markdown: str, title: str = "") -> str:
    def escape_html(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def process_inline(text: str) -> str:
        text = re.sub(r'`([^`]+)`',
            r'<code style="background:#f0f0f0;color:#e74c3c;padding:2px 6px;border-radius:3px;font-family:Menlo,monospace;font-size:14px;">\1</code>', text)
        text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',
            r'<img src="\2" alt="\1" style="max-width:100%;border-radius:8px;margin:16px 0;">', text)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
            r'<a href="\2" style="color:#576b95;text-decoration:none;">\1</a>', text)
        text = re.sub(r'\*\*(.+?)\*\*',
            r'<strong style="font-weight:700;color:#1a1a1a;">\1</strong>', text)
        text = re.sub(r'\*(.+?)\*',
            r'<em style="font-style:italic;color:#555;">\1</em>', text)
        return text

    lines = markdown.split("\n")
    html_parts = []
    in_code_block = False
    code_lines = []

    for line in lines:
        if line.strip().startswith("```"):
            if in_code_block:
                code_text = escape_html("\n".join(code_lines))
                html_parts.append(
                    f'<pre style="background:#2d2d2d;color:#f8f8f2;padding:20px;'
                    f'border-radius:8px;overflow-x:auto;font-size:13px;'
                    f'line-height:1.6;margin:16px 0;font-family:Menlo,monospace;">'
                    f'<code>{code_text}</code></pre>')
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            continue
        if in_code_block:
            code_lines.append(line)
            continue
        if not line.strip():
            continue

        s = line.strip()
        if s.startswith("# "):
            html_parts.append(f'<h1 style="font-size:24px;font-weight:700;color:#1a1a1a;margin:24px 0 16px;padding-bottom:8px;border-bottom:2px solid #07c160;line-height:1.4;">{process_inline(s[2:])}</h1>')
        elif s.startswith("## "):
            html_parts.append(f'<h2 style="font-size:20px;font-weight:700;color:#1a1a1a;margin:24px 0 12px;line-height:1.4;">{process_inline(s[3:])}</h2>')
        elif s.startswith("### "):
            html_parts.append(f'<h3 style="font-size:17px;font-weight:600;color:#333;margin:20px 0 10px;line-height:1.4;">{process_inline(s[4:])}</h3>')
        elif s in ("---", "***", "___"):
            html_parts.append('<hr style="border:none;border-top:1px solid #e0e0e0;margin:24px 0;">')
        elif s.startswith("> "):
            html_parts.append(f'<blockquote style="border-left:4px solid #07c160;background:#f6f9f7;padding:12px 16px;margin:12px 0;color:#555;font-size:15px;line-height:1.6;">{process_inline(s[2:])}</blockquote>')
        elif re.match(r'^[\-\*\+] ', s):
            html_parts.append(f'<li style="margin:6px 0;color:#333;font-size:15px;line-height:1.8;list-style-type:disc;">{process_inline(s[2:])}</li>')
        elif re.match(r'^\d+\. ', s):
            idx = s.index(". ")
            html_parts.append(f'<li style="margin:6px 0;color:#333;font-size:15px;line-height:1.8;">{process_inline(s[idx+2:])}</li>')
        else:
            html_parts.append(f'<p style="margin:10px 0;color:#333;font-size:15px;line-height:1.8;text-align:justify;">{process_inline(s)}</p>')

    body = "\n".join(html_parts)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="max-width:677px;margin:0 auto;padding:20px 16px;font-family:-apple-system,BlinkMacSystemFont,'PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;background:#fff;color:#333;">
<div style="margin-bottom:32px;"><h1 style="font-size:24px;font-weight:700;color:#1a1a1a;line-height:1.4;margin:0;">{escape_html(title) if title else ''}</h1></div>
{body}
<div style="margin-top:40px;padding-top:20px;border-top:1px solid #e0e0e0;color:#999;font-size:12px;text-align:center;">本文由 AI 辅助生成 · 公众号内容工坊</div>
</body></html>"""


# ── Frontend HTML ──────────────────────────────────────

FRONTEND_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>公众号内容工坊</title>
<style>
:root{--bg:#faf8f5;--surface:#fffef9;--border:#e8e4db;--text:#2c2416;--text-muted:#8c8273;--accent:#c84c3d;--accent2:#07c160;--accent2-light:#e8f8ef;--shadow:0 2px 24px rgba(44,36,22,0.06);--radius:12px;--font-display:'Noto Serif SC','STSong','Songti SC','SimSun',serif;--font-body:-apple-system,BlinkMacSystemFont,'PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;--font-mono:'SF Mono','Menlo','Consolas',monospace}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:var(--font-body);background:var(--bg);color:var(--text);min-height:100vh;background-image:radial-gradient(ellipse at 20% 0%,rgba(200,180,150,0.08) 0%,transparent 60%),radial-gradient(ellipse at 80% 100%,rgba(7,193,96,0.04) 0%,transparent 50%)}
.header{padding:24px 32px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}
.logo{font-family:var(--font-display);font-size:22px;font-weight:700;color:var(--text);letter-spacing:0.04em}
.logo span{color:var(--accent2)}
.status{font-size:12px;color:var(--text-muted);display:flex;align-items:center;gap:6px}
.status-dot{width:8px;height:8px;border-radius:50%}
.status-dot.online{background:var(--accent2)}
.status-dot.offline{background:#d4c5b9}
.main-grid{display:grid;grid-template-columns:340px 1fr;gap:24px;padding:16px 32px 32px;min-height:calc(100vh - 100px)}
@media(max-width:900px){.main-grid{grid-template-columns:1fr;padding:12px 16px}}
.left-panel{display:flex;flex-direction:column;gap:16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;box-shadow:var(--shadow)}
.card-label{font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);margin-bottom:12px;font-weight:600}
.input-group{margin-bottom:12px}
.input-group:last-child{margin-bottom:0}
.input-group label{display:block;font-size:13px;font-weight:600;color:var(--text);margin-bottom:5px}
.input-group input,.input-group select,.input-group textarea{width:100%;padding:10px 14px;border:1px solid var(--border);border-radius:8px;font-size:14px;font-family:var(--font-body);background:var(--bg);color:var(--text);transition:all 0.2s;resize:vertical}
.input-group input:focus,.input-group select:focus,.input-group textarea:focus{outline:none;border-color:var(--accent2);box-shadow:0 0 0 3px var(--accent2-light)}
.input-group textarea{min-height:60px}
.input-row{display:flex;gap:10px}
.input-row>*{flex:1}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:10px 20px;border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;transition:all 0.2s;font-family:var(--font-body)}
.btn-primary{background:var(--accent2);color:#fff}
.btn-primary:hover{background:#05a84e;transform:translateY(-1px)}
.btn-secondary{background:var(--bg);color:var(--text);border:1px solid var(--border)}
.btn-secondary:hover{background:var(--border)}
.btn-accent{background:var(--accent);color:#fff}
.btn-accent:hover{background:#b34032}
.btn-sm{padding:6px 14px;font-size:12px}
.btn-block{width:100%}
.btn:disabled{opacity:0.5;cursor:not-allowed;transform:none!important}
.topic-list{list-style:none;max-height:380px;overflow-y:auto}
.topic-item{padding:14px 16px;border:1px solid var(--border);border-radius:8px;margin-bottom:8px;cursor:pointer;transition:all 0.2s;background:var(--bg);position:relative}
.topic-item:hover{border-color:var(--accent2);background:var(--accent2-light)}
.topic-item.selected{border-color:var(--accent2);background:var(--accent2-light);box-shadow:0 0 0 2px var(--accent2-light)}
.topic-item .topic-title{font-size:14px;font-weight:600;margin-bottom:4px;color:var(--text)}
.topic-item .topic-desc{font-size:12px;color:var(--text-muted);line-height:1.5}
.topic-index{position:absolute;top:12px;right:14px;font-size:11px;color:var(--text-muted);font-family:var(--font-mono)}
.right-panel{display:flex;flex-direction:column;gap:16px;min-height:0}
.tabs{display:flex;gap:4px;background:var(--bg);border-radius:8px;padding:4px;border:1px solid var(--border);margin-bottom:-4px}
.tab{padding:8px 20px;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer;transition:all 0.2s;color:var(--text-muted);border:none;background:none;font-family:var(--font-body)}
.tab.active{background:var(--surface);color:var(--text);box-shadow:0 1px 3px rgba(0,0,0,0.06)}
.preview-area{flex:1;min-height:500px;overflow-y:auto;border:1px solid var(--border);border-radius:var(--radius);background:#fff;box-shadow:var(--shadow);position:relative}
.md-preview{padding:32px 40px;font-size:15px;line-height:1.8}
.md-preview h1{font-size:24px;margin:24px 0 16px;padding-bottom:8px;border-bottom:2px solid var(--accent2)}
.md-preview h2{font-size:20px;margin:20px 0 12px}
.md-preview h3{font-size:17px;margin:16px 0 8px;color:#555}
.md-preview p{margin:10px 0;text-align:justify}
.md-preview strong{color:var(--text);font-weight:700}
.md-preview blockquote{border-left:4px solid var(--accent2);background:var(--accent2-light);padding:12px 16px;margin:12px 0;color:#555;font-size:14px}
.md-preview code{background:#f0f0f0;color:#e74c3c;padding:2px 6px;border-radius:3px;font-family:var(--font-mono);font-size:13px}
.md-preview pre{background:#2d2d2d;color:#f8f8f2;padding:20px;border-radius:8px;overflow-x:auto;font-size:13px;line-height:1.6;margin:16px 0}
.md-preview pre code{background:none;color:inherit;padding:0}
.md-preview img{max-width:100%;border-radius:8px;margin:16px 0}
.md-preview li{margin:4px 0;line-height:1.8}
.md-preview hr{border:none;border-top:1px solid var(--border);margin:24px 0}
.wechat-preview{max-width:677px;margin:0 auto;padding:20px 16px;font-family:var(--font-body)}
.empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;height:400px;color:var(--text-muted);text-align:center;gap:12px}
.empty-state .empty-icon{font-size:48px;opacity:0.3;font-family:var(--font-display)}
.empty-state p{font-size:14px;max-width:280px;line-height:1.6}
.spinner{display:inline-block;width:18px;height:18px;border:2px solid rgba(255,255,255,0.3);border-top-color:#fff;border-radius:50%;animation:spin 0.6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.loading-overlay{position:absolute;inset:0;background:rgba(255,254,249,0.85);display:flex;align-items:center;justify-content:center;flex-direction:column;gap:16px;border-radius:var(--radius);z-index:10}
.loading-overlay .spinner{border-color:rgba(7,193,96,0.2);border-top-color:var(--accent2);width:32px;height:32px}
.loading-text{font-size:13px;color:var(--text-muted)}
.toast{position:fixed;top:24px;left:50%;transform:translateX(-50%);background:var(--text);color:#fff;padding:12px 24px;border-radius:8px;font-size:13px;z-index:1000;animation:toastIn 0.3s ease;box-shadow:0 8px 32px rgba(0,0,0,0.15)}
@keyframes toastIn{from{opacity:0;transform:translateX(-50%) translateY(-12px)}to{opacity:1;transform:translateX(-50%) translateY(0)}}
.toolbar{display:flex;gap:8px;padding:12px 20px;background:var(--surface);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:5}
.image-prompt-hint{padding:12px 16px;background:#fffbf0;border:1px solid #f0d88a;border-radius:8px;font-size:13px;color:#8a7440;line-height:1.6}
.image-prompt-hint strong{color:#6b5020}
.footer-row{display:flex;gap:8px;justify-content:flex-end}
</style>
</head>
<body>
<div class="header">
<div class="logo">公众号<span>内容</span>工坊</div>
<div class="status"><span class="status-dot" id="statusDot"></span><span id="statusText">检测中...</span></div>
</div>
<div class="main-grid">
<div class="left-panel">
<div class="card">
<div class="card-label">赛道设置</div>
<div class="input-group"><label>赛道 / 领域</label><input type="text" id="trackInput" placeholder="例如：AI 产品、新能源、消费零售..."></div>
<div class="input-group"><label>补充关键词</label><input type="text" id="keywordsInput" placeholder="逗号分隔多个关键词"></div>
<div class="input-group"><label>选题数量</label><select id="countSelect"><option value="3">3 个</option><option value="5" selected>5 个</option><option value="8">8 个</option></select></div>
<button class="btn btn-primary btn-block" id="genOutlineBtn" onclick="generateOutline()" disabled>生成选题</button>
</div>
<div class="card">
<div class="card-label">选题列表</div>
<ul class="topic-list" id="topicList"><li class="empty-state" style="height:150px"><p>填入赛道后点击「生成选题」</p></li></ul>
</div>
<div class="card">
<div class="card-label">生成设置</div>
<div class="input-group"><label>写作风格</label><select id="styleSelect"><option value="专业深度">专业深度</option><option value="轻松通俗">轻松通俗</option><option value="热点锐评">热点锐评</option><option value="故事叙事">故事叙事</option><option value="卡兹克">卡兹克风格</option></select></div>
<div class="input-group"><label>文章篇幅</label><select id="lengthSelect"><option value="short">短文 (800-1200字)</option><option value="medium" selected>中篇 (1500-2500字)</option><option value="long">长文 (3000-5000字)</option></select></div>
<div class="input-group" id="outlineGroup" style="display:none"><label>大纲要点</label><textarea id="outlineInput" rows="3" placeholder="选中选题后自动填充..."></textarea></div>
<button class="btn btn-primary btn-block" id="genArticleBtn" onclick="generateArticle()" disabled>生成图文</button>
</div>
</div>
<div class="right-panel">
<div class="tabs">
<button class="tab active" onclick="switchTab('markdown')">Markdown 预览</button>
<button class="tab" onclick="switchTab('wechat')">微信排版预览</button>
</div>
<div class="preview-area" id="previewArea">
<div class="empty-state" id="emptyState"><div class="empty-icon">&#10005;</div><p>生成选题后，选择一个感兴趣的，点击「生成图文」</p></div>
<div id="mdPreview" class="md-preview" style="display:none"></div>
<iframe id="wechatFrame" style="display:none;width:100%;height:100%;border:none;min-height:600px"></iframe>
<div class="loading-overlay" id="loadingOverlay" style="display:none"><div class="spinner"></div><div class="loading-text" id="loadingText">AI 正在创作中...</div></div>
</div>
<div class="image-prompt-hint" id="imagePromptHint" style="display:none"><strong>配图提示词：</strong><span id="imagePromptText"></span></div>
<div class="footer-row">
<button class="btn btn-secondary btn-sm" id="copyMdBtn" onclick="copyMarkdown()" disabled>复制 Markdown</button>
<button class="btn btn-secondary btn-sm" id="copyWechatBtn" onclick="copyWechatHtml()" disabled>复制微信排版</button>
<button class="btn btn-accent btn-sm" id="genImageBtn" onclick="generateImage()" disabled>生成配图</button>
</div>
</div>
</div>
<script>
// ── State ─────────────────────────────────────────────
const state = {
  apiBase: window.location.origin,
  currentTab: 'markdown',
  selectedTopic: null,
  topics: [],
  markdown: '',
  wechatHtml: '',
  imagePrompt: '',
  articleTitle: ''
};

// ── Init ──────────────────────────────────────────────
async function checkHealth() {
  const dot = document.getElementById('statusDot');
  const label = document.getElementById('statusText');
  try {
    const resp = await fetch(state.apiBase + '/api/health', { signal: AbortSignal.timeout(5000) });
    const data = await resp.json();
    if (data.status === 'ok') {
      dot.className = 'status-dot online';
      const parts = [];
      parts.push(data.deepseek ? 'Deepseek ✓' : 'Deepseek ✗');
      parts.push(data.doubao ? '豆包 ✓' : '豆包 ✗');
      label.textContent = parts.join(' · ');
      document.getElementById('genOutlineBtn').disabled = false;
    } else {
      dot.className = 'status-dot offline';
      label.textContent = '服务异常';
    }
  } catch {
    dot.className = 'status-dot offline';
    label.textContent = '未连接 — 请确认服务已启动';
  }
}
checkHealth();
setInterval(checkHealth, 30000);

// ── Helpers ────────────────────────────────────────────
function showLoading(text) { document.getElementById('loadingText').textContent = text; document.getElementById('loadingOverlay').style.display = 'flex'; }
function hideLoading() { document.getElementById('loadingOverlay').style.display = 'none'; }
function showToast(msg) { const t = document.createElement('div'); t.className = 'toast'; t.textContent = msg; document.body.appendChild(t); setTimeout(() => t.remove(), 2500); }
function escapeHtml(str) { if (!str) return ''; const d = document.createElement('div'); d.textContent = str; return d.innerHTML; }

function switchTab(tab) {
  state.currentTab = tab;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('mdPreview').style.display = tab === 'markdown' ? 'block' : 'none';
  document.getElementById('wechatFrame').style.display = tab === 'wechat' ? 'block' : 'none';
}

// ── Generate Outline ───────────────────────────────────
async function generateOutline() {
  const track = document.getElementById('trackInput').value.trim();
  if (!track) return showToast('请先输入赛道');
  showLoading('正在生成选题...');
  try {
    const resp = await fetch(state.apiBase + '/api/outline', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        track: track,
        keywords: document.getElementById('keywordsInput').value.trim(),
        count: parseInt(document.getElementById('countSelect').value)
      })
    });
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || resp.statusText); }
    const data = await resp.json();
    if (data.success) {
      state.topics = data.topics;
      renderTopics();
      showToast('已生成 ' + data.topics.length + ' 个选题');
    }
  } catch(e) {
    showToast('失败: ' + e.message);
    console.error(e);
  }
  hideLoading();
}

function renderTopics() {
  const list = document.getElementById('topicList');
  if (!state.topics.length) { list.innerHTML = '<li class="empty-state" style="height:150px"><p>暂无选题</p></li>'; return; }
  list.innerHTML = state.topics.map((t, i) => '<li class="topic-item' + (state.selectedTopic === i ? ' selected' : '') + '" onclick="selectTopic(' + i + ')"><span class="topic-index">' + String(i+1).padStart(2,'0') + '</span><div class="topic-title">' + escapeHtml(t.title) + '</div><div class="topic-desc">' + escapeHtml(t.desc || '') + '</div></li>').join('');
}

function selectTopic(index) {
  state.selectedTopic = index; state.markdown = ''; state.wechatHtml = ''; state.imagePrompt = '';
  renderTopics();
  const topic = state.topics[index];
  document.getElementById('outlineInput').value = (topic.points || []).join('\\\\n');
  document.getElementById('outlineGroup').style.display = 'block';
  document.getElementById('genArticleBtn').disabled = false;
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('mdPreview').style.display = 'none'; document.getElementById('mdPreview').innerHTML = '';
  document.getElementById('wechatFrame').style.display = 'none';
  document.getElementById('imagePromptHint').style.display = 'none';
  document.getElementById('copyMdBtn').disabled = true;
  document.getElementById('copyWechatBtn').disabled = true;
  document.getElementById('genImageBtn').disabled = true;
}

// ── Generate Article ───────────────────────────────────
async function generateArticle() {
  if (state.selectedTopic === null) return;
  const topic = state.topics[state.selectedTopic];
  showLoading('AI 正在创作文章和配图提示词...');
  try {
    const resp = await fetch(state.apiBase + '/api/publish-flow', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        topic: topic.title,
        track: document.getElementById('trackInput').value.trim(),
        outline: document.getElementById('outlineInput').value.trim(),
        style: document.getElementById('styleSelect').value,
        length: document.getElementById('lengthSelect').value
      })
    });
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || resp.statusText); }
    const data = await resp.json();
    if (data.success) {
      state.markdown = data.markdown;
      state.wechatHtml = data.wechat_html;
      state.imagePrompt = data.image_prompt;
      state.articleTitle = data.title;
      renderPreview();
      document.getElementById('emptyState').style.display = 'none';
      document.getElementById('copyMdBtn').disabled = false;
      document.getElementById('copyWechatBtn').disabled = false;
      document.getElementById('genImageBtn').disabled = !state.imagePrompt;
      if (state.imagePrompt) {
        document.getElementById('imagePromptText').textContent = state.imagePrompt;
        document.getElementById('imagePromptHint').style.display = 'block';
      }
      showToast('文章生成完成！');
    }
  } catch(e) {
    showToast('失败: ' + e.message);
    console.error(e);
  }
  hideLoading();
}

function renderPreview() {
  const mdDiv = document.getElementById('mdPreview');
  mdDiv.innerHTML = renderMarkdown(state.markdown);
  mdDiv.style.display = state.currentTab === 'markdown' ? 'block' : 'none';
  const frame = document.getElementById('wechatFrame');
  frame.style.display = state.currentTab === 'wechat' ? 'block' : 'none';
  frame.srcdoc = state.wechatHtml;
}

function renderMarkdown(md) {
  if (!md) return '';
  let html = md;
  html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  html = html.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
  html = html.replace(/\\*(.+?)\\*/g, '<em>$1</em>');
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
  html = html.replace(/^---$/gm, '<hr>');
  html = html.replace(/!\\[([^\\]]*)\\]\\(([^)]+)\\)/g, '<img src="$2" alt="$1">');
  html = html.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2">$1</a>');
  html = html.replace(/^[\\-\\*\\+] (.+)$/gm, '<li>$1</li>');
  const lines = html.split('\\\\n');
  const result = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) { result.push('<br>'); continue; }
    if (/^<(\\/)?(h[1-4]|strong|em|code|blockquote|hr|img|a|li|pre|ul|ol)/.test(line.trim())) { result.push(line); }
    else if (line.trim().startsWith('<')) { result.push(line); }
    else { result.push('<p>' + line + '</p>'); }
  }
  return result.join('\\\\n');
}

function copyMarkdown() {
  navigator.clipboard.writeText(state.markdown).then(() => showToast('已复制 Markdown'), () => showToast('复制失败'));
}

function copyWechatHtml() {
  navigator.clipboard.write([
    new ClipboardItem({
      'text/html': new Blob([state.wechatHtml], {type: 'text/html'}),
      'text/plain': new Blob([state.markdown], {type: 'text/plain'})
    })
  ]).then(() => showToast('微信排版已复制，可直接粘贴到公众号编辑器'), () => showToast('复制失败，请重试'));
}

async function generateImage() {
  if (!state.imagePrompt) return;
  showLoading('正在生成配图...');
  try {
    const resp = await fetch(state.apiBase + '/api/image', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt: state.imagePrompt, style: 'illustration'})
    });
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || resp.statusText); }
    const data = await resp.json();
    if (data.success) { showToast('配图生成请求已发送'); console.log('Image result:', data.data); }
  } catch(e) { showToast('失败: ' + e.message); }
  hideLoading();
}
</script>
</body>
</html>"""


# ── API 路由 ─────────────────────────────────────────────

@app.get("/")
async def index():
    return HTMLResponse(content=FRONTEND_HTML)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "deepseek": bool(DEEPSEEK_API_KEY),
        "doubao": bool(DOUBAO_ARK_API_KEY),
        "doubao_endpoint": bool(DOUBAO_ARK_ENDPOINT),
    }


@app.post("/api/outline")
async def generate_outline(req: OutlineRequest):
    count_hint = f"请生成{req.count}个选题" if req.count > 1 else "请生成1个选题"
    kw_hint = f"，结合关键词「{req.keywords}」" if req.keywords else ""
    prompt = f"""赛道/领域：{req.track}{kw_hint}

{count_hint}。每个选题包含 title（标题）、desc（一句话理由）、points（3-5个大纲要点，数组）。
请以 JSON 数组格式输出，只输出 JSON。"""

    try:
        result = call_deepseek(prompt, max_tokens=2048)
        result = result.strip()
        if result.startswith("```"):
            result = re.sub(r'^```\w*\n?', '', result)
            result = re.sub(r'\n?```$', '', result)
        topics = json.loads(result)
        return {"success": True, "topics": topics}
    except json.JSONDecodeError:
        return {"success": True, "topics": [{"title": "AI 建议", "desc": result, "points": []}]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成选题失败: {str(e)[:200]}")


@app.post("/api/article")
async def generate_article(req: ArticleRequest):
    length_map = {"short": "约 800-1200 字", "medium": "约 1500-2500 字", "long": "约 3000-5000 字"}
    length_hint = length_map.get(req.length, "约 1500-2500 字")
    outline_hint = f"\n文章大纲参考：{req.outline}" if req.outline else ""
    track_hint = f"\n赛道：{req.track}" if req.track else ""

    prompt = f"""请撰写一篇微信公众号文章。

选题：{req.topic}{track_hint}{outline_hint}
写作风格：{req.style}
篇幅要求：{length_hint}

请严格按照系统提示中的格式要求输出，使用 Markdown 格式。"""
    system = KHAZIX_SYSTEM_PROMPT if req.style == "卡兹克" else SYSTEM_PROMPT
    try:
        article = call_deepseek(prompt, system=system, max_tokens=8192)
        return {"success": True, "article": article}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成文章失败: {str(e)[:200]}")


@app.post("/api/image")
async def generate_image(req: ImageRequest):
    try:
        result = call_doubao_image(prompt=req.prompt, style=req.style, width=req.width, height=req.height)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成图片失败: {str(e)[:200]}")


@app.post("/api/convert")
async def convert_to_wechat(req: ConvertRequest):
    try:
        html = md_to_wechat_html(req.markdown, req.title)
        return {"success": True, "html": html}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"格式转换失败: {str(e)[:200]}")


@app.post("/api/publish-flow")
async def publish_flow(req: ArticleRequest):
    length_map = {"short": "约 800-1200 字", "medium": "约 1500-2500 字", "long": "约 3000-5000 字"}
    length_hint = length_map.get(req.length, "约 1500-2500 字")
    outline_hint = f"\n文章大纲参考：{req.outline}" if req.outline else ""
    track_hint = f"\n赛道：{req.track}" if req.track else ""

    article_prompt = f"""请撰写一篇微信公众号文章。

选题：{req.topic}{track_hint}{outline_hint}
写作风格：{req.style}
篇幅要求：{length_hint}

请严格按照系统提示中的格式要求输出，使用 Markdown 格式。

在文章末尾，单独一行用 [IMAGE_PROMPT: ...] 标记配图提示词（一段中文描述，用于 AI 文生图）。"""

    system = KHAZIX_SYSTEM_PROMPT if req.style == "卡兹克" else SYSTEM_PROMPT
    try:
        result = call_deepseek(article_prompt, system=system, max_tokens=8192)

        image_prompt = ""
        article = result
        match = re.search(r'\[IMAGE_PROMPT:\s*(.+?)\]', result, re.DOTALL)
        if match:
            image_prompt = match.group(1).strip()
            article = result[:match.start()].strip() + "\n\n" + result[match.end():].strip()

        title = req.topic
        title_match = re.match(r'^#\s+(.+?)$', article, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()

        wechat_html = md_to_wechat_html(article, title)

        return {
            "success": True,
            "title": title,
            "markdown": article,
            "wechat_html": wechat_html,
            "image_prompt": image_prompt,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成流程失败: {str(e)[:200]}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8765")))
