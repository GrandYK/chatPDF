# 📄 智能文档问答助手 - ChatPDF
基于 RAG (Retrieval-Augmented Generation) 架构的智能文档问答系统，支持多格式文件上传、多轮上下文对话、中文语义优化检索，解决大模型脱离文档编造答案的问题。

## 🌟 在线体验
[https://grandyk-chatpdf.streamlit.app](https://grandyk-chatpdf.streamlit.app)

## 🚀 核心功能
| 功能 | 详情 |
|------|------|
| 📁 多格式支持 | 支持 PDF/TXT/DOCX 三种常用文档格式 |
| 💬 多轮对话 | 记忆上下文，理解「这个/那个/它」等指代关系 |
| 🔍 精准检索 | 中文语义优化切块，动态调整检索参数，低幻觉 |
| ⚙️ 参数可调 | 支持调整检索条数（K值）、回答随机性（温度） |
| 🌐 公网访问 | 部署到 Streamlit Cloud，随时随地使用 |
| 🔒 安全保障 | API 密钥加密存储，不暴露敏感信息 |

## 🛠️ 技术栈
- **核心框架**：Python / Streamlit
- **LLM 框架**：LangChain 1.0
- **向量数据库**：Chroma
- **文档解析**：PyPDF2 / python-docx
- **大模型**：OpenAI GPT-3.5-turbo
- **版本管理**：Git
- **部署平台**：Streamlit Cloud

## 📖 使用指南
### 1. 上传文件
- 支持上传 PDF/TXT/DOCX 格式文件（可多选）
- 点击「添加文件」完成加载，系统会自动处理文本并生成向量库

### 2. 调整参数（侧边栏）
- **检索条数（K值）**：3-10，值越大召回率越高，推荐5-7
- **回答随机性（温度）**：0.0-0.5，值越低回答越准确，推荐0.1

### 3. 开始对话
- 输入问题，支持多轮上下文对话
- 例如：先问「产品基础版价格是多少？」，再问「这个价格包含服务费吗？」

## 📊 核心优化点
### 1. 中文语义文本切块
- 按「\n\n、。、！、？、，、」等中文标点分割
- 避免切断完整中文句子，提升上下文完整性

### 2. 动态切块策略
- 短文本（<5000字）：chunk_size=500
- 中等文本（5000-20000字）：chunk_size=1000
- 长文本（>20000字）：chunk_size=2000

### 3. 低幻觉提示词工程
- 明确禁止模型编造信息
- 要求优先使用文档原文回答
- 未找到相关内容时明确告知用户

## 🚗 本地运行
### 环境准备
```bash
# 克隆仓库
git clone https://github.com/GrandYK/chatPDF.git
cd chatPDF

# 创建虚拟环境
conda create -n lc_learner python=3.10
conda activate lc_learner

# 安装依赖
pip install -r requirements.txt
