import streamlit as st
from openai import OpenAI  # openai 1.0+ 新导入方式
import os
from PyPDF2 import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 页面标题
st.title("极简RAG - 最终版（适配openai 1.0+ / LangChain 1.0+）")


# ========== 替换成从秘钥读取（部署关键） ==========
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_BASE_URL = st.secrets["OPENAI_BASE_URL"]
# ===================================


# ----------------------
# 初始化openai 1.0+ 客户端（调用官方库的OpenAI类）
# ----------------------
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL  # 1.0+ 版本这么配置base_url
)

# 初始化Streamlit会话状态，保存已上传的文件（核心兼容逻辑）
if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []  # 存储已上传的文件
if "all_text" not in st.session_state:
    st.session_state.all_text = ""  # 存储所有文件的文本
if "chat_history" not in st.session_state:  # 新增：对话历史记忆
    st.session_state.chat_history = []  # 格式：[(问题1, 答案1), (问题2, 答案2), ...]
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
# ----------------------
# 多格式文件加载函数
# ----------------------
def load_file(file):
    """支持PDF/TXT/DOCX三种格式的文件加载"""
    file_ext = os.path.splitext(file.name)[1].lower()  # 获取文件后缀
    text = ""
    
    # 1. 处理PDF
    if file_ext == ".pdf":
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # 防止None
    
    # 2. 处理TXT
    elif file_ext == ".txt":
        # 按UTF-8读取，兼容中文
        text = file.read().decode("utf-8")
    
    # 3. 处理DOCX
    elif file_ext == ".docx":
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    
    else:
        st.error(f"不支持的文件格式：{file_ext}，仅支持PDF/TXT/DOCX")
        return ""
    
    return text

# ----------------------
# 核心RAG逻辑
# ----------------------
def init_vector_db(all_text):
    """初始化向量库（只执行一次，优化性能）"""
    if not all_text:
        return None
    # 优化1：动态切块（按文本长度调整，适配不同文件）
    text_length = len(all_text)
    if text_length < 5000:
        chunk_size = 500   # 短文本：小块，更精准
        chunk_overlap = 100
    elif text_length < 20000:
        chunk_size = 1000  # 中等文本：默认大小
        chunk_overlap = 200
    else:
        chunk_size = 2000  # 长文本：大块，减少检索次数
        chunk_overlap = 300

    # 优化2：按中文标点切块，更符合中文语义
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、"]  
    )
    texts = splitter.split_text(all_text)
    
    embedding = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL
    )
    db = Chroma.from_texts(texts, embedding)
    return db

# 优化3：默认检索5条，可调整
def rag_answer(vector_db, question, chat_history, top_k=5):  
    if not vector_db:
        st.error("向量库未初始化！请先上传文件。")
        return ""
    
    # 优化4：混合检索（相似度+关键词），提升召回率
    relevant_docs = vector_db.similarity_search(
        query=question,
        k=top_k,
        # 新增：关键词过滤，优先匹配包含核心词的文档
        filter=None  # 进阶可加关键词过滤，这里简化
    )
    
    # 优化5：按相关性排序，拼接上下文（只保留前3条核心内容，避免提示词过长）
    context = ""
    for idx, doc in enumerate(relevant_docs[:3]):
        context += f"相关内容{idx+1}：\n{doc.page_content}\n\n"
    
    # 拼接对话历史
    history_text = ""
    if chat_history:
        # 优化6：只保留最近3轮历史，避免提示词过长导致模型混乱
        recent_history = chat_history[-3:]
        for idx, (q, a) in enumerate(recent_history):
            history_text += f"历史对话{idx+1}：\n用户：{q}\n助手：{a}\n\n"
    
    # 优化7：更精准的提示词（减少幻觉）
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.1,  # 降低随机性，提升准确性
        messages=[
            {"role": "system", "content": """你是一个基于文档的问答助手，严格遵守以下规则：
                1. 仅使用提供的「相关内容」回答问题，绝对不编造任何信息；
                2. 如果相关内容中没有答案，明确说明「未在文档中找到相关信息」；
                3. 结合历史对话理解上下文，回答要简洁、准确、有针对性；
                4. 优先使用文档中的原文表述，避免意译导致偏差。"""},
            {"role": "user", "content": 
            f"""相关内容（仅基于此回答）：
                {context}

                历史对话（仅用于理解上下文）：
                {history_text}

                当前问题：
                {question}"""}
        ]
    )
    return response.choices[0].message.content

# ----------------------
# Streamlit交互界面
# ----------------------
# 侧边栏：检索优化参数（让用户可调整，体现灵活性）
with st.sidebar:
    st.subheader("检索优化参数")
    top_k = st.slider("检索条数（K值）", min_value=3, max_value=10, value=5, help="值越大召回率越高，但提示词越长")
    temp = st.slider("回答随机性（温度）", min_value=0.0, max_value=0.5, value=0.1, help="值越低回答越准确")

# 1. 单文件上传组件（旧版支持）
uploaded_file = st.file_uploader(
    "选择要上传的文件（支持PDF/TXT/DOCX）",
    type=["pdf", "txt", "docx"]
)

# 2. 添加文件按钮：把当前选择的文件加入列表
if st.button("添加文件") and uploaded_file:
    # 检查是否已上传过同名文件（避免重复）
    file_names = [f.name for f in st.session_state.uploaded_files_list]
    if uploaded_file.name in file_names:
        st.warning(f"文件 {uploaded_file.name} 已上传，无需重复添加！")
    else:
        # 加载文件文本并保存
        file_text = load_file(uploaded_file)
        if file_text:
            st.session_state.uploaded_files_list.append(uploaded_file)
            st.session_state.all_text += file_text + "\n\n"  # 拼接文本
            # 重新初始化向量库（优化性能：只在文件变化时执行）
            st.session_state.vector_db = init_vector_db(st.session_state.all_text)
            st.success(f"成功添加：{uploaded_file.name}")

# 3. 显示已上传的文件列表
if st.session_state.uploaded_files_list:
    st.subheader("已上传文件：")
    for idx, f in enumerate(st.session_state.uploaded_files_list):
        st.write(f"{idx+1}. {f.name}")
    
    # 清空文件按钮（可选）
    col_clear1, col_clear2 = st.columns(2)
    with col_clear1:
        if st.button("清空所有文件"):
            st.session_state.uploaded_files_list = []
            st.session_state.all_text = ""
            st.session_state.vector_db = None
            st.success("已清空所有文件和向量库！")
    with col_clear2:
        if st.button("清空对话历史") and st.session_state.chat_history:
            st.session_state.chat_history = []
            st.success("已清空对话历史！")

    # 4. 显示对话历史
    if st.session_state.chat_history:
        st.subheader("对话历史：")
        # 用聊天气泡样式显示，更直观
        for q, a in st.session_state.chat_history:
            st.chat_message("user").write(q)  # 用户问题（蓝色气泡）
            st.chat_message("assistant").write(a)  # 助手回答（灰色气泡）

    # 5. 提问功能
    question = st.text_input("请输入你的问题（支持上下文+精准检索）：")
    if st.button("获取答案") and question:
        with st.spinner("正在精准检索并生成答案..."):
            try:
                # 传入自定义检索参数
                answer = rag_answer(
                    vector_db=st.session_state.vector_db,
                    question=question,
                    chat_history=st.session_state.chat_history,
                    top_k=top_k
                )
                st.success("回答：\n" + answer)
                st.session_state.chat_history.append((question, answer))
            except Exception as e:
                st.error(f"运行出错：{str(e)}")
                st.code(f"详细错误信息：{repr(e)}", language="text")
else:
    st.info("请选择文件并点击「添加文件」，可多次添加不同格式的文件")

# 清理临时文件
if os.path.exists("temp.pdf"):
    os.remove("temp.pdf")