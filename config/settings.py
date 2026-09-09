import os

# 以当前settings.py文件位置为基准根目录，解决Windows相对路径乱码/错位问题
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 文档&存储路径配置
RAW_TXT_PATH = os.path.join(BASE_DIR, "data")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "data", "db")

# Ollama服务与模型配置
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "quentinz/bge-base-zh-v1.5"
CHAT_MODEL = "gemma3:1b"

# 文档切片参数
CHUNK_MAX_CHAR = 200
CHUNK_STEP = 100
SEMANTIC_BUFFER_SIZE = 1
SEMANTIC_BREAKPOINT_THRESHOLD = 90

# 向量入库、检索控制参数
BATCH_INSERT_SIZE = 20
TOP_N = 4
SIMILAR_THRESHOLD = 0.65

# 本地缓存文件路径（增量更新避免全量重切、重向量化）
BM25_CACHE_PATH = os.path.join(BASE_DIR, "data", "bm25_cache.pkl")
FILE_MD5_CACHE_PATH = os.path.join(BASE_DIR, "data", "file_md5_cache.pkl")

# 问答系统提示词
SYSTEM_PROMPT = """
你是本地知识库问答助手，请严格依据下方检索到的参考文档回答用户问题。
1. 如果参考文档无对应信息，直接回复：知识库未查询到相关内容，不要编造答案；
2. 回答简洁清晰，优先引用文档原文；
3. 禁止脱离上下文凭空推理、虚构数据。
参考上下文：
{context_str}
用户问题：{query_str}
"""
