import os
import json
import shutil
from pathlib import Path
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from config.settings import (
    OLLAMA_BASE_URL,
    EMBED_MODEL,
    VECTOR_DB_PATH,
    BASE_DIR,
    BATCH_INSERT_SIZE
)
from llama_index.embeddings.ollama import OllamaEmbedding

def get_ollama_embed_model() -> OllamaEmbedding:
    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
        ollama_additional_kwargs={
            "mirostat": 0,
            "temperature": 0.0
        },
        request_timeout=180
    )
    Settings.embed_model = embed_model
    return embed_model

def get_chroma_vector_store(collection_name: str = "local_knowledge_base") -> ChromaVectorStore:
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    chroma_collection = chroma_client.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store

# 修复1：统一缓存目录 config/index_cache
def get_index_cache_dir() -> str:
    cache_dir = os.path.join(BASE_DIR, "config", "index_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def build_vector_index(nodes, collection_name: str = "local_knowledge_base") -> VectorStoreIndex:
    if not nodes:
        raise ValueError("构建索引必须传入分块后的文档节点nodes")
    embed_model = get_ollama_embed_model()
    vec_store = get_chroma_vector_store(collection_name)
    index_cache_path = get_index_cache_dir()
    cache_p = Path(index_cache_path)
    if cache_p.exists():
        shutil.rmtree(cache_p)
    cache_p.mkdir(parents=True, exist_ok=True)

    # 修复2：开启store_nodes_override，强制将完整Node存入docstore，load_index_from_storage可正常读取
    index = VectorStoreIndex(
        nodes,
        vector_store=vec_store,
        embed_model=embed_model,
        insert_batch_size=BATCH_INSERT_SIZE,
        show_progress=True,
        store_nodes_override=True
    )
    index.storage_context.persist(persist_dir=index_cache_path)

    node_save_path = Path(index_cache_path) / "nodes_cache.json"
    dump_data = [{"text": n.text, "metadata": n.metadata} for n in nodes]
    with open(node_save_path, "w", encoding="utf-8") as f:
        json.dump(dump_data, f, ensure_ascii=False, indent=2)
    coll = vec_store._collection
    vec_count = coll.count()
    print(f"索引构建完成，缓存路径：{index_cache_path}，库内向量总数：{vec_count}")
    return index

def load_exist_vector_index(collection_name: str = "local_knowledge_base") -> VectorStoreIndex | None:
    embed_model = get_ollama_embed_model()
    index_cache_path = get_index_cache_dir()
    index_store_file = os.path.join(index_cache_path, "index_store.json")
    # 修复3：修正文件名 docstore.json，原代码写doc_store.json拼写错误
    doc_store_file = os.path.join(index_cache_path, "docstore.json")

    # 校验缓存文件存在
    if not (os.path.exists(index_store_file) and os.path.exists(doc_store_file)):
        print(f"⚠️ 元缓存缺失：{doc_store_file} / {index_store_file}，请执行【完全重构数据库】")
        return None
    # 拦截空文件
    fsize = Path(doc_store_file).stat().st_size
    if fsize <= 10:
        print(f"⚠️ docstore.json为空文件（大小{fsize}字节），缓存损坏，请完全重构")
        return None
    try:
        vec_store = get_chroma_vector_store(collection_name)
        storage_ctx = StorageContext.from_defaults(
            persist_dir=index_cache_path,
            vector_store=vec_store
        )
        index = load_index_from_storage(storage_ctx)
        coll = storage_ctx.vector_store._collection
        vec_count = coll.count()
        print(f"✅ 成功加载 config/index_cache 持久化向量索引，集合向量总数：{vec_count}")
        return index
    except Exception as e:
        print(f"❌ 索引加载异常，缓存损坏：{str(e)}")
        return None

def batch_insert_new_nodes(index: VectorStoreIndex, new_nodes, collection_name: str = "local_knowledge_base"):
    if not new_nodes:
        return
    index_cache_path = get_index_cache_dir()
    storage_ctx = index.storage_context
    for i in range(0, len(new_nodes), BATCH_INSERT_SIZE):
        batch_nodes = new_nodes[i:i + BATCH_INSERT_SIZE]
        for node in batch_nodes:
            index.insert(node)
    storage_ctx.persist(persist_dir=index_cache_path)
    node_save_path = Path(index_cache_path) / "nodes_cache.json"
    old_nodes = []
    if os.path.exists(node_save_path):
        with open(node_save_path, "r", encoding="utf-8") as f:
            old_nodes = json.load(f)
    new_dump = [{"text": n.text, "metadata": n.metadata} for n in new_nodes]
    all_dump = old_nodes + new_dump
    with open(node_save_path, "w", encoding="utf-8") as f:
        json.dump(all_dump, f, ensure_ascii=False, indent=2)
    coll = get_chroma_vector_store(collection_name)._collection
    total_vec = coll.count()
    print(f"✅ 增量插入 {len(new_nodes)} 个节点完成，当前总向量数：{total_vec}")

def get_or_build_index(nodes=None):
    index = load_exist_vector_index()
    if index is not None:
        return index
    if nodes is None:
        raise RuntimeError("无本地完整索引缓存，请点击【完全重构数据库】构建知识库")
    return build_vector_index(nodes)

if __name__ == "__main__":
    embed = get_ollama_embed_model()
    test_vec = embed.get_text_embedding("向量存储测试文本")
    print(f"嵌入向量维度：{len(test_vec)}")
    print("嵌入模型、Chroma向量库初始化正常")
