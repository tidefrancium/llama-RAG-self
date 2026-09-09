import sys
import os
import json
from pathlib import Path
# 项目根路径注入，解决config模块找不到
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.schema import TextNode
from config.settings import (
    OLLAMA_BASE_URL,
    CHAT_MODEL,
    TOP_N,
    SIMILAR_THRESHOLD,
    SYSTEM_PROMPT,
    BM25_CACHE_PATH,
    BASE_DIR
)
# ---------------------- 初始化全局问答LLM ----------------------
def get_chat_llm() -> LLM:
    llm = Ollama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        request_timeout=180
    )
    return llm
# ---------------------- 1. 查询澄清 ----------------------
def query_clarify(raw_query: str, llm: LLM) -> str:
    clarify_prompt = f"""
你是查询澄清助手，仅输出优化后的单条查询，不要多余解释。
规则：
1. 用户提问过短、模糊、有歧义时，补全完整语义，保留原问题核心；
2. 专业名词、缩写补充完整释义；
3. 若问题缺少限定条件，基于本地RAG知识库场景合理补全；
4. 输出仅一条优化后的查询语句，无其他文字。
用户原始问题：{raw_query}
优化后查询：
"""
    res = llm.complete(clarify_prompt)
    clean_query = res.text.strip()
    return clean_query if clean_query else raw_query
# ---------------------- 2. 查询重写 ----------------------
def query_rewrite(clear_query: str, llm: LLM) -> list[str]:
    rewrite_prompt = f"""
你是查询重写助手，基于下面问题生成3条语义相同、句式不同的检索问句，每条单独一行，不要编号、不要多余文字。
要求：替换同义词、变换语序、调整陈述/疑问句式，核心含义不变。
原始查询：{clear_query}
"""
    res = llm.complete(rewrite_prompt)
    rewrite_lines = [line.strip() for line in res.text.strip().split("\n") if line.strip()]
    query_list = list({clear_query} | set(rewrite_lines))
    return query_list

# ========== 新增最小改动：HyDE假设文档生成（查询构建扩展，仅新增函数，不修改原有逻辑） ==========
def query_hyde(clear_query: str, llm: LLM) -> str:
    hyde_prompt = f"""
针对下面RAG技术问题，生成一段完整客观的回答文本，无需严格贴合真实资料，仅用于语义检索增强。
问题：{clear_query}
回答：
"""
    hypo_result = llm.complete(hyde_prompt)
    return hypo_result.text.strip()
# =========================================================================================

# ---------------------- 3. BM25检索构建（读取本地nodes_cache.json） ----------------------
def build_bm25_retriever(index: VectorStoreIndex, top_k: int = None):
    use_topk = top_k if top_k is not None else TOP_N
    index_cache_path = Path(BASE_DIR) / "index_cache"
    node_json_path = index_cache_path / "nodes_cache.json"
    all_cached_nodes = None
    if node_json_path.exists():
        with open(node_json_path, "r", encoding="utf-8") as f:
            load_data = json.load(f)
            all_cached_nodes = [
                TextNode(text=item["text"], metadata=item["metadata"])
                for item in load_data
            ]
    if all_cached_nodes is None or len(all_cached_nodes) == 0:
        raise RuntimeError("索引无持久化nodes_cache.json，请删除index_cache、bm25_cache_dir、data/db，重新运行webui构建完整索引")
    # 修复：from_persist_path → from_persist_dir
    if os.path.isdir(BM25_CACHE_PATH):
        bm25_retriever = BM25Retriever.from_persist_dir(BM25_CACHE_PATH)
    else:
        bm25_retriever = BM25Retriever(
            nodes=all_cached_nodes,
            similarity_top_k=use_topk
        )
        bm25_retriever.persist(BM25_CACHE_PATH)
    return bm25_retriever
# ---------------------- 4. 混合检索构建（适配0.14.x新版参数） ----------------------
def build_fusion_retriever(
    index: VectorStoreIndex,
    top_k: int = None,
    filter_key: str = None,
    filter_value: str = None
) -> QueryFusionRetriever:
    use_topk = top_k if top_k is not None else TOP_N
    # 获取项目Ollama LLM，避免自动加载OpenAI
    llm = get_chat_llm()
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=use_topk,
        similarity_cutoff=SIMILAR_THRESHOLD
    )
    if filter_key and filter_value:
        filter_rules = MetadataFilters(filters=[ExactMatchFilter(key=filter_key, value=filter_value)])
        vector_retriever.filters = filter_rules
    bm25_retriever = build_bm25_retriever(index, use_topk)
    # 新增 llm=llm 关键参数，阻断OpenAI自动加载逻辑
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        num_queries=1,
        use_async=False,
        similarity_top_k=use_topk,
        mode="reciprocal_rerank",
        retriever_weights=[0.6, 0.4],
        llm=llm
    )
    return fusion_retriever
# ---------------------- 5. 基础问答引擎构建接口 ----------------------
def build_base_query_engine(
    index: VectorStoreIndex,
    top_k: int = None,
    filter_key: str = None,
    filter_value: str = None
):
    window_post_proc = MetadataReplacementPostProcessor(target_metadata_key="window")
    fusion_retriever = build_fusion_retriever(index, top_k, filter_key, filter_value)
    query_engine = index.as_query_engine(
        retriever=fusion_retriever,
        node_postprocessors=[window_post_proc],
        system_prompt=SYSTEM_PROMPT
    )
    return query_engine
# ---------------------- 6. 完整检索流水线入口（仅2行最小改动，其余原代码不动） ----------------------
def full_retrieve_pipeline(
    raw_user_query: str,
    index: VectorStoreIndex,
    top_k: int = None,
    filter_key: str = None,
    filter_value: str = None
) -> str:
    llm = get_chat_llm()
    clarified_q = query_clarify(raw_user_query, llm)
    query_candidates = query_rewrite(clarified_q, llm)
    
    # 改动1：最小集成HyDE查询构建，将假设文档加入检索候选
    hypo_document = query_hyde(clarified_q, llm)
    query_candidates.append(hypo_document)

    # 改动2：引擎初始化移出循环，消除重复构建开销，无业务逻辑变更
    qe = build_base_query_engine(index, top_k, filter_key, filter_value)
    
    all_answers = []
    for q in query_candidates:
        ans = qe.query(q)
        all_answers.append(str(ans))
    merge_prompt = f"""
整合下面多条知识库检索答案，合并重复信息，统一输出一份简洁完整回答；无相关内容则严格回复：知识库未查询到相关内容。
多条检索回答片段：
{all_answers}
用户原始问题：{raw_user_query}
整合后的最终回答：
"""
    final_resp = llm.complete(merge_prompt)
    return final_resp.text.strip()
# ---------------------- 7. 对外获取融合检索器接口 ----------------------
def get_raw_fusion_retriever(index: VectorStoreIndex, top_k: int = None, filter_key=None, filter_value=None):
    return build_fusion_retriever(index, top_k, filter_key, filter_value)
