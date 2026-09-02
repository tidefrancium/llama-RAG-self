from typing import List
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import NodeWithScore
from config.settings import (
    OLLAMA_BASE_URL,
    CHAT_MODEL,
    SYSTEM_PROMPT
)

# 全局单例LLM，仅初始化一次，消除重复加载显存开销
_GLOBAL_LLM_INSTANCE = None

def get_generator_llm() -> Ollama:
    global _GLOBAL_LLM_INSTANCE
    if _GLOBAL_LLM_INSTANCE is None:
        llm = Ollama(
            model=CHAT_MODEL,
            base_url="http://localhost:11434",
            temperature=0.1,
            request_timeout=600,
            num_gpu=-1,
            num_thread=8,
            num_ctx=2048,
            tfs_z=1,
            low_cpu_mem_usage=True
        )
        _GLOBAL_LLM_INSTANCE = llm
    return _GLOBAL_LLM_INSTANCE

def build_context_from_nodes(nodes: List[NodeWithScore]) -> str:
    """拼接完整知识库上下文，修复元数据file_name→file_path"""
    context_parts = []
    for idx, node in enumerate(nodes):
        score = node.score
        # 修复：文档入库元数据是file_path，原file_name读取不到
        source_file = node.node.metadata.get("file_path", "未知来源")
        text = node.node.get_text().strip()
        chunk_text = f"""【片段{idx+1}｜来源:{source_file}｜匹配分数:{score:.4f}】
{text}
"""
        context_parts.append(chunk_text)
    return "\n\n=====分割线=====\n\n".join(context_parts)

def generate_answer(
    user_query: str,
    retrieved_nodes: List[NodeWithScore],
    enable_citation: bool = True
) -> str:
    llm = get_generator_llm()
    context_str = build_context_from_nodes(retrieved_nodes)

    # 精简Prompt，删除冗余换行，减少LLM生成token开销
    user_prompt = f"""
仅依据下方知识库内容作答，禁止编造不存在信息。
无相关内容直接输出：知识库未查询到相关内容。
{f"回答标注片段编号溯源。" if enable_citation else "无需标注来源"}
{SYSTEM_PROMPT}
【知识库】
{context_str}
【用户问题】{user_query}
【最终回答】
"""
    # 核心修复：additional_kwargs强制下发options，num_ctx=2048永久生效
    resp = llm.complete(
        prompt=user_prompt,
        additional_kwargs={
            "options": {
                "num_ctx": 2048,
                "num_gpu": -1
            }
        }
    )
    return resp.text.strip()

def rag_full_workflow(user_query: str, index, top_k=None, filter_key=None, filter_value=None, enable_citation=True):
    print("[DEBUG] 1. 开始初始化融合检索器")
    from retriever.retrieve_engine import get_raw_fusion_retriever
    fusion_retriever = get_raw_fusion_retriever(index, top_k=top_k, filter_key=filter_key, filter_value=filter_value)
    print("[DEBUG] 2. 执行检索")
    raw_retrieve_nodes = fusion_retriever.retrieve(user_query)
    print(f"[DEBUG] 3. 检索完成，原始节点数量：{len(raw_retrieve_nodes)}")

    # 读取入库预存完整origin_full_chunk，规避Fusion检索截断短句
    fixed_retrieve_nodes = []
    for idx, node_with_score in enumerate(raw_retrieve_nodes):
        node = node_with_score.node
        full_origin_text = node.metadata.get("origin_full_chunk", node.text).strip()
        node.text = full_origin_text
        fixed_retrieve_nodes.append(node_with_score)
    retrieve_nodes = fixed_retrieve_nodes
    answer = generate_answer(user_query, retrieve_nodes, enable_citation=enable_citation)

    return {
        "query": user_query,
        "answer": answer,
        "retrieve_nodes": retrieve_nodes
    }

# 本地自测入口
if __name__ == "__main__":
    from embedder.vector_embed import get_or_build_index
    vec_index = get_or_build_index()
    if vec_index is None:
        print("请先全量构建向量索引！")
    else:
        question = "Chunk切片核心公理是什么？"
        result = rag_full_workflow(user_query=question, index=vec_index, enable_citation=True)
        print(f"【用户问题】{result['query']}")
        print(f"【RAG回答】\n{result['answer']}")
        print(f"\n【召回有效片段数量】{len(result['retrieve_nodes'])}")
