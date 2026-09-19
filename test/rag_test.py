import sys
from pathlib import Path

# ========== RAG 生产环境核心路径初始化 ==========
file_full = Path(__file__).resolve()
ROOT_DIR = file_full.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import RAW_TXT_PATH
from embedder.vector_embed import build_vector_index, load_exist_vector_index
from generator.generate_answer import rag_full_workflow
from data.data_loder import load_all_documents_full, split_docs_into_nodes

# ========== DeepEval 评测模块导入 ==========
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)

# RAG 向量库路径
INDEX_CACHE_DIR = ROOT_DIR / "index_cache"
db_folder = ROOT_DIR / "config" / "data" / "db"
chroma_db_file = db_folder / "chroma.sqlite3"

# DeepEval 打分配置【修复URL笔误】
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
JUDGE_MODEL = "qwen3:1.7b"
SCORE_THRESHOLD = 0.5

# 生产环境评测问题
EVAL_QUERY = "LlamaIndex是什么"

# RAG 全局状态
APP_STATE = {
    "vec_index": None,
    "all_nodes": None
}

def auto_load_rag():
    if chroma_db_file.exists():
        APP_STATE["vec_index"] = load_exist_vector_index()
        return True
    return False

def build_rag():
    docs = load_all_documents_full()
    if not docs:
        return False
    nodes = split_docs_into_nodes(docs, semantic_mode=False)
    APP_STATE["vec_index"] = build_vector_index(nodes)
    APP_STATE["all_nodes"] = nodes
    return True

def rag_infer(question: str):
    vec_index = APP_STATE["vec_index"]
    if not vec_index or not question.strip():
        return "", []
    rag_res = rag_full_workflow(question, vec_index)
    answer = rag_res.get("answer", "")
    retrieve_nodes = rag_res.get("retrieve_nodes", [])
    contexts = [
        node_item.node.metadata.get("window", node_item.node.text)
        for node_item in retrieve_nodes
    ]
    return answer, contexts

# ===================== DeepEval 评测封装 =====================
def eval_rag_output(question: str, answer: str, contexts: list):
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=contexts,
        expected_output="LlamaIndex是一套开源检索增强生成开发框架，提供文档加载、文本分块、向量存储、混合检索、重排序等全套RAG组件，用于私有化知识库问答、智能聊天机器人搭建。"
    )

    judge_llm = OllamaModel(
        model=JUDGE_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0
    )
    gen_metrics = [
        AnswerRelevancyMetric(model=judge_llm, threshold=SCORE_THRESHOLD, async_mode=False),
        FaithfulnessMetric(model=judge_llm, threshold=SCORE_THRESHOLD, async_mode=False)
    ]
    ret_metrics = [
        ContextualPrecisionMetric(model=judge_llm, threshold=SCORE_THRESHOLD, async_mode=False),
        ContextualRecallMetric(model=judge_llm, threshold=SCORE_THRESHOLD, async_mode=False)
    ]

    print("=" * 65)
    print("📊 生产环境RAG系统评测报告")
    print("=" * 65)
    print(f"评测问题：{question}")
    print(f"RAG回答摘要：{answer[:80]}...\n")

    print("\n【生成侧 · 回答质量】")
    print("-" * 55)
    for metric in gen_metrics:
        metric.measure(test_case)
        status = "✅ 达标" if metric.is_successful() else "❌ 未达标"
        print(f"\n指标：{metric.__class__.__name__}")
        print(f"得分：{metric.score:.4f}  {status}")
        print(f"依据：{metric.reason}")

    print("\n" + "=" * 65)
    print("\n【检索侧 · 召回质量】")
    print("-" * 55)
    for metric in ret_metrics:
        metric.measure(test_case)
        status = "✅ 达标" if metric.is_successful() else "❌ 未达标"
        print(f"\n指标：{metric.__class__.__name__}")
        print(f"得分：{metric.score:.4f}  {status}")
        print(f"依据：{metric.reason}")

# ===================== 主入口 =====================
if __name__ == "__main__":
    if not auto_load_rag():
        print("未检测到本地向量库，启动全量构建...")
        if not build_rag():
            print("知识库构建失败，文档目录无有效文件")
            sys.exit(1)
        print("RAG知识库构建完成")
    else:
        print("RAG知识库加载完成")

    answer, contexts = rag_infer(EVAL_QUERY)
    if not answer:
        print("RAG推理失败，知识库未初始化")
        sys.exit(1)

    eval_rag_output(EVAL_QUERY, answer, contexts)
    print("\n" + "=" * 65)
    print("全链路评测完成")
